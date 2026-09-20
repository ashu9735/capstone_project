"""Intent, urgency and confidence. (AC-A3, REQ-F02)

A keyword classifier backs the model call so that a provider outage degrades the
system's accuracy rather than stopping it.
"""

from __future__ import annotations

import json
import logging

from src.config import PROJECT_ROOT, Settings, get_settings
from src.llm import LLMClient, LLMUnavailable
from src.prompts import load_prompt
from src.schemas import Classification, NormalisedTicket, Urgency

log = logging.getLogger(__name__)

INTENTS: tuple[str, ...] = (
    "account_access",
    "api_key_issue",
    "api_usage_question",
    "authentication_failure",
    "billing_query",
    "compliance_request",
    "configuration_help",
    "data_export",
    "data_residency",
    "database_issue",
    "deployment_failure",
    "feature_request",
    "integration_help",
    "onboarding",
    "performance_degradation",
    "quota_or_overage",
    "rate_limit",
    "rollback_request",
    "security_incident",
    "sso_configuration",
    "unclear_request",
    "webhook_issue",
)

URGENCIES: tuple[str, ...] = tuple(u.value for u in Urgency)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "security_incident": ("breach", "compromised", "unauthorised access", "unauthorized access", "leaked", "intrusion", "phishing"),
    "compliance_request": ("gdpr", "audit", "compliance", "data processing agreement", "retention policy", "soc 2", "auditor"),
    "data_residency": ("data residency", "stored in", "region requirement", "sovereignty", "hosted in eu"),
    "authentication_failure": ("invalid credential", "cannot log in", "can't log in", "login fail", "mfa", "authenticator", "locked out"),
    "sso_configuration": ("sso", "saml", "identity provider", "single sign-on", "single sign on"),
    "api_key_issue": ("api key", "token expired", "rotate key", "revoke key", "scope"),
    "rate_limit": ("rate limit", "429", "throttl", "too many requests"),
    "quota_or_overage": ("quota", "overage", "usage limit", "exceeded the limit", "spend cap"),
    "billing_query": ("invoice", "charged", "billing", "refund", "payment", "subscription", "price"),
    "deployment_failure": ("deploy", "build fail", "pipeline fail", "dependency resolution", "rollout failed"),
    "rollback_request": ("roll back", "rollback", "revert to previous", "previous version"),
    "performance_degradation": ("slow", "latency", "timeout", "degraded", "performance"),
    "database_issue": ("database", "postgres", "mysql", "query fail", "connection pool", "deadlock"),
    "webhook_issue": ("webhook", "callback url", "event delivery", "not receiving events"),
    "integration_help": ("integrat", "connect to", "third party", "plugin"),
    "data_export": ("export", "download my data", "csv of", "extract records"),
    "account_access": ("permission", "role", "access to the project", "team member", "invite", "group membership"),
    "onboarding": ("getting started", "new to", "set up my account", "onboard", "first project"),
    "configuration_help": ("configure", "setting", "how do i set", "environment variable"),
    "api_usage_question": ("endpoint", "api call", "sdk", "documentation for the api", "request body"),
    "feature_request": ("would be useful", "feature request", "please add", "it would be great if", "roadmap", "suggestion"),
}

_HIGH_URGENCY = (
    "down", "outage", "production is", "cannot work", "can't work", "blocked",
    "urgent", "critical", "data loss", "breach", "immediately", "asap", "emergency",
)
_MEDIUM_URGENCY = ("deadline", "by friday", "intermittent", "failing", "broken", "error", "not working")

_CONFIDENCE_FLOOR = 0.05
_CONFIDENCE_CEILING = 0.99


def _extract_confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        conf = conf / 100.0 if conf <= 100.0 else 1.0
    return max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, conf))


def _coerce_intent(value: object) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in INTENTS:
        return text
    for intent in INTENTS:
        if intent in text or text in intent:
            return intent
    return None


def classify_by_keywords(ticket: NormalisedTicket) -> Classification:
    """Deterministic fallback. Deliberately low confidence so it routes to a human."""
    text = ticket.text.lower()
    scores: dict[str, int] = {}
    for intent, keywords in _KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            scores[intent] = hits

    if scores:
        intent = max(scores, key=lambda k: (scores[k], -INTENTS.index(k)))
        confidence = min(0.55, 0.30 + 0.10 * scores[intent])
    else:
        intent, confidence = "unclear_request", 0.20

    if any(kw in text for kw in _HIGH_URGENCY):
        urgency = Urgency.HIGH.value
    elif any(kw in text for kw in _MEDIUM_URGENCY):
        urgency = Urgency.MEDIUM.value
    else:
        urgency = Urgency.LOW.value

    return Classification(
        intent=intent,
        urgency=urgency,
        confidence=round(confidence, 3),
        rationale="keyword fallback classifier",
        source="keyword",
    )


class Classifier:
    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.prompt = load_prompt("build/classify_ticket.md")
        self._calibration = _load_calibration() if self.settings.use_calibration else []

    @property
    def prompt_version(self) -> str:
        return f"{self.prompt.name}@{self.prompt.version}"

    def classify(self, ticket: NormalisedTicket) -> Classification:
        result = self._classify_uncalibrated(ticket)
        result.confidence = self._apply_calibration(result.confidence)
        return result

    def _classify_uncalibrated(self, ticket: NormalisedTicket) -> Classification:
        if not self.llm.available:
            return classify_by_keywords(ticket)
        system = self.prompt.render(intents="\n".join(f"- {i}" for i in INTENTS))
        user = f"Channel: {ticket.channel}\nSubject: {ticket.subject}\n\n{ticket.body}"
        try:
            payload = self.llm.complete_json(system, user, max_tokens=250)
        except (LLMUnavailable, ValueError) as exc:
            log.warning("classification fell back to keywords for %s: %s", ticket.ticket_id, exc)
            fallback = classify_by_keywords(ticket)
            fallback.rationale = f"keyword fallback ({type(exc).__name__})"
            return fallback

        intent = _coerce_intent(payload.get("intent"))
        if intent is None:
            log.warning("model returned unknown intent %r for %s", payload.get("intent"), ticket.ticket_id)
            fallback = classify_by_keywords(ticket)
            fallback.rationale = "keyword fallback (unrecognised intent)"
            return fallback

        urgency = str(payload.get("urgency", "")).strip().lower()
        if urgency not in URGENCIES:
            urgency = classify_by_keywords(ticket).urgency

        confidence = _extract_confidence(payload.get("confidence"))
        return Classification(
            intent=intent,
            urgency=urgency,
            confidence=confidence,
            rationale=str(payload.get("rationale", ""))[:300],
            source="llm",
        )

    def _apply_calibration(self, confidence: float) -> float:
        """Map stated confidence onto accuracy observed on the development set."""
        if not self._calibration:
            return round(confidence, 3)
        for band in self._calibration:
            in_band = band["low"] <= confidence < band["high"]
            if in_band or (confidence >= 1.0 and band["high"] >= 1.0):
                return round(max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, band["observed"])), 3)
        return round(confidence, 3)


CALIBRATION_FILE = PROJECT_ROOT / "evaluation" / "calibration.json"


def _load_calibration() -> list[dict[str, float]]:
    if not CALIBRATION_FILE.exists():
        return []
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        return [b for b in data.get("bands", []) if b.get("count", 0) >= 10]
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("ignoring unreadable calibration file: %s", exc)
        return []
