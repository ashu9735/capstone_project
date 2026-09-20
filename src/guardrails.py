"""Checks that can block a response before it reaches a customer. (AC-A7, REQ-F07)

Two phases. Input guardrails run on the incoming ticket and can force escalation.
Output guardrails run on the drafted reply and can block it outright; a blocked
reply is never sent, the ticket escalates, and the block is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.schemas import GuardrailVerdict, NormalisedTicket, RetrievedPassage

# --- Detectors ---------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email_address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "payment_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "api_key_or_token": re.compile(
        r"\b(?:sk|pk|rk|ghp|gho|xox[baprs])[-_][A-Za-z0-9_\-]{16,}\b"
        r"|\bBearer\s+[A-Za-z0-9._\-]{20,}\b"
        r"|\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
    ),
    "password_disclosure": re.compile(r"\b(?:password|passphrase|secret)\s*(?:is|=|:)\s*\S{4,}", re.IGNORECASE),
    "national_id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Claims the system is never permitted to make. Taken from the must_not_claim field
# that appears on every reference response in ground_truth_responses.json.
FORBIDDEN_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refund_issued", re.compile(r"\b(?:refund(?:ed)?\s+(?:has been|was|is)\s+(?:issued|processed|applied)|we have refunded|i have refunded|your refund is on its way)\b", re.IGNORECASE)),  # noqa: E501
    ("fixed_on_our_side", re.compile(r"\b(?:(?:this|the issue|the problem|it)\s+(?:has been|is now|was)\s+(?:fixed|resolved|corrected)\s+(?:on our (?:side|end)|by (?:us|our team))|we have (?:now )?fixed (?:this|the issue))\b", re.IGNORECASE)),  # noqa: E501
    ("promised_fix_date", re.compile(r"\b(?:fix(?:ed)?|resolv(?:ed|ution)|patch(?:ed)?|releas(?:e|ed)|deploy(?:ed)?|ship(?:ped|ping)?|available)\b[^.\n]{0,60}?(?:\b(?:by|on|within|in)\s+)?\b(?:the\s+)?(?:next\s+|coming\s+|end\s+of\s+(?:the\s+)?)?(?:\d+\s+(?:hours?|days?|weeks?|months?)|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|week|month|quarter|q[1-4]|january|february|march|april|june|july|august|september|october|november|december)\b", re.IGNORECASE)),  # noqa: E501
    ("compensation_promise", re.compile(r"\b(?:we will|i will|we'll|i'll)\s+(?:credit|compensate|reimburse)\b", re.IGNORECASE)),
)

PROMPT_INJECTION = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"
    r"|disregard\s+(?:your|the)\s+(?:instructions|rules|system prompt)"
    r"|you\s+are\s+now\s+(?:a|an|in)\b"
    r"|reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions)"
    r"|print\s+(?:your|the)\s+system\s+prompt"
    r"|act\s+as\s+(?:a\s+)?(?:developer|dan|jailbreak))\b",
    re.IGNORECASE,
)

REDACTION = "[REDACTED]"


@dataclass
class GuardrailOutcome:
    verdicts: list[GuardrailVerdict]
    text: str

    @property
    def blocked(self) -> bool:
        return any(v.blocked for v in self.verdicts)

    @property
    def block_names(self) -> list[str]:
        return [v.name for v in self.verdicts if v.blocked]


def detect_pii(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text or "")
        if matches:
            found[name] = [m if isinstance(m, str) else str(m) for m in matches]
    return found


def redact_pii(text: str) -> tuple[str, list[str]]:
    redacted, names = text, []
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            names.append(name)
            redacted = pattern.sub(REDACTION, redacted)
    return redacted, names


# --- Input phase -------------------------------------------------------------------

def check_input(ticket: NormalisedTicket) -> GuardrailOutcome:
    verdicts: list[GuardrailVerdict] = []

    injection = bool(PROMPT_INJECTION.search(ticket.text))
    verdicts.append(
        GuardrailVerdict(
            name="prompt_injection",
            passed=not injection,
            blocked=injection,
            detail="instruction-override language in ticket body" if injection else "",
        )
    )

    found = detect_pii(ticket.text)
    # Inbound PII is the customer's own and is not a failure; it must simply never
    # be echoed back, so it is recorded here and enforced on the output side.
    verdicts.append(
        GuardrailVerdict(
            name="inbound_pii_present",
            passed=True,
            blocked=False,
            detail=", ".join(sorted(found)) if found else "",
        )
    )
    return GuardrailOutcome(verdicts=verdicts, text=ticket.text)


# --- Output phase ------------------------------------------------------------------

def check_output(
    response_text: str,
    passages: Sequence[RetrievedPassage],
    citations: Sequence[str],
) -> GuardrailOutcome:
    verdicts: list[GuardrailVerdict] = []
    text = response_text

    text, redacted_kinds = redact_pii(text)
    verdicts.append(
        GuardrailVerdict(
            name="pii_leak",
            passed=not redacted_kinds,
            blocked=bool(redacted_kinds),
            detail=f"redacted and blocked: {', '.join(redacted_kinds)}" if redacted_kinds else "",
        )
    )

    hits = [name for name, pattern in FORBIDDEN_CLAIMS if pattern.search(text)]
    verdicts.append(
        GuardrailVerdict(
            name="forbidden_claim",
            passed=not hits,
            blocked=bool(hits),
            detail=", ".join(hits),
        )
    )

    allowed = {p.doc_id for p in passages}
    unresolved = [c for c in citations if c not in allowed]
    verdicts.append(
        GuardrailVerdict(
            name="citation_resolves",
            passed=not unresolved,
            blocked=bool(unresolved),
            detail=f"citations not in retrieved set: {', '.join(unresolved)}" if unresolved else "",
        )
    )

    uncited = bool(text.strip()) and not citations
    verdicts.append(
        GuardrailVerdict(
            name="grounding_required",
            passed=not uncited,
            blocked=uncited,
            detail="reply makes claims with no citation" if uncited else "",
        )
    )

    empty = len(text.strip()) < 40
    verdicts.append(
        GuardrailVerdict(
            name="substantive_reply",
            passed=not empty,
            blocked=empty,
            detail="reply too short to be useful" if empty else "",
        )
    )

    return GuardrailOutcome(verdicts=verdicts, text=text)
