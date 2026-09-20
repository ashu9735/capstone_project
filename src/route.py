"""The escalation decision. (AC-A5, REQ-F04)

Deliberately a pure function of already-computed facts: no model call, no I/O, no
randomness. Routing is the decision a human will be asked to defend, so it has to be
reproducible from the log alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.config import Settings, get_settings
from src.schemas import Action, Classification, NormalisedTicket, RetrievedPassage

# These are never answered automatically regardless of confidence or retrieval quality.
# security_incident and compliance_request carry legal exposure; feature_request and
# unclear_request have no documentation that can resolve them (0% doc coverage).
ALWAYS_ESCALATE_INTENTS: frozenset[str] = frozenset(
    {"security_incident", "compliance_request", "feature_request", "unclear_request"}
)


class Reason:
    KILL_SWITCH = "kill_switch_engaged"
    MUST_NOT_AUTO_RESPOND = "ticket_flagged_must_not_auto_respond"
    ALWAYS_ESCALATE_INTENT = "intent_always_escalates"
    LOW_CONFIDENCE = "confidence_below_threshold"
    NO_RETRIEVAL = "no_supporting_documentation"
    GENERATION_FAILED = "generation_unavailable"
    INSUFFICIENT_CONTEXT = "model_declared_insufficient_context"
    GUARDRAIL_BLOCK = "guardrail_blocked_response"
    PIPELINE_ERROR = "pipeline_error"
    AUTO_RESPOND = "answerable_from_documentation"


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    reason: str
    threshold: float
    detail: str = ""

    @property
    def escalate(self) -> bool:
        return self.action == Action.ESCALATE.value


def decide(
    ticket: NormalisedTicket,
    classification: Classification,
    passages: Sequence[RetrievedPassage],
    settings: Settings | None = None,
) -> RoutingDecision:
    settings = settings or get_settings()
    threshold = settings.confidence_threshold

    if settings.kill_switch:
        return RoutingDecision(Action.ESCALATE.value, Reason.KILL_SWITCH, threshold,
                               "automatic responses disabled by operator")

    if ticket.must_not_auto_respond:
        return RoutingDecision(Action.ESCALATE.value, Reason.MUST_NOT_AUTO_RESPOND, threshold,
                               "ticket is marked as requiring a human")

    if classification.intent in ALWAYS_ESCALATE_INTENTS:
        return RoutingDecision(Action.ESCALATE.value, Reason.ALWAYS_ESCALATE_INTENT, threshold,
                               f"intent {classification.intent} is on the always-escalate list")

    if not passages:
        return RoutingDecision(Action.ESCALATE.value, Reason.NO_RETRIEVAL, threshold,
                               "no documentation passage cleared the relevance floor")

    if classification.confidence < threshold:
        return RoutingDecision(Action.ESCALATE.value, Reason.LOW_CONFIDENCE, threshold,
                               f"confidence {classification.confidence:.2f} < threshold {threshold:.2f}")

    return RoutingDecision(Action.AUTO_RESPOND.value, Reason.AUTO_RESPOND, threshold,
                           f"confidence {classification.confidence:.2f} with "
                           f"{len(passages)} supporting passage(s)")
