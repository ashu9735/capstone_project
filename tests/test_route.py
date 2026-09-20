"""The routing decision. (AC-A5)"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.ingest import normalise
from src.route import ALWAYS_ESCALATE_INTENTS, Reason, decide
from src.schemas import Action, Classification


@pytest.fixture
def base_settings():
    return Settings(confidence_threshold=0.80, enable_llm=False)


def _ticket(**overrides):
    raw = {"ticket_id": "T-1", "channel": "email", "subject": "s", "body": "a real problem to solve", **overrides}
    return normalise(raw)


def test_confident_and_grounded_ticket_is_answered(base_settings, passages, confident):
    decision = decide(_ticket(), confident, passages, base_settings)
    assert decision.action == Action.AUTO_RESPOND.value
    assert decision.reason == Reason.AUTO_RESPOND


def test_low_confidence_escalates(base_settings, passages, unconfident):
    decision = decide(_ticket(), unconfident, passages, base_settings)
    assert decision.escalate
    assert decision.reason == Reason.LOW_CONFIDENCE


def test_confidence_exactly_at_threshold_is_answered(base_settings, passages):
    classification = Classification(intent="billing_query", urgency="low", confidence=0.80)
    assert decide(_ticket(), classification, passages, base_settings).action == Action.AUTO_RESPOND.value


def test_no_retrieval_escalates(base_settings, confident):
    decision = decide(_ticket(), confident, [], base_settings)
    assert decision.escalate
    assert decision.reason == Reason.NO_RETRIEVAL


@pytest.mark.parametrize("intent", sorted(ALWAYS_ESCALATE_INTENTS))
def test_always_escalate_intents_ignore_confidence(base_settings, passages, intent):
    classification = Classification(intent=intent, urgency="low", confidence=0.99)
    decision = decide(_ticket(), classification, passages, base_settings)
    assert decision.escalate
    assert decision.reason == Reason.ALWAYS_ESCALATE_INTENT


def test_must_not_auto_respond_overrides_everything(base_settings, passages, confident):
    ticket = _ticket(labels={"must_not_auto_respond": True})
    decision = decide(ticket, confident, passages, base_settings)
    assert decision.escalate
    assert decision.reason == Reason.MUST_NOT_AUTO_RESPOND


def test_kill_switch_escalates_everything(passages, confident):
    settings = Settings(confidence_threshold=0.80, kill_switch=True, enable_llm=False)
    decision = decide(_ticket(), confident, passages, settings)
    assert decision.escalate
    assert decision.reason == Reason.KILL_SWITCH


def test_routing_is_deterministic(base_settings, passages, confident):
    first = decide(_ticket(), confident, passages, base_settings)
    for _ in range(20):
        assert decide(_ticket(), confident, passages, base_settings) == first
