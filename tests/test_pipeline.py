"""End-to-end pipeline behaviour with the model provider stubbed. (AC-A3, A8, A9, A11)"""

from __future__ import annotations

import pytest

from src.classify import INTENTS, classify_by_keywords
from src.llm import parse_json_object
from src.logging_store import DecisionLog
from src.pipeline import Pipeline
from src.retrieve import build_index
from src.route import Reason
from src.schemas import Action, Stage
from tests.stubs import StubLLM


@pytest.fixture(scope="module")
def indexed(settings):
    build_index(settings=settings)
    return settings


def _pipeline(settings, tmp_path, **stub):
    log = DecisionLog(settings, path=tmp_path / "decisions.db")
    return Pipeline(settings=settings, decision_log=log, llm=StubLLM(settings, **stub))


class TestClassification:
    def test_keyword_fallback_returns_a_known_intent(self, ticket_by_id):
        result = classify_by_keywords(ticket_by_id("FIX-PII-005"))
        assert result.intent in INTENTS
        assert result.urgency in {"low", "medium", "high"}
        assert 0.0 < result.confidence <= 1.0

    def test_keyword_fallback_stays_below_the_auto_respond_threshold(self, ticket_by_id, settings):
        result = classify_by_keywords(ticket_by_id("FIX-EMAIL-001"))
        assert result.confidence < settings.confidence_threshold

    def test_pipeline_reports_intent_urgency_and_confidence(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        result = pipeline.process(fixture_tickets[0])
        assert result.intent in INTENTS
        assert result.urgency in {"low", "medium", "high"}
        assert 0.0 < result.confidence <= 1.0
        pipeline.close()

    def test_malformed_json_from_the_model_is_recovered(self):
        assert parse_json_object('```json\n{"intent": "billing_query"}\n```')["intent"] == "billing_query"
        assert parse_json_object('Sure! {"a": 1} hope that helps')["a"] == 1
        with pytest.raises(ValueError):
            parse_json_object("no object here")


class TestEndToEnd:
    def test_a_confident_documented_ticket_is_answered_with_citations(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        result = pipeline.process(fixture_tickets[0])
        assert result.action == Action.AUTO_RESPOND.value
        assert result.citations
        assert set(result.citations).issubset(set(result.retrieved_doc_ids))
        pipeline.close()

    def test_a_flagged_ticket_never_auto_responds(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        result = pipeline.process(fixture_tickets[3])
        assert result.escalated
        assert result.reason in {Reason.MUST_NOT_AUTO_RESPOND, Reason.ALWAYS_ESCALATE_INTENT}
        assert "ESCALATED TO HUMAN" in result.response_text
        pipeline.close()

    def test_prompt_injection_escalates(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        result = pipeline.process(fixture_tickets[5])
        assert result.escalated
        assert result.blocked
        pipeline.close()

    def test_malformed_ticket_escalates_instead_of_crashing(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        result = pipeline.process(fixture_tickets[6])
        assert result.escalated
        assert result.error
        assert "malformed_input" in result.degradation_reasons
        pipeline.close()

    def test_provider_outage_degrades_to_escalation(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path, fail=True)
        result = pipeline.process(fixture_tickets[0])
        assert result.escalated
        assert result.degraded
        assert result.error is None
        pipeline.close()

    def test_retrieval_outage_degrades_to_escalation(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)

        def explode(_query):
            raise RuntimeError("vector store unreachable")

        pipeline.retriever.search = explode  # type: ignore[method-assign]
        result = pipeline.process(fixture_tickets[0])
        assert result.escalated
        assert result.reason == Reason.NO_RETRIEVAL
        assert "retrieval_unavailable" in result.degradation_reasons
        pipeline.close()

    def test_every_ticket_produces_exactly_one_final_decision(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        for raw in fixture_tickets:
            pipeline.process(raw)
        reconciliation = pipeline.decision_log.reconcile([t["ticket_id"] for t in fixture_tickets])
        assert reconciliation["reconciled"] is True
        assert reconciliation["tickets_with_final_decision"] == len(fixture_tickets)
        assert pipeline.decision_log.count(Stage.FINAL) == len(fixture_tickets)
        pipeline.close()

    def test_every_decision_row_records_its_reason(self, indexed, tmp_path, fixture_tickets):
        pipeline = _pipeline(indexed, tmp_path)
        pipeline.process(fixture_tickets[0])
        rows = pipeline.decision_log.rows_for(fixture_tickets[0]["ticket_id"])
        assert rows
        assert all(r["reason"] and r["action_taken"] and r["created_at"] for r in rows)
        assert any(r["stage"] == Stage.FINAL.value and r["prompt_version"] for r in rows)
        pipeline.close()
