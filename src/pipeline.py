"""The LangGraph pipeline. (REQ-F08)

ingest -> input guardrails -> classify -> retrieve -> route -> generate -> output
guardrails -> persist. Every node is wrapped: a node that fails records the reason
and the ticket escalates. Nothing in this graph is allowed to raise.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src import guardrails
from src.classify import Classifier, classify_by_keywords
from src.config import Settings, get_settings
from src.generate import InsufficientContext, ResponseGenerator, escalation_summary
from src.ingest import normalise
from src.llm import LLMClient, LLMUnavailable
from src.logging_store import DecisionLog
from src.observability import (
    CONFIDENCE,
    DEGRADED,
    GUARDRAIL,
    KILL_SWITCH_STATE,
    LATENCY,
    PROVIDER_FAILURES,
    TICKETS,
)
from src.retrieve import Retriever, unique_doc_ids
from src.route import Reason, RoutingDecision, decide
from src.schemas import (
    Action,
    Classification,
    GuardrailVerdict,
    NormalisedTicket,
    RetrievedPassage,
    Stage,
    TicketResult,
)

log = logging.getLogger(__name__)

REQUIREMENTS = {
    Stage.INGEST: ["REQ-F01"],
    Stage.GUARDRAIL: ["REQ-F07"],
    Stage.CLASSIFY: ["REQ-F02"],
    Stage.RETRIEVE: ["REQ-F03"],
    Stage.ROUTE: ["REQ-F04"],
    Stage.GENERATE: ["REQ-F05", "REQ-F06"],
    Stage.FINAL: ["REQ-G01"],
}


class State(TypedDict, total=False):
    raw: dict[str, Any]
    ticket: NormalisedTicket | None
    classification: Classification | None
    passages: list[RetrievedPassage]
    decision: RoutingDecision | None
    response_text: str
    citations: list[str]
    verdicts: list[GuardrailVerdict]
    blocked: bool
    degradation: list[str]
    error: str | None
    started: float


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        decision_log: DecisionLog | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.retriever = retriever or Retriever(self.settings)
        self.classifier = Classifier(llm=self.llm, settings=self.settings)
        self.generator = ResponseGenerator(llm=self.llm, settings=self.settings)
        self.decision_log = decision_log or DecisionLog(self.settings)
        self.graph = self._build_graph()
        KILL_SWITCH_STATE.set(1 if self.settings.kill_switch else 0)

    # --- Nodes ---------------------------------------------------------------------

    def _node_ingest(self, state: State) -> State:
        try:
            state["ticket"] = normalise(state["raw"])
        except Exception as exc:  # noqa: BLE001
            state["ticket"] = None
            state["error"] = f"ingest: {exc}"
            state.setdefault("degradation", []).append("malformed_input")
        return state

    def _node_input_guardrails(self, state: State) -> State:
        ticket = state.get("ticket")
        if ticket is None:
            return state
        outcome = guardrails.check_input(ticket)
        state.setdefault("verdicts", []).extend(outcome.verdicts)
        state["blocked"] = state.get("blocked", False) or outcome.blocked
        return state

    def _node_classify(self, state: State) -> State:
        ticket = state.get("ticket")
        if ticket is None:
            return state
        try:
            state["classification"] = self.classifier.classify(ticket)
        except Exception as exc:  # noqa: BLE001
            log.warning("classify failed for %s: %s", ticket.ticket_id, exc)
            state["classification"] = classify_by_keywords(ticket)
            state.setdefault("degradation", []).append("classifier_error")
        if state["classification"] and state["classification"].source != "llm":
            state.setdefault("degradation", []).append("classifier_degraded")
        return state

    def _node_retrieve(self, state: State) -> State:
        ticket = state.get("ticket")
        if ticket is None:
            state["passages"] = []
            return state
        try:
            state["passages"] = self.retriever.search(ticket.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("retrieval failed for %s: %s", ticket.ticket_id, exc)
            state["passages"] = []
            state.setdefault("degradation", []).append("retrieval_unavailable")
        return state

    def _node_route(self, state: State) -> State:
        ticket, classification = state.get("ticket"), state.get("classification")
        if ticket is None or classification is None:
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.PIPELINE_ERROR, self.settings.confidence_threshold,
                state.get("error") or "ticket could not be prepared",
            )
            return state
        if state.get("blocked"):
            blocked_names = [v.name for v in state.get("verdicts", []) if v.blocked]
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.GUARDRAIL_BLOCK, self.settings.confidence_threshold,
                f"input guardrail blocked: {', '.join(blocked_names)}",
            )
            return state
        state["decision"] = decide(ticket, classification, state.get("passages", []), self.settings)
        return state

    def _node_generate(self, state: State) -> State:
        ticket, decision = state["ticket"], state["decision"]
        assert ticket is not None and decision is not None
        try:
            generated = self.generator.generate(ticket, state.get("passages", []))
            state["response_text"] = generated.text
            state["citations"] = generated.citations
        except InsufficientContext:
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.INSUFFICIENT_CONTEXT, decision.threshold,
                "model judged the retrieved passages inadequate",
            )
        except LLMUnavailable as exc:
            PROVIDER_FAILURES.labels(provider="all").inc()
            state.setdefault("degradation", []).append("llm_unavailable")
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.GENERATION_FAILED, decision.threshold, str(exc)[:200],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("generation failed for %s: %s", ticket.ticket_id, exc)
            state.setdefault("degradation", []).append("generation_error")
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.GENERATION_FAILED, decision.threshold, str(exc)[:200],
            )
        return state

    def _node_output_guardrails(self, state: State) -> State:
        decision = state["decision"]
        assert decision is not None
        if decision.escalate or not state.get("response_text"):
            return state

        outcome = guardrails.check_output(
            state["response_text"], state.get("passages", []), state.get("citations", [])
        )
        state.setdefault("verdicts", []).extend(outcome.verdicts)
        state["response_text"] = outcome.text
        if outcome.blocked:
            for name in outcome.block_names:
                GUARDRAIL.labels(guardrail=name).inc()
            state["blocked"] = True
            state["decision"] = RoutingDecision(
                Action.ESCALATE.value, Reason.GUARDRAIL_BLOCK, decision.threshold,
                f"output guardrail blocked: {', '.join(outcome.block_names)}",
            )
        return state

    def _route_after_routing(self, state: State) -> str:
        decision = state.get("decision")
        return "generate" if decision and not decision.escalate else "output_guardrails"

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("ingest", self._node_ingest)
        graph.add_node("input_guardrails", self._node_input_guardrails)
        graph.add_node("classify", self._node_classify)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("route", self._node_route)
        graph.add_node("generate", self._node_generate)
        graph.add_node("output_guardrails", self._node_output_guardrails)

        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "input_guardrails")
        graph.add_edge("input_guardrails", "classify")
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "route")
        graph.add_conditional_edges(
            "route",
            self._route_after_routing,
            {"generate": "generate", "output_guardrails": "output_guardrails"},
        )
        graph.add_edge("generate", "output_guardrails")
        graph.add_edge("output_guardrails", END)
        return graph.compile()

    # --- Entry point ---------------------------------------------------------------

    def process(self, raw: dict[str, Any]) -> TicketResult:
        """Always returns a result. Never raises."""
        started = time.perf_counter()
        ticket_id = str(raw.get("ticket_id", "UNKNOWN")) if isinstance(raw, dict) else "UNKNOWN"
        try:
            with LATENCY.time():
                final: State = self.graph.invoke(
                    {"raw": raw, "passages": [], "verdicts": [], "degradation": [], "started": started}
                )
        except Exception as exc:  # noqa: BLE001 - the graph itself must never take down a run
            log.exception("pipeline crashed for %s", ticket_id)
            final = {
                "raw": raw,
                "ticket": None,
                "error": f"pipeline: {exc}",
                "degradation": ["pipeline_error"],
                "decision": RoutingDecision(
                    Action.ESCALATE.value, Reason.PIPELINE_ERROR, self.settings.confidence_threshold, str(exc)[:200]
                ),
            }

        result = self._to_result(ticket_id, raw, final, time.perf_counter() - started)
        self._persist(result, final)
        self._emit_metrics(result, final)
        return result

    # --- Assembly ------------------------------------------------------------------

    def _to_result(
        self, ticket_id: str, raw: dict[str, Any], state: State, elapsed: float
    ) -> TicketResult:
        ticket = state.get("ticket")
        classification = state.get("classification")
        decision = state.get("decision") or RoutingDecision(
            Action.ESCALATE.value, Reason.PIPELINE_ERROR, self.settings.confidence_threshold, "no decision recorded"
        )
        passages = state.get("passages", [])
        escalated = decision.escalate

        response_text = state.get("response_text", "")
        if escalated and ticket is not None and classification is not None:
            response_text = escalation_summary(ticket, classification, passages, decision.reason)

        received_at = ticket.received_at if ticket else None
        replied_at = received_at + timedelta(seconds=elapsed) if received_at else None
        labels = raw.get("labels", {}) if isinstance(raw, dict) else {}
        history = raw.get("history", {}) if isinstance(raw, dict) else {}

        return TicketResult(
            ticket_id=ticket.ticket_id if ticket else ticket_id,
            channel=ticket.channel if ticket else str(raw.get("channel", "unknown")),
            customer_tier=ticket.customer_tier if ticket else "unknown",
            customer_region=ticket.customer_region if ticket else "unknown",
            language_fluency=ticket.language_fluency if ticket else "unknown",
            intent=classification.intent if classification else None,
            urgency=classification.urgency if classification else None,
            confidence=classification.confidence if classification else 0.0,
            threshold=decision.threshold,
            retrieved_doc_ids=unique_doc_ids(passages),
            retrieved_chunk_ids=[p.chunk_id for p in passages],
            action=decision.action,
            reason=decision.reason,
            escalated=escalated,
            closed=not escalated,
            response_text=response_text,
            citations=state.get("citations", []) if not escalated else [],
            guardrails=state.get("verdicts", []),
            blocked=bool(state.get("blocked")),
            latency_seconds=round(elapsed, 4),
            received_at=received_at.isoformat() if received_at else None,
            replied_at=replied_at.isoformat() if replied_at else None,
            degraded=bool(state.get("degradation")),
            degradation_reasons=sorted(set(state.get("degradation", []))),
            error=state.get("error"),
            labels=labels if isinstance(labels, dict) else {},
            history=history if isinstance(history, dict) else {},
        )

    def _persist(self, result: TicketResult, state: State) -> None:
        tid = result.ticket_id
        classification = state.get("classification")
        decision = state.get("decision")

        if classification is not None:
            self.decision_log.record(
                ticket_id=tid, stage=Stage.CLASSIFY,
                action_taken="classified", reason=classification.rationale or classification.source,
                prediction={"intent": classification.intent, "urgency": classification.urgency},
                confidence=classification.confidence,
                prompt_version=self.classifier.prompt_version,
                requirement_ids=REQUIREMENTS[Stage.CLASSIFY],
            )

        self.decision_log.record(
            ticket_id=tid, stage=Stage.RETRIEVE,
            action_taken="retrieved" if result.retrieved_doc_ids else "no_passages",
            reason=f"{len(result.retrieved_chunk_ids)} passage(s) above relevance floor",
            sources_used=result.retrieved_doc_ids,
            requirement_ids=REQUIREMENTS[Stage.RETRIEVE],
        )

        self.decision_log.record(
            ticket_id=tid, stage=Stage.FINAL,
            action_taken=result.action, reason=result.reason,
            prediction={"intent": result.intent, "urgency": result.urgency},
            confidence=result.confidence, threshold=result.threshold,
            sources_used=result.citations or result.retrieved_doc_ids,
            guardrails=result.guardrails,
            prompt_version=(
                self.generator.prompt_version if not result.escalated else self.classifier.prompt_version
            ),
            requirement_ids=REQUIREMENTS[Stage.FINAL] + (REQUIREMENTS[Stage.GENERATE] if not result.escalated else []),
        )
        _ = decision

    def _emit_metrics(self, result: TicketResult, state: State) -> None:
        TICKETS.labels(channel=result.channel, outcome=result.action).inc()
        if result.confidence:
            CONFIDENCE.observe(result.confidence)
        for reason in result.degradation_reasons:
            DEGRADED.labels(reason=reason).inc()
        _ = state

    def close(self) -> None:
        self.decision_log.close()
