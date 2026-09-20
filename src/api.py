"""FastAPI application. (REQ-N04)

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Exposes a health check, single-ticket processing, the decision log, the kill switch
and Prometheus metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.config import get_settings
from src.observability import KILL_SWITCH_STATE, configure_logging
from src.pipeline import Pipeline
from src.schemas import TicketResult

log = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="CloudServe Support Automation",
    version=settings.system_version,
    description="Triage, retrieval, routing and grounded response drafting for support tickets.",
)

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(settings=settings)
    return _pipeline


class TicketRequest(BaseModel):
    ticket_id: str
    channel: str = "email"
    subject: str = ""
    body: str
    customer_tier: str | None = None
    customer_region: str | None = None
    language_fluency: str | None = None
    received_at: str | None = None
    labels: dict[str, Any] = Field(default_factory=dict)


class KillSwitchRequest(BaseModel):
    engaged: bool


@app.get("/health")
def health() -> dict[str, Any]:
    problems: list[str] = []
    try:
        collection_size = get_pipeline().retriever.collection.count()
    except Exception as exc:  # noqa: BLE001
        collection_size = 0
        problems.append(f"vector store unavailable: {exc}")

    if not settings.llm_available:
        problems.append("no model provider configured; all tickets will escalate")

    return {
        "status": "degraded" if problems else "ok",
        "version": settings.system_version,
        "indexed_passages": collection_size,
        "llm_enabled": settings.llm_available,
        "kill_switch": settings.kill_switch,
        "problems": problems,
    }


@app.post("/ticket", response_model=TicketResult)
def process_ticket(request: TicketRequest = Body(...)) -> TicketResult:
    return get_pipeline().process(request.model_dump())


@app.get("/decisions/{ticket_id}")
def decisions(ticket_id: str) -> dict[str, Any]:
    rows = get_pipeline().decision_log.rows_for(ticket_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no decisions logged for {ticket_id}")
    return {"ticket_id": ticket_id, "decisions": rows}


@app.post("/admin/kill-switch")
def set_kill_switch(request: KillSwitchRequest) -> dict[str, Any]:
    """Stops all automatic responses immediately; every ticket escalates to a human."""
    settings.kill_switch = request.engaged
    KILL_SWITCH_STATE.set(1 if request.engaged else 0)
    log.warning("kill switch %s", "ENGAGED" if request.engaged else "released")
    return {"kill_switch": settings.kill_switch}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
