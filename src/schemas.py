"""Canonical data shapes for tickets, documents, retrieval hits and pipeline results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Channel(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    DOCS_COMMENT = "docs_comment"
    FORUM = "forum"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class Stage(str, Enum):
    INGEST = "ingest"
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    ROUTE = "route"
    GENERATE = "generate"
    GUARDRAIL = "guardrail"
    FINAL = "final"


class TicketLabels(BaseModel):
    model_config = ConfigDict(extra="allow")
    intent: str | None = None
    urgency: str | None = None
    expected_route: str | None = None
    answerable_from_docs: bool | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_not_auto_respond: bool = False


class TicketHistory(BaseModel):
    model_config = ConfigDict(extra="allow")
    first_contact_resolution: bool | None = None
    resolution_time_minutes: float | None = None
    csat_rating: float | None = None
    escalated: bool | None = None
    repeat_contact: bool | None = None


class Ticket(BaseModel):
    """A ticket as it arrives. Only ticket_id and body are genuinely required."""

    model_config = ConfigDict(extra="allow")

    ticket_id: str
    channel: str = Channel.EMAIL.value
    subject: str = ""
    body: str = ""
    received_at: str | None = None
    customer_id: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None
    customer_region: str | None = None
    language_fluency: str | None = None
    labels: TicketLabels = Field(default_factory=TicketLabels)
    history: TicketHistory = Field(default_factory=TicketHistory)


class NormalisedTicket(BaseModel):
    """Channel differences are resolved here and nowhere downstream."""

    ticket_id: str
    channel: str
    subject: str
    body: str
    text: str
    received_at: datetime | None = None
    customer_tier: str = "unknown"
    customer_region: str = "unknown"
    language_fluency: str = "unknown"
    must_not_auto_respond: bool = False
    subject_was_synthesised: bool = False
    normalisation_notes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="allow")
    doc_id: str
    title: str
    category: str = ""
    applies_to: str = ""
    content: str
    related_docs: list[str] = Field(default_factory=list)
    last_reviewed_days_ago: int | None = None


class RetrievedPassage(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    category: str = ""
    text: str
    distance: float
    rank: int


class Classification(BaseModel):
    intent: str
    urgency: str
    confidence: float
    rationale: str = ""
    source: str = "llm"


class GuardrailVerdict(BaseModel):
    name: str
    passed: bool
    blocked: bool = False
    detail: str = ""


class GeneratedResponse(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    source: str = "llm"


class TicketResult(BaseModel):
    """Everything the harness, the metrics report and the decision log need."""

    ticket_id: str
    channel: str
    customer_tier: str = "unknown"
    customer_region: str = "unknown"
    language_fluency: str = "unknown"

    intent: str | None = None
    urgency: str | None = None
    confidence: float = 0.0
    threshold: float = 0.0

    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    action: str = Action.ESCALATE.value
    reason: str = ""
    escalated: bool = True
    closed: bool = False

    response_text: str = ""
    citations: list[str] = Field(default_factory=list)

    guardrails: list[GuardrailVerdict] = Field(default_factory=list)
    blocked: bool = False

    latency_seconds: float = 0.0
    received_at: str | None = None
    replied_at: str | None = None
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    error: str | None = None

    labels: dict[str, Any] = Field(default_factory=dict)
    history: dict[str, Any] = Field(default_factory=dict)
