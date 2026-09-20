"""Answer drafting with citations. (AC-A6, REQ-F05)

A citation is only kept when its document is in the passage set actually retrieved
for this ticket. An unresolvable citation manufactures confidence, so it is removed
rather than shown.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from src.config import Settings, get_settings
from src.llm import LLMClient, LLMUnavailable
from src.prompts import load_prompt
from src.schemas import (
    Classification,
    GeneratedResponse,
    NormalisedTicket,
    RetrievedPassage,
)

log = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
_CITATION = re.compile(r"\[([A-Z][A-Z0-9\-]{2,})\]")


def format_passages(passages: Sequence[RetrievedPassage]) -> str:
    blocks = []
    for p in passages:
        blocks.append(f"[{p.doc_id}] {p.title}\n{p.text}")
    return "\n\n---\n\n".join(blocks)


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for match in _CITATION.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def validate_citations(text: str, passages: Sequence[RetrievedPassage]) -> tuple[str, list[str], list[str]]:
    """Strip citations that do not resolve to a retrieved passage."""
    allowed = {p.doc_id for p in passages}
    kept, dropped = [], []
    for doc_id in extract_citations(text):
        (kept if doc_id in allowed else dropped).append(doc_id)

    for doc_id in dropped:
        text = text.replace(f"[{doc_id}]", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip(), kept, dropped


class ResponseGenerator:
    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.prompt = load_prompt("build/generate_response.md")

    @property
    def prompt_version(self) -> str:
        return f"{self.prompt.name}@{self.prompt.version}"

    def generate(
        self, ticket: NormalisedTicket, passages: Sequence[RetrievedPassage]
    ) -> GeneratedResponse:
        """Raise LLMUnavailable so the caller escalates rather than inventing a reply."""
        if not passages:
            raise LLMUnavailable("refusing to generate without supporting passages")

        system = self.prompt.render(
            channel=ticket.channel,
            subject=ticket.subject,
            body=ticket.body,
            passages=format_passages(passages),
        )
        raw = self.llm.complete(system, ticket.text, temperature=0.2, max_tokens=600)

        if INSUFFICIENT_CONTEXT in raw.upper():
            raise InsufficientContext(ticket.ticket_id)

        text, kept, dropped = validate_citations(raw, passages)
        if dropped:
            log.warning("ticket %s: dropped unresolvable citations %s", ticket.ticket_id, dropped)
        return GeneratedResponse(text=text, citations=kept, dropped_citations=dropped, source="llm")


class InsufficientContext(RuntimeError):
    """The model judged the retrieved passages inadequate. Escalate."""


def escalation_summary(
    ticket: NormalisedTicket,
    classification: Classification,
    passages: Sequence[RetrievedPassage],
    reason: str,
) -> str:
    """A handoff note, so the human does not restart the triage from nothing."""
    suggested = ", ".join(dict.fromkeys(p.doc_id for p in passages)) or "none above the relevance floor"
    return (
        f"ESCALATED TO HUMAN — {ticket.ticket_id}\n"
        f"Reason: {reason}\n"
        f"Channel: {ticket.channel} | Tier: {ticket.customer_tier} | Region: {ticket.customer_region}\n"
        f"Assessed intent: {classification.intent} (confidence {classification.confidence:.2f})\n"
        f"Assessed urgency: {classification.urgency}\n"
        f"Suggested documentation: {suggested}\n"
        f"Subject: {ticket.subject}"
    )
