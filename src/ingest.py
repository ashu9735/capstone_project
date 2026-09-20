"""Normalise tickets from all four channels into one internal shape.

Channel differences are resolved here so that no downstream module ever branches
on `channel` for parsing reasons. (AC-A2, REQ-F01)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import ValidationError

from src.schemas import Channel, NormalisedTicket, Ticket

log = logging.getLogger(__name__)

KNOWN_CHANNELS = {c.value for c in Channel}

_EMAIL_QUOTE = re.compile(r"^\s*>.*$", re.MULTILINE)
_EMAIL_REPLY_HEADER = re.compile(
    r"\n\s*(on .{0,80}wrote:|-{2,}\s*original message\s*-{2,}|from:\s).*",
    re.IGNORECASE | re.DOTALL,
)
_EMAIL_SIGNATURE = re.compile(r"\n\s*--\s*\n.*", re.DOTALL)
_CHAT_TURN = re.compile(r"^\s*(?:\[[^\]]{0,40}\]\s*)?(?:customer|user|agent|bot)\s*:\s*", re.IGNORECASE | re.MULTILINE)
_FORUM_QUOTE = re.compile(r"\[quote[^\]]*\].*?\[/quote\]", re.IGNORECASE | re.DOTALL)
_FORUM_META = re.compile(r"^\s*(re:|posted by .{0,60}|#\d+\s*)", re.IGNORECASE)
_DOCS_ANCHOR = re.compile(r"^\s*(commenting on|re:)\s+[\w\-/#.]+\s*[:\-]\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _normalise_email(subject: str, body: str, notes: list[str]) -> tuple[str, str]:
    before = body
    body = _EMAIL_REPLY_HEADER.sub("", body)
    body = _EMAIL_QUOTE.sub("", body)
    body = _EMAIL_SIGNATURE.sub("", body)
    if body != before:
        notes.append("email: stripped quoted history or signature")
    subject = re.sub(r"^\s*((re|fwd|fw)\s*:\s*)+", "", subject, flags=re.IGNORECASE)
    return subject, body


def _normalise_chat(subject: str, body: str, notes: list[str]) -> tuple[str, str]:
    if _CHAT_TURN.search(body):
        body = _CHAT_TURN.sub("", body)
        notes.append("chat: removed speaker turn markers")
    return subject, body


def _normalise_forum(subject: str, body: str, notes: list[str]) -> tuple[str, str]:
    if _FORUM_QUOTE.search(body):
        body = _FORUM_QUOTE.sub("", body)
        notes.append("forum: removed quoted posts")
    subject = _FORUM_META.sub("", subject)
    return subject, body


def _normalise_docs_comment(subject: str, body: str, notes: list[str]) -> tuple[str, str]:
    if _DOCS_ANCHOR.search(body):
        body = _DOCS_ANCHOR.sub("", body)
        notes.append("docs_comment: removed page anchor prefix")
    return subject, body


_NORMALISERS = {
    Channel.EMAIL.value: _normalise_email,
    Channel.CHAT.value: _normalise_chat,
    Channel.FORUM.value: _normalise_forum,
    Channel.DOCS_COMMENT.value: _normalise_docs_comment,
}


def _synthesise_subject(body: str) -> str:
    """Chat tickets never carry a subject; 31% of the corpus arrives this way."""
    first = re.split(r"(?<=[.!?])\s+|\n", body.strip(), maxsplit=1)[0]
    first = first.strip(" .!?,")
    if len(first) > 90:
        first = first[:87].rsplit(" ", 1)[0] + "..."
    return first or "(no subject)"


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    log.warning("unparseable received_at: %r", value)
    return None


def normalise(raw: dict[str, Any]) -> NormalisedTicket:
    """Raise ValueError for input that cannot form a ticket at all."""
    try:
        ticket = Ticket.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"malformed ticket: {exc.errors()[:1]}") from exc

    notes: list[str] = []
    channel = (ticket.channel or "").strip().lower()
    if channel not in KNOWN_CHANNELS:
        notes.append(f"unknown channel {channel!r} treated as email")
        channel = Channel.EMAIL.value

    subject = _clean(ticket.subject)
    body = _clean(ticket.body)

    subject, body = _NORMALISERS[channel](subject, body, notes)
    subject, body = _clean(subject), _clean(body)

    if not body and not subject:
        raise ValueError("ticket has neither subject nor body")

    synthesised = False
    if not subject:
        subject = _synthesise_subject(body)
        synthesised = True
        notes.append("subject synthesised from body")

    if not body:
        body = subject
        notes.append("body empty; subject used as body")

    return NormalisedTicket(
        ticket_id=str(ticket.ticket_id),
        channel=channel,
        subject=subject,
        body=body,
        text=f"{subject}\n\n{body}".strip(),
        received_at=_parse_timestamp(ticket.received_at),
        customer_tier=(ticket.customer_tier or "unknown").lower(),
        customer_region=(ticket.customer_region or "unknown").lower(),
        language_fluency=(ticket.language_fluency or "unknown").lower(),
        must_not_auto_respond=bool(ticket.labels.must_not_auto_respond),
        subject_was_synthesised=synthesised,
        normalisation_notes=notes,
        raw=raw,
    )


def load_tickets(path: str | Path) -> list[dict[str, Any]]:
    """Accept either a JSON array or newline-delimited JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        for key in ("tickets", "data", "items", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path} is an object with no recognised ticket array")
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list of tickets")
    return data


def normalise_all(raws: Iterable[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], NormalisedTicket | None, str | None]]:
    """Yield (raw, normalised, error). A bad ticket never stops the batch."""
    for index, raw in enumerate(raws):
        if not isinstance(raw, dict):
            yield {"ticket_id": f"UNPARSEABLE-{index}"}, None, "record is not an object"
            continue
        try:
            yield raw, normalise(raw), None
        except Exception as exc:  # noqa: BLE001 - malformed input must not halt the run
            log.warning("ticket %s failed normalisation: %s", raw.get("ticket_id", index), exc)
            yield raw, None, str(exc)
