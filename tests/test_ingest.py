"""Channel normalisation. (AC-A2)"""

from __future__ import annotations

import pytest

from src.ingest import load_tickets, normalise, normalise_all
from src.schemas import Channel


def test_all_four_channels_normalise(fixture_tickets):
    seen = set()
    for raw in fixture_tickets[:4]:
        ticket = normalise(raw)
        seen.add(ticket.channel)
        assert ticket.text
        assert ticket.subject
    assert seen == {c.value for c in Channel}


def test_email_quoted_history_and_signature_removed(ticket_by_id):
    ticket = ticket_by_id("FIX-EMAIL-001")
    assert "Please try clearing your cookies" not in ticket.body
    assert "Head of Platform" not in ticket.body
    assert ticket.subject == "Cannot log in to the console"
    assert "I still cannot log in" in ticket.body


def test_chat_subject_is_synthesised_from_body(ticket_by_id):
    ticket = ticket_by_id("FIX-CHAT-002")
    assert ticket.subject_was_synthesised is True
    assert ticket.subject
    assert "customer:" not in ticket.body
    assert "agent:" not in ticket.body


def test_forum_quotes_stripped(ticket_by_id):
    ticket = ticket_by_id("FIX-FORUM-003")
    assert "I had this last year" not in ticket.body
    assert "webhook endpoint stopped receiving events" in ticket.body


def test_docs_comment_anchor_stripped(ticket_by_id):
    ticket = ticket_by_id("FIX-DOCS-004")
    assert ticket.body.lower().startswith("our auditor")
    assert ticket.must_not_auto_respond is True


def test_unparseable_timestamp_does_not_raise():
    ticket = normalise({"ticket_id": "X", "channel": "email", "body": "hello there", "received_at": "nonsense"})
    assert ticket.received_at is None


def test_unknown_channel_falls_back_to_email():
    ticket = normalise({"ticket_id": "X", "channel": "carrier_pigeon", "body": "a real problem"})
    assert ticket.channel == Channel.EMAIL.value
    assert any("unknown channel" in note for note in ticket.normalisation_notes)


def test_empty_ticket_is_rejected():
    with pytest.raises(ValueError):
        normalise({"ticket_id": "X", "channel": "email", "subject": "", "body": ""})


def test_batch_survives_a_malformed_record(fixture_tickets):
    outcomes = list(normalise_all(fixture_tickets + ["not a dict"]))
    assert len(outcomes) == len(fixture_tickets) + 1
    assert sum(1 for _, t, _ in outcomes if t is not None) == len(fixture_tickets) - 1
    assert any(err for _, _, err in outcomes)


def test_load_tickets_reads_the_supplied_datasets(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('[{"ticket_id": "A", "body": "x"}]', encoding="utf-8")
    assert load_tickets(path)[0]["ticket_id"] == "A"


def test_load_tickets_accepts_newline_delimited(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"ticket_id": "A", "body": "x"}\n{"ticket_id": "B", "body": "y"}\n', encoding="utf-8")
    assert len(load_tickets(path)) == 2
