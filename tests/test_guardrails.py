"""Guardrails must demonstrably block, not merely warn. (AC-A7)

The 500 development tickets contain no naturally occurring private data, so the PII
guardrail is proven against engineered fixtures. That is a deliberate choice, not a
gap: a guardrail that has never fired has never been tested.
"""

from __future__ import annotations

import pytest

from src.guardrails import check_input, check_output, detect_pii, redact_pii
from src.schemas import RetrievedPassage


class TestPII:
    @pytest.mark.parametrize(
        "text,kind",
        [
            ("Reply to jane.okafor@example.com for details.", "email_address"),
            ("Your card 4111 1111 1111 1111 was charged.", "payment_card"),
            ("Use sk-live-4f9a2c7d1e8b6a3f5c0d9e2b to authenticate.", "api_key_or_token"),
            ("Your password is Hunter2Hunter2 and it works.", "password_disclosure"),
            ("Their reference is 123-45-6789 on file.", "national_id"),
            ("Connect to 192.168.14.22 and retry.", "ip_address"),
        ],
    )
    def test_each_pattern_is_detected(self, text, kind):
        assert kind in detect_pii(text)

    def test_clean_text_has_no_findings(self):
        assert detect_pii("Clear your cookies and sign in again from a private window.") == {}

    def test_redaction_removes_the_value(self):
        redacted, kinds = redact_pii("Contact jane@example.com now.")
        assert "jane@example.com" not in redacted
        assert "[REDACTED]" in redacted
        assert "email_address" in kinds


class TestOutputGuardrails:
    def test_pii_in_a_reply_is_blocked(self, passages):
        outcome = check_output(
            "Thanks for getting in touch. I can see your key sk-live-4f9a2c7d1e8b6a3f5c0d9e2b "
            "is rejected; please rotate it from the console [DOC-AUTH-001].",
            passages,
            ["DOC-AUTH-001"],
        )
        assert outcome.blocked is True
        assert "pii_leak" in outcome.block_names
        assert "sk-live-4f9a2c7d1e8b6a3f5c0d9e2b" not in outcome.text

    @pytest.mark.parametrize(
        "claim,expected",
        [
            ("A refund has been issued to your account [DOC-AUTH-001].", "forbidden_claim"),
            ("This has been fixed on our side already [DOC-AUTH-001].", "forbidden_claim"),
            ("The fix will be released next week [DOC-AUTH-001].", "forbidden_claim"),
            ("We will credit your account for the outage [DOC-AUTH-001].", "forbidden_claim"),
        ],
    )
    def test_forbidden_claims_are_blocked(self, passages, claim, expected):
        outcome = check_output(claim + " " * 40, passages, ["DOC-AUTH-001"])
        assert outcome.blocked is True
        assert expected in outcome.block_names

    def test_unresolvable_citation_is_blocked(self, passages):
        outcome = check_output(
            "Please clear your cookies and try again from a private window [DOC-BILLING-999].",
            passages,
            ["DOC-BILLING-999"],
        )
        assert outcome.blocked is True
        assert "citation_resolves" in outcome.block_names

    def test_uncited_reply_is_blocked(self, passages):
        outcome = check_output(
            "Your account is locked and it will unlock automatically after thirty minutes.",
            passages,
            [],
        )
        assert outcome.blocked is True
        assert "grounding_required" in outcome.block_names

    def test_a_clean_grounded_reply_passes(self, passages):
        outcome = check_output(
            "Thanks for getting in touch. A locked account unlocks automatically after thirty "
            "minutes, and the security page shows a red banner while it is locked [DOC-AUTH-001]. "
            "If the CLI is still failing, sign out and back in to discard the cached token.",
            passages,
            ["DOC-AUTH-001"],
        )
        assert outcome.blocked is False
        assert all(v.passed for v in outcome.verdicts)


class TestInputGuardrails:
    def test_prompt_injection_is_blocked(self, ticket_by_id):
        outcome = check_input(ticket_by_id("FIX-INJECT-006"))
        assert outcome.blocked is True
        assert "prompt_injection" in outcome.block_names

    def test_inbound_pii_is_recorded_but_does_not_block(self, ticket_by_id):
        outcome = check_input(ticket_by_id("FIX-PII-005"))
        recorded = next(v for v in outcome.verdicts if v.name == "inbound_pii_present")
        assert recorded.detail
        assert "prompt_injection" not in outcome.block_names

    def test_an_ordinary_ticket_passes(self, ticket_by_id):
        assert check_input(ticket_by_id("FIX-EMAIL-001")).blocked is False


def test_no_passages_means_citations_cannot_resolve():
    outcome = check_output("Some claim [DOC-AUTH-001]." + " " * 40, [], ["DOC-AUTH-001"])
    assert outcome.blocked is True


def test_passage_type_is_respected():
    passage = RetrievedPassage(
        chunk_id="D#0", doc_id="DOC-X-001", title="t", text="body", distance=0.1, rank=1
    )
    outcome = check_output(
        "A sufficiently long and grounded sentence about the documented behaviour [DOC-X-001].",
        [passage],
        ["DOC-X-001"],
    )
    assert outcome.blocked is False
