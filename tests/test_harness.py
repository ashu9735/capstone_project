"""The harness contract and the metrics report. (AC-A1, A9, A10)"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.harness import main
from evaluation.metrics import (
    RunContext,
    build_report,
    calibration_table,
    first_contact_resolution,
    render_markdown,
)
from src.config import PROJECT_ROOT
from src.schemas import Action, TicketResult


def _result(**overrides) -> TicketResult:
    base = dict(
        ticket_id="T-1", channel="email", action=Action.AUTO_RESPOND.value,
        reason="answerable_from_documentation", escalated=False, closed=True,
        confidence=0.9, intent="billing_query", urgency="low", latency_seconds=1.2,
        citations=["DOC-A-001"], retrieved_doc_ids=["DOC-A-001"],
        response_text="A grounded reply [DOC-A-001].",
        labels={"intent": "billing_query", "expected_route": Action.AUTO_RESPOND.value,
                "expected_doc_ids": ["DOC-A-001"], "urgency": "low"},
    )
    base.update(overrides)
    return TicketResult(**base)


class TestHarnessContract:
    def test_input_and_output_are_required_arguments(self):
        with pytest.raises(SystemExit):
            main([])

    def test_missing_input_file_exits_non_zero_without_traceback(self, tmp_path):
        assert main(["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "out")]) == 2

    def test_runs_an_arbitrary_input_path_unattended(self, tmp_path):
        """The hidden set is a file this code has never seen; nothing may be hardcoded."""
        tickets = json.loads((Path(__file__).parent / "fixtures" / "tickets.json").read_text(encoding="utf-8"))
        unseen = tmp_path / "an_unseen_file.json"
        unseen.write_text(json.dumps(tickets), encoding="utf-8")
        output = tmp_path / "results"

        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.harness", "--input", str(unseen),
             "--output", str(output), "--workers", "2"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL,
        )
        assert completed.returncode == 0, completed.stderr[-3000:]
        for name in ("results.jsonl", "responses.json", "metrics.json", "metrics.md", "decisions.db"):
            assert (output / name).exists(), f"{name} missing from {output}"

        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert report["run"]["tickets"] == len(tickets)
        assert report["run"]["input_file"].endswith("an_unseen_file.json")
        assert report["tier_three_governance_conditions"]["decision_logging"]["condition_holds"] is True
        assert len((output / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()) == len(tickets)


class TestMetrics:
    def test_first_contact_resolution_counts_only_unescalated(self):
        results = [_result(), _result(ticket_id="T-2", escalated=True, closed=False,
                                      action=Action.ESCALATE.value)]
        assert first_contact_resolution(results) == 50.0

    def test_report_carries_all_three_tiers_and_the_headline_table(self):
        report = build_report(
            [_result(), _result(ticket_id="T-2")],
            RunContext("x.json", "1.0.0", "stub", 0.8, True, hidden_set_runs=0),
            {"reconciled": True, "tickets_processed": 2, "tickets_with_final_decision": 2},
        )
        assert "tier_one_business_outcomes" in report
        assert "tier_two_technical_performance" in report
        assert "tier_three_governance_conditions" in report
        assert len(report["headline_table"]) == 10
        assert report["run"]["hidden_evaluation_set_runs"] == 0

    def test_markdown_states_the_provenance_and_the_caveat(self):
        report = build_report(
            [_result()],
            RunContext("x.json", "1.0.0", "stub", 0.8, True),
            {"reconciled": True, "tickets_processed": 1, "tickets_with_final_decision": 1},
        )
        markdown = render_markdown(report)
        assert "the figures above should be treated with caution because" in markdown.lower()
        assert "Runs against the hidden evaluation set" in markdown
        assert "| Measure | Baseline | Target | Achieved |" in markdown

    def test_calibration_compares_stated_confidence_to_observed_accuracy(self):
        correct = [_result(ticket_id=f"C{i}", confidence=0.95) for i in range(10)]
        wrong = [_result(ticket_id=f"W{i}", confidence=0.95, intent="rate_limit") for i in range(10)]
        table = calibration_table(correct + wrong)
        band = next(b for b in table["bands"] if b["count"] == 20)
        assert band["observed"] == pytest.approx(0.5)
        assert table["max_gap_points"] > 5
        assert table["condition_holds"] is False
