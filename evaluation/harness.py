"""Runs a ticket file end to end, unattended. (AC-A1, AC-A9, AC-A10)

    python -m evaluation.harness --input <tickets.json> --output <directory>

The input path is an argument, never a hardcoded filename: this is pointed at a file
the author has never seen. Nothing in here prompts for input or blocks on a terminal.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import RunContext, build_report, write_calibration, write_report
from src.config import PROJECT_ROOT, get_settings
from src.ingest import load_tickets
from src.logging_store import DecisionLog
from src.observability import configure_logging, start_metrics_server
from src.pipeline import Pipeline
from src.retrieve import build_index
from src.schemas import Stage, TicketResult

_STOP = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    global _STOP
    _STOP = True
    print(f"\nsignal {signum} received; finishing in-flight tickets and writing results", file=sys.stderr)


def _ensure_index(settings, rebuild: bool) -> None:
    marker = settings.chroma_dir / "chroma.sqlite3"
    if rebuild or not marker.exists():
        print("building vector index...", file=sys.stderr)
        count = build_index(settings=settings)
        print(f"indexed {count} passages", file=sys.stderr)


def _write_jsonl(results: list[TicketResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r.model_dump(), default=str) + "\n")


def _write_responses(results: list[TicketResult], path: Path) -> None:
    payload = [
        {
            "ticket_id": r.ticket_id,
            "action": r.action,
            "reason": r.reason,
            "intent": r.intent,
            "urgency": r.urgency,
            "confidence": r.confidence,
            "citations": r.citations,
            "response": r.response_text,
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(
    input_path: Path,
    output_dir: Path,
    limit: int | None = None,
    workers: int | None = None,
    rebuild_index: bool = False,
    hidden_set_runs: int = 0,
    metrics_port: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    workers = workers or settings.max_concurrency
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_index(settings, rebuild_index)
    if metrics_port:
        start_metrics_server(metrics_port)

    raw_tickets = load_tickets(input_path)
    if limit:
        raw_tickets = raw_tickets[:limit]
    total = len(raw_tickets)
    print(f"processing {total} tickets from {input_path} with {workers} worker(s)", file=sys.stderr)

    decision_log = DecisionLog(settings, path=output_dir / "decisions.db")
    pipeline = Pipeline(settings=settings, decision_log=decision_log)

    results: list[TicketResult] = []
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(pipeline.process, raw): raw for raw in raw_tickets}
            for done, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if done % 10 == 0 or done == total:
                    rate = done / max(time.perf_counter() - started, 1e-6)
                    print(f"  {done}/{total} ({rate:.1f}/s)", file=sys.stderr, flush=True)
                if _STOP:
                    break
    finally:
        elapsed = time.perf_counter() - started
        results.sort(key=lambda r: r.ticket_id)

        ticket_ids = [str(t.get("ticket_id", f"UNKNOWN-{i}")) for i, t in enumerate(raw_tickets)]
        reconciliation = decision_log.reconcile(ticket_ids[: len(results)])
        reconciliation["decision_rows_by_stage"] = {
            stage.value: decision_log.count(stage) for stage in (Stage.CLASSIFY, Stage.RETRIEVE, Stage.FINAL)
        }

        context = RunContext(
            input_path=str(input_path),
            system_version=settings.system_version,
            model_name=settings.model_name,
            confidence_threshold=settings.confidence_threshold,
            llm_enabled=settings.llm_available,
            hidden_set_runs=hidden_set_runs,
        )
        report = build_report(results, context, reconciliation)
        report["run"]["wall_clock_seconds"] = round(elapsed, 2)
        report["run"]["workers"] = workers
        report["run"]["interrupted"] = _STOP
        report["run"]["llm_provider_failures"] = pipeline.llm.stats.failures
        report["run"]["llm_fallback_calls"] = pipeline.llm.stats.fallback_calls

        _write_jsonl(results, output_dir / "results.jsonl")
        _write_responses(results, output_dir / "responses.json")
        json_path, md_path = write_report(report, output_dir)
        write_calibration(results, output_dir / "calibration.json")
        pipeline.close()

        print(
            f"\ndone: {len(results)}/{total} tickets in {elapsed:.1f}s\n"
            f"  decisions reconciled: {reconciliation['reconciled']}\n"
            f"  first contact resolution: {report['tier_one_business_outcomes']['first_contact_resolution_pct']}%\n"
            f"  report: {md_path}\n"
            f"  metrics: {json_path}",
            file=sys.stderr,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.harness",
        description="Process a ticket file end to end and write results and metrics.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to a tickets JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Directory for results and metrics")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tickets")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent tickets (default MAX_CONCURRENCY)")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the vector index first")
    parser.add_argument("--hidden-set-runs", type=int, default=0,
                        help="How many times the hidden set has been run, recorded in the report")
    parser.add_argument("--metrics-port", type=int, default=None, help="Expose Prometheus metrics on this port")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass  # not the main thread, or unsupported platform

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    output_dir = args.output
    if output_dir.exists() and any(output_dir.iterdir()) and output_dir.name != "results":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_dir / stamp
    try:
        run(
            input_path=args.input,
            output_dir=output_dir,
            limit=args.limit,
            workers=args.workers,
            rebuild_index=args.rebuild_index,
            hidden_set_runs=args.hidden_set_runs,
            metrics_port=args.metrics_port,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"harness failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    _ = PROJECT_ROOT
    raise SystemExit(main())
