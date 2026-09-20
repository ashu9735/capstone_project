"""Prometheus instrumentation and structured logging.

Metric names are those prescribed in the Setup Guide; the extra series exist because
the dashboard has to show confidence distribution and provider failures.
"""

from __future__ import annotations

import logging
import sys

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pythonjsonlogger import jsonlogger

TICKETS = Counter("tickets_processed_total", "Tickets processed", ["channel", "outcome"])
LATENCY = Histogram(
    "response_seconds",
    "End to end response time",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)
GUARDRAIL = Counter("guardrail_blocks_total", "Responses blocked", ["guardrail"])
CONFIDENCE = Histogram(
    "classification_confidence",
    "Distribution of classification confidence",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PROVIDER_FAILURES = Counter("llm_provider_failures_total", "Model provider failures", ["provider"])
DEGRADED = Counter("tickets_degraded_total", "Tickets completed on a degraded path", ["reason"])
KILL_SWITCH_STATE = Gauge("kill_switch_engaged", "1 when automatic responses are disabled")

_metrics_server_started = False


def start_metrics_server(port: int) -> bool:
    global _metrics_server_started
    if _metrics_server_started:
        return False
    try:
        start_http_server(port)
        _metrics_server_started = True
        return True
    except OSError as exc:
        logging.getLogger(__name__).warning("metrics server not started on %s: %s", port, exc)
        return False


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(
            jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
