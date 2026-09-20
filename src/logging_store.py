"""The decision log. (AC-A8, REQ-G01)

Schema is exactly the one prescribed in the Setup Guide. Every ticket produces a
`final` row on every path, including failures, so that logged decisions reconcile
to processed tickets exactly rather than approximately.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import Settings, get_settings
from src.schemas import GuardrailVerdict, Stage

SCHEMA = """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id     TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        ticket_id       TEXT NOT NULL,
        stage           TEXT NOT NULL,
        prediction      TEXT,
        confidence      REAL,
        threshold       REAL,
        action_taken    TEXT NOT NULL,
        reason          TEXT NOT NULL,
        sources_used    TEXT,
        guardrails      TEXT,
        prompt_version  TEXT,
        requirement_ids TEXT
    )
"""
INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(ticket_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_stage ON decisions(stage)",
)

_COLUMNS = (
    "decision_id", "created_at", "ticket_id", "stage", "prediction", "confidence",
    "threshold", "action_taken", "reason", "sources_used", "guardrails",
    "prompt_version", "requirement_ids",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _encode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return str(value)


class DecisionLog:
    """Thread-safe writer. The harness runs tickets concurrently."""

    def __init__(self, settings: Settings | None = None, path: str | Path | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.sqlite_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(SCHEMA)
            for statement in INDEXES:
                self._conn.execute(statement)
            self._conn.commit()

    def record(
        self,
        ticket_id: str,
        stage: Stage | str,
        action_taken: str,
        reason: str,
        prediction: Any = None,
        confidence: float | None = None,
        threshold: float | None = None,
        sources_used: Sequence[str] | None = None,
        guardrails: Sequence[GuardrailVerdict] | Sequence[dict[str, Any]] | None = None,
        prompt_version: str | None = None,
        requirement_ids: Sequence[str] | None = None,
    ) -> str:
        decision_id = str(uuid.uuid4())
        guardrail_payload = [
            g.model_dump() if isinstance(g, GuardrailVerdict) else g for g in (guardrails or [])
        ]
        row = (
            decision_id,
            _now(),
            ticket_id,
            stage.value if isinstance(stage, Stage) else str(stage),
            _encode(prediction),
            confidence,
            threshold,
            action_taken,
            reason,
            _encode(list(sources_used or [])),
            _encode(guardrail_payload),
            prompt_version,
            _encode(list(requirement_ids or [])),
        )
        with self._lock:
            self._conn.execute(
                f"INSERT INTO decisions ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_COLUMNS))})",
                row,
            )
            self._conn.commit()
        return decision_id

    # --- Reconciliation (AC-A8) ----------------------------------------------------

    def count(self, stage: Stage | str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM decisions"
        params: tuple[Any, ...] = ()
        if stage is not None:
            sql += " WHERE stage = ?"
            params = (stage.value if isinstance(stage, Stage) else str(stage),)
        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()[0])

    def distinct_tickets(self, stage: Stage | str | None = Stage.FINAL) -> set[str]:
        sql = "SELECT DISTINCT ticket_id FROM decisions"
        params: tuple[Any, ...] = ()
        if stage is not None:
            sql += " WHERE stage = ?"
            params = (stage.value if isinstance(stage, Stage) else str(stage),)
        with self._lock:
            return {r[0] for r in self._conn.execute(sql, params).fetchall()}

    def reconcile(self, ticket_ids: Iterable[str]) -> dict[str, Any]:
        expected = {str(t) for t in ticket_ids}
        logged = self.distinct_tickets(Stage.FINAL)
        missing = sorted(expected - logged)
        unexpected = sorted(logged - expected)
        return {
            "tickets_processed": len(expected),
            "tickets_with_final_decision": len(logged),
            "total_decision_rows": self.count(),
            "missing_ticket_ids": missing,
            "unexpected_ticket_ids": unexpected,
            "reconciled": not missing and not unexpected,
        }

    def rows_for(self, ticket_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM decisions WHERE ticket_id = ? ORDER BY created_at", (ticket_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DecisionLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
