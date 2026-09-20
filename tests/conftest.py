"""Shared fixtures. Nothing in the suite requires a model provider or a network call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import PROJECT_ROOT, Settings
from src.ingest import normalise
from src.schemas import Classification, RetrievedPassage

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    base = tmp_path_factory.mktemp("storage")
    return Settings(
        enable_llm=False,
        chroma_path=str(base / "chroma"),
        database_url=f"sqlite:///{base / 'decisions.db'}",
        confidence_threshold=0.80,
        use_calibration=False,
        docs_path=str(PROJECT_ROOT / "data" / "documentation.json"),
    )


@pytest.fixture(scope="session")
def fixture_tickets() -> list[dict[str, Any]]:
    return json.loads((FIXTURE_DIR / "tickets.json").read_text(encoding="utf-8"))


@pytest.fixture
def ticket_by_id(fixture_tickets: list[dict[str, Any]]):
    index = {t["ticket_id"]: t for t in fixture_tickets}

    def _get(ticket_id: str):
        return normalise(index[ticket_id])

    return _get


@pytest.fixture
def passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            chunk_id="DOC-AUTH-001#000",
            doc_id="DOC-AUTH-001",
            title="Resolving invalid credential errors on login",
            text="A locked account displays a red banner and unlocks automatically after thirty minutes.",
            distance=0.21,
            rank=1,
        ),
        RetrievedPassage(
            chunk_id="DOC-AUTH-002#000",
            doc_id="DOC-AUTH-002",
            title="Multi-factor authentication setup and recovery",
            text="Device clock drift of more than thirty seconds invalidates time-based codes.",
            distance=0.34,
            rank=2,
        ),
    ]


@pytest.fixture
def confident() -> Classification:
    return Classification(intent="authentication_failure", urgency="medium", confidence=0.93)


@pytest.fixture
def unconfident() -> Classification:
    return Classification(intent="authentication_failure", urgency="medium", confidence=0.41)
