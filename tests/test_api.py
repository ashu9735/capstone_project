"""The service surface: health, processing, decision lookup, kill switch, metrics."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.logging_store import DecisionLog
from src.pipeline import Pipeline
from src.retrieve import build_index
from src.schemas import Action
from tests.stubs import StubLLM


@pytest.fixture
def client(settings, tmp_path, monkeypatch):
    build_index(settings=settings)
    pipeline = Pipeline(
        settings=settings,
        decision_log=DecisionLog(settings, path=tmp_path / "api.db"),
        llm=StubLLM(settings),
    )
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(api, "_pipeline", pipeline)
    with TestClient(api.app) as test_client:
        yield test_client
    pipeline.close()


@pytest.fixture
def ticket_payload():
    return {
        "ticket_id": "API-001",
        "channel": "email",
        "subject": "Cannot log in to the console",
        "body": "I still cannot log in after resetting my password. The console says invalid credentials.",
        "customer_tier": "business",
    }


def test_health_reports_index_size_and_problems(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["indexed_passages"] > 0
    assert "kill_switch" in body


def test_ticket_returns_a_full_result(client, ticket_payload):
    response = client.post("/ticket", json=ticket_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "API-001"
    assert body["action"] in {Action.AUTO_RESPOND.value, Action.ESCALATE.value}
    assert body["reason"]
    assert set(body["citations"]).issubset(set(body["retrieved_doc_ids"]))


def test_ticket_rejects_a_payload_with_no_body(client):
    assert client.post("/ticket", json={"ticket_id": "X"}).status_code == 422


def test_decisions_are_retrievable_after_processing(client, ticket_payload):
    client.post("/ticket", json=ticket_payload)
    body = client.get("/decisions/API-001").json()
    assert body["ticket_id"] == "API-001"
    assert any(row["stage"] == "final" for row in body["decisions"])


def test_unknown_ticket_returns_404(client):
    assert client.get("/decisions/NOPE-999").status_code == 404


def test_kill_switch_forces_escalation(client, ticket_payload):
    try:
        assert client.post("/admin/kill-switch", json={"engaged": True}).json()["kill_switch"] is True
        body = client.post("/ticket", json={**ticket_payload, "ticket_id": "API-002"}).json()
        assert body["action"] == Action.ESCALATE.value
        assert body["reason"] == "kill_switch_engaged"
    finally:
        client.post("/admin/kill-switch", json={"engaged": False})


def test_kill_switch_release_restores_normal_routing(client, ticket_payload):
    client.post("/admin/kill-switch", json={"engaged": True})
    client.post("/admin/kill-switch", json={"engaged": False})
    body = client.post("/ticket", json={**ticket_payload, "ticket_id": "API-003"}).json()
    assert body["reason"] != "kill_switch_engaged"


def test_metrics_endpoint_exposes_the_prescribed_series(client, ticket_payload):
    client.post("/ticket", json=ticket_payload)
    text = client.get("/metrics").text
    for series in ("tickets_processed_total", "response_seconds", "guardrail_blocks_total",
                   "classification_confidence", "kill_switch_engaged"):
        assert series in text


def test_pii_in_an_inbound_ticket_is_never_echoed(client):
    response = client.post("/ticket", json={
        "ticket_id": "API-PII",
        "channel": "email",
        "subject": "Key rejected",
        "body": "My key sk-live-4f9a2c7d1e8b6a3f5c0d9e2b fails and my password is Hunter2Hunter2.",
    })
    body = response.json()
    assert "sk-live-4f9a2c7d1e8b6a3f5c0d9e2b" not in json.dumps(body)
    assert "Hunter2Hunter2" not in json.dumps(body)
