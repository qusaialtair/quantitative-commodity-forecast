#!/usr/bin/env python3
"""Admin token gate on mutating FastAPI control endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api.server import _DEV_ADMIN_TOKEN, app

client = TestClient(app)

PROTECTED = (
    ("/api/override", {"action": "HALT"}),
    ("/api/halt-trading", {"halted": False}),
    ("/api/run-pipeline", {}),
    ("/api/cache-invalidate", {}),
)

PUBLIC = (
    "/api/snapshot",
    "/api/executive-summary",
    "/api/phase14/trades",
)

AUTH_HEADER = {"X-QCTF-Admin-Token": _DEV_ADMIN_TOKEN}


@pytest.mark.parametrize("path,payload", PROTECTED)
def test_mutating_endpoints_reject_missing_token(path, payload):
    resp = client.post(path, json=payload)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"


@pytest.mark.parametrize("path,payload", PROTECTED)
def test_mutating_endpoints_reject_wrong_token(path, payload):
    resp = client.post(
        path,
        json=payload,
        headers={"X-QCTF-Admin-Token": "wrong-token"},
    )
    assert resp.status_code == 401


@pytest.mark.parametrize("path,payload", (
    ("/api/override", {"action": "HALT"}),
    ("/api/halt-trading", {"halted": False}),
    ("/api/cache-invalidate", {}),
))
def test_mutating_endpoints_accept_valid_header(path, payload, monkeypatch):
    monkeypatch.delenv("QCTF_ADMIN_TOKEN", raising=False)
    resp = client.post(path, json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 200


def test_run_pipeline_accepts_valid_token_without_spawning(monkeypatch):
    monkeypatch.delenv("QCTF_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr("api.server.asyncio.create_task", lambda _coro: None)
    resp = client.post("/api/run-pipeline", headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["status"] == "STARTED"


def test_run_pipeline_accepts_bearer_token(monkeypatch):
    monkeypatch.delenv("QCTF_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr("api.server.asyncio.create_task", lambda _coro: None)
    resp = client.post(
        "/api/run-pipeline",
        headers={"Authorization": f"Bearer {_DEV_ADMIN_TOKEN}"},
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("path", PUBLIC)
def test_read_endpoints_remain_public(path):
    resp = client.get(path)
    assert resp.status_code == 200
