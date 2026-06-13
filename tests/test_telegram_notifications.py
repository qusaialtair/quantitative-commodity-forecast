#!/usr/bin/env python3
"""Telegram notification broadcast — mocked HTTP / Bot API."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.order_router as order_router
import scripts.telegram_notifier as tn
from scripts.treasury_hedge_overlay import run as treasury_run


@pytest.fixture(autouse=True)
def _enable_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setattr(tn, "_BOT_TOKEN", "test-token")
    monkeypatch.setattr(tn, "_CHAT_ID", "12345")
    monkeypatch.setattr(tn, "_API_URL", "https://api.telegram.org/bottest-token/sendMessage")
    monkeypatch.setattr(tn, "_ENABLED", True)


def _mock_post_ok():
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.text = '{"ok": true}'
    return resp


# ── telegram_notifier unit tests ──────────────────────────────────────────────


def test_notify_execution_summary_posts_html(monkeypatch):
    calls: list[dict] = []

    def _capture_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _mock_post_ok()

    monkeypatch.setattr(tn.requests, "post", _capture_post)

    trades = [
        {
            "ticker": "GLD",
            "side": "LONG",
            "qty": 12.5,
            "notional_usd": 2500.0,
            "status": "NOOP",
            "strategy": "METAL_TREND",
        }
    ]
    ok = tn.notify_execution_summary(trades, source="test", book_equity_usd=100_000.0)

    assert ok is True
    assert len(calls) == 1
    payload = calls[0]["json"]
    assert payload["parse_mode"] == "HTML"
    assert payload["chat_id"] == "12345"
    text = payload["text"]
    assert "EXECUTION CONFIRMED" in text
    assert "GLD" in text
    assert "METAL_TREND" in text
    assert "$100,000.00" in text


def test_notify_execution_summary_empty_trades_noops():
    assert tn.notify_execution_summary([]) is False


def test_notify_compliance_shift_sovereign_to_gld(monkeypatch):
    calls: list[str] = []

    def _capture_post(url, json=None, timeout=None):
        calls.append(json["text"])
        return _mock_post_ok()

    monkeypatch.setattr(tn.requests, "post", _capture_post)

    ok = tn.notify_compliance_shift(
        "CLEARED_SOVEREIGN",
        "SHARIA_FALLBACK_GLD",
        allocation_pct=20.0,
        effective_instrument="GLD",
        regime_quadrant="DEFLATION",
        crisis_tier="CRISIS",
    )

    assert ok is True
    assert len(calls) == 1
    assert "COMPLIANCE SHIFT" in calls[0]
    assert "Sovereign Debt Gate Locked" in calls[0]
    assert "20% Defensive Allocation rerouted to Physical Gold" in calls[0]
    assert "DEFLATION" in calls[0]


def test_notify_compliance_shift_same_gate_skips():
    assert tn.notify_compliance_shift("SHARIA_FALLBACK_GLD", "SHARIA_FALLBACK_GLD") is False


def test_notify_system_halted(monkeypatch):
    calls: list[str] = []

    def _capture_post(url, json=None, timeout=None):
        calls.append(json["text"])
        return _mock_post_ok()

    monkeypatch.setattr(tn.requests, "post", _capture_post)

    ok = tn.notify_system_halted(source="EXECUTIVE_OVERRIDE")

    assert ok is True
    assert "SYSTEM HALTED BY OPERATOR" in calls[0]
    assert "All routing suspended" in calls[0]


def test_missing_credentials_noop_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(tn, "_ENABLED", False)
    ok = tn.notify_system_halted()
    assert ok is False
    err = capsys.readouterr().err
    assert "Disabled" in err


def test_notify_post_authorize_execution_reads_summary(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(tn, "ROOT", tmp_path)
    monkeypatch.setattr(tn, "_OVERRIDE_FILE", data_dir / "operator_override.json")

    (data_dir / "operator_override.json").write_text(json.dumps({
        "action": "AUTHORIZE",
        "cleared_for_date": datetime.now(timezone.utc).date().isoformat(),
    }))
    (data_dir / "multi_strategy_trader.json").write_text(json.dumps({
        "n_new_trades": 1,
        "n_hedge_trades": 0,
        "new_trade_ids": ["P14-1-GLD"],
        "strategy": "METAL_TREND",
        "book_equity_usd": 50_000.0,
    }))
    (data_dir / "phase14_book.json").write_text(json.dumps({
        "open_trades": [{
            "trade_id": "P14-1-GLD",
            "ticker": "GLD",
            "side": "LONG",
            "qty": 5.0,
            "notional_usd": 1000.0,
            "strategy": "METAL_TREND",
        }],
    }))

    calls: list[str] = []

    def _capture_post(url, json=None, timeout=None):
        calls.append(json["text"])
        return _mock_post_ok()

    monkeypatch.setattr(tn.requests, "post", _capture_post)
    monkeypatch.setattr("scripts.order_router.pop_routed_orders", lambda: [])

    ok = tn.notify_post_authorize_execution(data_dir / "multi_strategy_trader.json")
    assert ok is True
    assert "EXECUTION CONFIRMED" in calls[0]
    assert "AUTHORIZE pipeline" in calls[0]


# ── order_router session buffer ───────────────────────────────────────────────


def test_order_router_records_paper_internal_routes(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper_internal")
    monkeypatch.setattr(order_router, "is_halted", lambda: False)
    order_router._routed_this_session.clear()

    result = order_router.route_order("BUY", "GLD", 2.0, 200.0, "TEST", note="unit")
    assert result["status"] == "NOOP"

    routed = order_router.pop_routed_orders()
    assert len(routed) == 1
    assert routed[0]["ticker"] == "GLD"
    assert routed[0]["status"] == "NOOP"
    assert order_router.pop_routed_orders() == []


# ── treasury hedge compliance hook ────────────────────────────────────────────


def test_treasury_run_notifies_gate_transition(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TREASURY_SHARIA_CLEARED", "false")

    import scripts.treasury_hedge_overlay as tho
    monkeypatch.setattr(tho, "DATA_DIR", data_dir)
    monkeypatch.setattr(tho, "OUTPUT_FILE", data_dir / "treasury_hedge.json")
    monkeypatch.setattr(tho, "ROOT", tmp_path)

    (data_dir / "treasury_hedge.json").write_text(json.dumps({
        "gate_action": "CLEARED_SOVEREIGN",
        "effective_allocation_pct": 20.0,
    }))
    (data_dir / "macro_regime.json").write_text(json.dumps({
        "quadrant": "DEFLATION",
        "confidence": 0.9,
    }))
    (data_dir / "crisis_detector.json").write_text(json.dumps({
        "tier": "CRISIS",
        "score": 0.8,
    }))

    sent: list[tuple] = []

    def _fake_notify(prev, new, **kw):
        sent.append((prev, new, kw))
        return True

    monkeypatch.setattr("scripts.telegram_notifier.notify_compliance_shift", _fake_notify)

    result = tho.run(write=True)

    assert result["gate_action"] == "SHARIA_FALLBACK_GLD"
    assert len(sent) == 1
    assert sent[0][0] == "CLEARED_SOVEREIGN"
    assert sent[0][1] == "SHARIA_FALLBACK_GLD"


# ── API HALT hook ─────────────────────────────────────────────────────────────


def test_override_halt_fires_telegram(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from api.server import _DEV_ADMIN_TOKEN, app, DATA_DIR, HALT_FLAG_FILE

    halt_file = tmp_path / "trading_halted.flag"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("api.server.DATA_DIR", data_dir)
    monkeypatch.setattr("api.server.HALT_FLAG_FILE", halt_file)
    monkeypatch.setattr("api.server.ROOT", tmp_path)

    notified: list[str] = []

    def _fake_halt(source="EXECUTIVE_OVERRIDE", **kwargs):
        # **kwargs absorbs the briefing= payload the server now attaches.
        notified.append(source)
        return True

    monkeypatch.setattr("scripts.telegram_notifier.notify_system_halted", _fake_halt)
    monkeypatch.delenv("QCTF_ADMIN_TOKEN", raising=False)

    client = TestClient(app)
    resp = client.post(
        "/api/override",
        json={"action": "HALT"},
        headers={"X-QCTF-Admin-Token": _DEV_ADMIN_TOKEN},
    )

    assert resp.status_code == 200
    assert resp.json()["action"] == "HALT"
    assert notified == ["EXECUTIVE_OVERRIDE"]
