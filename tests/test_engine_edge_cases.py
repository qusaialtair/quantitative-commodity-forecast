#!/usr/bin/env python3
"""QA edge-case tests — book I/O, zero-equity sizing, DeepSeek offline paths."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.multi_strategy_trader as mst
from scripts.deepseek_explainer import (
    _call_deepseek_executive_summary,
    executive_summary_dumb_mode,
)


# ── Book I/O ──────────────────────────────────────────────────────────────────


def test_load_book_corrupt_json_returns_fresh_book(tmp_path, monkeypatch):
    book_path = tmp_path / "phase14_book.json"
    book_path.write_text("{not valid json")
    monkeypatch.setattr(mst, "BOOK_FILE", book_path)

    loaded = mst._load_book()

    assert loaded["starting_capital"] == mst.STARTING_CAPITAL
    assert loaded["open_trades"] == []


def test_load_book_repairs_non_list_trades(tmp_path, monkeypatch):
    book_path = tmp_path / "phase14_book.json"
    book_path.write_text(json.dumps({
        "schema_version": "1.0",
        "starting_capital": 50_000.0,
        "cash_usd": 50_000.0,
        "open_trades": "bad",
        "closed_trades": None,
    }))
    monkeypatch.setattr(mst, "BOOK_FILE", book_path)

    loaded = mst._load_book()

    assert loaded["open_trades"] == []
    assert loaded["closed_trades"] == []


def test_save_book_round_trip_atomic(tmp_path, monkeypatch):
    book_path = tmp_path / "phase14_book.json"
    monkeypatch.setattr(mst, "BOOK_FILE", book_path)

    book = mst._new_book()
    book["cash_usd"] = 42_000.0
    book["n_runs"] = 3
    mst._save_book(book)

    reloaded = mst._load_book()
    assert reloaded["cash_usd"] == 42_000.0
    assert reloaded["n_runs"] == 3
    assert book_path.exists()
    assert not book_path.with_suffix(".json.tmp").exists()


# ── Zero / negative equity sizing ─────────────────────────────────────────────


def test_safe_book_equity_floors_at_zero():
    book = mst._new_book()
    book["cash_usd"] = -500.0
    assert mst._safe_book_equity(book) == 0.0


def test_hedge_fraction_zero_equity():
    book = mst._new_book()
    assert mst._hedge_fraction(book, 0.0) == 0.0
    assert mst._hedge_fraction(book, -1.0) == 0.0


def test_execute_strategy_skips_when_equity_zero():
    book = mst._new_book()
    book["cash_usd"] = 0.0
    selector = {"direction": "BUY", "final_size_pct": 50.0}
    marks = {"GC=F": 2_000.0}

    trades = mst._execute_strategy(book, "TREND", selector, marks)

    assert trades == []
    assert book["open_trades"] == []


def test_execute_treasury_hedge_skips_when_equity_zero(monkeypatch):
    os.environ["TREASURY_HEDGE_MODE"] = "LOCAL_ACTIVE"
    os.environ["TREASURY_SHARIA_CLEARED"] = "false"
    book = mst._new_book()
    book["cash_usd"] = 0.0
    hedge = {
        "instrument": "TLT",
        "allocation_pct": 20.0,
        "effective_instrument": "GLD",
        "effective_allocation_pct": 20.0,
        "sub_tag": "sharia_fallback_gld",
        "gate_action": "SHARIA_FALLBACK_GLD",
        "regime_quadrant": "DEFLATION",
        "crisis_tier": "CRISIS",
        "mode": "LOCAL_ACTIVE",
    }
    marks = {"GLD": 200.0}

    opened = mst._execute_treasury_hedge(book, marks, hedge=hedge)

    assert opened == []


def test_lifetime_pl_pct_zero_starting_capital_dry_run(tmp_path, monkeypatch):
    book_path = tmp_path / "phase14_book.json"
    summary_path = tmp_path / "multi_strategy_trader.json"
    nav_path = tmp_path / "phase14_nav.csv"
    monkeypatch.setattr(mst, "BOOK_FILE", book_path)
    monkeypatch.setattr(mst, "SUMMARY_FILE", summary_path)
    monkeypatch.setattr(mst, "NAV_CSV", nav_path)

    book = mst._new_book()
    book["starting_capital"] = 0.0
    book["cash_usd"] = 0.0
    mst._save_book(book)

    def _fake_load(name: str) -> dict:
        if name == "strategy_selector.json":
            return {"strategy": "CASH", "direction": "HOLD", "final_size_pct": 0.0}
        if name == "treasury_hedge.json":
            return {}
        return {}

    monkeypatch.setattr(mst, "_load", _fake_load)
    monkeypatch.setattr(mst, "_fetch_price", lambda _t: 2_000.0)

    summary = mst.run_multi_strategy_trader(dry_run=True)

    assert summary["lifetime_pl_pct"] == 0.0
    assert summary["book_equity_usd"] == 0.0


# ── DeepSeek offline / timeout (graceful degrade, never crash) ────────────────


def test_deepseek_no_api_key_returns_offline_summary(monkeypatch):
    monkeypatch.setattr("scripts.deepseek_explainer.DEEPSEEK_API_KEY", "")
    result = executive_summary_dumb_mode()
    assert result["offline"] is True
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 20


def test_deepseek_timeout_returns_offline_not_exception(monkeypatch):
    monkeypatch.setattr("scripts.deepseek_explainer.DEEPSEEK_API_KEY", "sk-test")
    import requests

    def _timeout(*_a, **_k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", _timeout)

    text, err = _call_deepseek_executive_summary({"phase14_book": {}})
    assert text is None
    assert err == "timeout"

    result = executive_summary_dumb_mode()
    assert result["offline"] is True
    assert "offline" in result["summary"].lower() or "DeepSeek" in result["summary"]


def test_api_executive_summary_never_500_on_deepseek_failure(monkeypatch):
    from fastapi.testclient import TestClient
    from api.server import app

    def _boom(*_a, **_k):
        raise ConnectionError("network down")

    monkeypatch.setattr(
        "scripts.deepseek_explainer.executive_summary_dumb_mode",
        _boom,
    )
    client = TestClient(app)
    resp = client.get("/api/executive-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert len(body["summary"]) > 0


# ── API snapshot math with zero equity ────────────────────────────────────────


def test_dashboard_snapshot_zero_equity_no_crash(monkeypatch):
    from api.server import _build_dashboard_snapshot

    monkeypatch.setattr(
        "api.server.load_latest_trading_data",
        lambda: {
            "source": "live",
            "trader": {},
            "phase14_book": {"cash_usd": 0, "starting_capital": 0},
            "metrics": {
                "book_equity_usd": 0.0,
                "open_pl_usd": 0.0,
                "open_gross_notional_usd": 0.0,
                "hedge_state": {},
            },
            "treasury_hedge": {},
            "trading_halted": False,
        },
    )
    monkeypatch.setattr("api.server._load_json", lambda _n: {})

    snap = _build_dashboard_snapshot()

    assert snap["total_equity"] == 0.0
    assert snap["daily_pnl"] == 0.0
    assert snap["gross_exposure"] == 0.0
