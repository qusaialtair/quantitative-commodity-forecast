#!/usr/bin/env python3
"""Market-data ingestion loop and silent MTM refresh."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.multi_strategy_trader as mst
from api.server import (
    _market_data_ingest_should_run,
    _run_market_data_ingest_cycle,
    _us_equity_market_open,
)


def test_collect_ingest_universe_includes_core_and_open_legs():
    book = mst._new_book()
    book["open_trades"] = [{
        "ticker": "DUST",
        "strategy": "TAIL_HEDGE",
        "side": "SHORT",
        "qty": 1.0,
        "notional_usd": 500.0,
        "entry_price": 10.0,
    }]
    universe = mst.collect_ingest_universe(book)
    assert "GC=F" in universe
    assert "GLD" in universe
    assert "TLT" in universe
    assert "DUST" in universe


def test_run_mtm_refresh_persists_book_and_summary(tmp_path, monkeypatch):
    book_path = tmp_path / "phase14_book.json"
    summary_path = tmp_path / "multi_strategy_trader.json"
    nav_path = tmp_path / "phase14_nav.csv"
    monkeypatch.setattr(mst, "BOOK_FILE", book_path)
    monkeypatch.setattr(mst, "SUMMARY_FILE", summary_path)
    monkeypatch.setattr(mst, "NAV_CSV", nav_path)
    # Hermetic: block live yfinance top-ups so marks_fetched stays at the
    # single provided mark regardless of network availability.
    monkeypatch.setattr(mst, "_fetch_price", lambda sym: None)

    # Relative entry date — a hardcoded date eventually crosses the 5-day
    # TIMEOUT boundary and the trade gets closed before the MTM assertion.
    entry_day = (datetime.now().date() - timedelta(days=1)).isoformat()
    book = mst._new_book()
    book["open_trades"] = [{
        "trade_id": "t1",
        "ticker": "GC=F",
        "strategy": "TREND",
        "side": "LONG",
        "qty": 1.0,
        "notional_usd": 4_000.0,
        "entry_price": 2_000.0,
        "entry_fee": 2.0,
        "entry_at": f"{entry_day}T12:00:00+00:00",
        "entry_date": entry_day,
    }]
    book["cash_usd"] = 96_000.0
    mst._save_book(book)

    marks = {"GC=F": 2_100.0}
    result = mst.run_mtm_refresh(marks=marks)

    assert result["status"] == "OK"
    assert result["marks_fetched"] == 1
    saved = json.loads(book_path.read_text())
    assert saved["n_mtm_runs"] == 1
    assert saved["latest_marks"]["GC=F"] == 2_100.0
    summary = json.loads(summary_path.read_text())
    assert summary["mtm_refresh"] is True
    assert summary["open_pl_usd"] > 0


def test_fetch_live_marks_skips_failures(monkeypatch):
    def _fake_fetch(sym: str) -> float | None:
        if sym == "GLD":
            raise TimeoutError("yahoo timeout")
        if sym == "TLT":
            return 95.0
        return None

    monkeypatch.setattr(mst, "_fetch_price", _fake_fetch)
    marks = mst.fetch_live_marks({"GLD", "TLT", "GC=F"})
    assert marks == {"TLT": 95.0}


def test_market_hours_gate(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_INGEST_ENABLED", "true")
    monkeypatch.setenv("MARKET_DATA_INGEST_CONTINUOUS", "false")
    saturday = datetime(2026, 6, 6, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert _us_equity_market_open(saturday) is False
    assert _market_data_ingest_should_run(saturday) is False

    tuesday_open = datetime(2026, 6, 2, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    assert _us_equity_market_open(tuesday_open) is True
    assert _market_data_ingest_should_run(tuesday_open) is True


def test_ingest_cycle_survives_trader_failure(monkeypatch):
    monkeypatch.setattr(
        "scripts.multi_strategy_trader.run_mtm_refresh",
        lambda: (_ for _ in ()).throw(TimeoutError("network")),
    )
    result = asyncio.run(_run_market_data_ingest_cycle())
    assert result is None


def test_ingest_cycle_broadcasts_on_success(monkeypatch):
    monkeypatch.setattr(
        "scripts.multi_strategy_trader.run_mtm_refresh",
        lambda: {
            "status": "OK",
            "book_equity_usd": 100_000.0,
            "open_pl_usd": 50.0,
            "n_exits": 0,
            "marks_fetched": 6,
            "universe": ["GC=F", "GLD"],
        },
    )
    broadcast = AsyncMock()
    monkeypatch.setattr("api.server.manager.broadcast", broadcast)

    result = asyncio.run(_run_market_data_ingest_cycle())

    assert result is not None
    broadcast.assert_awaited_once()
    payload = broadcast.await_args.args[0]
    assert payload["type"] == "trading_updated"
    assert payload["source"] == "market_data_ingest"
