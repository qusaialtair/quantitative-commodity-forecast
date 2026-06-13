#!/usr/bin/env python3
"""
Phase XXV — Treasury Hedge Sleeve toggle test
=============================================
Independent, network-free verification that flipping TREASURY_SHARIA_CLEARED
toggles the EXECUTED payload in the internal paper book:

    TREASURY_SHARIA_CLEARED=true   ->  TLT/IEF   (sovereign duration)
    TREASURY_SHARIA_CLEARED=false  ->  GLD       (sub_tag=sharia_fallback_gld)

It also asserts the 20% sleeve cap is respected and that a stable regime does
not churn the book. Uses dependency injection (synthetic recommendation + marks)
so it never hits yfinance or the real data/ files.

Run:
    python3 scripts/test_treasury_hedge_sleeve.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import scripts.treasury_hedge_overlay as tho
import scripts.multi_strategy_trader as mst

# DEFLATION/CRISIS warrants the maximum sovereign hedge (TLT 20%).
REGIME = {"quadrant": "DEFLATION", "confidence": 0.9}
CRISIS = {"tier": "CRISIS", "score": 0.95}
# Synthetic marks — no network.
MARKS = {"TLT": 90.0, "IEF": 95.0, "GLD": 200.0, "GC=F": 4000.0}


def _hedge_legs(book: dict) -> list[dict]:
    return [t for t in book["open_trades"] if t.get("strategy") == "TREASURY_HEDGE"]


def _equity(book: dict) -> float:
    return book["cash_usd"] + sum(t["notional_usd"] for t in book["open_trades"])


def run() -> int:
    os.environ["TREASURY_HEDGE_MODE"] = "LOCAL_ACTIVE"
    os.environ["TREASURY_HEDGE_MAX_PCT"] = "20"
    book = mst._new_book()

    # ── 1) Sharia CLEARED -> sovereign TLT ───────────────────────────────────
    os.environ["TREASURY_SHARIA_CLEARED"] = "true"
    reco = tho.compute_recommendation(REGIME, CRISIS)
    assert reco["effective_instrument"] == "TLT", reco
    assert reco["sub_tag"] is None, reco
    assert reco["gate_action"] == "CLEARED_SOVEREIGN", reco

    mst._execute_treasury_hedge(book, MARKS, hedge=reco)
    legs = _hedge_legs(book)
    assert len(legs) == 1, legs
    assert legs[0]["ticker"] == "TLT", legs[0]
    assert legs[0]["strategy"] == "TREASURY_HEDGE", legs[0]
    assert legs[0].get("sub_tag") is None, legs[0]
    assert "ibkr_status" not in legs[0], "sleeve must be local-only (no routing)"
    tlt_notional = legs[0]["notional_usd"]
    assert tlt_notional <= 0.2001 * _equity(book), (tlt_notional, _equity(book))
    print(f"[1] cleared  -> {legs[0]['ticker']:<3s} "
          f"${tlt_notional:,.0f} ({tlt_notional/_equity(book):.1%} of equity)  ✓")

    # ── 2) Sharia REVOKED -> fallback to GLD (gate flip = transition) ─────────
    os.environ["TREASURY_SHARIA_CLEARED"] = "false"
    reco2 = tho.compute_recommendation(REGIME, CRISIS)
    assert reco2["effective_instrument"] == "GLD", reco2
    assert reco2["sub_tag"] == "sharia_fallback_gld", reco2
    assert reco2["gate_action"] == "SHARIA_FALLBACK_GLD", reco2

    mst._execute_treasury_hedge(book, MARKS, hedge=reco2)
    legs2 = _hedge_legs(book)
    assert len(legs2) == 1, legs2
    assert legs2[0]["ticker"] == "GLD", legs2[0]
    assert legs2[0]["strategy"] == "TREASURY_HEDGE", legs2[0]
    assert legs2[0]["sub_tag"] == "sharia_fallback_gld", legs2[0]
    # The forbidden sovereign instruments must be fully out of the book.
    assert not any(t["ticker"] in {"TLT", "IEF"} for t in book["open_trades"]), book["open_trades"]
    gld_notional = legs2[0]["notional_usd"]
    assert gld_notional <= 0.2001 * _equity(book), (gld_notional, _equity(book))
    print(f"[2] revoked  -> {legs2[0]['ticker']:<3s} "
          f"${gld_notional:,.0f} ({gld_notional/_equity(book):.1%} of equity)  "
          f"sub_tag={legs2[0]['sub_tag']}  ✓")

    # ── 3) Stable regime -> no churn ─────────────────────────────────────────
    n_closed_before = len(book["closed_trades"])
    mst._execute_treasury_hedge(book, MARKS, hedge=reco2)
    assert len(_hedge_legs(book)) == 1, "should still hold exactly one sleeve leg"
    assert len(book["closed_trades"]) == n_closed_before, "stable regime must not churn"
    print("[3] stable   -> held GLD, no churn  ✓")

    # ── 4) Mode off -> sleeve flattened ──────────────────────────────────────
    os.environ["TREASURY_HEDGE_MODE"] = "SIGNAL_ONLY"
    mst._execute_treasury_hedge(book, MARKS, hedge=reco2)
    assert len(_hedge_legs(book)) == 0, "SIGNAL_ONLY must hold no local sleeve"
    print("[4] mode off -> sleeve flattened  ✓")

    print("\nALL ASSERTIONS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
