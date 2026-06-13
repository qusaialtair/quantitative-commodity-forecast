#!/usr/bin/env python3
"""
Active Pairs Trader  (Phase XIII Stage 70)
============================================
Converts the cointegration_engine's actionable signals into concrete
spread-trade orders ready for IBKR.

For each actionable pair the engine produces:
  - leg 1   long ticker, qty, entry price
  - leg 2   short ticker, qty, entry price (notional-matched)
  - z-entry, z-target, z-stop
  - expected half-life
  - per-leg dollar size

Sizing applies the same risk-cap stack as trade_idea_generator:
  base = abs(z_score) × 5% of NAV
  × kelly_mult × drawdown_mult × leverage_cap × physical capacity

Open spread trades persisted to data/pairs_open.json with status tracking.

Output: data/pairs_trader.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "pairs_trader.json"
OPEN_TRADES = DATA_DIR / "pairs_open.json"

LINE_W = 62
SEP = "━" * LINE_W

Z_TARGET_RATIO = 0.25  # exit at 25% of entry z (mean reversion)
Z_STOP_RATIO = 1.5     # stop if z extends 50% further

# Base sizing (% of NAV per unit |z|)
BASE_SIZE_PCT_PER_Z = 5.0
MAX_PAIR_NOTIONAL_PCT = 15.0


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_open() -> list:
    if not OPEN_TRADES.exists():
        return []
    try:
        d = json.loads(OPEN_TRADES.read_text())
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_open(trades: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_TRADES.write_text(json.dumps(trades, indent=2, default=str))


def _price(ticker: str) -> float | None:
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        v = raw["Close"].dropna().iloc[-1]
        return float(v)
    except Exception:
        return None


def _size_caps(nav: float) -> float:
    """Return max pair notional in USD."""
    pct = MAX_PAIR_NOTIONAL_PCT
    vt = _load("vol_target_budget.json")
    if vt:
        pct *= float(vt.get("leverage_capped") or 1.0)
    dd = _load("drawdown_controller.json")
    if dd:
        pct *= float(dd.get("sizing_multiplier") or 1.0)
    vs = _load("vol_surface.json")
    if vs:
        pct *= float(vs.get("actions", {}).get("kelly_fraction_multiplier") or 1.0)
    return max(0, nav * min(pct, MAX_PAIR_NOTIONAL_PCT) / 100.0)


# ---------------------------------------------------------------------------
# Construct trades
# ---------------------------------------------------------------------------
def _build_pair_trade(signal: dict, nav: float, max_notional: float) -> dict | None:
    """
    cointegration_engine emits signals like:
      {
        "name":          "GC=F_SI=F",
        "signal":        "LONG_SPREAD" or "SHORT_SPREAD",
        "z_score":       2.31,
        "half_life_days":12.5,
        ...
      }
    """
    pair_name = signal.get("name") or ""
    if "__" in pair_name:
        parts = pair_name.split("__")
    elif "_" in pair_name:
        parts = pair_name.split("_", 1)
    else:
        return None
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    direction = signal.get("signal", "")
    z = float(signal.get("z_score", 0))
    hl = float(signal.get("half_life_days", 0))

    pa = _price(a)
    pb = _price(b)
    if not pa or not pb:
        return None

    # Notional sizing
    target_notional = nav * BASE_SIZE_PCT_PER_Z * abs(z) / 100.0
    notional = min(target_notional, max_notional)
    qty_a = round(notional / pa, 4)
    qty_b = round(notional / pb, 4)

    # Direction:
    #   LONG_SPREAD  = long A, short B   (residual = A - β·B is low; expect to rise)
    #   SHORT_SPREAD = short A, long B
    if direction == "LONG_SPREAD":
        leg1 = {"ticker": a, "side": "BUY",  "qty": qty_a, "entry": pa}
        leg2 = {"ticker": b, "side": "SELL", "qty": qty_b, "entry": pb}
    elif direction == "SHORT_SPREAD":
        leg1 = {"ticker": a, "side": "SELL", "qty": qty_a, "entry": pa}
        leg2 = {"ticker": b, "side": "BUY",  "qty": qty_b, "entry": pb}
    else:
        return None

    z_target = round(z * Z_TARGET_RATIO, 3)
    z_stop = round(z * Z_STOP_RATIO, 3)

    return {
        "id":           datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + pair_name,
        "pair":         pair_name,
        "direction":    direction,
        "z_entry":      round(z, 3),
        "z_target":     z_target,
        "z_stop":       z_stop,
        "half_life_days":hl,
        "notional_usd": round(notional, 2),
        "leg1":         leg1,
        "leg2":         leg2,
        "status":       "READY",
        "created_ts":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pairs_trader(register: bool = False) -> dict:
    ce = _load("cointegration_engine.json")
    signals = ce.get("actionable_signals", []) or []

    ps = _load("pipeline_state.json")
    pf = ps.get("portfolio", {})
    nav = float(pf.get("portfolio_value", 100_000) or 100_000)
    max_notional = _size_caps(nav)

    trades = []
    for sig in signals:
        t = _build_pair_trade(sig, nav, max_notional)
        if t:
            trades.append(t)

    # Persist open trades if register=True
    open_trades = _load_open()
    new_open = []
    if register and trades:
        for t in trades:
            new_open.append(t)
        open_trades.extend(new_open)
        _save_open(open_trades)

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nav_usd":        nav,
        "max_pair_notional_usd": round(max_notional, 2),
        "n_signals":      len(signals),
        "n_trades_built": len(trades),
        "trades":         trades,
        "n_open":         len(open_trades),
        "open_trades":    open_trades,
        "registered":     len(new_open),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  ACTIVE PAIRS TRADER\n{SEP}")
    print(f"  NAV:                ${r['nav_usd']:,.2f}")
    print(f"  Max pair notional:  ${r['max_pair_notional_usd']:,.2f}")
    print(f"  Cointegration signals: {r['n_signals']}")
    print(f"  Trades built:       {r['n_trades_built']}")
    print(f"  Open spread trades: {r['n_open']}")
    print()
    if r["trades"]:
        print(f"  TRADES READY")
        print(f"  {'─' * 58}")
        for t in r["trades"]:
            print(f"  {t['pair']:<14s}  {t['direction']:<12s}  "
                  f"z={t['z_entry']:+.2f} → {t['z_target']:+.2f}  "
                  f"½={t['half_life_days']:.0f}d")
            print(f"    LEG1: {t['leg1']['side']} {t['leg1']['ticker']} "
                  f"qty {t['leg1']['qty']} @ ${t['leg1']['entry']:.2f}")
            print(f"    LEG2: {t['leg2']['side']} {t['leg2']['ticker']} "
                  f"qty {t['leg2']['qty']} @ ${t['leg2']['entry']:.2f}")
            print(f"    Notional: ${t['notional_usd']:,.2f}")
    else:
        print(f"  No actionable cointegration signals")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Pairs Trader")
    parser.add_argument("--register", action="store_true",
                        help="Persist trades to pairs_open.json")
    args = parser.parse_args()
    run_pairs_trader(register=args.register)
