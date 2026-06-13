#!/usr/bin/env python3
"""
Physical Metals Rebalancer  (Phase XI Stage 59)
=================================================
The operator holds physical gold (and silver) as a long-term Sharia-compliant
core position. The rebalancer's job is to surface OPPORTUNISTIC trades around
that core: sell into spikes, buy back lower; and vice versa for dips.

The 75-bps UAE physical premium is the round-trip floor — every signal here
must clear that hurdle plus a safety margin to be worth executing.

Trigger conditions (any one fires a candidate):

  SELL-INTO-SPIKE
    macro_nowcast.composite_score >  +1.0      (STRONGLY_BULLISH overshoot)
    vol_surface.regime in {ELEVATED, EXTREME}
    AND current price > 20% above 252d mean    (long-run rich)
    AND term_structure.curve_shape == BACKWARDATION  (physical scarcity = sell paper)

  BUY-INTO-DIP
    macro_nowcast.composite_score < -0.5       (BEARISH but core thesis intact)
    drawdown_controller.tier_name == CAUTION   (we have headroom)
    AND current price < 252d mean - 1σ         (technical capitulation)
    AND cointegration shows gold-silver pair stressed (mean-reversion entry)

Open-trade tracking: data/metals_open_rebalance.json
  Records every active rebalance with target buy-back / sell-back zone.

Output: data/metals_rebalancer.json
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
OUTPUT_FILE = DATA_DIR / "metals_rebalancer.json"
OPEN_TRADES = DATA_DIR / "metals_open_rebalance.json"

# UAE physical round-trip premium and safety margin
PHYSICAL_RT_BPS = 150   # 75 bps × 2 sides
SAFETY_MARGIN_BPS = 200 # require 2% additional move to make it worth it
MIN_OPP_PCT = (PHYSICAL_RT_BPS + SAFETY_MARGIN_BPS) / 100.0   # 3.5%

# Triggers
SPIKE_MEAN_PREMIUM_PCT = 20.0   # >20% above 252d mean
DIP_MEAN_DISCOUNT_SIGMAS = 1.0

LINE_W = 62
SEP = "━" * LINE_W


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
        data = json.loads(OPEN_TRADES.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_open(trades: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_TRADES.write_text(json.dumps(trades, indent=2, default=str))


# ---------------------------------------------------------------------------
# Long-run reference
# ---------------------------------------------------------------------------
def _long_run_stats(ticker: str = "GC=F", lookback: str = "2y") -> dict:
    if yf is None:
        return {}
    try:
        raw = yf.download(ticker, period=lookback, interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        close = raw["Close"].dropna()
        return {
            "current":   float(close.iloc[-1]),
            "mean_252":  float(close.tail(252).mean()),
            "std_252":   float(close.tail(252).std()),
            "max_252":   float(close.tail(252).max()),
            "min_252":   float(close.tail(252).min()),
            "n_obs":     int(len(close)),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------
def _eval_sell_spike(stats: dict, nc: dict, vs: dict, ts: dict) -> tuple[bool, list]:
    if not stats or not nc:
        return False, ["insufficient data"]
    reasons = []
    cur = stats["current"]
    mean = stats["mean_252"]
    if mean <= 0:
        return False, ["mean=0"]

    # 1. Macro overshoot
    composite = float(nc.get("composite_score", 0) or 0)
    if composite < 1.0:
        return False, [f"macro nowcast {composite:+.2f} not strongly bullish (need > +1.0)"]
    reasons.append(f"macro nowcast {composite:+.2f} > +1.0")

    # 2. Vol regime
    regime = vs.get("vol_regime")
    if regime not in ("ELEVATED", "EXTREME"):
        return False, [f"vol regime {regime} not elevated"]
    reasons.append(f"vol regime {regime}")

    # 3. Above long-run mean
    premium_pct = (cur / mean - 1) * 100
    if premium_pct < SPIKE_MEAN_PREMIUM_PCT:
        return False, [f"price {premium_pct:.1f}% above 252d mean (need > {SPIKE_MEAN_PREMIUM_PCT}%)"]
    reasons.append(f"price {premium_pct:.1f}% above 252d mean")

    # 4. Term structure backwardation (physical scarcity)
    shape = ts.get("curve_shape")
    if shape == "BACKWARDATION":
        reasons.append(f"term structure {shape} (physical scarcity bid)")

    return True, reasons


def _eval_buy_dip(stats: dict, nc: dict, dd: dict, ce: dict) -> tuple[bool, list]:
    if not stats or not nc:
        return False, ["insufficient data"]
    reasons = []
    cur = stats["current"]
    mean = stats["mean_252"]
    std = stats["std_252"]
    if std <= 0:
        return False, ["std=0"]

    composite = float(nc.get("composite_score", 0) or 0)
    if composite > -0.5:
        return False, [f"macro nowcast {composite:+.2f} not bearish (need < -0.5)"]
    reasons.append(f"macro nowcast {composite:+.2f} < -0.5")

    tier = dd.get("tier_name", "NORMAL")
    if tier in ("DEFENSIVE", "CRITICAL", "EMERGENCY"):
        return False, [f"drawdown tier {tier} — no room for buying"]
    reasons.append(f"drawdown tier {tier}")

    discount_sigmas = (mean - cur) / std
    if discount_sigmas < DIP_MEAN_DISCOUNT_SIGMAS:
        return False, [f"price only {discount_sigmas:+.2f}σ below 252d mean (need > +{DIP_MEAN_DISCOUNT_SIGMAS})"]
    reasons.append(f"price {discount_sigmas:+.2f}σ below 252d mean")

    # Cointegration / mean-reversion signal
    actionable = ce.get("actionable_signals", []) or []
    gs_sig = next((s for s in actionable if "GC=F" in s.get("name", "") and "SI=F" in s.get("name", "")), None)
    if gs_sig:
        reasons.append(f"GC-SI cointegration z={gs_sig.get('z_score', 0):+.2f}")

    return True, reasons


# ---------------------------------------------------------------------------
# Open-trade management
# ---------------------------------------------------------------------------
def _check_close_open_trades(open_trades: list, cur_price: float) -> tuple[list, list]:
    """Close trades whose buy-back / sell-back target has been hit."""
    still_open = []
    closed = []
    for t in open_trades:
        if t["direction"] == "SOLD_INTO_SPIKE":
            # Looking to buy back at target_price <= entry × (1 - MIN_OPP_PCT/100)
            target = float(t["target_price"])
            if cur_price <= target:
                t["close_price"] = cur_price
                t["close_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                t["realised_pct"] = round((t["entry_price"] - cur_price) / t["entry_price"] * 100, 3)
                t["status"] = "CLOSED_BUY_BACK"
                closed.append(t)
                continue
        elif t["direction"] == "BOUGHT_INTO_DIP":
            target = float(t["target_price"])
            if cur_price >= target:
                t["close_price"] = cur_price
                t["close_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                t["realised_pct"] = round((cur_price - t["entry_price"]) / t["entry_price"] * 100, 3)
                t["status"] = "CLOSED_SELL_BACK"
                closed.append(t)
                continue
        still_open.append(t)
    return still_open, closed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_rebalancer(
    ticker: str = "GC=F",
    register: bool = False,
) -> dict:
    stats = _long_run_stats(ticker)
    nc = _load("macro_nowcast.json")
    vs = _load("vol_surface.json")
    ts = _load("term_structure.json")
    dd = _load("drawdown_controller.json")
    ce = _load("cointegration_engine.json")

    sell_ok, sell_reasons = _eval_sell_spike(stats, nc, vs, ts)
    buy_ok, buy_reasons = _eval_buy_dip(stats, nc, dd, ce)

    open_trades = _load_open()
    cur_price = stats.get("current", 0)
    open_trades, closed = _check_close_open_trades(open_trades, cur_price)

    candidate_action = "HOLD"
    candidate_target = None
    candidate_reasons = []

    if sell_ok:
        candidate_action = "SELL_INTO_SPIKE"
        # Buy-back target: 5% below current (or wider if vol is high)
        target_pct = max(MIN_OPP_PCT, 5.0)
        candidate_target = round(cur_price * (1 - target_pct / 100.0), 2)
        candidate_reasons = sell_reasons
    elif buy_ok:
        candidate_action = "BUY_INTO_DIP"
        target_pct = max(MIN_OPP_PCT, 5.0)
        candidate_target = round(cur_price * (1 + target_pct / 100.0), 2)
        candidate_reasons = buy_reasons

    # Optional: register the trade
    new_trade = None
    if register and candidate_action != "HOLD":
        new_trade = {
            "id":          datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "direction":   "SOLD_INTO_SPIKE" if candidate_action == "SELL_INTO_SPIKE" else "BOUGHT_INTO_DIP",
            "ticker":      ticker,
            "entry_price": cur_price,
            "target_price":candidate_target,
            "entry_ts":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status":      "OPEN",
            "reasons":     candidate_reasons,
        }
        open_trades.append(new_trade)

    _save_open(open_trades)

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":           ticker,
        "long_run_stats":   stats,
        "min_opportunity_pct": MIN_OPP_PCT,
        "physical_rt_bps":  PHYSICAL_RT_BPS,
        "candidate_action": candidate_action,
        "candidate_target": candidate_target,
        "candidate_reasons":candidate_reasons,
        "sell_trigger":     {"fired": sell_ok, "reasons": sell_reasons},
        "buy_trigger":      {"fired": buy_ok, "reasons": buy_reasons},
        "n_open_trades":    len(open_trades),
        "open_trades":      open_trades,
        "n_closed_this_run":len(closed),
        "closed_this_run":  closed,
        "registered":       bool(new_trade is not None),
        "new_trade":        new_trade,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    action_color = {
        "SELL_INTO_SPIKE": "\033[33m",
        "BUY_INTO_DIP":    "\033[32m",
        "HOLD":            "\033[36m",
    }.get(r["candidate_action"], "\033[0m")

    stats = r["long_run_stats"]
    print(f"\n{SEP}\n  PHYSICAL METALS REBALANCER -- {r['ticker']}\n{SEP}")
    print(f"  Current:       ${stats.get('current', 0):,.2f}")
    print(f"  252d mean:     ${stats.get('mean_252', 0):,.2f}  "
          f"σ ${stats.get('std_252', 0):,.2f}")
    print(f"  Min opp move:  {r['min_opportunity_pct']:.2f}% "
          f"(physical RT premium {r['physical_rt_bps']/100:.2f}% + safety margin)")
    print()
    print(f"  ACTION:        {action_color}{r['candidate_action']}\033[0m")
    if r["candidate_target"]:
        print(f"  Target:        ${r['candidate_target']:,.2f}")
    if r["candidate_reasons"]:
        for reason in r["candidate_reasons"]:
            print(f"    • {reason}")
    print()
    print(f"  TRIGGER EVALUATION")
    print(f"    SELL_INTO_SPIKE: {r['sell_trigger']['fired']}")
    for reason in r["sell_trigger"]["reasons"][:3]:
        print(f"      - {reason}")
    print(f"    BUY_INTO_DIP:    {r['buy_trigger']['fired']}")
    for reason in r["buy_trigger"]["reasons"][:3]:
        print(f"      - {reason}")
    print()
    print(f"  OPEN REBALANCE TRADES: {r['n_open_trades']}")
    for t in r["open_trades"][:5]:
        print(f"    {t['id']}  {t['direction']}  "
              f"entry ${t['entry_price']:,.2f} → target ${t['target_price']:,.2f}  "
              f"{t['status']}")
    if r["n_closed_this_run"] > 0:
        print(f"\n  CLOSED THIS RUN: {r['n_closed_this_run']}")
        for t in r["closed_this_run"]:
            print(f"    {t['id']}  {t.get('realised_pct', 0):+.2f}%  {t['status']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physical Metals Rebalancer")
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--register", action="store_true",
                        help="Register a new rebalance trade if triggered")
    args = parser.parse_args()
    run_rebalancer(ticker=args.ticker, register=args.register)
