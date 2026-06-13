#!/usr/bin/env python3
"""
Gold Carry Trade Analyzer
==========================
Decomposes gold total return into spot, carry, and roll components, then
compares the market-implied carry against the theoretical no-arbitrage
fair value.

Components:

  spot_return     21d % change in spot proxy
  market_carry    (front-futures − spot) / spot, annualised
                    > 0  contango  (paper gold costs money to hold)
                    < 0  backwardation (paper gold is being bid)
  fair_carry      USD risk-free rate − gold lease rate − storage cost
                    (≈ r_USD − r_lease − 0.40%)
  carry_spread    market_carry − fair_carry
                    > 0  contango stronger than fair (sell paper, buy spot)
                    < 0  backwardation richer than fair (buy paper)

Inputs:
  - GC=F  generic front-month gold futures
  - GLD   ETF, spot proxy at $/oz / 10
  - r_USD  4.50% (configurable)
  - r_lease 0.50% gold lease (configurable)
  - storage 0.40% (GLD-implied)

Output: data/carry_analyzer.json
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
OUTPUT_FILE = DATA_DIR / "carry_analyzer.json"

DEFAULT_LOOKBACK = "1y"
DEFAULT_USD_RATE = 0.045
DEFAULT_LEASE_RATE = 0.005
DEFAULT_STORAGE = 0.004

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_close(ticker: str, lookback: str) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].dropna()


def _build_panel(lookback: str) -> pd.DataFrame:
    """
    GC=F is the front-month gold futures generic and is the most reliable
    "spot-equivalent" price available through yfinance. True LBMA spot is
    not directly exposed, and GLD has accumulated tracking error from fund
    expenses that makes a GLD×10 proxy unreliable.
    """
    futures = _fetch_close("GC=F", lookback)
    df = pd.DataFrame({"price": futures}).dropna()
    df.attrs["price_source"] = "GC=F front-month"
    return df


def _load_real_yield_now() -> float | None:
    """Pull latest 10y real yield from alt_data.csv."""
    alt_path = DATA_DIR / "alt_data.csv"
    if not alt_path.exists():
        return None
    try:
        alt = pd.read_csv(alt_path, index_col=0, parse_dates=True)
        if "real_yield_10y" in alt.columns:
            v = alt["real_yield_10y"].dropna()
            if len(v):
                return float(v.iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def _spot_return_pct(spot: pd.Series, lookback_days: int = 21) -> float:
    s = spot.dropna()
    if len(s) <= lookback_days:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-lookback_days - 1] - 1) * 100)


def _fair_carry_pct(usd_rate: float, lease_rate: float, storage_pct: float) -> float:
    """
    Cost of carry to hold gold paper for a year:
        carry = r_USD − r_lease − storage_cost
    Higher real yields → higher carry cost → bearish for gold.
    """
    return float(usd_rate - lease_rate - storage_pct)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_carry_analyzer(
    lookback: str = DEFAULT_LOOKBACK,
    usd_rate: float = DEFAULT_USD_RATE,
    lease_rate: float = DEFAULT_LEASE_RATE,
    storage: float = DEFAULT_STORAGE,
    days_to_expiry: int = 60,
) -> dict:
    df = _build_panel(lookback)
    price_now = float(df["price"].iloc[-1])

    spot_ret_21 = _spot_return_pct(df["price"], 21)
    spot_ret_63 = _spot_return_pct(df["price"], 63)
    spot_ret_252 = _spot_return_pct(df["price"], 252)

    # Use real yield as the dynamic carry driver if available
    real_yield = _load_real_yield_now()
    if real_yield is not None:
        # real yield is already in % (e.g. 1.94 means 1.94%)
        dynamic_usd_rate = real_yield / 100.0 + 0.025  # add inflation premium ~2.5%
    else:
        dynamic_usd_rate = usd_rate

    fair_carry = _fair_carry_pct(dynamic_usd_rate, lease_rate, storage)

    # Carry burden classification
    if fair_carry > 0.05:
        burden = "HIGH_CARRY_HEADWIND"
    elif fair_carry > 0.02:
        burden = "MODERATE_CARRY_HEADWIND"
    elif fair_carry > 0:
        burden = "LOW_CARRY_HEADWIND"
    else:
        burden = "NEGATIVE_CARRY_TAILWIND"

    # Realised return vs fair carry expectation
    annualised_realised_21d = (spot_ret_21 / 21.0) * 252.0
    annualised_realised_63d = (spot_ret_63 / 63.0) * 252.0
    excess_vs_carry_21d = annualised_realised_21d - fair_carry * 100
    excess_vs_carry_63d = annualised_realised_63d - fair_carry * 100

    result = {
        "generated_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback":            lookback,
        "n_obs":               int(len(df)),
        "price_source":        df.attrs.get("price_source", "GC=F"),
        "price_now":           round(price_now, 2),
        "spot_returns": {
            "21d_pct":  round(spot_ret_21, 3),
            "63d_pct":  round(spot_ret_63, 3),
            "252d_pct": round(spot_ret_252, 3),
            "ann_realised_21d_pct": round(annualised_realised_21d, 3),
            "ann_realised_63d_pct": round(annualised_realised_63d, 3),
        },
        "carry": {
            "fair_pct":          round(fair_carry * 100, 4),
            "dynamic_usd_rate":  round(dynamic_usd_rate * 100, 3),
            "lease_rate_pct":    round(lease_rate * 100, 3),
            "storage_pct":       round(storage * 100, 3),
            "real_yield_used":   real_yield,
            "burden":            burden,
        },
        "excess_vs_carry": {
            "21d_pct":  round(excess_vs_carry_21d, 3),
            "63d_pct":  round(excess_vs_carry_63d, 3),
        },
        "parameters": {
            "fallback_usd_rate": usd_rate,
            "lease_rate":        lease_rate,
            "storage":           storage,
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    burden_color = {
        "HIGH_CARRY_HEADWIND":     "\033[31;1m",
        "MODERATE_CARRY_HEADWIND": "\033[31m",
        "LOW_CARRY_HEADWIND":      "\033[33m",
        "NEGATIVE_CARRY_TAILWIND": "\033[32;1m",
    }.get(r["carry"]["burden"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  GOLD CARRY TRADE ANALYZER")
    print(SEP)
    print(f"  Price source:    {r['price_source']}")
    print(f"  Price:           ${r['price_now']:,.2f}")
    print()

    print(f"  SPOT RETURNS")
    sr = r["spot_returns"]
    print(f"  21d:               {sr['21d_pct']:+.2f}%  (ann {sr['ann_realised_21d_pct']:+.2f}%)")
    print(f"  63d:               {sr['63d_pct']:+.2f}%  (ann {sr['ann_realised_63d_pct']:+.2f}%)")
    print(f"  252d:              {sr['252d_pct']:+.2f}%")
    print()

    c = r["carry"]
    print(f"  CARRY DECOMPOSITION  (annualised %)")
    print(f"  {'─' * 50}")
    print(f"  USD rate (dyn):    {c['dynamic_usd_rate']:.3f}%  "
          f"(real_yield={c['real_yield_used'] if c['real_yield_used'] is not None else 'n/a'})")
    print(f"  Lease rate:        −{c['lease_rate_pct']:.3f}%")
    print(f"  Storage:           −{c['storage_pct']:.3f}%")
    print(f"  Fair carry cost:   {c['fair_pct']:+.3f}%")
    print(f"  Burden:            {burden_color}{c['burden']}\033[0m")
    print()

    ex = r["excess_vs_carry"]
    print(f"  EXCESS RETURN vs CARRY EXPECTATION")
    print(f"  21d ann:  {ex['21d_pct']:+.3f}%")
    print(f"  63d ann:  {ex['63d_pct']:+.3f}%")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Carry Trade Analyzer")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--usd-rate", type=float, default=DEFAULT_USD_RATE)
    parser.add_argument("--lease", type=float, default=DEFAULT_LEASE_RATE)
    parser.add_argument("--storage", type=float, default=DEFAULT_STORAGE)
    parser.add_argument("--dte", type=int, default=60)
    args = parser.parse_args()
    run_carry_analyzer(
        lookback=args.lookback,
        usd_rate=args.usd_rate,
        lease_rate=args.lease,
        storage=args.storage,
        days_to_expiry=args.dte,
    )
