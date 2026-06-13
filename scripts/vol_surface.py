#!/usr/bin/env python3
"""
Volatility Surface Monitor
===========================
Tracks the realized-vol term structure across multiple horizons
(5d / 10d / 21d / 63d / 252d), classifies the current vol regime,
detects curve shape (contango vs backwardation) and the expansion /
contraction phase, then maps the result to actionable Kelly multipliers
and stop-loss widths.

Outputs:
  - Realized vol at each horizon, annualised, plus its history percentile
  - Vol-of-vol (rolling std of the 21d vol)
  - Vol regime classification (LOW / NORMAL / ELEVATED / EXTREME)
  - Term-structure shape (CONTANGO / BACKWARDATION / FLAT)
  - Phase (EXPANDING / CONTRACTING / STABLE)
  - Suggested Kelly fraction multiplier
  - Suggested stop-loss width multiplier (× ATR)

Saved to: data/vol_surface.json

Usage:
    python3 scripts/vol_surface.py
    python3 scripts/vol_surface.py --ticker SI=F --lookback 8y
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
OUTPUT_FILE = DATA_DIR / "vol_surface.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"

HORIZONS = [5, 10, 21, 63, 252]
SQ252 = float(np.sqrt(252))

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


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualised realized vol."""
    return returns.rolling(window).std() * SQ252


def _percentile_of(value: float, history: pd.Series) -> float:
    """Percentile rank of value in history (0..100)."""
    h = history.dropna()
    if len(h) < 2 or not np.isfinite(value):
        return 50.0
    return float((h < value).mean() * 100.0)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def _classify_regime(vol_21d_pctile: float) -> str:
    if vol_21d_pctile < 25:
        return "LOW"
    if vol_21d_pctile < 75:
        return "NORMAL"
    if vol_21d_pctile < 90:
        return "ELEVATED"
    return "EXTREME"


def _classify_curve(rv_5d: float, rv_21d: float, rv_63d: float) -> tuple[str, float]:
    """Compare short, medium, long vol. Returns (shape, slope_pct)."""
    if rv_63d <= 0:
        return "FLAT", 0.0
    short = rv_5d
    long_ = rv_63d
    slope_pct = (short - long_) / long_ * 100.0
    if slope_pct > 10:
        return "BACKWARDATION", slope_pct
    if slope_pct < -10:
        return "CONTANGO", slope_pct
    return "FLAT", slope_pct


def _classify_phase(vol_21d: pd.Series) -> tuple[str, float]:
    """Compare current 21d vol to its 21d-trailing average."""
    recent = vol_21d.dropna().tail(42)
    if len(recent) < 22:
        return "STABLE", 0.0
    cur = float(recent.iloc[-1])
    prior_avg = float(recent.iloc[-22:-1].mean())
    if prior_avg <= 0:
        return "STABLE", 0.0
    change_pct = (cur - prior_avg) / prior_avg * 100.0
    if change_pct > 15:
        return "EXPANDING", change_pct
    if change_pct < -15:
        return "CONTRACTING", change_pct
    return "STABLE", change_pct


def _kelly_multiplier(regime: str, phase: str) -> float:
    """
    Map regime + phase to a Kelly fraction multiplier in [0.25, 1.0].
    Lower in turbulent / expanding regimes; full only when calm and stable.
    """
    base = {"LOW": 1.00, "NORMAL": 0.85, "ELEVATED": 0.55, "EXTREME": 0.30}[regime]
    if phase == "EXPANDING":
        base *= 0.70
    elif phase == "CONTRACTING":
        base *= 1.10
    return float(round(max(0.25, min(1.0, base)), 3))


def _stop_atr_multiplier(regime: str, phase: str) -> float:
    """
    Map regime + phase to a stop-loss width (× ATR).
    Wider stops in high-vol regimes to avoid being whipsawed.
    """
    base = {"LOW": 1.5, "NORMAL": 2.0, "ELEVATED": 2.5, "EXTREME": 3.0}[regime]
    if phase == "EXPANDING":
        base *= 1.15
    elif phase == "CONTRACTING":
        base *= 0.90
    return float(round(base, 2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_vol_surface(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    close = _fetch_close(ticker, lookback)
    returns = close.pct_change().dropna()
    cur_price = float(close.iloc[-1])

    # Compute term structure
    term_structure = {}
    term_structure_pctile = {}
    rv_series = {}
    for h in HORIZONS:
        rv = _realized_vol(returns, h)
        rv_series[h] = rv
        cur_v = rv.dropna().iloc[-1] if not rv.dropna().empty else 0.0
        term_structure[f"rv_{h}d"] = round(float(cur_v) * 100, 3)
        term_structure_pctile[f"rv_{h}d_pctile"] = round(
            _percentile_of(float(cur_v), rv), 1
        )

    # Vol-of-vol: rolling std of 21d vol (annualised)
    vov = rv_series[21].rolling(21).std()
    vov_cur = float(vov.dropna().iloc[-1]) if not vov.dropna().empty else 0.0
    vov_pctile = _percentile_of(vov_cur, vov)

    # Regime classification (using 21d vol percentile)
    vol_21d_pctile = term_structure_pctile["rv_21d_pctile"]
    regime = _classify_regime(vol_21d_pctile)

    # Curve shape
    curve, slope_pct = _classify_curve(
        term_structure["rv_5d"],
        term_structure["rv_21d"],
        term_structure["rv_63d"],
    )

    # Expansion vs contraction phase
    phase, phase_change_pct = _classify_phase(rv_series[21])

    # Action mapping
    kelly_mult = _kelly_multiplier(regime, phase)
    stop_mult = _stop_atr_multiplier(regime, phase)

    actions = {
        "kelly_fraction_multiplier": kelly_mult,
        "stop_atr_multiplier":       stop_mult,
        "trade_size_guidance":       (
            "Reduce sizing materially" if regime in ("ELEVATED", "EXTREME")
            else "Standard sizing" if regime == "NORMAL"
            else "Full sizing permitted"
        ),
        "stop_width_guidance":       f"{stop_mult:.2f}× ATR",
    }

    result = {
        "generated_at":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":                 ticker,
        "current_price":          round(cur_price, 2),
        "lookback":               lookback,
        "n_obs":                  int(len(returns)),
        "term_structure":         term_structure,
        "term_structure_pctile":  term_structure_pctile,
        "vol_of_vol":             round(vov_cur * 100, 3),
        "vol_of_vol_pctile":      round(vov_pctile, 1),
        "vol_regime":             regime,
        "vol_21d_pctile":         vol_21d_pctile,
        "curve_shape":            curve,
        "curve_slope_pct":        round(slope_pct, 2),
        "phase":                  phase,
        "phase_change_pct":       round(phase_change_pct, 2),
        "actions":                actions,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    regime_color = {
        "LOW":      "\033[32m",
        "NORMAL":   "\033[36m",
        "ELEVATED": "\033[33m",
        "EXTREME":  "\033[31;1m",
    }.get(r["vol_regime"], "\033[0m")

    phase_color = {
        "EXPANDING":   "\033[31m",
        "STABLE":      "\033[36m",
        "CONTRACTING": "\033[32m",
    }.get(r["phase"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  VOLATILITY SURFACE -- {r['ticker']}")
    print(SEP)
    print(f"  Current Price:  ${r['current_price']:,.2f}")
    print(f"  Observations:   {r['n_obs']}")
    print()

    print(f"  TERM STRUCTURE (annualised %)")
    print(f"  {'─' * 40}")
    for h in HORIZONS:
        v = r["term_structure"][f"rv_{h}d"]
        p = r["term_structure_pctile"][f"rv_{h}d_pctile"]
        bar = "█" * int(p / 5)
        print(f"  rv_{h:>3d}d:  {v:6.2f}%   p{p:5.1f}  {bar}")
    print()

    print(f"  VOL-OF-VOL")
    print(f"  {'─' * 40}")
    print(f"  Current:        {r['vol_of_vol']:.3f}%  (p{r['vol_of_vol_pctile']:.0f})")
    print()

    print(f"  REGIME / PHASE / SHAPE")
    print(f"  {'─' * 40}")
    print(f"  Regime:         {regime_color}{r['vol_regime']}\033[0m  (21d vol p{r['vol_21d_pctile']:.0f})")
    print(f"  Phase:          {phase_color}{r['phase']}\033[0m  ({r['phase_change_pct']:+.1f}% vs prior 21d avg)")
    print(f"  Curve Shape:    {r['curve_shape']}  (slope {r['curve_slope_pct']:+.1f}%)")
    print()

    a = r["actions"]
    print(f"  RECOMMENDED ACTIONS")
    print(f"  {'─' * 40}")
    print(f"  Kelly Mult:     {a['kelly_fraction_multiplier']:.2f}×")
    print(f"  Stop Width:     {a['stop_width_guidance']}")
    print(f"  Sizing:         {a['trade_size_guidance']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Surface Monitor")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_vol_surface(ticker=args.ticker, lookback=args.lookback)
