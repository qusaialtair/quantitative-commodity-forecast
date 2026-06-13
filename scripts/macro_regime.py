#!/usr/bin/env python3
"""
Macro Regime Classifier (Bridgewater-style 4-quadrant)
=======================================================
Classifies the current macro environment into one of four quadrants based on
growth direction × inflation direction:

    GROWTH ↑ / INFL ↓   →  GOLDILOCKS    (risk-on, weakest for gold)
    GROWTH ↑ / INFL ↑   →  REFLATION     (commodities, gold neutral+)
    GROWTH ↓ / INFL ↑   →  STAGFLATION   (gold's best quadrant)
    GROWTH ↓ / INFL ↓   →  DEFLATION     (bonds, gold safe-haven spikes)

Growth proxies (composite z-score):
  - SPY 21d momentum             (+ growth)
  - Copper-gold ratio z-score    (+ growth)
  - DXY 21d momentum             (+ growth — strong dollar = US-led growth)

Inflation proxies (composite z-score):
  - Real yield 21d change inv.   (lower real yields = rising inflation)
  - DXY 21d momentum inverted    (weaker USD = imported inflation)
  - Gold 21d momentum            (gold rising often = inflation pricing)

Each quadrant maps to recommended asset tilts and a confidence score.

Output: data/macro_regime.json
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
OUTPUT_FILE = DATA_DIR / "macro_regime.json"
ALT_CSV = DATA_DIR / "alt_data.csv"

DEFAULT_LOOKBACK = "5y"

# Quadrant tilts (qualitative; -2 = strong avoid, +2 = strong overweight)
QUADRANT_TILTS = {
    "GOLDILOCKS":   {"GC=F": -1, "SI=F":  0, "SPY": +2, "TLT": -1, "DXY":  0},
    "REFLATION":    {"GC=F": +1, "SI=F": +2, "SPY": +1, "TLT": -2, "DXY": -1},
    "STAGFLATION":  {"GC=F": +2, "SI=F": +1, "SPY": -2, "TLT": -1, "DXY":  0},
    "DEFLATION":    {"GC=F": +1, "SI=F": -1, "SPY": -1, "TLT": +2, "DXY": +1},
}

QUADRANT_DESCRIPTIONS = {
    "GOLDILOCKS":  "Growth rising, inflation falling — risk-on. Sell gold, buy equities.",
    "REFLATION":   "Growth and inflation both rising — buy commodities and silver.",
    "STAGFLATION": "Growth falling, inflation rising — gold's strongest quadrant.",
    "DEFLATION":   "Growth and inflation both falling — bonds rally, gold as safe-haven.",
}

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def _fetch_close(ticker: str, lookback: str) -> pd.Series:
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].dropna()


def _fetch_panel(lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    g = _fetch_close("GC=F", lookback)
    s = _fetch_close("SPY", lookback)
    d = _fetch_close("DX-Y.NYB", lookback)
    return pd.DataFrame({"gold": g, "spy": s, "dxy": d}).ffill().dropna()


def _load_alt() -> pd.DataFrame | None:
    if not ALT_CSV.exists():
        return None
    try:
        df = pd.read_csv(ALT_CSV, index_col=0, parse_dates=True)
        return df.dropna(how="all")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------
def _zscore(series: pd.Series, lookback: int = 252) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < 21 or s.std() <= 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / s.std())


def _momentum_pct(series: pd.Series, days: int = 21) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-days - 1] - 1.0) * 100)


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------
def compute_growth_inflation(
    panel: pd.DataFrame, alt: pd.DataFrame | None
) -> dict:
    """
    Return composite growth and inflation scores (each in roughly [-3, +3])
    along with their component breakdowns.
    """
    # --- Growth components ---
    spy_mom = _momentum_pct(panel["spy"], 21)
    spy_mom_z = float(np.clip(spy_mom / 6.0, -3, 3))  # 6% / 21d ≈ +1σ

    dxy_mom = _momentum_pct(panel["dxy"], 21)
    dxy_mom_z = float(np.clip(dxy_mom / 2.5, -3, 3))  # 2.5% / 21d ≈ +1σ

    cg_z = 0.0
    if alt is not None and "copper_gold_ratio_zscore" in alt.columns:
        last = alt["copper_gold_ratio_zscore"].dropna().tail(1)
        if len(last):
            cg_z = float(last.iloc[0])

    growth_components = {
        "spy_21d_mom_pct":         round(spy_mom, 3),
        "spy_21d_mom_z":           round(spy_mom_z, 3),
        "dxy_21d_mom_pct":         round(dxy_mom, 3),
        "dxy_21d_mom_z":           round(dxy_mom_z, 3),
        "copper_gold_ratio_z":     round(cg_z, 3),
    }
    growth_score = round(spy_mom_z * 0.45 + dxy_mom_z * 0.20 + cg_z * 0.35, 3)

    # --- Inflation components ---
    real_yield_change_z = 0.0
    if alt is not None and "real_yield_10y" in alt.columns:
        ry = alt["real_yield_10y"].dropna()
        if len(ry) > 21:
            ry_chg = ry.diff(21)
            real_yield_change_z = _zscore(ry_chg)

    inflation_score_z = -real_yield_change_z  # rising real yields = lower expected inflation

    dxy_weak_z = -dxy_mom_z

    gold_mom = _momentum_pct(panel["gold"], 21)
    gold_mom_z = float(np.clip(gold_mom / 6.0, -3, 3))

    inflation_components = {
        "real_yield_21d_chg_z":  round(real_yield_change_z, 3),
        "real_yield_signal_z":   round(inflation_score_z, 3),
        "dxy_weakness_z":        round(dxy_weak_z, 3),
        "gold_21d_mom_pct":      round(gold_mom, 3),
        "gold_21d_mom_z":        round(gold_mom_z, 3),
    }
    inflation_score = round(
        inflation_score_z * 0.50 + dxy_weak_z * 0.25 + gold_mom_z * 0.25, 3
    )

    return {
        "growth_score":         growth_score,
        "inflation_score":      inflation_score,
        "growth_components":    growth_components,
        "inflation_components": inflation_components,
    }


# ---------------------------------------------------------------------------
# Quadrant classification
# ---------------------------------------------------------------------------
def classify_quadrant(growth: float, inflation: float) -> tuple[str, float]:
    """Return (quadrant_name, confidence in [0, 1])."""
    if growth >= 0 and inflation <= 0:
        name = "GOLDILOCKS"
    elif growth >= 0 and inflation > 0:
        name = "REFLATION"
    elif growth < 0 and inflation > 0:
        name = "STAGFLATION"
    else:
        name = "DEFLATION"

    magnitude = float(np.sqrt(growth ** 2 + inflation ** 2))
    confidence = float(np.clip(magnitude / 2.5, 0.0, 1.0))  # 2.5 = high-confidence
    return name, confidence


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_macro_regime(lookback: str = DEFAULT_LOOKBACK) -> dict:
    panel = _fetch_panel(lookback)
    alt = _load_alt()

    scores = compute_growth_inflation(panel, alt)
    g = scores["growth_score"]
    i = scores["inflation_score"]
    quadrant, conf = classify_quadrant(g, i)

    tilts = QUADRANT_TILTS[quadrant]
    description = QUADRANT_DESCRIPTIONS[quadrant]

    result = {
        "generated_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback":             lookback,
        "n_obs":                int(len(panel)),
        "growth_score":         g,
        "inflation_score":      i,
        "growth_components":    scores["growth_components"],
        "inflation_components": scores["inflation_components"],
        "quadrant":             quadrant,
        "confidence":           round(conf, 3),
        "description":          description,
        "asset_tilts":          tilts,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    quad_color = {
        "GOLDILOCKS":  "\033[32m",
        "REFLATION":   "\033[33m",
        "STAGFLATION": "\033[31m",
        "DEFLATION":   "\033[36m",
    }.get(r["quadrant"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  MACRO REGIME CLASSIFIER (4-quadrant)")
    print(SEP)
    print(f"  Observations:   {r['n_obs']}")
    print()

    print(f"  COMPOSITE SCORES")
    print(f"  {'─' * 50}")
    print(f"  Growth:         {r['growth_score']:+.3f}")
    print(f"  Inflation:      {r['inflation_score']:+.3f}")
    print()

    print(f"  GROWTH BREAKDOWN")
    for k, v in r["growth_components"].items():
        print(f"    {k:<28s} {v:+.3f}")
    print()

    print(f"  INFLATION BREAKDOWN")
    for k, v in r["inflation_components"].items():
        print(f"    {k:<28s} {v:+.3f}")
    print()

    print(f"  QUADRANT:       {quad_color}{r['quadrant']}\033[0m  "
          f"(confidence {r['confidence']:.1%})")
    print(f"  {r['description']}")
    print()

    print(f"  ASSET TILTS  (-2 strong avoid → +2 strong overweight)")
    print(f"  {'─' * 50}")
    for asset, tilt in r["asset_tilts"].items():
        bar = "█" * abs(tilt) if tilt != 0 else "·"
        side = "+" if tilt > 0 else "-" if tilt < 0 else " "
        print(f"  {asset:<10s}  {side}{bar:<4s}  ({tilt:+d})")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Regime Classifier")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_macro_regime(lookback=args.lookback)
