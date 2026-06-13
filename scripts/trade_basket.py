#!/usr/bin/env python3
"""
Multi-Asset Trade Basket  (Phase XII Stage 63)
================================================
Generalises the single trade idea (Stage 57) into a top-N basket spanning
the halal universe + metals core. For each candidate ticker the engine
computes:

  - 21d momentum                       directional signal
  - 21d realised vol                   risk floor
  - cross-sectional score              (momentum / vol) z-score
  - sector / class tilt                from macro_regime quadrant
  - position fraction                  from cross-section + vol target

Output:
  - top_long_basket    high-score names with positive directional bias
  - top_short_basket   low-score names (only when macro allows shorts)
  - cash_pct           1 − sum(long + short)
  - per-ticker reasoning

Honors halal_universe.json + the trade-idea size stack so the basket
total weight respects every existing risk cap.

Saved to: data/trade_basket.json
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
OUTPUT_FILE = DATA_DIR / "trade_basket.json"

DEFAULT_TOP_N_LONG = 5
DEFAULT_TOP_N_SHORT = 0  # operator-toggled; default keep long-only
DEFAULT_GROSS_EXPOSURE_CAP_PCT = 25.0

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


def _fetch_panel(tickers: list[str], lookback: str = "3mo") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers, period=lookback, interval="1d",
            progress=False, auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = tickers[:1]
        return close.dropna(how="all").ffill().dropna()
    except Exception:
        return pd.DataFrame()


def _score_universe(close: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional score = 21d momentum / 21d vol."""
    returns = close.pct_change()
    mom_21 = (close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else pd.Series(0, index=close.columns)
    vol_21 = returns.tail(21).std() * np.sqrt(252)
    score = mom_21 / vol_21.replace(0, np.nan)

    df = pd.DataFrame({
        "mom_21":  mom_21,
        "vol_21":  vol_21,
        "score":   score,
    }).dropna()
    # Z-score normalisation
    if len(df) >= 3 and df["score"].std() > 0:
        df["score_z"] = (df["score"] - df["score"].mean()) / df["score"].std()
    else:
        df["score_z"] = 0.0
    return df


def _macro_tilt() -> dict:
    """Read macro_regime asset tilts; default neutral."""
    mr = _load("macro_regime.json")
    return mr.get("asset_tilts", {}) or {}


def _size_caps() -> tuple[float, float]:
    """Compute total gross-exposure cap from existing risk engines."""
    cap_pct = DEFAULT_GROSS_EXPOSURE_CAP_PCT
    # Vol target leverage
    vt = _load("vol_target_budget.json")
    if vt:
        lev = float(vt.get("leverage_capped") or 1.0)
        cap_pct *= lev
    # Drawdown tier
    dd = _load("drawdown_controller.json")
    if dd:
        cap_pct *= float(dd.get("sizing_multiplier") or 1.0)
    # Vol surface
    vs = _load("vol_surface.json")
    if vs:
        cap_pct *= float(vs.get("actions", {}).get("kelly_fraction_multiplier") or 1.0)
    return max(0.0, min(40.0, cap_pct)), DEFAULT_GROSS_EXPOSURE_CAP_PCT


def run_trade_basket(
    top_n_long: int = DEFAULT_TOP_N_LONG,
    top_n_short: int = DEFAULT_TOP_N_SHORT,
) -> dict:
    halal = _load("halal_universe.json")
    halal_tickers = halal.get("tickers", []) or []
    metals = ["GC=F", "SI=F"]

    candidates = list(set(halal_tickers + metals))
    if not candidates:
        result = {
            "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warning":       "No halal-screened universe available; run halal_screener.py",
            "candidates":    0,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        return result

    close = _fetch_panel(candidates)
    if close.empty:
        return {"warning": "Price panel empty", "candidates": len(candidates)}

    scored = _score_universe(close)
    tilts = _macro_tilt()
    # Apply macro tilt: bonus z-score for assets the macro quadrant favours
    for t in scored.index:
        bias = 0
        if t.startswith("GC=F") or t.startswith("SI=F"):
            bias = float(tilts.get("GC=F", 0))
        elif t in ("SPY", "IVV", "VOO"):
            bias = float(tilts.get("SPY", 0))
        scored.loc[t, "tilted_score"] = scored.loc[t, "score_z"] + bias * 0.3

    scored = scored.sort_values("tilted_score", ascending=False)
    gross_cap, gross_target = _size_caps()

    # Long basket = top N positive
    longs = scored.head(top_n_long).copy()
    longs = longs[longs["tilted_score"] > 0]
    # Short basket
    shorts = scored.tail(top_n_short).copy() if top_n_short > 0 else pd.DataFrame()
    if not shorts.empty:
        shorts = shorts[shorts["tilted_score"] < 0]

    # Weight = score-proportional within the gross cap
    long_total = float(longs["tilted_score"].sum()) if not longs.empty else 0.0
    short_total = float(abs(shorts["tilted_score"]).sum()) if not shorts.empty else 0.0

    long_alloc_pct = (long_total / (long_total + short_total + 1e-9)) * gross_cap if (long_total + short_total) > 0 else 0
    short_alloc_pct = gross_cap - long_alloc_pct

    long_basket = []
    if not longs.empty and long_total > 0:
        for tick, row in longs.iterrows():
            w = (row["tilted_score"] / long_total) * long_alloc_pct
            long_basket.append({
                "ticker":          tick,
                "weight_pct":      round(float(w), 3),
                "score_z":         round(float(row["score_z"]), 3),
                "tilted_score":    round(float(row["tilted_score"]), 3),
                "mom_21_pct":      round(float(row["mom_21"]) * 100, 3),
                "ann_vol_pct":     round(float(row["vol_21"]) * 100, 3),
            })

    short_basket = []
    if not shorts.empty and short_total > 0:
        for tick, row in shorts.iterrows():
            w = (abs(row["tilted_score"]) / short_total) * short_alloc_pct
            short_basket.append({
                "ticker":       tick,
                "weight_pct":   round(-float(w), 3),
                "score_z":      round(float(row["score_z"]), 3),
                "tilted_score": round(float(row["tilted_score"]), 3),
            })

    deployed_pct = sum(b["weight_pct"] for b in long_basket) - sum(b["weight_pct"] for b in short_basket)
    cash_pct = round(100 - abs(deployed_pct), 3)

    result = {
        "generated_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates":       len(candidates),
        "n_scored":           int(len(scored)),
        "gross_exposure_cap_pct":  gross_cap,
        "gross_exposure_target_pct":gross_target,
        "long_alloc_pct":     round(long_alloc_pct, 3),
        "short_alloc_pct":    round(short_alloc_pct, 3),
        "cash_pct":           cash_pct,
        "long_basket":        long_basket,
        "short_basket":       short_basket,
        "n_long":             len(long_basket),
        "n_short":            len(short_basket),
        "macro_tilts":        tilts,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  TRADE BASKET (multi-asset)\n{SEP}")
    print(f"  Candidates:         {r.get('n_candidates', 0)}")
    print(f"  Scored:             {r.get('n_scored', 0)}")
    print(f"  Gross cap:          {r.get('gross_exposure_cap_pct', 0):.1f}% "
          f"(target {r.get('gross_exposure_target_pct', 0):.1f}%)")
    print(f"  Long allocation:    {r.get('long_alloc_pct', 0):.2f}%")
    print(f"  Short allocation:   {r.get('short_alloc_pct', 0):.2f}%")
    print(f"  Cash:               {r.get('cash_pct', 0):.2f}%")
    print()
    if r.get("long_basket"):
        print(f"  LONG BASKET ({len(r['long_basket'])} names)")
        print(f"  {'─' * 58}")
        print(f"  {'ticker':<10s}  {'weight':>7s}  {'mom_21':>7s}  {'vol':>6s}  {'z':>6s}")
        for b in r["long_basket"]:
            print(f"  {b['ticker']:<10s}  {b['weight_pct']:>6.2f}%  "
                  f"{b['mom_21_pct']:>+6.2f}%  {b['ann_vol_pct']:>5.1f}%  "
                  f"{b['tilted_score']:>+6.2f}")
    if r.get("short_basket"):
        print()
        print(f"  SHORT BASKET ({len(r['short_basket'])} names)")
        print(f"  {'─' * 58}")
        for b in r["short_basket"]:
            print(f"  {b['ticker']:<10s}  {b['weight_pct']:>6.2f}%  "
                  f"score_z {b['score_z']:+.2f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-asset Trade Basket")
    parser.add_argument("--longs", type=int, default=DEFAULT_TOP_N_LONG)
    parser.add_argument("--shorts", type=int, default=DEFAULT_TOP_N_SHORT)
    args = parser.parse_args()
    run_trade_basket(top_n_long=args.longs, top_n_short=args.shorts)
