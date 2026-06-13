#!/usr/bin/env python3
"""
Central Bank Speech Analyzer
==============================
Hawkish-dovish scoring of central-bank communications. Pulls the live
`pplx_fed` score from oracle_scout (already used by the daily pipeline)
and maintains its own time-series in data/cb_speech_history.csv.

Score conventions (from oracle_scout):
    pplx_fed:        -1.0 (hawkish) → +1.0 (dovish)
    pplx_geo_risk:    0.0 (calm)    → +1.0 (crisis)
    pplx_phys_demand: 0.0 (weak)    → +1.0 (strong)
    pplx_macro:      -1.0 (bear)    → +1.0 (bull)

For pplx_fed the engine reports:
    - latest score & 7d EWMA
    - regime: HAWKISH / LEANING_HAWKISH / NEUTRAL / LEANING_DOVISH / DOVISH
    - shift detection: z-score vs full history
    - cumulative direction (running average over the last 21 entries)

Output: data/cb_speech.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "cb_speech.json"
HISTORY_CSV = DATA_DIR / "cb_speech_history.csv"

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Live snapshot
# ---------------------------------------------------------------------------
def _pull_live_pplx() -> dict:
    """Pull current pplx_* scores via PerplexityOracle (cached 6h by the SDK)."""
    import os
    try:
        from agents.perplexity_oracle import PerplexityOracle  # type: ignore
        oracle = PerplexityOracle(
            metal_name="gold",
            api_key=os.getenv("PERPLEXITY_API_KEY", ""),
        )
        scores = oracle.get_scores() or {}
        return {
            "pplx_fed":         float(scores.get("pplx_fed",         0.0)),
            "pplx_geo_risk":    float(scores.get("pplx_geo_risk",    0.5)),
            "pplx_phys_demand": float(scores.get("pplx_phys_demand", 0.0)),
            "pplx_macro":       float(scores.get("pplx_macro",       0.0)),
        }
    except Exception:
        return {}


def _append_history(snapshot: dict) -> None:
    """Append today's snapshot to cb_speech_history.csv."""
    if not snapshot:
        return
    today = date.today().isoformat()
    row = {"date": today, **snapshot}
    if HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV)
        # Avoid duplicate rows for the same day
        df = df[df["date"] != today]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("date")
    df.to_csv(HISTORY_CSV, index=False)


def _load_history() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=[
            "date", "pplx_fed", "pplx_geo_risk", "pplx_phys_demand", "pplx_macro"
        ])
    df = pd.read_csv(HISTORY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def _classify_fed(score: float) -> str:
    if score >= 0.5:
        return "DOVISH"
    if score >= 0.15:
        return "LEANING_DOVISH"
    if score <= -0.5:
        return "HAWKISH"
    if score <= -0.15:
        return "LEANING_HAWKISH"
    return "NEUTRAL"


def _analyze_series(values: pd.Series, name: str) -> dict:
    s = values.dropna()
    if len(s) == 0:
        return {"name": name, "n_obs": 0}
    latest = float(s.iloc[-1])
    if len(s) < 2:
        return {
            "name": name, "n_obs": int(len(s)),
            "latest": round(latest, 4),
            "ewma_7": round(latest, 4),
            "momentum": 0.0,
            "mean": round(latest, 4),
            "z_score": 0.0,
            "regime": _classify_fed(latest) if name == "pplx_fed" else "n/a",
        }
    ewma_short = float(s.ewm(span=min(7, len(s)), adjust=False).mean().iloc[-1])
    ewma_med = float(s.ewm(span=min(21, len(s)), adjust=False).mean().iloc[-1])
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    z = (latest - mean) / std if std > 1e-6 else 0.0
    regime = _classify_fed(latest) if name == "pplx_fed" else "n/a"
    return {
        "name": name,
        "n_obs": int(len(s)),
        "latest": round(latest, 4),
        "ewma_7": round(ewma_short, 4),
        "ewma_21": round(ewma_med, 4),
        "momentum": round(latest - ewma_med, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "z_score": round(z, 3),
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_cb_speech(record: bool = True) -> dict:
    # Optionally pull live + persist
    if record:
        snapshot = _pull_live_pplx()
        if snapshot:
            _append_history(snapshot)

    df = _load_history()
    n = len(df)

    series_summary = {}
    for col in ["pplx_fed", "pplx_geo_risk", "pplx_phys_demand", "pplx_macro"]:
        if col in df.columns:
            series_summary[col] = _analyze_series(df[col], col)

    fed = series_summary.get("pplx_fed", {})
    fed_regime = fed.get("regime", "NEUTRAL")

    # Determine if there's been a regime shift (compare last 5 to prior 5)
    shift_flag = False
    if "pplx_fed" in df.columns and len(df) >= 10:
        last5 = df["pplx_fed"].tail(5).mean()
        prior5 = df["pplx_fed"].iloc[-10:-5].mean()
        if abs(last5 - prior5) > 0.3:
            shift_flag = True

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_obs_history":  int(n),
        "history_path":   str(HISTORY_CSV),
        "series":         series_summary,
        "fed_regime":     fed_regime,
        "regime_shift_detected": bool(shift_flag),
        "fed_latest":     fed.get("latest"),
        "fed_z":          fed.get("z_score"),
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
        "HAWKISH":          "\033[31;1m",
        "LEANING_HAWKISH":  "\033[31m",
        "NEUTRAL":          "\033[36m",
        "LEANING_DOVISH":   "\033[32m",
        "DOVISH":           "\033[32;1m",
    }.get(r["fed_regime"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  CENTRAL BANK SPEECH ANALYZER")
    print(SEP)
    print(f"  History rows:   {r['n_obs_history']}")
    print(f"  History CSV:    {r['history_path']}")
    print()

    print(f"  PER-SERIES SUMMARY")
    print(f"  {'─' * 58}")
    print(
        f"  {'series':<18s}  {'latest':>7s}  {'7d':>6s}  {'21d':>6s}  "
        f"{'mom':>7s}  {'z':>6s}  {'regime':>14s}"
    )
    for name, s in r["series"].items():
        print(
            f"  {name:<18s}  "
            f"{s.get('latest', 0):>+7.3f}  "
            f"{s.get('ewma_7', 0):>+6.3f}  "
            f"{s.get('ewma_21', 0):>+6.3f}  "
            f"{s.get('momentum', 0):>+7.3f}  "
            f"{s.get('z_score', 0):>+6.2f}  "
            f"{s.get('regime', 'n/a'):>14s}"
        )
    print()

    print(f"  HEADLINE FED REGIME: {regime_color}{r['fed_regime']}\033[0m")
    if r["regime_shift_detected"]:
        print(f"  ⚠ REGIME SHIFT detected — last 5 vs prior 5 differ by >0.3")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central Bank Speech Analyzer")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip live API call; analyze existing history only")
    args = parser.parse_args()
    run_cb_speech(record=not args.no_record)
