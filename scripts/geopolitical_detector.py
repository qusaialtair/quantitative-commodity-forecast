#!/usr/bin/env python3
"""
Geopolitical Event Detector
=============================
Classifies the current geopolitical risk regime from the Perplexity
`pplx_geo_risk` score and detects day-over-day spikes that signal a fresh
event (sanctions, escalation, summit collapse, etc.).

Regimes (pplx_geo_risk ∈ [0, 1]):
    CALM        < 0.30
    ELEVATED    0.30 – 0.50
    HIGH        0.50 – 0.75
    CRISIS      ≥ 0.75

Event detection:
    Δ > 0.15 day-over-day  → EVENT_DETECTED (high-priority)
    z-score > +2σ vs window → REGIME_SHIFT (medium-priority)

Reads from data/cb_speech_history.csv (populated by cb_speech_analyzer)
so no extra API call is made.

Output: data/geopolitical_events.json
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

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "geopolitical_events.json"
HISTORY_CSV = DATA_DIR / "cb_speech_history.csv"

LINE_W = 62
SEP = "━" * LINE_W


def _classify(score: float) -> str:
    if score < 0.30:
        return "CALM"
    if score < 0.50:
        return "ELEVATED"
    if score < 0.75:
        return "HIGH"
    return "CRISIS"


def _load_geo_history() -> pd.Series:
    if not HISTORY_CSV.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(HISTORY_CSV)
    if "pplx_geo_risk" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["pplx_geo_risk"].dropna().sort_index()


def run_geopolitical_detector() -> dict:
    series = _load_geo_history()
    n = len(series)

    if n == 0:
        result = {
            "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_obs":         0,
            "current_score": None,
            "regime":        "NO_DATA",
            "event_flag":    False,
            "warning":       "No history in cb_speech_history.csv — run cb_speech_analyzer first.",
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _print_report(result)
        return result

    current = float(series.iloc[-1])
    regime = _classify(current)

    # Day-over-day delta
    delta_dod = float(series.iloc[-1] - series.iloc[-2]) if n >= 2 else 0.0
    event_dod = bool(abs(delta_dod) > 0.15)

    # Z-score vs full window
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else 0.0
    z = (current - mean) / std if std > 1e-6 else 0.0
    regime_shift = bool(abs(z) > 2.0)

    # 7-day momentum
    if n >= 7:
        mean7 = float(series.tail(7).mean())
        mom7 = current - mean7
    else:
        mean7 = mean
        mom7 = 0.0

    # Past spike dates (any |delta| > 0.15 in history)
    spikes = []
    if n >= 2:
        diffs = series.diff().dropna()
        for d, v in diffs.items():
            if abs(v) > 0.15:
                spikes.append({
                    "date":  str(d.date()),
                    "delta": round(float(v), 4),
                    "score": round(float(series.loc[d]), 4),
                })

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_obs":          int(n),
        "current_score":  round(current, 4),
        "regime":         regime,
        "delta_dod":      round(delta_dod, 4),
        "event_flag":     event_dod,
        "z_score":        round(z, 3),
        "regime_shift":   regime_shift,
        "mean_full":      round(mean, 4),
        "std_full":       round(std, 4),
        "mean_7d":        round(mean7, 4),
        "momentum_7d":    round(mom7, 4),
        "spike_history":  spikes,
        "priority":       (
            "HIGH"   if event_dod
            else "MEDIUM" if regime_shift or regime in ("HIGH", "CRISIS")
            else "LOW"
        ),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    regime_color = {
        "CALM":     "\033[32m",
        "ELEVATED": "\033[33m",
        "HIGH":     "\033[31m",
        "CRISIS":   "\033[31;1m",
        "NO_DATA":  "\033[90m",
    }.get(r["regime"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  GEOPOLITICAL EVENT DETECTOR")
    print(SEP)
    if r["n_obs"] == 0:
        print(f"  ⚠ {r.get('warning', 'No data')}")
        print(SEP)
        return

    print(f"  History rows:   {r['n_obs']}")
    print(f"  Current score:  {r['current_score']:.3f}")
    print(f"  Regime:         {regime_color}{r['regime']}\033[0m")
    print(f"  Day-over-day:   {r['delta_dod']:+.3f}")
    print(f"  Z-score:        {r['z_score']:+.2f}")
    print(f"  7d momentum:    {r['momentum_7d']:+.3f}")
    print()
    print(f"  Priority:       {r['priority']}")
    if r["event_flag"]:
        print(f"  ⚠ EVENT_DETECTED — geo_risk moved {r['delta_dod']:+.3f} day-over-day")
    if r["regime_shift"]:
        print(f"  ⚠ REGIME_SHIFT — z-score {r['z_score']:+.2f}σ from mean")
    if r["spike_history"]:
        print()
        print(f"  HISTORICAL SPIKES ({len(r['spike_history'])})")
        for sp in r["spike_history"][-5:]:
            print(f"    {sp['date']}  Δ={sp['delta']:+.3f}  score={sp['score']:.3f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geopolitical Event Detector")
    args = parser.parse_args()
    run_geopolitical_detector()
