#!/usr/bin/env python3
"""
Macro Nowcasting Composite
============================
Single fused score that captures every macro signal the system can see
right now. Combines eight independent components and converts each to a
gold-bullish (+) / gold-bearish (-) z-score, then averages.

Components (positive = bullish gold):
  1. real_yields_inv     real_yield_10y z-score, inverted
  2. copper_gold_inv     copper-gold ratio z-score, inverted (low growth)
  3. cot_contrarian      COT MM-net z-score, inverted (avoid crowded longs)
  4. dcc_stress          # of DCC-stressed pairs (risk-off proxy)
  5. geo_risk            (pplx_geo_risk − 0.5) × 2     (-1..+1)
  6. vol_regime          {LOW: -0.5, NORMAL: 0, ELEVATED: +0.5, EXTREME: +1.5}
  7. etf_flows           {NET_INFLOW: +1, MIXED: 0, NET_OUTFLOW: -1}
  8. sentiment           (avg_oracle_sentiment − 0.5) × 2

Normalised composite ∈ roughly [-2, +2] →
    > +1.0       STRONGLY_BULLISH
    +0.4 .. +1   BULLISH
    -0.4 .. +0.4 NEUTRAL
    -1.0 .. -0.4 BEARISH
    < -1.0       STRONGLY_BEARISH

Output: data/macro_nowcast.json
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
OUTPUT_FILE = DATA_DIR / "macro_nowcast.json"
ALT_CSV = DATA_DIR / "alt_data.csv"

LINE_W = 62
SEP = "━" * LINE_W


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _zscore_last(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 20:
        return 0.0
    mean = s.mean()
    std = s.std(ddof=1)
    if std <= 1e-9:
        return 0.0
    return float((s.iloc[-1] - mean) / std)


def _component_real_yields(alt: pd.DataFrame) -> tuple[float, dict]:
    if "real_yield_10y" not in alt.columns:
        return 0.0, {"available": False}
    z = _zscore_last(alt["real_yield_10y"])
    return -z, {"z": round(z, 3), "interpretation": "inverted; lower yields → bullish"}


def _component_copper_gold(alt: pd.DataFrame) -> tuple[float, dict]:
    col = "copper_gold_ratio_zscore"
    if col in alt.columns:
        v = float(alt[col].dropna().iloc[-1]) if alt[col].dropna().size else 0.0
        return -v, {"z": round(v, 3), "interpretation": "inverted; low growth → bullish"}
    return 0.0, {"available": False}


def _component_cot(alt: pd.DataFrame) -> tuple[float, dict]:
    col = "cot_gold_mm_net_zscore"
    if col in alt.columns:
        v = float(alt[col].dropna().iloc[-1]) if alt[col].dropna().size else 0.0
        return -v * 0.5, {"z": round(v, 3), "interpretation": "inverted×0.5; avoid crowded longs"}
    return 0.0, {"available": False}


def _component_dcc_stress() -> tuple[float, dict]:
    dcc = _safe_load_json(DATA_DIR / "dcc_garch.json")
    n = int(dcc.get("n_stressed", 0))
    if dcc:
        return n * 0.4, {"n_stressed": n, "interpretation": "stress → risk-off → bullish gold"}
    return 0.0, {"available": False}


def _component_geo_risk() -> tuple[float, dict]:
    ge = _safe_load_json(DATA_DIR / "geopolitical_events.json")
    score = ge.get("current_score")
    if score is None:
        return 0.0, {"available": False}
    val = (float(score) - 0.5) * 2.0
    return val, {"score": round(float(score), 3),
                 "regime": ge.get("regime"),
                 "interpretation": "elevated → bullish gold"}


def _component_vol_regime() -> tuple[float, dict]:
    vs = _safe_load_json(DATA_DIR / "vol_surface.json")
    regime = vs.get("vol_regime")
    mapping = {"LOW": -0.5, "NORMAL": 0.0, "ELEVATED": 0.5, "EXTREME": 1.5}
    if regime in mapping:
        return mapping[regime], {"regime": regime,
                                 "interpretation": "vol regime → flight-to-quality bid"}
    return 0.0, {"available": False}


def _component_etf_flows() -> tuple[float, dict]:
    ef = _safe_load_json(DATA_DIR / "etf_flows.json")
    headline = ef.get("headline")
    mapping = {"NET_INFLOW": 1.0, "MIXED": 0.0, "NET_OUTFLOW": -1.0}
    if headline in mapping:
        return mapping[headline], {
            "headline": headline,
            "gold_bucket_regime": ef.get("gold_bucket", {}).get("bucket_regime"),
            "interpretation": "ETF inflow → bullish",
        }
    return 0.0, {"available": False}


def _component_sentiment() -> tuple[float, dict]:
    ns = _safe_load_json(DATA_DIR / "news_sentiment.json")
    avg = ns.get("aggregate", {}).get("avg_sentiment")
    if avg is None:
        return 0.0, {"available": False}
    val = (float(avg) - 0.5) * 2.0
    return val, {"avg_sentiment": round(float(avg), 3),
                 "interpretation": "oracle sentiment > 0.5 → bullish"}


def _classify(composite: float) -> str:
    if composite > 1.0:  return "STRONGLY_BULLISH"
    if composite > 0.4:  return "BULLISH"
    if composite > -0.4: return "NEUTRAL"
    if composite > -1.0: return "BEARISH"
    return "STRONGLY_BEARISH"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_macro_nowcast() -> dict:
    # Load alt_data once
    if ALT_CSV.exists():
        alt = pd.read_csv(ALT_CSV, index_col=0, parse_dates=True)
    else:
        alt = pd.DataFrame()

    components = {}
    diagnostics = {}

    for name, fn in [
        ("real_yields_inv",     lambda: _component_real_yields(alt)),
        ("copper_gold_inv",     lambda: _component_copper_gold(alt)),
        ("cot_contrarian",      lambda: _component_cot(alt)),
        ("dcc_stress",          _component_dcc_stress),
        ("geo_risk",            _component_geo_risk),
        ("vol_regime",          _component_vol_regime),
        ("etf_flows",           _component_etf_flows),
        ("sentiment",           _component_sentiment),
    ]:
        try:
            val, diag = fn()
            components[name] = round(float(val), 4)
            diagnostics[name] = diag
        except Exception as exc:
            components[name] = 0.0
            diagnostics[name] = {"error": str(exc)}

    # Average over available components
    active = {k: v for k, v in components.items()
              if abs(v) > 1e-9 or diagnostics.get(k, {}).get("available", True) is not False}
    if active:
        composite = float(np.mean(list(active.values())))
    else:
        composite = 0.0

    regime = _classify(composite)

    # Identify top drivers (by abs value)
    drivers = sorted(active.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "composite_score":   round(composite, 4),
        "regime":            regime,
        "n_components":      len(components),
        "n_active":          len(active),
        "components":        components,
        "diagnostics":       diagnostics,
        "top_drivers":       [{"name": k, "value": v} for k, v in drivers],
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
        "STRONGLY_BULLISH": "\033[32;1m",
        "BULLISH":          "\033[32m",
        "NEUTRAL":          "\033[36m",
        "BEARISH":          "\033[31m",
        "STRONGLY_BEARISH": "\033[31;1m",
    }.get(r["regime"], "\033[0m")

    print(f"\n{SEP}")
    print(f"  MACRO NOWCASTING COMPOSITE")
    print(SEP)
    print(f"  Components active:  {r['n_active']} / {r['n_components']}")
    print(f"  Composite score:    {r['composite_score']:+.3f}")
    print(f"  Regime:             {regime_color}{r['regime']}\033[0m")
    print()

    print(f"  COMPONENT BREAKDOWN")
    print(f"  {'─' * 58}")
    print(f"  {'component':<20s}  {'value':>8s}   notes")
    for name, val in r["components"].items():
        diag = r["diagnostics"].get(name, {})
        note = ""
        for k in ("z", "score", "regime", "n_stressed", "avg_sentiment", "headline"):
            if k in diag:
                note = f"{k}={diag[k]}"
                break
        if not note and diag.get("available") is False:
            note = "no data"
        if not note and diag.get("interpretation"):
            note = diag["interpretation"][:32]
        print(f"  {name:<20s}  {val:>+8.3f}   {note}")
    print()

    if r["top_drivers"]:
        print(f"  TOP DRIVERS")
        for d in r["top_drivers"]:
            print(f"    {d['name']:<20s}  {d['value']:>+.3f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Nowcasting Composite")
    args = parser.parse_args()
    run_macro_nowcast()
