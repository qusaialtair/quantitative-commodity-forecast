#!/usr/bin/env python3
"""
Volatility Targeting & Risk Budgeting Engine
=============================================
Top-down portfolio sizing:
  - Set a target annualised portfolio vol (default 12%)
  - Compute the leverage factor that maps current realised vol → target
  - Allocate the resulting "risk budget" across alpha sources two ways:
      * ir_weighted   weights ∝ Information Ratio (favour edges that paid)
      * equal_risk    weights inverse-proportional to source vol
                       (each source contributes equal portfolio variance)
  - Cap leverage in [0.25, 2.0] to avoid extremes

Inputs:
  - data/vol_surface.json        for current 21d realised vol of gold
  - data/alpha_attribution.json  for per-source vol and IR

Outputs:
  data/vol_target_budget.json with:
    - target_vol_pct, current_vol_pct
    - leverage_factor (raw and capped)
    - per-source ir_weighted and equal_risk allocations
    - per-source ex-ante risk contribution
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "vol_target_budget.json"

DEFAULT_TARGET_VOL_PCT = 12.0
LEVERAGE_FLOOR = 0.25
LEVERAGE_CEILING = 2.0
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _load_current_vol() -> float | None:
    """Pull 21d realised vol (annualised pct) from vol_surface.json."""
    vs_path = DATA_DIR / "vol_surface.json"
    try:
        if vs_path.exists():
            vs = json.loads(vs_path.read_text())
            v = vs.get("term_structure", {}).get("rv_21d")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return None


def _load_source_metrics() -> dict:
    """Pull per-source IR + vol from alpha_attribution.json."""
    aa_path = DATA_DIR / "alpha_attribution.json"
    out = {}
    try:
        if aa_path.exists():
            aa = json.loads(aa_path.read_text())
            full = aa.get("full_history", {})
            irs = aa.get("information_ratios", {})
            for src in aa.get("sources", []):
                out[src] = {
                    "ann_vol_pct": float(full.get(src, {}).get("ann_vol_pct", 0)),
                    "sharpe":      float(full.get(src, {}).get("sharpe", 0)),
                    "ir":          float(irs.get(src, {}).get("information_ratio", 0)),
                    "ann_return_pct": float(full.get(src, {}).get("ann_return_pct", 0)),
                }
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _ir_weighted_allocation(source_metrics: dict) -> dict:
    """Weights proportional to max(IR, 0). Normalised to sum to 1."""
    raw = {s: max(0.0, m["ir"]) for s, m in source_metrics.items()}
    total = sum(raw.values())
    if total <= 1e-9:
        # No positive IR — fall back to equal-weight
        n = max(1, len(source_metrics))
        return {s: 1.0 / n for s in source_metrics}
    return {s: v / total for s, v in raw.items()}


def _equal_risk_allocation(source_metrics: dict) -> dict:
    """Inverse-vol weights so each source contributes equal portfolio variance
    (assuming zero pairwise correlation — first-order risk parity)."""
    inv = {s: 1.0 / max(m["ann_vol_pct"], 1e-3) for s, m in source_metrics.items()}
    total = sum(inv.values())
    if total <= 1e-9:
        n = max(1, len(source_metrics))
        return {s: 1.0 / n for s in source_metrics}
    return {s: v / total for s, v in inv.items()}


def _risk_contribution(weights: dict, source_metrics: dict) -> dict:
    """
    Ex-ante variance contribution of each source assuming zero correlation
    (a first-order approximation; correlations live in alpha_attribution).
    """
    var_contribs = {}
    total_var = 0.0
    for s, w in weights.items():
        v = source_metrics[s]["ann_vol_pct"] / 100.0
        contrib = (w * v) ** 2
        var_contribs[s] = contrib
        total_var += contrib
    if total_var <= 0:
        return {s: 0.0 for s in weights}
    return {s: round(c / total_var * 100, 2) for s, c in var_contribs.items()}


def _portfolio_vol_estimate(weights: dict, source_metrics: dict) -> float:
    """Zero-correlation portfolio vol estimate (annualised %)."""
    var = 0.0
    for s, w in weights.items():
        v = source_metrics[s]["ann_vol_pct"] / 100.0
        var += (w * v) ** 2
    return float(np.sqrt(var) * 100)


def compute_leverage(
    target_vol_pct: float,
    current_vol_pct: float,
    floor: float = LEVERAGE_FLOOR,
    ceiling: float = LEVERAGE_CEILING,
) -> tuple[float, float]:
    """Return (raw_leverage, capped_leverage)."""
    if current_vol_pct <= 1e-3:
        return 1.0, 1.0
    raw = target_vol_pct / current_vol_pct
    capped = float(np.clip(raw, floor, ceiling))
    return float(raw), capped


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_vol_target(
    target_vol_pct: float = DEFAULT_TARGET_VOL_PCT,
) -> dict:
    current_vol = _load_current_vol()
    source_metrics = _load_source_metrics()

    if current_vol is None or current_vol <= 0:
        # Fallback: assume current vol equals target so leverage = 1
        current_vol = target_vol_pct

    raw_lev, capped_lev = compute_leverage(target_vol_pct, current_vol)

    if not source_metrics:
        # No source data — degenerate output
        result = {
            "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target_vol_pct":   target_vol_pct,
            "current_vol_pct":  current_vol,
            "leverage_raw":     round(raw_lev, 3),
            "leverage_capped":  round(capped_lev, 3),
            "leverage_floor":   LEVERAGE_FLOOR,
            "leverage_ceiling": LEVERAGE_CEILING,
            "n_sources":        0,
            "ir_weighted":      {},
            "equal_risk":       {},
            "warning":          "No alpha_attribution data — run scripts/alpha_attribution.py first",
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _print_report(result)
        return result

    ir_w = _ir_weighted_allocation(source_metrics)
    er_w = _equal_risk_allocation(source_metrics)

    ir_contribs = _risk_contribution(ir_w, source_metrics)
    er_contribs = _risk_contribution(er_w, source_metrics)

    ir_blend_vol = _portfolio_vol_estimate(ir_w, source_metrics)
    er_blend_vol = _portfolio_vol_estimate(er_w, source_metrics)

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_vol_pct":    target_vol_pct,
        "current_vol_pct":   round(current_vol, 3),
        "leverage_raw":      round(raw_lev, 3),
        "leverage_capped":   round(capped_lev, 3),
        "leverage_floor":    LEVERAGE_FLOOR,
        "leverage_ceiling":  LEVERAGE_CEILING,
        "n_sources":         len(source_metrics),
        "source_metrics":    {
            s: {
                "ir":            round(m["ir"], 3),
                "sharpe":        round(m["sharpe"], 3),
                "ann_vol_pct":   round(m["ann_vol_pct"], 3),
                "ann_return_pct":round(m["ann_return_pct"], 3),
            } for s, m in source_metrics.items()
        },
        "ir_weighted": {
            "weights":             {s: round(w, 4) for s, w in ir_w.items()},
            "risk_contrib_pct":    ir_contribs,
            "blend_vol_pct":       round(ir_blend_vol, 3),
        },
        "equal_risk": {
            "weights":             {s: round(w, 4) for s, w in er_w.items()},
            "risk_contrib_pct":    er_contribs,
            "blend_vol_pct":       round(er_blend_vol, 3),
        },
        "guidance": {
            "deploy_pct_of_capital": round(min(1.0, capped_lev) * 100, 2),
            "leverage_action":       (
                "DELEVERAGE" if raw_lev < 0.85
                else "LEVER_UP" if raw_lev > 1.15
                else "MAINTAIN"
            ),
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
    print(f"\n{SEP}")
    print(f"  VOLATILITY TARGETING & RISK BUDGETING")
    print(SEP)
    print(f"  Target Vol:     {r['target_vol_pct']:.1f}% annualised")
    print(f"  Current Vol:    {r.get('current_vol_pct', 0):.2f}% (gold 21d realised)")
    print()
    print(f"  LEVERAGE")
    print(f"  {'─' * 40}")
    print(f"  Raw:            {r['leverage_raw']:.3f}×")
    print(f"  Capped:         {r['leverage_capped']:.3f}×  "
          f"(floor {r['leverage_floor']:.2f}, ceiling {r['leverage_ceiling']:.2f})")
    print(f"  Action:         {r.get('guidance', {}).get('leverage_action', 'MAINTAIN')}")
    print()

    if r.get("warning"):
        print(f"  ⚠ {r['warning']}")
        print(SEP)
        return

    sm = r["source_metrics"]
    print(f"  PER-SOURCE METRICS")
    print(f"  {'─' * 58}")
    print(f"  {'source':<16s}  {'IR':>7s}  {'Sharpe':>7s}  {'Vol %':>7s}  {'Ret %':>7s}")
    for s, m in sm.items():
        print(
            f"  {s:<16s}  {m['ir']:>+7.3f}  {m['sharpe']:>+7.3f}  "
            f"{m['ann_vol_pct']:>7.2f}  {m['ann_return_pct']:>+7.2f}"
        )
    print()

    print(f"  IR-WEIGHTED RISK BUDGET (pre-leverage)")
    print(f"  {'─' * 58}")
    irw = r["ir_weighted"]
    for s, w in irw["weights"].items():
        rc = irw["risk_contrib_pct"].get(s, 0)
        print(f"  {s:<16s}  weight={w:>7.2%}  risk_contrib={rc:>5.1f}%")
    print(f"  Blend vol estimate: {irw['blend_vol_pct']:.2f}%")
    print()

    print(f"  EQUAL-RISK BUDGET (inverse-vol)")
    print(f"  {'─' * 58}")
    erw = r["equal_risk"]
    for s, w in erw["weights"].items():
        rc = erw["risk_contrib_pct"].get(s, 0)
        print(f"  {s:<16s}  weight={w:>7.2%}  risk_contrib={rc:>5.1f}%")
    print(f"  Blend vol estimate: {erw['blend_vol_pct']:.2f}%")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vol Targeting & Risk Budgeting")
    parser.add_argument("--target-vol", type=float, default=DEFAULT_TARGET_VOL_PCT,
                        help="Annualised target vol in percent (default 12)")
    args = parser.parse_args()
    run_vol_target(target_vol_pct=args.target_vol)
