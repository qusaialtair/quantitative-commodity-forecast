#!/usr/bin/env python3
"""
Treasury Hedge Overlay — Phase XXV
====================================
Recommends a Treasury sleeve allocation (TLT long-duration or IEF
intermediate-duration) sized by macro regime quadrant and crisis tier.

The honest finding from Phase XX (multi-asset stress test) was that SPY +
gold alone did NOT rescue the 2008 GFC or 2022 inflation crisis windows —
only Treasuries hedge those regimes.  This engine produces the
recommendation; the operator decides whether to act on it (the active
allocation requires a one-time Sharia decision on US Treasury ETFs).

Inputs
------
- data/macro_regime.json    quadrant ∈ {GOLDILOCKS, REFLATION, STAGFLATION, DEFLATION}
- data/crisis_detector.json tier     ∈ {NORMAL, ELEVATED, STRESS, CRISIS, EXTREME}

Output
------
data/treasury_hedge.json with fields:
    mode                "SIGNAL_ONLY" (default) or "ACTIVE"
    instrument          "TLT" | "IEF" | null
    allocation_pct      0.0–20.0
    regime_quadrant     pass-through
    crisis_tier         pass-through
    reason              human-readable
    rule_matrix         the full lookup table (for transparency)
    sharia_note         caveat for the operator
    generated_at        ISO 8601

Modes
-----
SIGNAL_ONLY (default) — emits the recommendation but the multi-strategy
    trader will NOT execute it. Use for backtest / observation only.
ACTIVE — multi_strategy_trader will route the recommended allocation to
    IBKR (or stay in paper_internal book if EXECUTION_MODE=paper_internal).
    Set via env: TREASURY_HEDGE_MODE=ACTIVE

CLI
---
    python3 scripts/treasury_hedge_overlay.py                # run + write JSON
    python3 scripts/treasury_hedge_overlay.py --quiet        # suppress stdout
    python3 scripts/treasury_hedge_overlay.py --explain      # show rule matrix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.env_utils import env_bool, env_float

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "treasury_hedge.json"

SEP = "━" * 62

# Valid execution modes. LOCAL_ACTIVE executes the sleeve into the internal
# paper_internal book only. ACTIVE (IBKR routing) is DEPRECATED by the IBKR pivot.
VALID_MODES = {"SIGNAL_ONLY", "ACTIVE", "LOCAL_ACTIVE"}

# Default hard cap on sleeve size (% of book equity). Overridable via env.
DEFAULT_MAX_ALLOCATION_PCT = 20.0

# Sharia compliance fallback: while TREASURY_SHARIA_CLEARED is false we refuse
# coupon-bearing sovereign debt (TLT/IEF) and reroute the SAME sleeve budget to
# a physical-gold proxy. Tagged downstream so attribution can see the pivot.
FALLBACK_INSTRUMENT = "GLD"
FALLBACK_SUB_TAG = "sharia_fallback_gld"
SOVEREIGN_INSTRUMENTS = frozenset({"TLT", "IEF"})


def apply_sharia_gate(
    sovereign: str | None,
    pct: float,
    *,
    sharia_cleared: bool,
) -> dict[str, Any]:
    """Resolve the effective hedge instrument from sovereign reco and Sharia gate.

    Returns ``effective_instrument``, ``effective_allocation_pct``, ``sub_tag``,
    and ``gate_action`` for the trader and API layers.
    """
    if pct <= 0 or sovereign is None:
        return {
            "effective_instrument": None,
            "effective_allocation_pct": 0.0,
            "sub_tag": None,
            "gate_action": "NO_HEDGE",
        }
    if sharia_cleared:
        return {
            "effective_instrument": sovereign,
            "effective_allocation_pct": pct,
            "sub_tag": None,
            "gate_action": "CLEARED_SOVEREIGN",
        }
    return {
        "effective_instrument": FALLBACK_INSTRUMENT,
        "effective_allocation_pct": pct,
        "sub_tag": FALLBACK_SUB_TAG,
        "gate_action": "SHARIA_FALLBACK_GLD",
    }


# Rule matrix: (quadrant, crisis_tier) -> (instrument, allocation_pct, reason)
# Long-duration TLT is the panic hedge; IEF (7-10y) is the mild hedge.
# Both rates-fall = bond-prices-rise instruments — the bet pays off when
# the Fed eases or flight-to-quality compresses real yields.
HEDGE_RULES: dict[tuple[str, str], tuple[str | None, float, str]] = {
    # GOLDILOCKS — growth up, inflation down. Bonds are mediocre but not toxic.
    ("GOLDILOCKS", "NORMAL"):   (None, 0.0,
        "Risk-on regime, no defensive hedge needed."),
    ("GOLDILOCKS", "ELEVATED"): ("IEF", 5.0,
        "Risk-on but vol elevated — small intermediate-duration cushion."),
    ("GOLDILOCKS", "STRESS"):   ("TLT", 10.0,
        "Risk-on regime cracking — duration hedge against regime flip."),
    ("GOLDILOCKS", "CRISIS"):   ("TLT", 15.0,
        "Regime flipping out of risk-on — defensive duration."),
    ("GOLDILOCKS", "EXTREME"):  ("TLT", 20.0,
        "Extreme stress in risk-on regime — max duration hedge."),

    # REFLATION — growth up, inflation up. Bonds get hammered. AVOID.
    ("REFLATION", "NORMAL"):    (None, 0.0,
        "Growth + inflation rising — bonds underperform, no hedge."),
    ("REFLATION", "ELEVATED"):  (None, 0.0,
        "Reflation regime — bonds still net-negative, no hedge."),
    ("REFLATION", "STRESS"):    ("IEF", 5.0,
        "Reflation with stress — small intermediate position only."),
    ("REFLATION", "CRISIS"):    ("IEF", 8.0,
        "Reflation crisis — short-duration cushion only."),
    ("REFLATION", "EXTREME"):   ("IEF", 10.0,
        "Reflation extreme — capped intermediate duration."),

    # STAGFLATION — growth down, inflation up. The 2022 nightmare.
    # Long bonds get murdered (rates rise), but intermediate is less bad
    # and provides some hedge against the growth-down side.
    ("STAGFLATION", "NORMAL"):    ("IEF", 5.0,
        "Stagflation — IEF as partial hedge (long bonds too rate-sensitive)."),
    ("STAGFLATION", "ELEVATED"):  ("IEF", 10.0,
        "Stagflation w/ vol — intermediate duration buffer."),
    ("STAGFLATION", "STRESS"):    ("IEF", 12.0,
        "Stagflation stress — bigger IEF cushion, avoid long duration."),
    ("STAGFLATION", "CRISIS"):    ("TLT", 15.0,
        "Stagflation crisis — Fed-pivot bet via long duration."),
    ("STAGFLATION", "EXTREME"):   ("TLT", 20.0,
        "Stagflation panic — max TLT (Fed-pivot bet)."),

    # DEFLATION — growth down, inflation down. The classic bonds-shine regime.
    # 2008 GFC, 2020 COVID Q1. TLT is the textbook hedge.
    ("DEFLATION", "NORMAL"):    ("TLT", 10.0,
        "Deflation regime — classic TLT flight-to-quality."),
    ("DEFLATION", "ELEVATED"):  ("TLT", 15.0,
        "Deflation w/ stress — heavier TLT allocation."),
    ("DEFLATION", "STRESS"):    ("TLT", 18.0,
        "Deflation stress — near-max TLT."),
    ("DEFLATION", "CRISIS"):    ("TLT", 20.0,
        "Deflation crisis (2008-style) — max TLT."),
    ("DEFLATION", "EXTREME"):   ("TLT", 20.0,
        "Deflation extreme — capped at 20% TLT."),
}

SHARIA_NOTE = (
    "US Treasury ETFs (TLT/IEF) hold coupon-bearing sovereign debt. "
    "Many scholars permit short-duration sovereign hedging under dharurah "
    "(necessity for risk management); some prohibit any interest-bearing "
    "instrument. The strict gate TREASURY_SHARIA_CLEARED (default false) must "
    "be set true by the operator (after a fatwa) before any TLT/IEF allocation. "
    "While false, the sleeve does NOT sit idle in a hedge-warranted regime: it "
    "reroutes the same budget to the physical-gold proxy GLD "
    "(sub_tag=sharia_fallback_gld). Note: GLD is a paper-gold ETF and is itself "
    "subject to the project's physical-only metals review; it is used here only "
    "as a local-simulation defensive proxy and is never routed to a broker."
)


def _load_json(name: str) -> dict:
    """Load a JSON file from ``data/``; return ``{}`` on missing or corrupt input."""
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def compute_recommendation(regime: dict, crisis: dict) -> dict:
    """Build a treasury hedge recommendation from macro regime and crisis tier."""
    quadrant = (regime.get("quadrant") or "UNKNOWN").upper()
    tier = (crisis.get("tier") or "NORMAL").upper()

    key = (quadrant, tier)
    if key in HEDGE_RULES:
        instrument, pct, reason = HEDGE_RULES[key]
    else:
        instrument, pct, reason = (None, 0.0,
            f"No rule for quadrant={quadrant} tier={tier} — defaulting to no hedge.")

    # Defense-in-depth cap (the rule matrix already tops out at 20%).
    max_pct = env_float("TREASURY_HEDGE_MAX_PCT", DEFAULT_MAX_ALLOCATION_PCT)
    pct = min(pct, max_pct)

    mode = os.environ.get("TREASURY_HEDGE_MODE", "SIGNAL_ONLY").upper()
    if mode not in VALID_MODES:
        mode = "SIGNAL_ONLY"

    sharia_cleared = env_bool("TREASURY_SHARIA_CLEARED", False)
    gate = apply_sharia_gate(instrument, pct, sharia_cleared=sharia_cleared)

    return {
        "schema_version": "1.1",
        "engine":          "treasury_hedge_overlay",
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode":            mode,
        "instrument":      instrument,
        "allocation_pct":  pct,
        "max_allocation_pct": max_pct,
        "sharia_cleared":  sharia_cleared,
        "effective_instrument":     gate["effective_instrument"],
        "effective_allocation_pct": gate["effective_allocation_pct"],
        "sub_tag":         gate["sub_tag"],
        "gate_action":     gate["gate_action"],
        "regime_quadrant": quadrant,
        "regime_confidence": regime.get("confidence"),
        "crisis_tier":     tier,
        "crisis_score":    crisis.get("score"),
        "reason":          reason,
        "sharia_note":     SHARIA_NOTE,
        "rule_matrix_size": len(HEDGE_RULES),
        "inputs_fresh": bool(regime) and bool(crisis),
    }


def sanitize_hedge_recommendation(hedge: dict) -> dict:
    """Re-apply the Sharia gate from env — never trust on-disk effective_* fields.

    Prevents state injection via a tampered treasury_hedge.json (e.g. forcing
    effective_instrument=TLT while TREASURY_SHARIA_CLEARED is false).
    """
    if not hedge:
        return hedge

    out = dict(hedge)
    sharia_cleared = env_bool("TREASURY_SHARIA_CLEARED", False)
    out["sharia_cleared"] = sharia_cleared

    sovereign = out.get("instrument")
    pct = min(
        float(out.get("allocation_pct") or out.get("effective_allocation_pct") or 0),
        env_float("TREASURY_HEDGE_MAX_PCT", DEFAULT_MAX_ALLOCATION_PCT),
    )
    gate = apply_sharia_gate(sovereign, pct, sharia_cleared=sharia_cleared)
    out.update(gate)

    # Defense-in-depth: block injected sovereign tickers on the effective slot.
    eff = out.get("effective_instrument")
    if not sharia_cleared and eff in SOVEREIGN_INSTRUMENTS:
        out["effective_instrument"] = FALLBACK_INSTRUMENT
        out["effective_allocation_pct"] = pct
        out["sub_tag"] = FALLBACK_SUB_TAG
        out["gate_action"] = "SHARIA_FALLBACK_GLD"

    prev_gate = hedge.get("gate_action")
    new_gate = out.get("gate_action")
    if prev_gate and new_gate and str(prev_gate).upper() != str(new_gate).upper():
        _notify_gate_transition(str(prev_gate), out)

    return out


def _load_previous_gate_action() -> str | None:
    """Read gate_action from the on-disk treasury_hedge.json, if present."""
    if not OUTPUT_FILE.exists():
        return None
    try:
        prior = json.loads(OUTPUT_FILE.read_text())
    except Exception:
        return None
    gate = prior.get("gate_action")
    return str(gate) if gate else None


def _notify_gate_transition(previous_gate: str | None, result: dict) -> None:
    """Fire Telegram compliance alert when gate_action changes (Event B)."""
    new_gate = result.get("gate_action")
    if not new_gate or (previous_gate or "").upper() == str(new_gate).upper():
        return
    try:
        from scripts.telegram_notifier import notify_compliance_shift
        notify_compliance_shift(
            previous_gate,
            str(new_gate),
            allocation_pct=float(result.get("effective_allocation_pct") or 0),
            effective_instrument=result.get("effective_instrument"),
            regime_quadrant=result.get("regime_quadrant"),
            crisis_tier=result.get("crisis_tier"),
        )
    except Exception:
        pass


def run(write: bool = True) -> dict:
    """Load regime inputs, compute recommendation, and optionally persist JSON."""
    previous_gate = _load_previous_gate_action()
    regime = _load_json("macro_regime.json")
    crisis = _load_json("crisis_detector.json")
    result = compute_recommendation(regime, crisis)
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _notify_gate_transition(previous_gate, result)
    return result


def _print_report(r: dict) -> None:
    instrument = r["instrument"] or "—"
    pct = r["allocation_pct"]
    color_red    = "\033[31;1m"
    color_amber  = "\033[33m"
    color_green  = "\033[32m"
    color_off    = "\033[0m"
    if pct >= 15.0:
        color = color_red
    elif pct >= 5.0:
        color = color_amber
    else:
        color = color_green
    eff = r.get("effective_instrument") or "—"
    eff_pct = r.get("effective_allocation_pct", 0.0)
    cleared = "YES" if r.get("sharia_cleared") else "NO"
    print(f"\n{SEP}\n  TREASURY HEDGE OVERLAY  (Phase XXV)\n{SEP}")
    print(f"  Mode:           {r['mode']}")
    print(f"  Regime:         {r['regime_quadrant']}  ({r.get('regime_confidence')})")
    print(f"  Crisis:         {r['crisis_tier']}  (score={r.get('crisis_score')})")
    print(f"  Recommendation: {color}{instrument}  {pct:.1f}%{color_off}")
    print(f"  Sharia cleared: {cleared}")
    print(f"  Gate action:    {r.get('gate_action')}")
    print(f"  EFFECTIVE:      {color}{eff}  {eff_pct:.1f}%{color_off}"
          + (f"  [sub_tag={r.get('sub_tag')}]" if r.get("sub_tag") else ""))
    print(f"  Reason:         {r['reason']}")
    print(f"  Inputs fresh:   {r['inputs_fresh']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")


def _print_matrix() -> None:
    print(f"\n{SEP}\n  HEDGE RULE MATRIX  ({len(HEDGE_RULES)} entries)\n{SEP}")
    print(f"  {'Quadrant':<13s} {'CrisisTier':<10s} {'Instr':<5s} {'Pct':>6s}  Reason")
    print(f"  {'-'*13} {'-'*10} {'-'*5} {'-'*6}  {'-'*40}")
    for (q, t), (ins, pct, rsn) in HEDGE_RULES.items():
        instr = ins if ins else "—"
        print(f"  {q:<13s} {t:<10s} {instr:<5s} {pct:>5.1f}%  {rsn}")
    print(SEP)


def main() -> int:
    """CLI entrypoint: run overlay, print report, or show rule matrix."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="suppress stdout report")
    ap.add_argument("--explain", action="store_true", help="print full rule matrix and exit")
    args = ap.parse_args()
    if args.explain:
        _print_matrix()
        return 0
    r = run(write=True)
    if not args.quiet:
        _print_report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
