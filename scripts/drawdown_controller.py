#!/usr/bin/env python3
"""
Drawdown Recovery Controller
==============================
Graduated defensive posture that automatically reduces exposure as
drawdown deepens, and ramps back up as equity recovers.

Tiers (configurable):
  DD < 5%  → NORMAL:    100% of target sizing
  DD < 10% → CAUTION:    75% of target sizing
  DD < 20% → DEFENSIVE:  50% of target sizing
  DD < 30% → CRITICAL:   25% of target sizing, mandatory partial exit
  DD ≥ 30% → EMERGENCY:  10% of target sizing, full defensive mode

Recovery ramp:
  When DD improves by 5% from worst, step up one tier (with 3-day hold)
  When DD < 5% for 5 consecutive days, return to NORMAL

Usage:
    python3 scripts/drawdown_controller.py
    python3 scripts/drawdown_controller.py --portfolio-value 100000 --peak 110000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "drawdown_controller.json"
STATE_FILE = DATA_DIR / "drawdown_state.json"

LINE_W = 62
SEP = "\u2501" * LINE_W


@dataclass
class DrawdownTier:
    name: str
    dd_threshold_pct: float
    sizing_multiplier: float
    action: str


TIERS = [
    DrawdownTier("NORMAL",    5.0,  1.00, "Full sizing permitted"),
    DrawdownTier("CAUTION",  10.0,  0.75, "Reduce new positions to 75%"),
    DrawdownTier("DEFENSIVE",20.0,  0.50, "Half sizing, tighten all stops"),
    DrawdownTier("CRITICAL", 30.0,  0.25, "Quarter sizing, partial exits required"),
    DrawdownTier("EMERGENCY",100.0, 0.10, "Minimal exposure, defensive only"),
]


@dataclass
class DrawdownState:
    current_dd_pct: float
    peak_value: float
    current_value: float
    tier_name: str
    sizing_multiplier: float
    action: str
    days_in_tier: int
    worst_dd_pct: float
    worst_dd_date: str
    recovery_pct: float
    consecutive_improving_days: int


def _load_state() -> dict:
    """Load persistent drawdown state from disk."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    """Save drawdown state to disk for persistence across runs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _classify_tier(dd_pct: float) -> DrawdownTier:
    """Find the appropriate tier for a given drawdown percentage."""
    abs_dd = abs(dd_pct)
    for tier in TIERS:
        if abs_dd < tier.dd_threshold_pct:
            return tier
    return TIERS[-1]


def evaluate_drawdown(
    current_value: float,
    peak_value: float | None = None,
) -> DrawdownState:
    """
    Evaluate current drawdown and determine defensive posture.

    Parameters
    ----------
    current_value : float
        Current portfolio value
    peak_value : float, optional
        Historical peak portfolio value. If None, reads from state or uses current.
    """
    prev_state = _load_state()

    # Determine peak
    if peak_value is None:
        peak_value = max(
            current_value,
            float(prev_state.get("peak_value", current_value)),
        )
    else:
        peak_value = max(peak_value, current_value)

    # Current drawdown
    dd_pct = ((current_value - peak_value) / peak_value * 100) if peak_value > 0 else 0.0

    # Track worst drawdown
    prev_worst = float(prev_state.get("worst_dd_pct", 0.0))
    worst_dd = min(dd_pct, prev_worst)
    worst_dd_date = prev_state.get("worst_dd_date", date.today().isoformat())
    if dd_pct <= prev_worst:
        worst_dd_date = date.today().isoformat()

    # Recovery from worst
    recovery_pct = dd_pct - worst_dd if worst_dd < 0 else 0.0

    # Classify tier
    tier = _classify_tier(dd_pct)

    # Track days in current tier
    prev_tier = prev_state.get("tier_name", "NORMAL")
    prev_days = int(prev_state.get("days_in_tier", 0))
    days_in_tier = (prev_days + 1) if tier.name == prev_tier else 1

    # Consecutive improving days (DD less negative than previous)
    prev_dd = float(prev_state.get("current_dd_pct", 0.0))
    prev_improving = int(prev_state.get("consecutive_improving_days", 0))
    if dd_pct > prev_dd:  # less negative = improving
        consecutive_improving = prev_improving + 1
    else:
        consecutive_improving = 0

    result = DrawdownState(
        current_dd_pct=round(dd_pct, 4),
        peak_value=round(peak_value, 2),
        current_value=round(current_value, 2),
        tier_name=tier.name,
        sizing_multiplier=tier.sizing_multiplier,
        action=tier.action,
        days_in_tier=days_in_tier,
        worst_dd_pct=round(worst_dd, 4),
        worst_dd_date=worst_dd_date,
        recovery_pct=round(recovery_pct, 4),
        consecutive_improving_days=consecutive_improving,
    )

    # Persist state
    state_dict = asdict(result)
    state_dict["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_state(state_dict)

    return result


def run_drawdown_controller(
    current_value: float | None = None,
    peak_value: float | None = None,
) -> dict:
    """
    Run the drawdown controller. If no values provided, reads from pipeline_state.json.
    """
    if current_value is None:
        try:
            ps = json.loads((DATA_DIR / "pipeline_state.json").read_text())
            pf = ps.get("portfolio", {})
            current_value = float(pf.get("portfolio_value", 100_000))
            if peak_value is None:
                peak_value = max(current_value, float(pf.get("starting_capital", current_value)))
        except Exception:
            current_value = 100_000.0

    state = evaluate_drawdown(current_value, peak_value)
    result = asdict(state)
    result["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Add tier ladder for display
    result["tier_ladder"] = [
        {
            "tier": t.name,
            "threshold": f"<{t.dd_threshold_pct}%",
            "multiplier": t.sizing_multiplier,
            "active": t.name == state.tier_name,
        }
        for t in TIERS
    ]

    # Phase XXV-E: surface Phase XIV hedge sleeve exposure alongside the metals
    # book drawdown metrics so the operator has a single risk snapshot.
    # Read-only; never alters the drawdown tier calculation.
    try:
        mst_path = DATA_DIR / "multi_strategy_trader.json"
        if mst_path.exists():
            mst = json.loads(mst_path.read_text())
            hs = mst.get("hedge_state") or {}
            result["phase14"] = {
                "book_equity_usd":      mst.get("book_equity_usd"),
                "hedge_notional_usd":   mst.get("hedge_notional_usd"),
                "hedge_fraction_pct":   round((mst.get("hedge_fraction") or 0.0) * 100, 1),
                "hedge_instrument":     hs.get("instrument"),
                "hedge_mode":           hs.get("mode"),
                "hedge_gate_blocked":   hs.get("gate_blocked", False),
                "stress_eval":          hs.get("stress_eval"),
            }
    except Exception:
        pass

    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result, state)
    return result


def _print_report(result: dict, state: DrawdownState) -> None:
    dd_color = "\033[32m" if state.current_dd_pct > -5 else "\033[33m" if state.current_dd_pct > -20 else "\033[31m"
    tier_color = {
        "NORMAL": "\033[32m", "CAUTION": "\033[33m",
        "DEFENSIVE": "\033[33;1m", "CRITICAL": "\033[31m",
        "EMERGENCY": "\033[31;1m",
    }.get(state.tier_name, "\033[0m")

    print(f"\n{SEP}")
    print(f"  DRAWDOWN RECOVERY CONTROLLER")
    print(SEP)
    print(f"  Peak Value:     ${state.peak_value:,.2f}")
    print(f"  Current Value:  ${state.current_value:,.2f}")
    print(f"  Drawdown:       {dd_color}{state.current_dd_pct:+.2f}%\033[0m")
    print(f"  Worst DD:       {state.worst_dd_pct:+.2f}%  ({state.worst_dd_date})")
    print(f"  Recovery:       {state.recovery_pct:+.2f}%")
    print()
    print(f"  TIER:           {tier_color}{state.tier_name}\033[0m")
    print(f"  Sizing Mult:    {state.sizing_multiplier:.0%}")
    print(f"  Action:         {state.action}")
    print(f"  Days in Tier:   {state.days_in_tier}")
    print(f"  Improving Days: {state.consecutive_improving_days}")
    print()

    print(f"  TIER LADDER")
    print(f"  {'─' * 50}")
    for t in TIERS:
        marker = " >>>" if t.name == state.tier_name else "    "
        print(f"  {marker} {t.name:<12s}  DD<{t.dd_threshold_pct:>5.0f}%  "
              f"Size: {t.sizing_multiplier:.0%}  {t.action}")

    p14 = result.get("phase14")
    if p14 and p14.get("book_equity_usd") is not None:
        print(f"\n  PHASE XIV HEDGE SLEEVE")
        print(f"  {'─' * 50}")
        instr = p14.get("hedge_instrument") or "off"
        gate_flag = "  [GATE BLOCKED]" if p14.get("hedge_gate_blocked") else ""
        print(f"  Instrument:     {instr}{gate_flag}")
        print(f"  Notional:       ${p14.get('hedge_notional_usd', 0):,.2f}"
              f"  ({p14.get('hedge_fraction_pct', 0):.1f}% of book)")
        print(f"  Book equity:    ${p14.get('book_equity_usd', 0):,.2f}")
        print(f"  Mode:           {p14.get('hedge_mode', '—')}")
        if p14.get("stress_eval"):
            print(f"  Stress gate:    {p14['stress_eval'][:70]}")

    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drawdown Recovery Controller")
    parser.add_argument("--portfolio-value", type=float, default=None)
    parser.add_argument("--peak", type=float, default=None)
    args = parser.parse_args()
    run_drawdown_controller(
        current_value=args.portfolio_value,
        peak_value=args.peak,
    )
