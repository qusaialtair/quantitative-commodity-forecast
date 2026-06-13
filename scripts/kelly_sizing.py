#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing
=================================
Computes optimal position size using Kelly criterion with multiple
safety overlays for real-world application.

Variants:
  - Full Kelly:      maximizes geometric growth (too aggressive for real use)
  - Half Kelly:      industry standard — balances growth with drawdown tolerance
  - Fractional Kelly: configurable fraction (default 0.25 for conservative mandate)

Inputs:
  - Win rate from prediction log (or Monte Carlo)
  - Average win / average loss ratio
  - Current portfolio heat (total risk exposure)
  - Regime overlay (reduce in VOLATILE/BEARISH)

Usage:
    python3 scripts/kelly_sizing.py
    python3 scripts/kelly_sizing.py --fraction 0.5
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
OUTPUT_FILE = DATA_DIR / "kelly_sizing.json"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"
MC_FILE = DATA_DIR / "monte_carlo_simulation.json"
PIPELINE_STATE = DATA_DIR / "pipeline_state.json"

DEFAULT_FRACTION = 0.25  # conservative: quarter-Kelly
MAX_POSITION_PCT = 40.0  # hard cap regardless of Kelly output
MIN_POSITION_PCT = 5.0   # minimum meaningful position
PORTFOLIO_HEAT_MAX = 60.0  # max total portfolio heat before throttling

LINE_W = 62
SEP = "\u2501" * LINE_W


def _load_win_loss_stats() -> dict:
    """
    Extract win rate and payoff ratio from prediction log.
    Falls back to Monte Carlo probabilities if prediction log is empty.
    """
    stats = {"source": "default", "win_rate": 0.55, "avg_win": 2.0, "avg_loss": 1.5, "n_trades": 0}

    # Try prediction log first
    try:
        import pandas as pd
        if PREDICTION_LOG.exists():
            df = pd.read_csv(PREDICTION_LOG, dtype=str)
            df["direction_correct"] = pd.to_numeric(df["direction_correct"], errors="coerce")
            df["predicted_pct"] = pd.to_numeric(df["predicted_pct"], errors="coerce")
            df["actual_pct"] = pd.to_numeric(df["actual_pct"], errors="coerce")

            evaluated = df[df["direction_correct"].notna()].copy()
            if len(evaluated) >= 5:
                win_rate = float(evaluated["direction_correct"].mean())
                wins = evaluated[evaluated["actual_pct"] > 0]["actual_pct"]
                losses = evaluated[evaluated["actual_pct"] < 0]["actual_pct"].abs()

                avg_win = float(wins.mean()) if len(wins) > 0 else 2.0
                avg_loss = float(losses.mean()) if len(losses) > 0 else 1.5

                stats = {
                    "source": "prediction_log",
                    "win_rate": round(win_rate, 4),
                    "avg_win": round(avg_win, 4),
                    "avg_loss": round(avg_loss, 4),
                    "n_trades": len(evaluated),
                }
                return stats
    except Exception:
        pass

    # Fallback: Monte Carlo probabilities
    try:
        if MC_FILE.exists():
            mc = json.loads(MC_FILE.read_text())
            prob_pos = mc.get("probabilities", {}).get("positive_return", 0.55)
            terminal = mc.get("terminal", {})
            mean_ret = terminal.get("mean_return_pct", 2.0)
            std_ret = terminal.get("std_return_pct", 5.0)

            # Approximate win/loss from distribution
            avg_win = abs(mean_ret) + 0.5 * std_ret
            avg_loss = 0.5 * std_ret

            stats = {
                "source": "monte_carlo",
                "win_rate": round(prob_pos, 4),
                "avg_win": round(avg_win, 4),
                "avg_loss": round(max(avg_loss, 0.5), 4),
                "n_trades": 0,
            }
    except Exception:
        pass

    return stats


def _load_regime() -> str:
    """Load current HMM regime state."""
    try:
        if PIPELINE_STATE.exists():
            ps = json.loads(PIPELINE_STATE.read_text())
            return ps.get("regime", {}).get("hmm_state", "UNKNOWN")
    except Exception:
        pass
    return "UNKNOWN"


def _regime_multiplier(regime: str) -> float:
    """Reduce position in unfavorable regimes."""
    return {
        "BULLISH": 1.0,
        "VOLATILE": 0.6,
        "BEARISH": 0.3,
        "RANGING": 0.7,
    }.get(regime, 0.5)


def compute_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = DEFAULT_FRACTION,
) -> dict:
    """
    Compute Kelly criterion position sizes.

    Full Kelly: f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win / avg_loss (payoff ratio)
    """
    p = max(0.01, min(0.99, win_rate))
    q = 1.0 - p
    b = avg_win / max(avg_loss, 0.01)  # payoff ratio

    # Full Kelly fraction
    full_kelly = (p * b - q) / b

    # Clamp to [0, 1] — negative Kelly means don't trade
    full_kelly = max(0.0, min(1.0, full_kelly))

    half_kelly = full_kelly * 0.5
    frac_kelly = full_kelly * fraction

    # Edge calculation (expected value per unit risked)
    edge = p * b - q
    edge_per_dollar = edge / b if b > 0 else 0

    return {
        "payoff_ratio": round(b, 4),
        "edge": round(edge, 4),
        "edge_per_dollar": round(edge_per_dollar, 4),
        "full_kelly_pct": round(full_kelly * 100, 2),
        "half_kelly_pct": round(half_kelly * 100, 2),
        "fractional_kelly_pct": round(frac_kelly * 100, 2),
        "fraction_used": fraction,
        "should_trade": full_kelly > 0.01,
    }


def compute_position_size(
    portfolio_value: float,
    kelly_result: dict,
    regime: str = "UNKNOWN",
    current_heat_pct: float = 0.0,
) -> dict:
    """
    Convert Kelly fraction to dollar position size with safety overlays.
    """
    raw_pct = kelly_result["fractional_kelly_pct"]

    # Regime adjustment
    regime_mult = _regime_multiplier(regime)
    adjusted_pct = raw_pct * regime_mult

    # Portfolio heat throttle
    available_heat = max(0, PORTFOLIO_HEAT_MAX - current_heat_pct)
    heat_capped_pct = min(adjusted_pct, available_heat)

    # Hard bounds
    final_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, heat_capped_pct))

    # If Kelly says don't trade, respect it
    if not kelly_result["should_trade"]:
        final_pct = 0.0

    deploy_usd = portfolio_value * final_pct / 100.0

    return {
        "raw_kelly_pct": round(raw_pct, 2),
        "regime_adjusted_pct": round(adjusted_pct, 2),
        "heat_capped_pct": round(heat_capped_pct, 2),
        "final_position_pct": round(final_pct, 2),
        "deploy_usd": round(deploy_usd, 2),
        "regime": regime,
        "regime_multiplier": regime_mult,
        "current_heat_pct": round(current_heat_pct, 2),
        "max_heat_pct": PORTFOLIO_HEAT_MAX,
    }


def run_kelly(
    portfolio_value: float = 100_000.0,
    fraction: float = DEFAULT_FRACTION,
    current_heat_pct: float = 0.0,
) -> dict:
    """Full Kelly sizing pipeline."""
    wl_stats = _load_win_loss_stats()
    regime = _load_regime()

    kelly = compute_kelly(
        win_rate=wl_stats["win_rate"],
        avg_win=wl_stats["avg_win"],
        avg_loss=wl_stats["avg_loss"],
        fraction=fraction,
    )

    sizing = compute_position_size(
        portfolio_value=portfolio_value,
        kelly_result=kelly,
        regime=regime,
        current_heat_pct=current_heat_pct,
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio_value": portfolio_value,
        "win_loss_stats": wl_stats,
        "kelly": kelly,
        "sizing": sizing,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))

    _print_report(result)
    return result


def _print_report(result: dict) -> None:
    wl = result["win_loss_stats"]
    k = result["kelly"]
    s = result["sizing"]

    print(f"\n{SEP}")
    print(f"  KELLY CRITERION POSITION SIZING")
    print(SEP)
    print(f"  Data Source:    {wl['source']}  ({wl['n_trades']} evaluated trades)")
    print(f"  Win Rate:       {wl['win_rate']:.1%}")
    print(f"  Avg Win:        {wl['avg_win']:.2f}%")
    print(f"  Avg Loss:       {wl['avg_loss']:.2f}%")
    print(f"  Payoff Ratio:   {k['payoff_ratio']:.2f}x")
    print()
    print(f"  KELLY FRACTIONS")
    print(f"  {'─' * 40}")
    print(f"  Edge:           {k['edge']:+.4f}")
    print(f"  Full Kelly:     {k['full_kelly_pct']:.2f}%")
    print(f"  Half Kelly:     {k['half_kelly_pct']:.2f}%")
    print(f"  {k['fraction_used']:.0%} Kelly:     {k['fractional_kelly_pct']:.2f}%")
    print(f"  Should Trade:   {'YES' if k['should_trade'] else 'NO'}")
    print()
    print(f"  POSITION SIZING (${result['portfolio_value']:,.0f} portfolio)")
    print(f"  {'─' * 40}")
    print(f"  Regime:         {s['regime']}  (mult: {s['regime_multiplier']:.1f}x)")
    print(f"  Raw Kelly:      {s['raw_kelly_pct']:.2f}%")
    print(f"  Regime Adj:     {s['regime_adjusted_pct']:.2f}%")
    print(f"  Heat Cap:       {s['heat_capped_pct']:.2f}%  (heat: {s['current_heat_pct']:.1f}/{s['max_heat_pct']:.0f}%)")
    print(f"  Final Size:     {s['final_position_pct']:.2f}%")
    print(f"  Deploy:         ${s['deploy_usd']:,.2f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kelly Criterion Position Sizing")
    parser.add_argument("--portfolio", type=float, default=100_000.0)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--heat", type=float, default=0.0)
    args = parser.parse_args()

    run_kelly(
        portfolio_value=args.portfolio,
        fraction=args.fraction,
        current_heat_pct=args.heat,
    )
