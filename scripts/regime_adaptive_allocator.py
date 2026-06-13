#!/usr/bin/env python3
"""
Regime-Adaptive Allocator  (Phase XXI Stage 82)
=================================================
Single source-of-truth for daily metals/equity target weights based on
the current crisis tier.

Phase XX proved that fixed 60/40 metals/equity weighting helps in calm
regimes but hurts in deep crises (2008 GFC, 2022 inflation rout — both
asset classes fell together).  This module shifts the weights *toward
metals* as crisis intensity rises, recognising that:

  - In NORMAL regimes, equities provide independent edge — overweight them.
  - In ELEVATED stress, lean defensive but keep some equity participation.
  - In full STRESS / CRISIS, gold becomes the haven; equities are correlated
    risk that should be cut hard.

Default weight schedule
-----------------------

    Tier        Metals   Equity   Logic
    NORMAL      40%      60%      Balanced — equities lead
    ELEVATED    55%      45%      Defensive tilt
    STRESS      75%      25%      Heavy gold; minimal equity exposure
    CRISIS      95%       5%      Almost pure gold (residual equity for
                                  liquidity / regime-recovery upside)

The schedule is exposed as `WEIGHT_SCHEDULE` so future calibration can
override it.  Smoothing optional: enable via `smoothing=True` to apply
an EMA over the last 5 tier observations (avoids whipsaws on borderline
NORMAL↔ELEVATED transitions).

Public API
----------
    weights_for_tier(tier)                  — instant lookup
    adaptive_series(tier_series, smoothing) — for backtest replay
    run_allocator()                         — live snapshot for the UI

Output: data/regime_adaptive_allocator.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "regime_adaptive_allocator.json"


# ──────────────────────────────────────────────────────────────────────────────
# Weight schedule — public so callers can override (e.g. a UI A/B test)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHT_SCHEDULE: dict[str, tuple[float, float]] = {
    # Calibrated 2026-05-19 after the v1 schedule (40/55/75/95) overshot
    # in 2011 Euro and 2022 inflation windows.  The metals book already
    # has a Phase XVIII crisis guard that suppresses its own size; an
    # *additional* aggressive lean-to-metals double-counts the defence.
    # The current schedule is gentler — the operator can override per-run.
    "NORMAL":   (0.40, 0.60),
    "ELEVATED": (0.50, 0.50),
    "STRESS":   (0.62, 0.38),
    "CRISIS":   (0.80, 0.20),
}

DEFAULT_TIER = "NORMAL"
SMOOTHING_WINDOW = 5  # days


def weights_for_tier(tier: str) -> tuple[float, float]:
    """Returns (metals_weight, equity_weight) for a given tier."""
    return WEIGHT_SCHEDULE.get((tier or "").upper(), WEIGHT_SCHEDULE[DEFAULT_TIER])


def adaptive_series(
    tier_series: list[str], smoothing: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Translate a per-day tier sequence into per-day (metals, equity) weights.

    With `smoothing=True`, the realised weight is an EMA of the last N
    tier-implied weights so transitions don't whipsaw.  Without smoothing,
    the allocator switches instantly on tier change.
    """
    metals_weights: list[float] = []
    equity_weights: list[float] = []
    buf_m: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
    buf_e: deque[float] = deque(maxlen=SMOOTHING_WINDOW)

    for tier in tier_series:
        m, e = weights_for_tier(tier)
        if smoothing:
            buf_m.append(m); buf_e.append(e)
            metals_weights.append(sum(buf_m) / len(buf_m))
            equity_weights.append(sum(buf_e) / len(buf_e))
        else:
            metals_weights.append(m)
            equity_weights.append(e)

    return metals_weights, equity_weights


# ──────────────────────────────────────────────────────────────────────────────
# Live snapshot
# ──────────────────────────────────────────────────────────────────────────────
def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def run_allocator() -> dict:
    """Live snapshot for the UI panel — reads crisis_detector.json."""
    crisis = _load("crisis_detector.json")
    tier = (crisis.get("tier") or DEFAULT_TIER).upper()
    score = float(crisis.get("score", 0.0))
    m, e = weights_for_tier(tier)

    out = {
        "schema_version": "1.0",
        "engine":         "regime_adaptive_allocator",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_tier":   tier,
        "crisis_score":   round(score, 4),
        "target_weights": {
            "metals": m,
            "equity": e,
        },
        "weight_schedule": {
            k: {"metals": v[0], "equity": v[1]}
            for k, v in WEIGHT_SCHEDULE.items()
        },
        "smoothing_window_days": SMOOTHING_WINDOW,
        "rationale": _rationale_for_tier(tier),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    return out


def _rationale_for_tier(tier: str) -> str:
    return {
        "NORMAL":   "Balanced posture — equities lead, gold provides diversification.",
        "ELEVATED": "Defensive tilt — vol elevated, fade equity exposure 15pp.",
        "STRESS":   "Heavy gold — stress regime; cut equity exposure to a third of normal.",
        "CRISIS":   "Almost pure gold — crisis regime; equities are correlated risk.",
    }.get(tier, "Unknown tier — defaulting to NORMAL.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = run_allocator()
    if args.quiet:
        return 0
    print("=" * 64)
    print(f"REGIME-ADAPTIVE ALLOCATOR  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Current tier   : {out['current_tier']}  (score {out['crisis_score']})")
    print(f"  Target metals  : {out['target_weights']['metals']*100:.0f}%")
    print(f"  Target equity  : {out['target_weights']['equity']*100:.0f}%")
    print(f"  Rationale      : {out['rationale']}")
    print()
    print("  Full schedule:")
    for k, v in out["weight_schedule"].items():
        print(f"    {k:<10s}  metals {v['metals']*100:>3.0f}%  /  equity {v['equity']*100:>3.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
