#!/usr/bin/env python3
"""
Model Risk Management — Champion / Challenger Tournament
==========================================================
Maintains a roster of candidate strategies, scores each against the live
champion, and recommends a promotion when a challenger beats the champion
by a configurable margin over a configurable lookback.

Inputs:
  - alpha_attribution.json   per-source Sharpe and IR
  - purged_kfold.json        cross-validated stability
  - bma_weights.json         posterior weights
  - decision_quality.json    Brier skill
  - ic_ir_tracker.json       deployable signals + IR

Scoring (weighted average; tunable):
  score = 0.35·norm(Sharpe) + 0.25·norm(IR) + 0.20·stability_ratio
         + 0.10·Brier_skill + 0.10·deployable_flag

Promotion rule:
  challenger_score − champion_score > 0.10  → promote
  challenger_score − champion_score < −0.10 → retire challenger
  otherwise → keep monitoring

State stored in: data/mrm_champion.json
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
OUTPUT_FILE = DATA_DIR / "mrm_champion.json"
STATE_FILE = DATA_DIR / "mrm_state.json"

PROMOTION_MARGIN = 0.10
RETIRE_MARGIN = -0.10
LINE_W = 62
SEP = "━" * LINE_W


def _safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _norm(x: float, lo: float = -1.0, hi: float = 2.0) -> float:
    if hi == lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _score_signal(name: str, aa: dict, kf: dict, bma: dict, dq: dict, ii: dict) -> dict:
    full = aa.get("full_history", {}).get(name, {})
    irs = aa.get("information_ratios", {}).get(name, {})
    sharpe = float(full.get("sharpe", 0))
    ir = float(irs.get("information_ratio", 0))
    stability = float(kf.get("summary", {}).get("stability_ratio") or 0)
    bma_weight = float(bma.get("weights", {}).get("bma", {}).get(name, 0))
    brier_skill = float(
        dq.get("per_signal", {}).get(name, {}).get("skill_brier") or 0
    )
    ii_data = ii.get("per_signal", {}).get(name, {})
    deployable = 1.0 if ii_data.get("deployable") else 0.0

    score = (
        0.35 * _norm(sharpe) +
        0.25 * _norm(ir, -1.0, 1.5) +
        0.20 * _norm(stability, 0, 2) +
        0.10 * _norm(brier_skill, -0.2, 0.2) +
        0.10 * deployable
    )
    return {
        "name":        name,
        "sharpe":      round(sharpe, 3),
        "ir":          round(ir, 3),
        "stability":   round(stability, 3),
        "bma_weight":  round(bma_weight, 4),
        "brier_skill": round(brier_skill, 4),
        "deployable":  bool(deployable),
        "score":       round(score, 4),
    }


def run_mrm() -> dict:
    aa = _safe(DATA_DIR / "alpha_attribution.json")
    kf = _safe(DATA_DIR / "purged_kfold.json")
    bma = _safe(DATA_DIR / "bma_weights.json")
    dq = _safe(DATA_DIR / "decision_quality.json")
    ii = _safe(DATA_DIR / "ic_ir_tracker.json")

    signals = aa.get("sources", []) or list(
        aa.get("full_history", {}).keys()
    )
    rankings = [_score_signal(s, aa, kf, bma, dq, ii) for s in signals]
    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Load champion state
    state = _safe(STATE_FILE)
    current_champion = state.get("champion")
    if not current_champion and rankings:
        current_champion = rankings[0]["name"]

    # Decision logic
    decision = "MONITOR"
    new_champion = current_champion
    delta = 0.0
    if rankings:
        top = rankings[0]
        champ_row = next(
            (r for r in rankings if r["name"] == current_champion),
            rankings[0],
        )
        delta = top["score"] - champ_row["score"]
        if delta > PROMOTION_MARGIN and top["name"] != current_champion:
            decision = "PROMOTE"
            new_champion = top["name"]
        elif delta < RETIRE_MARGIN:
            decision = "RETIRE_CHAMPION"
            new_champion = top["name"]

    # Persist state
    new_state = {
        "champion":           new_champion,
        "previous_champion":  current_champion if new_champion != current_champion else state.get("previous_champion"),
        "promotion_date":     datetime.now(timezone.utc).isoformat(timespec="seconds")
                                 if decision == "PROMOTE" else state.get("promotion_date"),
        "last_evaluated":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(new_state, indent=2, default=str))

    result = {
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_champion":current_champion,
        "new_champion":    new_champion,
        "decision":        decision,
        "score_delta":     round(float(delta), 4),
        "promotion_margin":PROMOTION_MARGIN,
        "rankings":        rankings,
        "previous_state":  state,
        "new_state":       new_state,
    }
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    decision_color = {
        "PROMOTE":         "\033[32;1m",
        "RETIRE_CHAMPION": "\033[31m",
        "MONITOR":         "\033[36m",
    }.get(r["decision"], "\033[0m")

    print(f"\n{SEP}\n  MRM CHAMPION / CHALLENGER\n{SEP}")
    print(f"  Current champion: {r['current_champion']}")
    print(f"  New champion:     {r['new_champion']}")
    print(f"  Score delta:      {r['score_delta']:+.4f}")
    print(f"  Decision:         {decision_color}{r['decision']}\033[0m")
    print()
    print(f"  RANKINGS")
    print(f"  {'─' * 64}")
    print(
        f"  {'rank':>4s}  {'signal':<16s}  {'score':>7s}  "
        f"{'Sharpe':>7s}  {'IR':>7s}  {'stab':>5s}  {'depl':>5s}"
    )
    for i, row in enumerate(r["rankings"], 1):
        marker = " *" if row["name"] == r["new_champion"] else "  "
        print(
            f"  {marker} {i:>2d}  {row['name']:<16s}  "
            f"{row['score']:>7.4f}  "
            f"{row['sharpe']:>+7.3f}  "
            f"{row['ir']:>+7.3f}  "
            f"{row['stability']:>5.2f}  "
            f"{'YES' if row['deployable'] else 'no':>5s}"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRM Champion / Challenger")
    args = parser.parse_args()
    run_mrm()
