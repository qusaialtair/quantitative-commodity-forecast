#!/usr/bin/env python3
"""
ML Walk-Forward — Phase XXVI-c
================================
The honest reality check after the ML conviction PoC.  Slides 252-day
rolling windows forward 21 days at a time from 2010-01-01 over the
out-of-sample LightGBM predictions and computes per-window Sharpe + ann
return for BOTH the ML signal and the rule baseline.

Compares the aggregate verdict against Phase XXII's rule walk-forward
baseline:
    Phase XXII rule:   62% positive, median annual +3.96%, avg Sharpe +0.49

Acceptance bar for the ML wiring (Phase XXVI gate):
    Median annual ≥ +8%
    Positive share ≥ 70%
    Avg Sharpe ≥ +0.70

The strategy is sign-based: long when prediction > 0, short when < 0,
flat at zero. Forward 21d returns drive the per-window stats; same
mechanic as the existing walk-forward.

Input
-----
data/ml_conviction_predictions.csv  (produced by ml_conviction_poc.py)

Output
------
data/ml_walk_forward.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
INPUT_CSV = DATA_DIR / "ml_conviction_predictions.csv"
OUTPUT_FILE = DATA_DIR / "ml_walk_forward.json"

WINDOW_DAYS = 252
STEP_DAYS = 21
START_DATE = "2010-01-01"

SEP = "━" * 78


def _window_stats(direction, fwd_returns, horizon=21):
    """Compute window-level (sharpe, ann_pct, hit_rate_pct, win_pct).

    direction ∈ {-1, 0, +1}; fwd_returns are 21d cumulative returns.
    Daily strategy return ≈ direction * fwd_return / horizon (approximation;
    same mechanic as Phase XXII).
    """
    import numpy as np
    n = len(direction)
    if n < 60:
        return None
    daily = direction * fwd_returns / horizon
    if daily.std() < 1e-9:
        return None
    mean_d = daily.mean()
    std_d  = daily.std()
    sharpe = (mean_d / std_d) * math.sqrt(252)
    ann    = ((1 + mean_d) ** 252 - 1) * 100
    # Hit rate: of days with a direction signal, what % were correct?
    active = direction != 0
    n_active = int(active.sum())
    hit = (((direction == np.sign(fwd_returns)) & active).sum() / max(n_active, 1)) * 100
    win = float((daily > 0).mean() * 100)
    return {
        "sharpe":  float(sharpe),
        "ann_pct": float(ann),
        "hit_pct": float(hit),
        "win_pct": float(win),
        "n_days":  int(n),
        "n_active": n_active,
    }


def _verdict(positive_share, avg_sharpe, sigma_sharpe):
    if positive_share >= 0.75 and sigma_sharpe < 1.0:
        return "STABLE"
    if positive_share >= 0.50 and sigma_sharpe < 1.5:
        return "DRIFTING"
    return "UNSTABLE"


def run() -> dict:
    import numpy as np, pandas as pd
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found — run ml_conviction_poc.py first"
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.dropna(subset=["ml_pred", "rule_pred", "fwd_21d_return"])
    df = df[df["date"] >= START_DATE].reset_index(drop=True)

    n = len(df)
    ml_dir   = np.sign(df["ml_pred"].values)
    rule_dir = np.sign(df["rule_pred"].values)
    fwd      = df["fwd_21d_return"].values

    windows: list[dict] = []
    for i in range(0, n - WINDOW_DAYS + 1, STEP_DAYS):
        end = i + WINDOW_DAYS
        ml_w   = _window_stats(ml_dir[i:end],   fwd[i:end])
        rule_w = _window_stats(rule_dir[i:end], fwd[i:end])
        if not ml_w or not rule_w:
            continue
        windows.append({
            "start": str(df["date"].iloc[i].date()),
            "end":   str(df["date"].iloc[end - 1].date()),
            "ml":    ml_w,
            "rule":  rule_w,
            "delta_sharpe": ml_w["sharpe"] - rule_w["sharpe"],
            "delta_ann":    ml_w["ann_pct"] - rule_w["ann_pct"],
        })

    if not windows:
        raise RuntimeError("No windows produced — input too short")

    def _agg(key):
        sharps = np.array([w[key]["sharpe"] for w in windows])
        anns   = np.array([w[key]["ann_pct"] for w in windows])
        pos    = (sharps > 0).mean()
        return {
            "n_windows": int(len(windows)),
            "positive_share": round(float(pos), 4),
            "avg_sharpe":     round(float(sharps.mean()), 4),
            "median_sharpe":  round(float(np.median(sharps)), 4),
            "sigma_sharpe":   round(float(sharps.std(ddof=1)), 4),
            "sharpe_min":     round(float(sharps.min()), 4),
            "sharpe_max":     round(float(sharps.max()), 4),
            "avg_ann_pct":    round(float(anns.mean()), 3),
            "median_ann_pct": round(float(np.median(anns)), 3),
            "worst_ann_pct":  round(float(anns.min()), 3),
            "best_ann_pct":   round(float(anns.max()), 3),
            "verdict":        _verdict(pos, sharps.mean(), sharps.std(ddof=1)),
        }

    ml_agg   = _agg("ml")
    rule_agg = _agg("rule")

    # Phase XXVI gate
    accept = (
        ml_agg["median_ann_pct"] >= 8.0 and
        ml_agg["positive_share"] >= 0.70 and
        ml_agg["avg_sharpe"] >= 0.70
    )
    if accept:
        gate = "PASS"
        note = (
            f"ML median ann {ml_agg['median_ann_pct']:.2f}% ≥ 8%, "
            f"positive share {ml_agg['positive_share']*100:.1f}% ≥ 70%, "
            f"avg Sharpe {ml_agg['avg_sharpe']:.2f} ≥ 0.70. "
            "Justifies wiring LightGBM as the conviction core."
        )
    elif ml_agg["median_ann_pct"] > rule_agg["median_ann_pct"] + 1.0:
        gate = "PARTIAL"
        note = (
            f"ML beats rule on median annual "
            f"({ml_agg['median_ann_pct']:.2f}% > {rule_agg['median_ann_pct']:.2f}%) "
            "but doesn't clear the +8%/y gate. Wire cautiously or improve features first."
        )
    else:
        gate = "FAIL"
        note = (
            f"ML median {ml_agg['median_ann_pct']:.2f}% does not clearly beat "
            f"rule {rule_agg['median_ann_pct']:.2f}%. Walk-forward signal weaker "
            "than the PoC IC implied. Need feature work or different target."
        )

    result = {
        "schema_version": "1.0",
        "engine":          "ml_walk_forward",
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":          str(INPUT_CSV.name),
        "config": {
            "window_days":    WINDOW_DAYS,
            "step_days":      STEP_DAYS,
            "start_date":     START_DATE,
            "horizon_days":   21,
            "n_windows":      len(windows),
        },
        "phase_xxii_baseline": {
            "positive_share": 0.622,
            "median_ann_pct": 3.96,
            "avg_sharpe":     0.493,
            "note": "Phase XXII rule walk-forward — the baseline ML must beat",
        },
        "ml":   ml_agg,
        "rule": rule_agg,
        "deltas": {
            "median_ann_pp":  round(ml_agg["median_ann_pct"] - rule_agg["median_ann_pct"], 2),
            "avg_sharpe":     round(ml_agg["avg_sharpe"]    - rule_agg["avg_sharpe"], 3),
            "positive_share": round(ml_agg["positive_share"] - rule_agg["positive_share"], 4),
        },
        "phase_xxvi_gate": gate,
        "note":             note,
        "windows":          windows,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    return result


def _print(r: dict) -> None:
    cfg = r["config"]
    print(f"\n{SEP}\n  ML WALK-FORWARD VALIDATION  (Phase XXVI-c)\n{SEP}")
    print(f"  Windows: {cfg['n_windows']}  "
          f"(252d rolling, 21d step from {cfg['start_date']})")
    print(SEP)
    header = f"  {'':<14s} {'pos %':>7s} {'med ann':>9s} {'avg Sh':>8s} {'σ Sh':>7s} {'min Sh':>8s} {'max Sh':>8s}  verdict"
    print(header); print("  " + "-" * (len(header) - 2))
    for k, label in [("ml", "LightGBM"), ("rule", "Rule baseline"), ("phase_xxii_baseline", "Phase XXII")]:
        m = r[k]
        pos = m.get("positive_share", 0) * 100
        med = m.get("median_ann_pct", 0)
        avg = m.get("avg_sharpe", 0)
        sig = m.get("sigma_sharpe", 0) if k != "phase_xxii_baseline" else 0
        mn = m.get("sharpe_min", 0) if k != "phase_xxii_baseline" else 0
        mx = m.get("sharpe_max", 0) if k != "phase_xxii_baseline" else 0
        v  = m.get("verdict", "ref") if k != "phase_xxii_baseline" else "ref"
        print(f"  {label:<14s} {pos:>6.1f}% {med:>+8.2f}% {avg:>+8.2f} {sig:>7.3f} "
              f"{mn:>+8.2f} {mx:>+8.2f}  {v}")
    d = r["deltas"]
    print(SEP)
    print(f"  Δ (ML − rule):  median ann {d['median_ann_pp']:+.2f}pp · "
          f"avg Sharpe {d['avg_sharpe']:+.3f} · "
          f"positive share {d['positive_share']*100:+.2f}pp")
    print(SEP)
    print(f"  Phase XXVI gate: {r['phase_xxvi_gate']}")
    print(f"  Note:            {r['note']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    res = run()
    _print(res)
