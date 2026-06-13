#!/usr/bin/env python3
"""
audit_hmm_sensitivity.py
========================
Temporary diagnostic — HMM veto sensitivity audit for GC=F (Gold).

Answers three questions:
  1. What % of the last 6 months did P(BEARISH) > 0.65? (current veto threshold)
     If >40% the model is too tight — it is in "permanent fear" mode.

  2. Threshold sensitivity table — how does veto rate change if we tighten or
     loosen the threshold from 0.50 → 0.85?

  3. Streak analysis — how many consecutive VETO days are typical?
     Long unbroken streaks suggest the model is trapped, not detecting transitions.

Usage:
    python3 audit_hmm_sensitivity.py
    python3 audit_hmm_sensitivity.py --months 12   # extend window
    python3 audit_hmm_sensitivity.py --ticker SI=F  # silver
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

HISTORY_CSV = ROOT / "data" / "regime_history.csv"
CURRENT_VETO_THRESHOLD = 0.65    # matches regime_detector.py REGIME_VETO_THRESHOLD
DANGER_VETO_RATE       = 0.40    # >40% veto days = "permanent fear" flag

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t

RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
GREEN  = lambda t: _c("32;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_history(ticker: str, months: int) -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        print(RED(f"FAIL: {HISTORY_CSV} not found."))
        print("      Run: python3 scripts/regime_detector.py --fit GC=F")
        sys.exit(1)

    df = pd.read_csv(HISTORY_CSV)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    if df.empty:
        print(RED(f"FAIL: No history rows for ticker={ticker}"))
        sys.exit(1)

    cutoff = pd.Timestamp(date.today() - timedelta(days=int(months * 30.44)))
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    if df.empty:
        print(RED(f"FAIL: No data in last {months} months for {ticker}"))
        sys.exit(1)

    return df


# ── Streak analysis ───────────────────────────────────────────────────────────

def streak_stats(veto_series: pd.Series) -> dict:
    """
    Given a boolean series, compute streak statistics for True runs.
    Returns max_streak, mean_streak, current_streak, n_episodes.
    """
    streaks = []
    current = 0
    for v in veto_series:
        if v:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    current_streak = current if veto_series.iloc[-1] else 0

    return {
        "max_streak":     max(streaks, default=0),
        "mean_streak":    round(float(np.mean(streaks)), 1) if streaks else 0.0,
        "current_streak": current_streak,
        "n_episodes":     len(streaks),
    }


# ── Main audit ────────────────────────────────────────────────────────────────

def run_audit(ticker: str, months: int) -> None:
    df = load_history(ticker, months)
    n  = len(df)

    date_start = df["date"].iloc[0].date()
    date_end   = df["date"].iloc[-1].date()

    print()
    print(BOLD("═" * 68))
    print(BOLD(f"  HMM SENSITIVITY AUDIT — {ticker}"))
    print(BOLD(f"  Window: {date_start}  →  {date_end}  ({n} trading days ≈ {months}mo)"))
    print(BOLD("═" * 68))

    # ── Section 1: Regime occupancy ───────────────────────────────────────────
    print(BOLD("\n[1/3]  Regime Occupancy (Viterbi hard labels)"))
    print("─" * 52)

    for label in ["BEARISH", "RANGING", "BULLISH"]:
        count = (df["regime_label"] == label).sum()
        pct   = count / n * 100
        bar   = "█" * int(pct / 2.5)
        color = RED if label == "BEARISH" else (YELLOW if label == "RANGING" else GREEN)
        print(f"  {color(f'{label:8s}')}  {bar:<40}  {pct:5.1f}%  ({count:4d} days)")

    # ── Section 2: Veto rate at current threshold ─────────────────────────────
    print(BOLD(f"\n[2/3]  Veto Rate — Current threshold P(BEARISH) > {CURRENT_VETO_THRESHOLD}"))
    print("─" * 52)

    veto_mask  = df["p_bearish"] > CURRENT_VETO_THRESHOLD
    veto_days  = veto_mask.sum()
    veto_rate  = veto_days / n

    if veto_rate > DANGER_VETO_RATE:
        verdict = RED(f"DANGER — {veto_rate:.1%} exceeds {DANGER_VETO_RATE:.0%} threshold")
        verdict += RED("  (model is in 'permanent fear' mode)")
    elif veto_rate > 0.30:
        verdict = YELLOW(f"ELEVATED — {veto_rate:.1%}  (monitor closely)")
    else:
        verdict = GREEN(f"HEALTHY — {veto_rate:.1%}  (within expected range)")

    print(f"  Veto days:  {veto_days} / {n}")
    print(f"  Veto rate:  {verdict}")

    # Streak analysis at current threshold
    streaks = streak_stats(veto_mask)
    streak_color = RED if streaks["current_streak"] > 10 else (YELLOW if streaks["current_streak"] > 5 else GREEN)
    print(f"\n  Veto streak analysis:")
    print(f"    Max consecutive veto days:     {RED(str(streaks['max_streak']))}")
    print(f"    Mean veto episode length:      {streaks['mean_streak']} days")
    print(f"    Number of veto episodes:       {streaks['n_episodes']}")
    print(f"    Current veto streak:           {streak_color(str(streaks['current_streak']))} days  ← live now")

    if streaks["current_streak"] > 0:
        streak_start = df[veto_mask]["date"].iloc[-streaks["current_streak"]].date()
        print(f"    Current streak began:          {streak_start}")

    # ── Section 3: Threshold sensitivity table ────────────────────────────────
    print(BOLD("\n[3/3]  Threshold Sensitivity Table"))
    print("─" * 68)
    print(f"  {'Threshold':>10}  {'Veto Days':>10}  {'Veto Rate':>10}  {'Max Streak':>12}  {'Verdict':}")
    print("  " + "─" * 64)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for thr in thresholds:
        mask   = df["p_bearish"] > thr
        days   = mask.sum()
        rate   = days / n
        st     = streak_stats(mask)
        active = "  ◄ CURRENT" if thr == CURRENT_VETO_THRESHOLD else ""

        if rate > DANGER_VETO_RATE:
            rate_str = RED(f"{rate:9.1%}")
            verdict  = RED("TOO TIGHT")
        elif rate > 0.30:
            rate_str = YELLOW(f"{rate:9.1%}")
            verdict  = YELLOW("ELEVATED")
        else:
            rate_str = GREEN(f"{rate:9.1%}")
            verdict  = GREEN("OK")

        print(f"  {thr:>10.2f}  {days:>10}  {rate_str}  {st['max_streak']:>12}  {verdict}{active}")

    # ── Section 4: P(BEARISH) distribution ───────────────────────────────────
    print(BOLD("\n[4]  P(BEARISH) Posterior Distribution"))
    print("─" * 52)

    pct_list = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    pctls    = np.percentile(df["p_bearish"].values, pct_list)
    for p, v in zip(pct_list, pctls):
        bar = "█" * int(v * 40)
        print(f"  P{p:>3}  {bar:<40}  {v:.3f}")

    print(f"\n  Mean P(BEARISH):   {df['p_bearish'].mean():.3f}")
    print(f"  Median P(BEARISH): {df['p_bearish'].median():.3f}")
    print(f"  Days P(BEARISH)=1.00 (fully committed bearish): "
          f"{(df['p_bearish'] >= 0.999).sum()}")

    # ── Summary verdict ───────────────────────────────────────────────────────
    print()
    print(BOLD("═" * 68))
    print(BOLD("  SUMMARY VERDICT"))
    print(BOLD("═" * 68))

    if veto_rate > DANGER_VETO_RATE:
        print(RED(f"\n  FAIL — Veto fires on {veto_rate:.1%} of days (limit: {DANGER_VETO_RATE:.0%})"))
        print(RED(  "  The model is too sensitive. Recommend raising REGIME_VETO_THRESHOLD"))

        # Find the lowest threshold that brings rate below 40%
        for thr in thresholds:
            mask = df["p_bearish"] > thr
            if mask.sum() / n <= DANGER_VETO_RATE:
                print(YELLOW(f"  Suggested threshold: {thr:.2f} → {mask.sum()/n:.1%} veto rate"))
                break

    elif veto_rate > 0.30:
        print(YELLOW(f"\n  WARN — Veto rate {veto_rate:.1%} is elevated but below danger threshold."))
        print(YELLOW( "  Monitor over the next 2 weeks before adjusting."))
    else:
        print(GREEN(f"\n  PASS — Veto rate {veto_rate:.1%} is healthy."))
        print(GREEN( "  The HMM is not in permanent fear mode."))

    if streaks["current_streak"] > 15:
        print(RED(f"\n  WARNING: Current streak of {streaks['current_streak']} consecutive veto days"))
        print(RED( "  suggests the model may be stuck. Consider re-fitting:"))
        print(RED( "  python3 scripts/regime_detector.py --fit GC=F"))

    print(BOLD("\n" + "═" * 68 + "\n"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM Veto Sensitivity Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--months",  type=int,   default=6,     help="Lookback in months (default: 6)")
    parser.add_argument("--ticker",  type=str,   default="GC=F", help="Ticker to audit (default: GC=F)")
    args = parser.parse_args()

    run_audit(args.ticker.upper(), args.months)


if __name__ == "__main__":
    main()
