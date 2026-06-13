#!/usr/bin/env python3
"""
scripts/model_evaluator.py
==========================
Weekly self-improving feedback loop — Dynamic Committee Weight Adjustment.

What it does
------------
1. Reads data/decision_log.json and selects entries that are ≥7 trading days old
   AND have both quant_conviction and macro_conviction recorded.

2. For each qualifying entry, fetches the actual gold (or silver) price
   7 trading days after the decision date via yfinance.

3. Grades each agent (Quant / Macro) independently:
   - A directional call is made when |conviction| ≥ 2.
   - The market "moved" when the 7-day price change exceeds ±0.5%.
   - Correct = conviction sign matches actual price direction.
   - Abstain (|conviction| < 2) → not graded for that agent.

4. Computes rolling accuracy over the last MAX_WINDOW=20 graded decisions
   for each agent separately.

5. Applies EMA weight update (α = 0.15, anti-overfit damping):
       target_w = 0.50 + 0.20 × (accuracy − 0.50)
       new_w    = (1 − α) × old_w + α × target_w
   Hard bounds: [W_MIN=0.30, W_MAX=0.70].
   Maximum single-update shift: ±(0.20 × 0.15) = ±3 pp per week.

6. Writes data/committee_weights.json with the updated weights.

7. metal_logic.py reads committee_weights.json via _load_committee_weights()
   and uses the weights when combining Quant+Macro conviction before the CIO.

Usage
-----
    python3 scripts/model_evaluator.py
    python3 scripts/model_evaluator.py --ticker GC=F --window 30 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DECISION_LOG     = ROOT / "data" / "decision_log.json"
WEIGHTS_FILE     = ROOT / "data" / "committee_weights.json"
GRADE_LOG_FILE   = ROOT / "data" / "evaluator_grade_log.json"

# ── Tuning constants ──────────────────────────────────────────────────────────
LOOKFORWARD_DAYS    = 7       # calendar days ahead to measure outcome
CONVICTION_MIN      = 2       # |conviction| must meet this to count as a call
MOVE_THRESHOLD_PCT  = 0.5     # price must move > this % to avoid noise-grade
MAX_WINDOW          = 20      # rolling accuracy window (number of graded decisions)
EMA_ALPHA           = 0.15    # weight update speed — lower = more stable
W_MIN               = 0.30    # floor: no agent drops below 30%
W_MAX               = 0.70    # ceiling: no agent exceeds 70%
DEFAULT_WEIGHT      = 0.50    # starting weight when no history exists

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t


RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
GREEN  = lambda t: _c("32;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)


# ── Trading-day calendar ──────────────────────────────────────────────────────

def add_trading_days(start: date, n: int) -> date:
    """Advance `start` by exactly n trading days (Mon-Fri, no holiday filter)."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:   # Mon=0 … Fri=4
            added += 1
    return d


# ── Price fetcher ─────────────────────────────────────────────────────────────

def fetch_close_price(ticker: str, target_date: date, retries: int = 3) -> float | None:
    """
    Fetch the closing price for `ticker` on `target_date` using yfinance.
    Falls back up to ±2 trading days if the exact date has no data
    (weekends, holidays).  Returns None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        print(RED("yfinance not installed — run: pip install yfinance"))
        return None

    # Try ±2 trading days around target
    for offset in range(0, 3):
        check = target_date + timedelta(days=offset)
        end   = check + timedelta(days=1)
        for attempt in range(retries):
            try:
                tkr  = yf.Ticker(ticker)
                hist = tkr.history(start=check.isoformat(), end=end.isoformat())
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
                break  # no data for this offset — try next
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

    return None


# ── Loader + filter ───────────────────────────────────────────────────────────

def load_gradeable_entries(ticker: str) -> list[dict]:
    """
    Load decision_log entries for `ticker` that:
      - Are ≥ LOOKFORWARD_DAYS calendar days old (outcome should be settled)
      - Have both quant_conviction and macro_conviction fields
    """
    if not DECISION_LOG.exists():
        print(RED(f"FAIL: {DECISION_LOG} not found."))
        sys.exit(1)

    raw: list[dict] = json.loads(DECISION_LOG.read_text())
    cutoff = date.today() - timedelta(days=LOOKFORWARD_DAYS)

    entries = []
    for e in raw:
        if e.get("ticker", "") != ticker:
            continue
        if "quant_conviction" not in e or "macro_conviction" not in e:
            continue
        try:
            entry_date = date.fromisoformat(e["date"])
        except (KeyError, ValueError):
            continue
        if entry_date > cutoff:
            continue
        entries.append({**e, "_entry_date": entry_date})

    return entries


# ── Grading logic ─────────────────────────────────────────────────────────────

def grade_entry(entry: dict, ticker: str) -> dict | None:
    """
    For a single decision entry, fetch the outcome price and grade each agent.

    Returns a grade dict or None if outcome price unavailable.
    """
    entry_date   = entry["_entry_date"]
    price_then   = float(entry["price_at_time"])
    q_conv       = int(entry["quant_conviction"])
    m_conv       = int(entry["macro_conviction"])

    outcome_date = add_trading_days(entry_date, LOOKFORWARD_DAYS)
    price_after  = fetch_close_price(ticker, outcome_date)

    if price_after is None:
        return None

    pct_change = (price_after - price_then) / price_then * 100.0
    if abs(pct_change) < MOVE_THRESHOLD_PCT:
        direction = 0   # noise — grade as neutral
    elif pct_change > 0:
        direction = +1  # price rose
    else:
        direction = -1  # price fell

    def _grade(conv: int) -> str | None:
        """Returns 'correct', 'wrong', or None (abstain)."""
        if abs(conv) < CONVICTION_MIN:
            return None   # not a directional call — skip
        if direction == 0:
            return None   # flat move — too noisy to grade
        predicted = +1 if conv > 0 else -1
        return "correct" if predicted == direction else "wrong"

    q_grade = _grade(q_conv)
    m_grade = _grade(m_conv)

    return {
        "entry_date":    entry_date.isoformat(),
        "outcome_date":  outcome_date.isoformat(),
        "price_then":    round(price_then, 2),
        "price_after":   round(price_after, 2),
        "pct_change":    round(pct_change, 3),
        "direction":     direction,
        "quant_conviction": q_conv,
        "macro_conviction": m_conv,
        "quant_grade":   q_grade,
        "macro_grade":   m_grade,
        "action_taken":  entry.get("action_taken", ""),
    }


# ── Rolling accuracy ──────────────────────────────────────────────────────────

def rolling_accuracy(grades: list[dict], agent: str, window: int) -> tuple[float | None, int]:
    """
    Compute rolling accuracy for `agent` (quant|macro) over the last `window`
    GRADED decisions (None grades are skipped).

    Returns (accuracy, n_graded).
    """
    key = f"{agent}_grade"
    graded = [g[key] for g in grades if g[key] is not None][-window:]
    if not graded:
        return None, 0
    correct = sum(1 for g in graded if g == "correct")
    return correct / len(graded), len(graded)


# ── Weight update ─────────────────────────────────────────────────────────────

def load_current_weights() -> dict:
    """Load committee_weights.json or return defaults."""
    if WEIGHTS_FILE.exists():
        try:
            return json.loads(WEIGHTS_FILE.read_text())
        except Exception:
            pass
    return {
        "quant_weight":      DEFAULT_WEIGHT,
        "macro_weight":      DEFAULT_WEIGHT,
        "last_updated":      None,
        "quant_graded_n":    0,
        "macro_graded_n":    0,
        "quant_accuracy":    None,
        "macro_accuracy":    None,
    }


def ema_update(old_w: float, accuracy: float) -> float:
    """
    Apply EMA weight update bounded to [W_MIN, W_MAX].

    target_w = 0.50 + 0.20 × (accuracy − 0.50)
      → accuracy=1.00 → target=0.70
      → accuracy=0.50 → target=0.50
      → accuracy=0.00 → target=0.30

    new_w = (1 − α) × old_w + α × target_w
    """
    target_w = 0.50 + 0.20 * (accuracy - 0.50)
    target_w = max(W_MIN, min(W_MAX, target_w))
    new_w    = (1.0 - EMA_ALPHA) * old_w + EMA_ALPHA * target_w
    return round(max(W_MIN, min(W_MAX, new_w)), 4)


# ── Main evaluator ────────────────────────────────────────────────────────────

def run_evaluator(ticker: str, window: int, dry_run: bool) -> None:
    print()
    print(BOLD("═" * 68))
    print(BOLD(f"  COMMITTEE WEIGHT EVALUATOR — {ticker}"))
    print(BOLD(f"  Date: {date.today()}  |  window={window}  |  dry_run={dry_run}"))
    print(BOLD("═" * 68))

    # ── Load entries ──────────────────────────────────────────────────────────
    entries = load_gradeable_entries(ticker)
    print(f"\n  Gradeable entries found: {len(entries)}")

    if not entries:
        print(YELLOW("  No entries with conviction fields and settled outcome — skipping update."))
        print(BOLD("\n" + "═" * 68 + "\n"))
        return

    # ── Grade each entry (fetch outcome prices) ───────────────────────────────
    print(f"  Fetching outcome prices ({LOOKFORWARD_DAYS}-day forward)…\n")

    # Load existing grade log to avoid re-fetching known outcomes
    existing_grades: list[dict] = []
    if GRADE_LOG_FILE.exists():
        try:
            existing_grades = json.loads(GRADE_LOG_FILE.read_text())
        except Exception:
            existing_grades = []

    graded_keys = {(g["entry_date"], g.get("action_taken", "")) for g in existing_grades}
    new_grades:  list[dict] = []

    for e in entries:
        key = (e["date"], e.get("action_taken", ""))
        if key in graded_keys:
            continue   # already graded in a previous run

        print(f"  Grading {e['date']}  ({e.get('action_taken','?')})  "
              f"Q={e['quant_conviction']:+d}  M={e['macro_conviction']:+d}  ", end="", flush=True)

        grade = grade_entry(e, ticker)
        if grade is None:
            print(DIM("→ no outcome price (skipped)"))
            continue

        pct = grade["pct_change"]
        arrow = ("↑" if pct > 0 else "↓") if abs(pct) >= MOVE_THRESHOLD_PCT else "→"
        print(f"→ {arrow} {pct:+.2f}%  "
              f"Q:{grade['quant_grade'] or 'abstain':7s}  "
              f"M:{grade['macro_grade'] or 'abstain':7s}")
        new_grades.append(grade)

    all_grades = existing_grades + new_grades

    # Persist updated grade log
    if not dry_run and new_grades:
        GRADE_LOG_FILE.write_text(json.dumps(all_grades, indent=2))
        print(f"\n  Grade log updated: {len(all_grades)} total entries.")
    elif new_grades:
        print(f"\n  [DRY-RUN] Would write {len(all_grades)} grades to {GRADE_LOG_FILE.name}.")

    # ── Rolling accuracy ──────────────────────────────────────────────────────
    q_acc, q_n = rolling_accuracy(all_grades, "quant", window)
    m_acc, m_n = rolling_accuracy(all_grades, "macro", window)

    print()
    print(BOLD("[1/2]  Rolling Accuracy"))
    print("─" * 52)

    def _acc_line(name: str, acc: float | None, n: int) -> None:
        if acc is None:
            print(f"  {name:6s}: {DIM('no graded data yet')}")
            return
        bar   = "█" * int(acc * 40)
        color = GREEN if acc >= 0.60 else (YELLOW if acc >= 0.45 else RED)
        print(f"  {name:6s}: {color(f'{acc:.1%}'):15s}  {bar:<40}  ({n} decisions)")

    _acc_line("Quant", q_acc, q_n)
    _acc_line("Macro", m_acc, m_n)

    # ── Weight update ─────────────────────────────────────────────────────────
    print()
    print(BOLD("[2/2]  Weight Update (EMA α=0.15)"))
    print("─" * 52)

    current = load_current_weights()
    old_qw  = current["quant_weight"]
    old_mw  = current["macro_weight"]

    if q_acc is not None:
        new_qw = ema_update(old_qw, q_acc)
    else:
        new_qw = old_qw

    if m_acc is not None:
        new_mw = ema_update(old_mw, m_acc)
    else:
        new_mw = old_mw

    def _weight_line(name: str, old: float, new: float) -> None:
        delta = new - old
        delta_str = f"{delta:+.4f}"
        color = GREEN if delta > 0.001 else (RED if delta < -0.001 else DIM)
        print(f"  {name:6s}: {old:.4f} → {color(f'{new:.4f}')}  ({color(delta_str)})")

    _weight_line("Quant", old_qw, new_qw)
    _weight_line("Macro", old_mw, new_mw)

    # Warn if a weight hits its bound
    if new_qw <= W_MIN + 0.001:
        print(YELLOW(f"  WARNING: Quant weight at floor {W_MIN} — model may be systematically wrong"))
    if new_mw <= W_MIN + 0.001:
        print(YELLOW(f"  WARNING: Macro weight at floor {W_MIN} — model may be systematically wrong"))
    if new_qw >= W_MAX - 0.001:
        print(YELLOW(f"  WARNING: Quant weight at ceiling {W_MAX} — consider re-fitting HMM/LSTM"))
    if new_mw >= W_MAX - 0.001:
        print(YELLOW(f"  WARNING: Macro weight at ceiling {W_MAX} — check Perplexity data quality"))

    # ── Write output ──────────────────────────────────────────────────────────
    payload = {
        "ticker":           ticker,
        "last_updated":     date.today().isoformat(),
        "quant_weight":     new_qw,
        "macro_weight":     new_mw,
        "quant_accuracy":   round(q_acc, 4) if q_acc is not None else None,
        "macro_accuracy":   round(m_acc, 4) if m_acc is not None else None,
        "quant_graded_n":   q_n,
        "macro_graded_n":   m_n,
        "window_used":      window,
        "ema_alpha":        EMA_ALPHA,
        "bounds":           [W_MIN, W_MAX],
    }

    if dry_run:
        print(YELLOW(f"\n  [DRY-RUN] Would write to {WEIGHTS_FILE.name}:"))
        print(f"    {json.dumps(payload, indent=4)}")
    else:
        WEIGHTS_FILE.write_text(json.dumps(payload, indent=2))
        print(GREEN(f"\n  Weights written to {WEIGHTS_FILE.name}"))

    print(BOLD("\n" + "═" * 68 + "\n"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic Committee Weight Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",  default="GC=F",  help="Ticker to evaluate (default: GC=F)")
    parser.add_argument("--window",  type=int, default=MAX_WINDOW,
                        help=f"Rolling accuracy window (default: {MAX_WINDOW})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print updated weights without writing files")
    args = parser.parse_args()

    run_evaluator(args.ticker.upper(), args.window, args.dry_run)


if __name__ == "__main__":
    main()
