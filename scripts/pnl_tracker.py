#!/usr/bin/env python3
"""
Daily P&L Tracker  (Phase XIII Stage 71)
==========================================
Persists a day-by-day mark-to-market series of the platform's portfolio
value so we can compute realised performance, rolling Sharpe, drawdown,
and YTD return — all from actual book state, not paper backtest.

Sources of daily NAV:
  shadow_book.db  portfolio_value     (metals core + cash)
  portfolio.json  sum of equity holdings × last spot

Outputs a rolling stats file consumed by tear_sheet + UI:
  - latest_nav, prev_nav, day_pnl_usd, day_pnl_pct
  - cumulative_return_pct since first row
  - 30d / 90d / 252d Sharpe
  - 30d / 90d / 252d max drawdown
  - winning days / losing days ratio

Persistent history: data/nav_history.csv
Snapshot:           data/pnl_tracker.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "pnl_tracker.json"
NAV_HISTORY = DATA_DIR / "nav_history.csv"
SHADOW_DB = DATA_DIR / "shadow_book.db"

SQ252 = float(np.sqrt(252))
LINE_W = 62
SEP = "━" * LINE_W


def _read_shadow_nav() -> tuple[float, float, float]:
    if not SHADOW_DB.exists():
        return 0.0, 0.0, 0.0
    try:
        conn = sqlite3.connect(SHADOW_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM portfolio_state WHERE id = 1"
        ).fetchone()
        conn.close()
        if not row:
            return 0.0, 0.0, 0.0
        gold_oz = float(row["gold_oz"] or 0)
        cash = float(row["cash_usd"] or 0)
        pv = float(row["portfolio_value"] or 0)
        return gold_oz, cash, pv
    except Exception:
        return 0.0, 0.0, 0.0


def _read_equity_nav() -> float:
    pj = DATA_DIR / "portfolio.json"
    if not pj.exists():
        return 0.0
    try:
        data = json.loads(pj.read_text())
        if not isinstance(data, list):
            return 0.0
        nav = 0.0
        for entry in data:
            qty = float(entry.get("shares") or entry.get("qty") or 0)
            price = float(entry.get("price_usd") or entry.get("last") or 0)
            nav += qty * price
        return nav
    except Exception:
        return 0.0


def _load_nav_history() -> pd.DataFrame:
    if not NAV_HISTORY.exists():
        return pd.DataFrame(columns=["date", "nav_usd", "metals_pv", "equity_pv"])
    try:
        df = pd.read_csv(NAV_HISTORY)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "nav_usd", "metals_pv", "equity_pv"])


def _save_nav_row(today: date, metals_pv: float, equity_pv: float) -> pd.DataFrame:
    df = _load_nav_history()
    total = metals_pv + equity_pv
    # Replace today's row if it exists (defensive on empty df)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.date != today]
    new_row = pd.DataFrame([{
        "date":      pd.Timestamp(today),
        "nav_usd":   total,
        "metals_pv": metals_pv,
        "equity_pv": equity_pv,
    }])
    df = pd.concat([df, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NAV_HISTORY, index=False)
    return df


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _stats_for_window(returns: pd.Series, window: int, label: str) -> dict:
    r = returns.dropna().tail(window)
    if len(r) < 5:
        return {"window": label, "n": len(r), "sharpe": None, "max_dd_pct": None,
                "total_return_pct": None, "win_days_pct": None}
    ann_r = float(r.mean() * 252)
    ann_v = float(r.std() * SQ252)
    sharpe = ann_r / ann_v if ann_v > 1e-9 else 0.0
    cum = (1 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    win_days = float((r > 0).mean())
    total_return = float(cum.iloc[-1] - 1)
    return {
        "window":            label,
        "n":                 int(len(r)),
        "sharpe":            round(sharpe, 3),
        "max_dd_pct":        round(max_dd * 100, 3),
        "total_return_pct":  round(total_return * 100, 3),
        "win_days_pct":      round(win_days * 100, 2),
    }


def run_pnl_tracker() -> dict:
    today = date.today()
    _, _, metals_pv = _read_shadow_nav()
    equity_pv = _read_equity_nav()
    df = _save_nav_row(today, metals_pv, equity_pv)

    if len(df) < 2:
        result = {
            "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "today":                 today.isoformat(),
            "latest_nav_usd":        float(df["nav_usd"].iloc[-1]),
            "metals_pv":             metals_pv,
            "equity_pv":             equity_pv,
            "day_pnl_usd":           0.0,
            "day_pnl_pct":           0.0,
            "cumulative_return_pct": 0.0,
            "n_history":             int(len(df)),
            "windows":               {},
            "warning":               "insufficient history for stats (< 2 rows)",
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
        _print_report(result)
        return result

    # Daily returns
    df["return"] = df["nav_usd"].pct_change()
    returns = df["return"]
    latest_nav = float(df["nav_usd"].iloc[-1])
    prev_nav = float(df["nav_usd"].iloc[-2])
    day_pnl_usd = latest_nav - prev_nav
    day_pnl_pct = (latest_nav / prev_nav - 1) * 100 if prev_nav > 0 else 0

    first_nav = float(df["nav_usd"].iloc[0])
    cum_return_pct = (latest_nav / first_nav - 1) * 100 if first_nav > 0 else 0

    windows = {
        "30d":  _stats_for_window(returns, 30, "30d"),
        "90d":  _stats_for_window(returns, 90, "90d"),
        "252d": _stats_for_window(returns, 252, "252d"),
        "all":  _stats_for_window(returns, len(returns), "all"),
    }

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today":            today.isoformat(),
        "n_history":        int(len(df)),
        "first_history":    str(df["date"].iloc[0].date()),
        "latest_nav_usd":   round(latest_nav, 2),
        "prev_nav_usd":     round(prev_nav, 2),
        "day_pnl_usd":      round(day_pnl_usd, 2),
        "day_pnl_pct":      round(float(day_pnl_pct), 4),
        "metals_pv":        round(metals_pv, 2),
        "equity_pv":        round(equity_pv, 2),
        "cumulative_return_pct": round(float(cum_return_pct), 3),
        "windows":          windows,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  DAILY P&L TRACKER\n{SEP}")
    print(f"  Today:           {r.get('today', '—')}")
    print(f"  History rows:    {r.get('n_history', 0)}")
    if r.get("warning"):
        print(f"  ⚠ {r['warning']}")
        print(SEP)
        return
    print(f"  Latest NAV:      ${r['latest_nav_usd']:,.2f}")
    print(f"  Prev NAV:        ${r['prev_nav_usd']:,.2f}")
    pnl_color = "\033[32m" if r["day_pnl_usd"] >= 0 else "\033[31m"
    print(f"  Day P&L:         {pnl_color}${r['day_pnl_usd']:+,.2f}  "
          f"({r['day_pnl_pct']:+.4f}%)\033[0m")
    print(f"  Cumulative:      {r['cumulative_return_pct']:+.3f}%")
    print(f"  Metals book:     ${r['metals_pv']:,.2f}")
    print(f"  Equity book:     ${r['equity_pv']:,.2f}")
    print()
    print(f"  ROLLING WINDOWS")
    print(f"  {'─' * 56}")
    print(f"  {'window':<8s}  {'n':>4s}  {'Sharpe':>7s}  {'maxDD%':>7s}  {'win%':>5s}  {'totRet%':>8s}")
    for w in r["windows"].values():
        if w.get("sharpe") is None:
            print(f"  {w['window']:<8s}  {w['n']:>4d}  (insufficient data)")
            continue
        print(f"  {w['window']:<8s}  {w['n']:>4d}  "
              f"{w['sharpe']:>+7.3f}  {w['max_dd_pct']:>+7.2f}  "
              f"{w['win_days_pct']:>5.1f}  {w['total_return_pct']:>+8.3f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}  (history: {NAV_HISTORY})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily P&L Tracker")
    args = parser.parse_args()
    run_pnl_tracker()
