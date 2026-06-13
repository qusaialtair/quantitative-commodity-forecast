#!/usr/bin/env python3
"""
Earnings Calendar  (Phase XIII Stage 68)
==========================================
For every ticker in the halal universe, fetches the next earnings date via
yfinance and emits a per-ticker position guard.

Earnings windows are dangerous for equity trades — IV spikes pre-event and
post-event gaps routinely run ±10%. Standard institutional practice is to
exit positions ~2 days before earnings and avoid new entries until the day
after.

Outputs per ticker:
  - next_earnings_date   ISO yyyy-mm-dd or null
  - days_until           int or null
  - in_blackout          True if within [-2, +1] days
  - action               EXIT_BEFORE / HOLD_NEW / NORMAL

Top-level:
  blocked_tickers        list of tickers in any blackout
  filtered_universe      halal universe minus blackouts

Cached results live in data/earnings_cache.json (24h TTL) — yfinance.calendar
is slow per-ticker so we don't re-fetch every pipeline run.

Output: data/earnings_calendar.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "earnings_calendar.json"
CACHE_FILE = DATA_DIR / "earnings_cache.json"

PRE_BLACKOUT_DAYS = 2
POST_BLACKOUT_DAYS = 1
CACHE_TTL_SECONDS = 24 * 3600

LINE_W = 62
SEP = "━" * LINE_W


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))


def _is_fresh(cache_entry: dict) -> bool:
    return (
        cache_entry
        and "_cached_ts" in cache_entry
        and (time.time() - cache_entry["_cached_ts"]) < CACHE_TTL_SECONDS
    )


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------
def _fetch_next_earnings(ticker: str) -> str | None:
    """Best-effort: yfinance.calendar can be a dict or DataFrame depending
    on version. Return ISO date or None."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None
        # Modern yfinance returns dict
        if isinstance(cal, dict):
            for key in ("Earnings Date", "earningsDate", "earnings_date"):
                if key in cal:
                    val = cal[key]
                    if isinstance(val, (list, tuple)) and val:
                        val = val[0]
                    if hasattr(val, "isoformat"):
                        return val.isoformat()[:10]
                    if isinstance(val, str):
                        return val[:10]
        # DataFrame fallback (older yfinance)
        try:
            row = cal.loc["Earnings Date"] if hasattr(cal, "loc") else None
            if row is not None and len(row) > 0:
                v = row.iloc[0] if hasattr(row, "iloc") else row[0]
                if hasattr(v, "isoformat"):
                    return v.isoformat()[:10]
        except Exception:
            pass
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _action_for_days_until(days_until: int | None) -> tuple[bool, str]:
    if days_until is None:
        return False, "NORMAL"
    if -POST_BLACKOUT_DAYS <= days_until <= PRE_BLACKOUT_DAYS:
        if days_until > 0:
            return True, "EXIT_BEFORE"
        if days_until == 0:
            return True, "HOLD_NEW"
        return True, "HOLD_NEW"
    return False, "NORMAL"


def _load_halal_tickers() -> list[str]:
    hu = DATA_DIR / "halal_universe.json"
    if not hu.exists():
        return []
    try:
        data = json.loads(hu.read_text())
        return list(data.get("tickers", []))
    except Exception:
        return []


def run_earnings_calendar(tickers: list[str] | None = None) -> dict:
    cache = _load_cache()
    today = date.today()

    if tickers is None:
        tickers = _load_halal_tickers()
    if not tickers:
        result = {
            "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warning":        "No tickers (run halal_screener.py first)",
            "tickers_checked":0,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        return result

    per_ticker = []
    for tk in tickers:
        cache_entry = cache.get(tk, {})
        if _is_fresh(cache_entry):
            ed = cache_entry.get("earnings_date")
        else:
            ed = _fetch_next_earnings(tk)
            cache[tk] = {
                "earnings_date": ed,
                "_cached_ts":    time.time(),
            }

        days_until = None
        if ed:
            try:
                d = date.fromisoformat(ed)
                days_until = (d - today).days
            except Exception:
                pass
        in_blackout, action = _action_for_days_until(days_until)

        per_ticker.append({
            "ticker":              tk,
            "next_earnings_date":  ed,
            "days_until":          days_until,
            "in_blackout":         in_blackout,
            "action":              action,
        })

    _save_cache(cache)

    blocked = [r["ticker"] for r in per_ticker if r["in_blackout"]]
    filtered = [r["ticker"] for r in per_ticker if not r["in_blackout"]]

    # Most-imminent earnings (skip None and past)
    upcoming = [r for r in per_ticker
                if r["days_until"] is not None and r["days_until"] >= 0]
    upcoming.sort(key=lambda r: r["days_until"])
    next_n = upcoming[:5]

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today":             today.isoformat(),
        "tickers_checked":   len(per_ticker),
        "n_with_data":       sum(1 for r in per_ticker if r["next_earnings_date"]),
        "blackout_tickers":  blocked,
        "filtered_universe": filtered,
        "n_blocked":         len(blocked),
        "n_filtered":        len(filtered),
        "next_5_earnings":   next_n,
        "per_ticker":        per_ticker,
        "pre_blackout_days": PRE_BLACKOUT_DAYS,
        "post_blackout_days":POST_BLACKOUT_DAYS,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  EARNINGS CALENDAR\n{SEP}")
    print(f"  Today:            {r['today']}")
    print(f"  Tickers checked:  {r['tickers_checked']}")
    print(f"  With data:        {r['n_with_data']}")
    print(f"  Blocked:          {r['n_blocked']}  ({', '.join(r['blackout_tickers']) or 'none'})")
    print(f"  Tradable:         {r['n_filtered']}")
    print()
    print(f"  NEXT 5 UPCOMING EARNINGS")
    print(f"  {'─' * 56}")
    for row in r["next_5_earnings"]:
        action_color = {
            "EXIT_BEFORE": "\033[31m", "HOLD_NEW": "\033[31;1m", "NORMAL": "\033[32m"
        }.get(row["action"], "")
        print(f"  {row['ticker']:<8s}  {row['next_earnings_date']}  "
              f"T+{row['days_until']:<3d}d  "
              f"{action_color}{row['action']}\033[0m")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Calendar")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers (default: halal_universe)")
    args = parser.parse_args()
    tlist = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )
    run_earnings_calendar(tickers=tlist)
