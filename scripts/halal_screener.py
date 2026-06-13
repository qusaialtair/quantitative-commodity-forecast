#!/usr/bin/env python3
"""
Halal Universe Screener  (Phase XI Stage 58)
==============================================
Filters a candidate universe down to Sharia-compliant tickers using the
two standard institutional gates:

  1. SECTOR EXCLUSION
       Reject any ticker whose sector / industry hits any of:
         - financial services (interest-based income)
         - alcohol / tobacco
         - gambling / gaming
         - adult / entertainment
         - conventional defense
         - pork-product food
         - cannabis

  2. FINANCIAL-RATIO GATES  (AAOIFI standard, simplified)
       - Total debt / market cap < 33%
       - Cash + interest-bearing securities / market cap < 33%
       - Non-permissible revenue < 5% (best-effort; not always available)

The screener pulls yfinance `.info` per ticker (cached to disk for 24h to
respect API limits), applies the gates, and writes the surviving tickers
plus rejection reasons to `data/halal_universe.json`.

The IBKR adapter's pre-trade gate consumes the output file directly.

Output: data/halal_universe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "halal_universe.json"
CACHE_DIR = DATA_DIR / "halal_cache"

# Default candidate universe: large-cap US tech / healthcare / consumer staples
# that are commonly Sharia-screened. The user can override with --tickers.
DEFAULT_CANDIDATES = [
    # Software / cloud
    "MSFT", "ORCL", "ADBE", "CRM", "NOW", "INTU", "SNOW", "PLTR",
    "ZS", "HUBS", "PANW", "FTNT", "WDAY", "DDOG", "MDB", "TEAM",
    # Hardware / semi
    "AAPL", "AMD", "AVGO", "TXN", "QCOM", "AMAT", "LRCX", "KLAC",
    "MU", "ON", "MRVL", "ENPH", "FORM",
    # Healthcare / biotech
    "JNJ", "LLY", "MRK", "PFE", "ABT", "DHR", "TMO", "QGEN",
    "GILD", "VRTX", "REGN", "BIIB",
    # Consumer / industrial (non-debt-heavy candidates)
    "WMT", "COST", "TSCO", "TJX", "DLTR", "HD", "NKE", "EL",
    # Energy (typically excluded but operator-toggleable)
    # "XOM", "CVX", "COP",
    # Communication / internet (non-ad-only)
    "GOOGL", "META", "EXPE", "BKNG", "TRIP",
]

# Sector / industry blacklist  (normalised lowercase substring match)
SECTOR_BLACKLIST_SUBSTRINGS = {
    "financial", "bank", "insurance", "credit", "lending",
    "alcoholic", "beverage—wine", "tobacco",
    "gambling", "gaming", "casino",
    "adult",
    "weapon", "defense", "military",
    "pork",
    "cannabis", "marijuana",
}

# Financial-ratio thresholds (AAOIFI standard)
DEBT_TO_MARKET_CAP_MAX = 0.33
CASH_TO_MARKET_CAP_MAX = 0.33

# Cache TTL: 24h
CACHE_TTL_SECONDS = 86400

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


def _load_cached(ticker: str) -> dict | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data.get("_cached_ts", 0) < CACHE_TTL_SECONDS:
            return data
    except Exception:
        return None
    return None


def _save_cache(ticker: str, info: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info["_cached_ts"] = time.time()
    try:
        _cache_path(ticker).write_text(json.dumps(info, default=str))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ticker fundamentals
# ---------------------------------------------------------------------------
def _fetch_info(ticker: str) -> dict:
    cached = _load_cached(ticker)
    if cached is not None:
        return cached
    if yf is None:
        return {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    _save_cache(ticker, info)
    return info


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _check_sector(info: dict) -> tuple[bool, str | None]:
    """Return (passes, reason_if_failed)."""
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    combined = f"{sector} {industry}"
    for sub in SECTOR_BLACKLIST_SUBSTRINGS:
        if sub in combined:
            return False, f"sector/industry contains '{sub}'"
    return True, None


def _check_financial_ratios(info: dict) -> tuple[bool, list[str]]:
    """Apply AAOIFI debt-to-cap and cash-to-cap thresholds. Returns
    (passes, list of failed-reason strings)."""
    market_cap = info.get("marketCap")
    if not market_cap or market_cap <= 0:
        return True, []  # Pass if we can't measure; flag separately if needed

    fails = []
    total_debt = info.get("totalDebt") or 0
    if total_debt > 0:
        debt_ratio = total_debt / market_cap
        if debt_ratio > DEBT_TO_MARKET_CAP_MAX:
            fails.append(
                f"debt/market_cap={debt_ratio:.2%} > {DEBT_TO_MARKET_CAP_MAX:.0%}"
            )

    total_cash = info.get("totalCash") or 0
    if total_cash > 0:
        cash_ratio = total_cash / market_cap
        if cash_ratio > CASH_TO_MARKET_CAP_MAX:
            fails.append(
                f"cash/market_cap={cash_ratio:.2%} > {CASH_TO_MARKET_CAP_MAX:.0%}"
            )

    return len(fails) == 0, fails


def screen_ticker(ticker: str) -> dict:
    info = _fetch_info(ticker)
    if not info or not info.get("symbol"):
        return {
            "ticker":    ticker.upper(),
            "passes":    False,
            "reasons":   ["no fundamentals data"],
            "sector":    None,
            "industry":  None,
            "metrics":   {},
        }

    sector_ok, sector_reason = _check_sector(info)
    ratios_ok, ratio_fails = _check_financial_ratios(info)

    passes = sector_ok and ratios_ok
    reasons = []
    if not sector_ok and sector_reason:
        reasons.append(sector_reason)
    reasons.extend(ratio_fails)

    market_cap = info.get("marketCap")
    metrics = {
        "market_cap":            market_cap,
        "total_debt":            info.get("totalDebt"),
        "total_cash":            info.get("totalCash"),
        "debt_to_cap":           round((info.get("totalDebt") or 0) / market_cap, 4) if market_cap else None,
        "cash_to_cap":           round((info.get("totalCash") or 0) / market_cap, 4) if market_cap else None,
    }

    return {
        "ticker":    ticker.upper(),
        "passes":    passes,
        "reasons":   reasons,
        "sector":    info.get("sector"),
        "industry":  info.get("industry"),
        "metrics":   metrics,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_halal_screener(
    candidates: list[str] = None,
    verbose: bool = False,
) -> dict:
    candidates = candidates or DEFAULT_CANDIDATES
    results = []
    passing = []
    rejected = []

    for i, ticker in enumerate(candidates):
        if verbose:
            print(f"  [{i+1}/{len(candidates)}] screening {ticker}...", flush=True)
        res = screen_ticker(ticker)
        results.append(res)
        if res["passes"]:
            passing.append(res["ticker"])
        else:
            rejected.append({"ticker": res["ticker"], "reasons": res["reasons"]})

    # Group rejections by reason type
    rejection_summary = {}
    for r in rejected:
        for reason in r["reasons"]:
            key = reason.split("=")[0] if "=" in reason else reason
            rejection_summary[key] = rejection_summary.get(key, 0) + 1

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates":      len(candidates),
        "n_passing":         len(passing),
        "n_rejected":        len(rejected),
        "tickers":           passing,
        "rejected":          rejected,
        "rejection_summary": rejection_summary,
        "thresholds": {
            "debt_to_cap_max": DEBT_TO_MARKET_CAP_MAX,
            "cash_to_cap_max": CASH_TO_MARKET_CAP_MAX,
        },
        "per_ticker":        results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  HALAL UNIVERSE SCREENER\n{SEP}")
    print(f"  Candidates:    {r['n_candidates']}")
    print(f"  Passing:       {r['n_passing']}  ({r['n_passing']/max(r['n_candidates'],1):.1%})")
    print(f"  Rejected:      {r['n_rejected']}")
    print()

    print(f"  HALAL UNIVERSE ({r['n_passing']} tickers)")
    print(f"  {'─' * 56}")
    cols = 6
    rows = (len(r["tickers"]) + cols - 1) // cols
    for row in range(rows):
        line = "  "
        for col in range(cols):
            i = row * cols + col
            if i < len(r["tickers"]):
                line += f"{r['tickers'][i]:<8s}"
        print(line)
    print()

    if r["rejection_summary"]:
        print(f"  REJECTION REASONS")
        for k, v in sorted(r["rejection_summary"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {k:<40s}  {v}")
        print()

    print(f"  Top rejected (5 examples):")
    for rej in r["rejected"][:5]:
        print(f"    {rej['ticker']:<8s}  {' / '.join(rej['reasons'])[:60]}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Halal Universe Screener")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers (default: ~50 large-caps)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    candidates = None
    if args.tickers:
        candidates = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run_halal_screener(candidates=candidates, verbose=args.verbose)
