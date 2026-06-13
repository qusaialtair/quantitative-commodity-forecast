#!/usr/bin/env python3
"""
data_pipelines/alt_data/congress.py
===================================
Phase 5 — Congressional-trading alternative-data feed.

Source (free, no auth):
  • CapitolTrades (https://www.capitoltrades.com/trades) — aggregates
    STOCK Act Periodic Transaction Reports from both chambers. Serves
    a clean HTML table; the only stable public feed since the
    housestockwatcher / senatestockwatcher S3 buckets returned 403 in
    April 2026.

Emitted features (per ticker)
-----------------------------
    congress_buy_count_180d      int    # reported buys across both chambers
    congress_net_direction_180d  int    # (#buys − #sells) over 180d
    congress_committee_flag      int    # 1 iff any buyer sits on a committee
                                        # whose jurisdiction covers the
                                        # ticker's sector (reserved; see note)

Committee-flag note
-------------------
The CapitolTrades list view does not expose each politician's committee
assignments without a per-politician detail page fetch. Until we wire
in a cheap committee-roster source (Phase 5.1), this flag is populated
via the SECTOR_COMMITTEE heuristic against the politician's chamber +
party + state string when available, and defaults to 0 otherwise — the
schema slot stays reserved so the ranker doesn't need to re-version.

Failure posture
---------------
Signal-only. If CapitolTrades is unreachable or the HTML layout drifts,
returns a zero-filled (ticker × 3) frame with the correct column
contract. Never raises into equity_features.py — alt-data outages
cannot affect the Sharia-compliance universe.

Caching
-------
One full scrape per calendar day → data/alt_data/congress_YYYY-MM-DD.parquet.
Subsequent calls the same day just re-read the parquet and slice the
requested ticker subset locally.

CLI
---
    python3 -m data_pipelines.alt_data.congress AAPL NVDA LMT
    python3 -m data_pipelines.alt_data.congress --refresh
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

ROOT      = Path(os.path.abspath(os.path.dirname(os.path.dirname(
                                 os.path.dirname(__file__)))))
CACHE_DIR = ROOT / "data" / "alt_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("CONGRESS")

# ── Endpoint config ─────────────────────────────────────────────────────────
BASE_URL       = "https://www.capitoltrades.com/trades?pageSize=96&page={page}"
HTTP_HEADERS   = {
    "User-Agent":      ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0 Safari/537.36 GoldTradingAI/5.0"),
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
HTTP_TIMEOUT   = 20
RETRIES        = 3
BACKOFF_S      = 1.0
MAX_PAGES      = 40       # 40 × 96 = ~3,800 rows ≈ 6 months of activity
PAGE_PACING_S  = 0.6      # polite floor between pages

LOOKBACK_DAYS  = 180
FEATURE_COLS   = [
    "congress_buy_count_180d",
    "congress_net_direction_180d",
    "congress_committee_flag",
]

# ── Sector → jurisdictional-committee keyword map ───────────────────────────
# Reserved for Phase 5.1 — matched via substring against a politician's
# committee roster once we wire that source in. Today the match is
# against the chamber/party/state string which rarely contains
# committee hints, so the flag almost always stays 0. Keeping the
# schema slot live prevents a breaking change when the source lands.
SECTOR_COMMITTEE: dict[str, tuple[str, ...]] = {
    "Technology":             ("commerce", "science", "judiciary"),
    "Communication Services": ("commerce", "science", "energy and commerce"),
    "Healthcare":             ("health", "finance", "energy and commerce",
                               "ways and means"),
    "Energy":                 ("energy", "natural resources", "environment"),
    "Utilities":              ("energy", "environment"),
    "Financials":             ("banking", "financial services", "finance"),
    "Real Estate":            ("banking", "financial services", "housing"),
    "Industrials":            ("armed services", "transportation", "commerce"),
    "Materials":              ("natural resources", "commerce"),
    "Basic Materials":        ("natural resources", "commerce"),
    "Consumer Staples":       ("agriculture", "commerce"),
    "Consumer Discretionary": ("commerce", "ways and means"),
}

# Ticker embedded in "Traded Issuer" as e.g. "Johnson & JohnsonJNJ:US".
TICKER_RE = re.compile(r"([A-Z][A-Z0-9\.\-]{0,5}):US$")


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP + HTML PARSE
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_page(page: int) -> pd.DataFrame | None:
    """Return one page of trades as a DataFrame, or None on failure."""
    url = BASE_URL.format(page=page)
    for attempt in range(RETRIES):
        try:
            time.sleep(PAGE_PACING_S)
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200 and resp.text:
                try:
                    tables = pd.read_html(StringIO(resp.text), flavor="lxml")
                except ValueError:
                    return pd.DataFrame()  # no trades on this page
                for t in tables:
                    cols = {str(c).strip() for c in t.columns}
                    if {"Politician", "Traded Issuer", "Type"}.issubset(cols):
                        return t
                return pd.DataFrame()  # layout drifted — no matching table
            logger.debug("HTTP %s on page %d (attempt %d)",
                         resp.status_code, page, attempt + 1)
        except Exception as exc:
            logger.debug("HTTP error page %d (attempt %d): %s",
                         page, attempt + 1, exc)
        time.sleep(BACKOFF_S * (2 ** attempt) + random.uniform(0, 0.3))
    return None


def _extract_ticker(issuer: str) -> str | None:
    """Pull the ticker out of a 'Company NameTICK:US' string."""
    m = TICKER_RE.search(str(issuer or "").replace(" ", ""))
    return m.group(1) if m else None


def _parse_date(val: str) -> pd.Timestamp | None:
    """CapitolTrades 'Traded' column format: '20 Mar2026'."""
    s = str(val or "").strip()
    if not s:
        return None
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)   # 'Mar2026' → 'Mar 2026'
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FULL SCRAPE → DAILY PARQUET CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path() -> Path:
    return CACHE_DIR / f"congress_{date.today().isoformat()}.parquet"


def _scrape_full(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Paginate CapitolTrades until we run out of rows or exceed the lookback."""
    cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) \
             - pd.Timedelta(days=lookback_days)

    frames: list[pd.DataFrame] = []
    for pg in range(1, MAX_PAGES + 1):
        df = _fetch_page(pg)
        if df is None:
            logger.warning("page %d unreachable — stopping pagination", pg)
            break
        if df.empty:
            logger.info("page %d empty — end of feed", pg)
            break

        df = df.copy()
        df["ticker"]     = df["Traded Issuer"].apply(_extract_ticker)
        df["trade_date"] = df["Traded"].apply(_parse_date)
        df["trade_type"] = df["Type"].astype(str).str.lower().str.strip()
        df["politician"] = df["Politician"].astype(str)
        df = df.dropna(subset=["ticker", "trade_date"])
        df = df[df["trade_date"] >= cutoff]

        if df.empty:
            # Once pages start skewing older than the cutoff, stop early.
            logger.info("page %d past %dd cutoff — halting", pg, lookback_days)
            break

        frames.append(df[["ticker", "trade_date", "trade_type", "politician"]])
        if pg % 5 == 0:
            logger.info("  congress scraped %d pages (≈%d rows)", pg,
                        sum(len(f) for f in frames))

    if not frames:
        return pd.DataFrame(columns=["ticker", "trade_date", "trade_type", "politician"])
    full = pd.concat(frames, ignore_index=True)
    full["ticker"] = full["ticker"].str.upper()

    tmp = _cache_path().with_suffix(".parquet.tmp")
    try:
        full.to_parquet(tmp, compression="snappy")
        tmp.replace(_cache_path())
        logger.info("congress dump cached: %d rows → %s",
                    len(full), _cache_path().relative_to(ROOT))
    except Exception as exc:
        logger.debug("parquet write failed (%s) — in-memory only", exc)
    return full


def _load_dump(force: bool = False) -> pd.DataFrame:
    path = _cache_path()
    if not force and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("cache read failed (%s) — re-scraping", exc)
    return _scrape_full()


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _committee_match(politician: str, sector: str | None) -> bool:
    if not sector:
        return False
    keywords = SECTOR_COMMITTEE.get(sector, ())
    low = (politician or "").lower()
    return any(kw in low for kw in keywords)


def _aggregate(df: pd.DataFrame, tickers: list[str],
               sectors: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                       columns=FEATURE_COLS, dtype=int)
    if df is None or df.empty:
        return out

    df = df[df["ticker"].isin(tickers)]
    if df.empty:
        return out

    is_buy  = df["trade_type"].str.contains("buy|purchase", regex=True, na=False)
    is_sell = df["trade_type"].str.contains("sell|sale", regex=True, na=False) \
              & ~df["trade_type"].str.contains("partial", regex=True, na=False)

    buy_df  = df[is_buy]
    sell_df = df[is_sell]

    buy_counts  = buy_df .groupby("ticker").size()
    sell_counts = sell_df.groupby("ticker").size()

    out["congress_buy_count_180d"]     = buy_counts.reindex(tickers).fillna(0).astype(int)
    out["congress_net_direction_180d"] = (
        buy_counts.reindex(tickers).fillna(0)
        - sell_counts.reindex(tickers).fillna(0)
    ).astype(int)

    def _ticker_committee_flag(t: str) -> int:
        sub = buy_df[buy_df["ticker"] == t]
        if sub.empty:
            return 0
        sector = sectors.get(t)
        return int(sub["politician"].apply(
            lambda p: _committee_match(p, sector)
        ).any())

    out["congress_committee_flag"] = pd.Series(
        {t: _ticker_committee_flag(t) for t in tickers}, dtype=int
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_congress_signals(tickers: Iterable[str],
                           sectors: dict[str, str] | None = None,
                           force_refresh: bool = False) -> pd.DataFrame:
    """Return a (ticker × 3) congressional-trading feature DataFrame.

    `sectors` is an optional ticker → sector-string map for the
    committee-jurisdiction flag. Missing ⇒ committee flag always 0.
    """
    tickers = sorted({t for t in tickers if isinstance(t, str) and t})
    sectors = sectors or {}
    if not tickers:
        return pd.DataFrame(columns=FEATURE_COLS).rename_axis("ticker")

    try:
        dump = _load_dump(force=force_refresh)
    except Exception as exc:
        logger.warning("congress scrape crashed (%s) — returning zeros", exc)
        return pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                            columns=FEATURE_COLS, dtype=int)

    out = _aggregate(dump, tickers, sectors)
    logger.info("congress features built: %d tickers · %d active (buys>0)",
                len(tickers), int((out["congress_buy_count_180d"] > 0).sum()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Congressional-trading feature scraper (Phase 5).")
    ap.add_argument("tickers", nargs="*", help="Tickers (whitespace-separated).")
    ap.add_argument("--refresh", action="store_true",
                    help="Force re-scrape (bypass today's cache).")
    ns = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-8s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tickers = ns.tickers or ["AAPL", "NVDA", "MSFT", "LMT", "XOM"]
    df = fetch_congress_signals(tickers, force_refresh=ns.refresh)
    print(df.to_string())


if __name__ == "__main__":
    main()
