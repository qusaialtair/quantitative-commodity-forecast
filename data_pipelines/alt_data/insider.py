#!/usr/bin/env python3
"""
data_pipelines/alt_data/insider.py
==================================
Phase 5 — Insider-trading alternative-data feed.

Pulls Form-4 filings (C-suite / 10%-owner transactions) from OpenInsider
and aggregates them into three per-ticker features for the ranker:

    insider_buy_count_90d    int    # distinct insider BUYS in last 90d
    insider_net_usd_90d      float  # net $ flow (buys − sells), 90d, $M
    insider_cluster_flag     int    # 1 iff ≥2 officers bought on
                                    # distinct trade-dates within last 30d

Why OpenInsider?
----------------
OpenInsider normalises SEC EDGAR Form-4 XML into a clean HTML table.
It is:
  • free, no auth required
  • maintained with <24h latency on filings
  • stable column schema (verified manually 2026-04)
  • polite rate-limit tolerance (1 req/s is fine)

Failure posture
---------------
This module is **signal-only** — it NEVER raises into the caller. If
OpenInsider is down, returns column-present DataFrame of all zeros.
The Sharia gate in equity_features.py never reads these columns;
missing alt-data cannot eject a ticker from the investable universe.

Caching
-------
One parquet per calendar day under data/alt_data/:
    insider_YYYY-MM-DD.parquet
If today's cache exists, fetch_insider_signals() returns it
immediately — no network traffic, idempotent across re-runs.

CLI
---
    python3 -m data_pipelines.alt_data.insider AAPL MSFT NVDA
    python3 -m data_pipelines.alt_data.insider --file tickers.txt
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logger = logging.getLogger("INSIDER")

# ── OpenInsider endpoint contract ───────────────────────────────────────────
#  Columns (left-to-right) as served today:
#    X / Filing Date / Trade Date / Ticker / Insider Name / Title /
#    Trade Type / Price / Qty / Owned / ΔOwn / Value / [more …]
#
#  Trade Type codes we care about:
#    P - Purchase          (counts as BUY)
#    S - Sale              (counts as SELL; excludes option exercises)
#    S - Sale+OE           (option-exercise-and-sell — treat as NEUTRAL,
#                           not a directional signal)
# ─────────────────────────────────────────────────────────────────────────────
OPENINSIDER_URL = (
    # Filing-date window via `fd=<days-ago>` preset — the custom date-range
    # param on this screener is flaky; the days-ago preset is stable.
    # xp=1/xs=1 → show purchases AND sales; cnt=100 → page cap.
    "http://openinsider.com/screener?s={ticker}&fd={days}&xp=1&xs=1&cnt=100"
)
HTTP_HEADERS   = {
    "User-Agent":      "GoldTradingAI/5.0 (sharia-equity-fund; research)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml",
}
LOOKBACK_DAYS  = 90
CLUSTER_WINDOW = 30
CLUSTER_MIN_N  = 2
HTTP_TIMEOUT   = 12
RETRIES        = 3
BACKOFF_S      = 0.8
WORKERS        = 2       # be polite — two in flight at most
REQ_PACING_S   = 0.5     # floor between requests per worker

FEATURE_COLS = [
    "insider_buy_count_90d",
    "insider_net_usd_90d",
    "insider_cluster_flag",
]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE HTTP + PARSE
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_raw_html(ticker: str, days: int) -> str | None:
    url = OPENINSIDER_URL.format(ticker=ticker, days=int(days))
    for attempt in range(RETRIES):
        try:
            time.sleep(REQ_PACING_S)
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200 and resp.text:
                return resp.text
            logger.debug("openinsider HTTP %s for %s (attempt %d)",
                         resp.status_code, ticker, attempt + 1)
        except Exception as exc:
            logger.debug("openinsider error %s for %s (attempt %d): %s",
                         ticker, ticker, attempt + 1, exc)
        # Exponential backoff with jitter.
        time.sleep(BACKOFF_S * (2 ** attempt) + random.uniform(0, 0.3))
    return None


def _parse_table(html: str) -> pd.DataFrame:
    """Extract the insider-transaction table from an OpenInsider HTML page.

    Returns an empty DataFrame (not None) if the page had no table — that
    is the common case for mid-cap names with zero recent Form-4 activity.
    """
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        # "No tables found" — a legitimately empty result, not an error.
        return pd.DataFrame()
    except Exception as exc:
        logger.debug("pandas.read_html parse failure: %s", exc)
        return pd.DataFrame()

    # The page serves one large table; filter to the one with the columns
    # we recognise (guards against layout drift). OpenInsider uses U+00A0
    # (non-breaking space) in column headers — normalise before matching.
    for t in tables:
        cols = {str(c).replace("\xa0", " ").strip() for c in t.columns}
        if {"Ticker", "Trade Type", "Value"}.issubset(cols):
            t = t.rename(columns=lambda c: str(c).replace("\xa0", " ").strip())
            return t
    return pd.DataFrame()


def _signals_from_frame(df: pd.DataFrame, as_of: datetime) -> dict:
    """Aggregate one ticker's raw Form-4 rows → three numeric features."""
    blank = {
        "insider_buy_count_90d":  0,
        "insider_net_usd_90d":    0.0,
        "insider_cluster_flag":   0,
    }
    if df is None or df.empty:
        return blank

    # Normalise column names (OpenInsider uses mixed case + emoji Δ).
    rename_map = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=rename_map)

    trade_col = next((c for c in df.columns if c.lower() == "trade date"), None)
    type_col  = next((c for c in df.columns if "trade type" in c.lower()), None)
    val_col   = next((c for c in df.columns if c.lower() == "value"), None)
    name_col  = next((c for c in df.columns if "insider" in c.lower() and "name" in c.lower()), None)
    if not all([trade_col, type_col, val_col]):
        return blank

    df = df.copy()
    df[trade_col] = pd.to_datetime(df[trade_col], errors="coerce")
    # Value is "$123,456" — strip non-numerics.
    df[val_col] = (df[val_col].astype(str)
                   .str.replace(r"[\$,]", "", regex=True)
                   .str.replace(r"[^\d\-\.]", "", regex=True))
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[trade_col, val_col])

    cutoff_90 = as_of - pd.Timedelta(days=LOOKBACK_DAYS)
    cutoff_30 = as_of - pd.Timedelta(days=CLUSTER_WINDOW)
    df90 = df[df[trade_col] >= cutoff_90]
    if df90.empty:
        return blank

    tt = df90[type_col].astype(str).str.upper()
    is_buy  = tt.str.startswith("P")                 # Purchase
    is_sell = tt.str.startswith("S") & ~tt.str.contains("OE")  # exclude option-exercise-and-sell

    buys_90 = int(is_buy.sum())
    # OpenInsider encodes buys as POSITIVE Value and sells as NEGATIVE
    # Value (e.g., "-$167,935" for a sale). Summing directly yields the
    # signed net dollar flow — no subtraction needed.
    net_usd = float(df90.loc[is_buy | is_sell, val_col].sum())

    # Cluster-buy: ≥2 DISTINCT insiders buying on ≥2 DISTINCT trade dates
    # within the last 30 calendar days. The "distinct" guard prevents a
    # single insider's split fills from inflating the flag.
    cluster_flag = 0
    if name_col is not None:
        recent = df90[(df90[trade_col] >= cutoff_30) & is_buy]
        if not recent.empty:
            n_insiders = recent[name_col].nunique()
            n_days     = recent[trade_col].dt.date.nunique()
            if n_insiders >= CLUSTER_MIN_N and n_days >= CLUSTER_MIN_N:
                cluster_flag = 1

    return {
        "insider_buy_count_90d":  buys_90,
        "insider_net_usd_90d":    net_usd / 1e6,   # scale to $M
        "insider_cluster_flag":   cluster_flag,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_one(ticker: str, days: int, as_of: datetime) -> tuple[str, dict]:
    html = _fetch_raw_html(ticker, days)
    if not html:
        return ticker, {"insider_buy_count_90d": 0,
                        "insider_net_usd_90d":   0.0,
                        "insider_cluster_flag":  0}
    return ticker, _signals_from_frame(_parse_table(html), as_of)


def fetch_insider_signals(tickers: Iterable[str],
                          lookback_days: int = LOOKBACK_DAYS,
                          use_cache: bool = True) -> pd.DataFrame:
    """Return a (ticker × 3) feature DataFrame.

    On any catastrophic failure (network, parse, etc.) returns a frame
    with the correct columns filled with zeros — the pipeline never dies.
    """
    tickers = sorted({t for t in tickers if isinstance(t, str) and t})
    if not tickers:
        return pd.DataFrame(columns=FEATURE_COLS).rename_axis("ticker")

    stamp = date.today().isoformat()
    cache = CACHE_DIR / f"insider_{stamp}.parquet"
    if use_cache and cache.exists():
        try:
            cached = pd.read_parquet(cache)
            missing = [t for t in tickers if t not in cached.index]
            if not missing:
                logger.info("insider cache hit: %d tickers from %s",
                            len(tickers), cache.name)
                return cached.loc[tickers, FEATURE_COLS]
            logger.info("insider cache partial: %d new tickers need fetch",
                        len(missing))
        except Exception as exc:
            logger.warning("insider cache read failed (%s) — refetching", exc)
            cached = None
            missing = tickers
    else:
        cached = None
        missing = tickers

    # tz-naive: pandas parses OpenInsider trade dates without tz, and
    # `Timestamp.naive vs Timestamp.tz-aware` raises TypeError.
    as_of = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    logger.info("Fetching insider signals for %d tickers (OpenInsider, %dd lookback)…",
                len(missing), lookback_days)

    rows: dict[str, dict] = {}
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(_fetch_one, t, lookback_days, as_of): t
                    for t in missing}
            done = 0
            for fut in as_completed(futs):
                t, feat = fut.result()
                rows[t] = feat
                done += 1
                if done % 25 == 0:
                    logger.info("  insider %d/%d", done, len(missing))
    except Exception as exc:
        logger.warning("insider pool crashed: %s — returning zeros", exc)

    new_df = pd.DataFrame.from_dict(rows, orient="index")
    if new_df.empty:
        new_df = pd.DataFrame(columns=FEATURE_COLS,
                              index=pd.Index(missing, name="ticker"))
    new_df.index.name = "ticker"
    # Ensure all columns present
    for c in FEATURE_COLS:
        if c not in new_df.columns:
            new_df[c] = 0
    new_df = new_df[FEATURE_COLS].fillna(0)

    if cached is not None and not cached.empty:
        full = pd.concat([cached, new_df])
        full = full[~full.index.duplicated(keep="last")]
    else:
        full = new_df

    try:
        tmp = cache.with_suffix(".parquet.tmp")
        full.to_parquet(tmp, compression="snappy")
        tmp.replace(cache)
    except Exception as exc:
        logger.debug("insider cache write failed: %s", exc)

    return full.loc[[t for t in tickers if t in full.index], FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Insider-trading feature scraper (Phase 5).")
    ap.add_argument("tickers", nargs="*", help="Tickers (whitespace-separated).")
    ap.add_argument("--file", help="Path to a file with one ticker per line.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-fetch (bypass today's cache).")
    ns = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-8s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tickers = list(ns.tickers)
    if ns.file:
        tickers += [line.strip() for line in Path(ns.file).read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    if not tickers:
        ap.error("no tickers given (positional args or --file required)")

    df = fetch_insider_signals(tickers, use_cache=not ns.no_cache)
    print(df.to_string())


if __name__ == "__main__":
    main()
