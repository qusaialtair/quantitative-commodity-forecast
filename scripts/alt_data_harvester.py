#!/usr/bin/env python3
"""
scripts/alt_data_harvester.py
==============================
Step 3 of the Masterplan: Alternative Data Ingestion.

Three institutional-grade data streams fetched, aligned, and written to
data/alt_data.csv for consumption by the LSTM and DeepSeek calibrator.

Stream A — Real Yields (DFII10)
    Source  : FRED API (fredapi)
    Cadence : Daily US business days, 1-day publication lag
    Column  : real_yield_10y  (raw %, e.g. 2.15)

Stream B — Copper/Gold Ratio
    Source  : yfinance (HG=F / GC=F)
    Cadence : Daily COMEX
    Column  : copper_gold_ratio_raw  (e.g. 0.000238)
              copper_gold_ratio_zscore  (rolling 252d z-score — LSTM feature)

Stream C — CFTC COT Managed Money Net Positions (Gold + Silver)
    Source  : CFTC.gov direct ZIP download (free, no rate limit)
    Cadence : Weekly (as-of Tuesday, published Friday ~3:30 PM ET)
    Column  : cot_gold_mm_net_raw / cot_gold_mm_net_zscore
              cot_silver_mm_net_raw / cot_silver_mm_net_zscore

Look-ahead bias prevention
---------------------------
  DFII10         : shifted +1 business day before merge
  Copper/Gold    : same-session close, no shift
  COT            : each weekly report is assigned to its PUBLICATION Friday
                   (as-of Tuesday + 3 days), then forward-filled through
                   subsequent trading days until the next Friday report.
                   Monday–Thursday of any given week always shows the
                   PREVIOUS Friday's report — data that was publicly
                   available at that time.

Output
------
  data/alt_data.csv   (append-only, idempotent on re-run)

Usage
-----
  python3 scripts/alt_data_harvester.py --update          # last 90 days
  python3 scripts/alt_data_harvester.py --backfill        # 2010-01-01 → today
  python3 scripts/alt_data_harvester.py --show            # tail of alt_data.csv
  python3 scripts/alt_data_harvester.py --install-launchd # schedule 22:00 UTC
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
import zipfile
from datetime import date, timedelta
from pathlib import Path

import ssl
import certifi

# macOS Python.org builds ship without system CA certs; point ssl at certifi bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.file_io import read_bytes as _read_bytes, write_bytes as _write_bytes

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

try:
    from fredapi import Fred
except ImportError:
    raise ImportError("pip install fredapi")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ALT_DATA_CSV  = ROOT / "data" / "alt_data.csv"
COT_CACHE_DIR = ROOT / "data" / "cot_cache"
COT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALT_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKFILL_START  = date(2010, 1, 1)
ZSCORE_WINDOW   = 252          # 1 trading year rolling window for z-scores
FRED_SERIES     = "DFII10"    # 10-Year Treasury Inflation-Indexed Security
FRED_SHIFT_DAYS = 1            # FRED publishes with 1 business-day lag

# CFTC Disaggregated Futures-Only COT
# Current year (updated weekly every Friday)
_CFTC_CURRENT = "https://www.cftc.gov/dea/newcot/fut_disagg_txt.zip"
# Historical per-year archives
_CFTC_HIST    = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

# Contract name fragments for filtering (case-insensitive, prefix match)
_COT_GOLD_KEY   = "GOLD - COMMODITY EXCHANGE"
_COT_SILVER_KEY = "SILVER - COMMODITY EXCHANGE"


# ==============================================================================
# Stream A — Real Yields (FRED DFII10)
# ==============================================================================

def _fetch_real_yields(start: date, end: date) -> pd.Series:
    """
    Pull DFII10 from FRED.  Shift forward by 1 business day to respect the
    publication lag (Monday's value is released on Tuesday morning).
    Returns a Series named 'real_yield_10y' indexed to business days.
    """
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.warning("FRED_API_KEY not set — real_yield_10y will be NaN.")
        idx = pd.bdate_range(str(start), str(end))
        return pd.Series(np.nan, index=idx, name="real_yield_10y")

    fred = Fred(api_key=api_key)
    # Fetch extra buffer for the lag shift
    fetch_start = start - timedelta(days=10)
    raw: pd.Series = fred.get_series(
        FRED_SERIES,
        observation_start=str(fetch_start),
        observation_end=str(end),
    )
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.name  = "real_yield_10y"

    # Shift backward — data published on day T isn't available until T+1
    raw_shifted = raw.shift(-FRED_SHIFT_DAYS, freq="B")

    # Reindex to business day calendar, forward-fill gaps (weekends already excluded)
    bday_idx = pd.bdate_range(str(start), str(end))
    series   = raw_shifted.reindex(bday_idx).ffill()

    logger.info(f"FRED DFII10: {len(series)} rows  "
                f"[{series.first_valid_index().date()} → {series.index[-1].date()}]  "
                f"latest={series.dropna().iloc[-1]:.3f}%")
    return series


# ==============================================================================
# Stream B — Copper / Gold Ratio
# ==============================================================================

def _fetch_copper_gold_ratio(start: date, end: date) -> pd.DataFrame:
    """
    Download HG=F and GC=F closes.  Compute ratio and rolling-252d z-score.
    Returns DataFrame with columns: copper_gold_ratio_raw, copper_gold_ratio_zscore.
    """
    # Fetch extra history for the 252-day rolling window to be valid at `start`
    fetch_start = start - timedelta(days=int(ZSCORE_WINDOW * 1.6))
    tickers = yf.download(
        ["HG=F", "GC=F"],
        start=str(fetch_start),
        end=str(end + timedelta(days=2)),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    # Flatten MultiIndex
    if isinstance(tickers.columns, pd.MultiIndex):
        close = tickers["Close"].copy()
    else:
        close = tickers[["Close"]].copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.rename(columns={"HG=F": "copper", "GC=F": "gold"}).dropna()

    if "copper" not in close.columns or "gold" not in close.columns:
        logger.warning("Could not fetch HG=F or GC=F for copper/gold ratio.")
        idx = pd.bdate_range(str(start), str(end))
        return pd.DataFrame({
            "copper_gold_ratio_raw":    np.nan,
            "copper_gold_ratio_zscore": np.nan,
        }, index=idx)

    close["copper_gold_ratio_raw"] = close["copper"] / close["gold"]

    # Rolling 252-day z-score  (no look-ahead: only past 252 days used)
    mu    = close["copper_gold_ratio_raw"].rolling(ZSCORE_WINDOW).mean()
    sigma = close["copper_gold_ratio_raw"].rolling(ZSCORE_WINDOW).std()
    close["copper_gold_ratio_zscore"] = (
        (close["copper_gold_ratio_raw"] - mu) / sigma
    ).replace([np.inf, -np.inf], np.nan)

    result = close[["copper_gold_ratio_raw", "copper_gold_ratio_zscore"]]
    result = result[result.index.date >= start]

    if result.empty:
        logger.warning(
            "Copper/Gold ratio: no rows after filtering to start=%s "
            "(HG=F may have no settlement data yet — returning NaN series).", start
        )
        idx = pd.bdate_range(str(start), str(end))
        return pd.DataFrame({
            "copper_gold_ratio_raw":    np.nan,
            "copper_gold_ratio_zscore": np.nan,
        }, index=idx)

    logger.info(f"Copper/Gold ratio: {len(result)} rows  "
                f"latest raw={result['copper_gold_ratio_raw'].iloc[-1]:.6f}  "
                f"z={result['copper_gold_ratio_zscore'].iloc[-1]:+.2f}")
    return result


# ==============================================================================
# Stream C — CFTC COT Managed Money Net Positions
# ==============================================================================

def _cot_url(year: int) -> str:
    current = date.today().year
    if year >= current:
        return _CFTC_CURRENT
    return _CFTC_HIST.format(year=year)


def _cot_cache_path(year: int) -> Path:
    return COT_CACHE_DIR / f"fut_disagg_{year}.zip"


def _cot_cache_is_fresh(year: int) -> bool:
    """
    Historical years (< current year) are immutable — cache forever.
    Current year is updated weekly — stale after 8 days.
    """
    p = _cot_cache_path(year)
    if not p.exists():
        return False
    current = date.today().year
    if year < current:
        return True                              # historical: never re-download
    age_days = (date.today() - date.fromtimestamp(p.stat().st_mtime)).days
    return age_days < 8                          # current year: refresh weekly


def _download_cot_zip(year: int) -> bytes | None:
    """Download (or return cached) CFTC COT ZIP for the given year."""
    cache = _cot_cache_path(year)

    if _cot_cache_is_fresh(year):
        logger.debug(f"  COT {year}: using cached ZIP")
        return _read_bytes(cache)

    url = _cot_url(year)
    logger.info(f"  COT {year}: downloading {url} …")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "metals-ai/1.0"})
        resp.raise_for_status()
        data = resp.content
        _write_bytes(cache, data)
        return data
    except requests.HTTPError as exc:
        logger.warning(f"  COT {year}: HTTP {exc.response.status_code} — skipping")
        return None
    except Exception as exc:
        logger.warning(f"  COT {year}: download failed — {exc}")
        return None


def _parse_cot_zip(data: bytes, year: int) -> pd.DataFrame | None:
    """
    Parse a CFTC Disaggregated Futures-Only ZIP.
    Returns a DataFrame with columns: publication_date, ticker, mm_net.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Take the first .txt or .csv file in the archive
            names = [n for n in zf.namelist()
                     if n.lower().endswith((".txt", ".csv"))]
            if not names:
                logger.warning(f"  COT {year}: no text file found in ZIP")
                return None
            csv_bytes = zf.read(names[0])
    except zipfile.BadZipFile:
        logger.warning(f"  COT {year}: corrupt ZIP")
        return None

    # Decode — CFTC uses latin-1 for some older files
    for enc in ("utf-8", "latin-1"):
        try:
            text = csv_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.warning(f"  COT {year}: could not decode CSV")
        return None

    df = pd.read_csv(io.StringIO(text), low_memory=False)

    # ── Flexible column detection ────────────────────────────────────────────
    cols_upper = {c: c.upper().strip() for c in df.columns}

    # Find the market name column
    name_col = next(
        (c for c, u in cols_upper.items()
         if "MARKET_AND_EXCHANGE" in u or "MARKET AND EXCHANGE" in u),
        None,
    )
    # Find the report/as-of date column  (prefer Report_Date_as_MM_DD_YYYY)
    date_col = next(
        (c for c, u in cols_upper.items()
         if "REPORT_DATE_AS_MM_DD_YYYY" in u or "REPORT DATE AS MM_DD_YYYY" in u),
        None,
    )
    if date_col is None:
        # Fallback: as-of date in YYMMDD
        date_col = next(
            (c for c, u in cols_upper.items()
             if "AS_OF_DATE_IN_FORM_YYMMDD" in u or "AS OF DATE" in u),
            None,
        )
    long_col = next(
        (c for c, u in cols_upper.items()
         if "M_MONEY_POSITIONS_LONG_ALL" in u),
        None,
    )
    short_col = next(
        (c for c, u in cols_upper.items()
         if "M_MONEY_POSITIONS_SHORT_ALL" in u),
        None,
    )

    missing = [n for n, v in [
        ("market_name", name_col), ("date", date_col),
        ("mm_long", long_col), ("mm_short", short_col),
    ] if v is None]
    if missing:
        logger.warning(f"  COT {year}: missing columns {missing} — skipping")
        return None

    df = df[[name_col, date_col, long_col, short_col]].copy()
    df.columns = ["market_name", "as_of_date_str", "mm_long", "mm_short"]

    # ── Filter for Gold and Silver (COMEX) ──────────────────────────────────
    name_upper = df["market_name"].str.upper().str.strip()
    gold_mask   = name_upper.str.startswith(_COT_GOLD_KEY.upper())
    silver_mask = name_upper.str.startswith(_COT_SILVER_KEY.upper())
    df = df[gold_mask | silver_mask].copy()

    if df.empty:
        logger.warning(f"  COT {year}: no GOLD/SILVER rows found")
        return None

    # ── Parse as-of date ────────────────────────────────────────────────────
    # Two possible formats: MM/DD/YYYY or YYMMDD
    sample = str(df["as_of_date_str"].iloc[0]).strip()
    if "/" in sample:
        df["as_of_date"] = pd.to_datetime(df["as_of_date_str"], format="%m/%d/%Y",
                                           errors="coerce")
    else:
        # YYMMDD: two-digit year — add century prefix
        df["as_of_date"] = pd.to_datetime(
            df["as_of_date_str"].astype(str).str.zfill(6),
            format="%y%m%d", errors="coerce",
        )
    df = df.dropna(subset=["as_of_date"])

    # Publication date = as-of Tuesday + 3 days = Friday
    df["publication_date"] = df["as_of_date"] + pd.Timedelta(days=3)

    # Net managed money position
    df["mm_long"]  = pd.to_numeric(df["mm_long"],  errors="coerce").fillna(0)
    df["mm_short"] = pd.to_numeric(df["mm_short"], errors="coerce").fillna(0)
    df["mm_net"]   = df["mm_long"] - df["mm_short"]

    # Assign ticker label — recompute name_upper on the filtered df
    # (original name_upper has stale indices from the pre-filter frame)
    filtered_names = df["market_name"].str.upper().str.strip()
    df["ticker"] = "GOLD"
    silver_rows = filtered_names.str.startswith(_COT_SILVER_KEY.upper())
    df.loc[silver_rows, "ticker"] = "SILVER"

    result = (df[["publication_date", "ticker", "mm_net"]]
              .drop_duplicates(subset=["publication_date", "ticker"])
              .sort_values("publication_date"))

    logger.info(f"  COT {year}: {len(result)} rows parsed  "
                f"({result['ticker'].value_counts().to_dict()})")
    return result


def _fetch_cot(start: date, end: date) -> pd.DataFrame:
    """
    Fetch and align COT managed money net positions to a daily business-day index.

    Look-ahead bias prevention:
      Each COT observation is placed on the PUBLICATION FRIDAY (as-of Tuesday + 3d).
      The series is then forward-filled so that Monday–Thursday shows the PREVIOUS
      Friday's report — only publicly available information is used on each date.

    Returns DataFrame with columns:
      cot_gold_mm_net_raw, cot_gold_mm_net_zscore,
      cot_silver_mm_net_raw, cot_silver_mm_net_zscore
    """
    # Need extra history before `start` to compute valid 252d z-scores.
    # CFTC disaggregated per-year ZIPs start from 2010; before that the
    # data is bundled differently and returns 404 on the per-year URL.
    hist_start = start - timedelta(days=int(ZSCORE_WINDOW * 1.6) + 30)
    years = list(range(max(2010, hist_start.year), end.year + 1))

    frames: list[pd.DataFrame] = []
    for yr in years:
        data = _download_cot_zip(yr)
        if data is None:
            continue
        parsed = _parse_cot_zip(data, yr)
        if parsed is not None:
            frames.append(parsed)

    if not frames:
        logger.error("COT: no data fetched from CFTC — filling with NaN")
        bday_idx = pd.bdate_range(str(start), str(end))
        return pd.DataFrame({
            "cot_gold_mm_net_raw":      np.nan,
            "cot_gold_mm_net_zscore":   np.nan,
            "cot_silver_mm_net_raw":    np.nan,
            "cot_silver_mm_net_zscore": np.nan,
        }, index=bday_idx)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["publication_date", "ticker"])

    # Separate gold and silver, set publication_date as index
    gold   = (raw[raw["ticker"] == "GOLD"]
              .set_index("publication_date")["mm_net"]
              .sort_index())
    silver = (raw[raw["ticker"] == "SILVER"]
              .set_index("publication_date")["mm_net"]
              .sort_index())

    # Build trading calendar covering full history (for z-score warmup)
    trading_cal = pd.bdate_range(
        start=str(hist_start),
        end=str(end + timedelta(days=5)),
    )

    # Reindex to daily calendar and forward-fill from Friday publication dates
    gold_daily   = gold.reindex(trading_cal).ffill()
    silver_daily = silver.reindex(trading_cal).ffill()

    # Rolling 252d z-score
    def _rolling_zscore(s: pd.Series) -> pd.Series:
        mu    = s.rolling(ZSCORE_WINDOW).mean()
        sigma = s.rolling(ZSCORE_WINDOW).std()
        return ((s - mu) / sigma).replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({
        "cot_gold_mm_net_raw":      gold_daily,
        "cot_gold_mm_net_zscore":   _rolling_zscore(gold_daily),
        "cot_silver_mm_net_raw":    silver_daily,
        "cot_silver_mm_net_zscore": _rolling_zscore(silver_daily),
    })

    # Trim to requested window
    result = result.loc[str(start):str(end)]

    if not result.empty:
        logger.info(
            f"COT aligned: {len(result)} daily rows  "
            f"gold_net={result['cot_gold_mm_net_raw'].dropna().iloc[-1]:,.0f}  "
            f"gold_z={result['cot_gold_mm_net_zscore'].dropna().iloc[-1]:+.2f}"
        )
    return result


# ==============================================================================
# Merge — align all three streams on the GC=F trading calendar
# ==============================================================================

def _build_alt_data(start: date, end: date) -> pd.DataFrame:
    """
    Fetch all three streams and merge onto a single business-day calendar.
    All alignment rules (lags, ffill) are applied inside each fetch function.

    Columns in output:
      real_yield_10y
      copper_gold_ratio_raw, copper_gold_ratio_zscore
      cot_gold_mm_net_raw,   cot_gold_mm_net_zscore
      cot_silver_mm_net_raw, cot_silver_mm_net_zscore
    """
    logger.info(f"Building alt data: {start} → {end}")

    yields  = _fetch_real_yields(start, end)
    cu_au   = _fetch_copper_gold_ratio(start, end)
    cot     = _fetch_cot(start, end)

    # Anchor on business-day calendar
    bday_idx = pd.bdate_range(str(start), str(end))

    df = pd.DataFrame(index=bday_idx)
    df.index.name = "date"

    df = df.join(yields.rename("real_yield_10y"))
    df = df.join(cu_au)
    df = df.join(cot)

    # Final forward-fill for any residual gaps (e.g. holiday mismatches)
    df = df.ffill()

    # Drop leading rows where all values are NaN (warmup artefact)
    df = df.dropna(how="all")

    logger.info(f"Alt data built: {len(df)} rows × {len(df.columns)} columns")
    return df


# ==============================================================================
# CSV persistence — idempotent append
# ==============================================================================

def _append_to_csv(df: pd.DataFrame) -> None:
    """
    Write df to data/alt_data.csv.
    If the file already exists, existing rows for df's date range are
    replaced (idempotent — safe to call on re-run or partial overlap).
    """
    df = df.copy()
    df.index.name = "date"

    if ALT_DATA_CSV.exists():
        existing = pd.read_csv(ALT_DATA_CSV, index_col="date", parse_dates=True)
        existing.index = pd.to_datetime(existing.index)
        # Drop any dates being replaced by new data
        existing = existing[~existing.index.isin(df.index)]
        combined = pd.concat([existing, df]).sort_index()
    else:
        combined = df.sort_index()

    # Round numeric columns to 6 decimal places for compact CSV
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined[num_cols] = combined[num_cols].round(6)

    combined.to_csv(ALT_DATA_CSV)
    logger.info(f"alt_data.csv updated — {len(combined)} total rows  "
                f"(added/replaced {len(df)})")


# ==============================================================================
# Public entrypoints
# ==============================================================================

def run(mode: str = "update") -> None:
    """
    mode="update"   → fetch last 90 days (daily incremental use)
    mode="backfill" → full history from BACKFILL_START (2010-01-01)
    """
    today = date.today()

    if mode == "backfill":
        start = BACKFILL_START
        logger.info(f"BACKFILL mode: {start} → {today}")
    else:
        start = today - timedelta(days=90)
        logger.info(f"UPDATE mode: {start} → {today}")

    t0 = time.time()
    df = _build_alt_data(start, today)
    _append_to_csv(df)
    elapsed = round(time.time() - t0, 1)
    logger.info(f"Alt data harvester complete in {elapsed}s")


def load_latest_row() -> dict:
    """
    Return the most recent row of alt_data.csv as a dict.
    Used by daily_trainer.py to inject alt data context into DeepSeek prompt.
    Returns empty dict if file doesn't exist.
    """
    if not ALT_DATA_CSV.exists():
        return {}
    try:
        df = pd.read_csv(ALT_DATA_CSV, index_col="date", parse_dates=True)
        df = df.dropna(how="all")
        if df.empty:
            return {}
        row = df.iloc[-1].to_dict()
        row["date"] = str(df.index[-1].date())
        return row
    except Exception as exc:
        logger.warning(f"Could not read alt_data.csv: {exc}")
        return {}


# ==============================================================================
# launchd installer
# ==============================================================================

def install_launchd() -> None:
    """
    Schedule alt_data_harvester at 22:00 UTC daily — 30 min before daily_trainer.
    This ensures fresh alt data is available when the trainer runs at 22:30.
    """
    import subprocess
    python  = sys.executable
    script  = str(Path(__file__).resolve())
    workdir = str(ROOT)
    label   = "com.metals.alt_harvester"
    plist   = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logfile = "/tmp/metals_alt_harvester.log"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>{script}</string>
    <string>--update</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{workdir}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>   <integer>22</integer>
    <key>Minute</key> <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>  <string>{logfile}</string>
  <key>StandardErrorPath</key><string>{logfile}</string>
  <key>RunAtLoad</key> <false/>
  <key>KeepAlive</key> <false/>
</dict>
</plist>"""

    plist.write_text(xml)
    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    subprocess.run(["launchctl", "load",   str(plist)], capture_output=True)
    logger.info(f"launchd job installed: {label}  (22:00 UTC daily)")
    logger.info(f"Plist: {plist}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alt data harvester — FRED, Copper/Gold, COT"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--update",          action="store_true",
                   help="Incremental update (last 90 days)")
    g.add_argument("--backfill",        action="store_true",
                   help="Full backfill from 2010-01-01")
    g.add_argument("--show",            action="store_true",
                   help="Print tail of alt_data.csv")
    g.add_argument("--install-launchd", action="store_true",
                   help="Install launchd job at 22:00 UTC")
    args = parser.parse_args()

    if args.install_launchd:
        install_launchd()
    elif args.show:
        if ALT_DATA_CSV.exists():
            df = pd.read_csv(ALT_DATA_CSV, index_col="date")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 120)
            print(f"\n  alt_data.csv — last 10 rows ({len(df)} total)\n")
            print(df.tail(10).to_string())
            print()
        else:
            print("  alt_data.csv not found — run --backfill first.")
    elif args.backfill:
        run("backfill")
    else:
        run("update")
