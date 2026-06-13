#!/usr/bin/env python3
"""
PHASE 1 — The Multivariate Harvester
======================================
Pulls 8 years of daily OHLCV for GC=F and SI=F, merged with
macro context (DXY, VIX).  Saves two clean CSVs ready for the Tri-Horizon
GPU Forge (metals_trainer.py).

Usage:
    python3 scripts/metals_harvester.py
"""

import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

METALS  = ["GC=F", "SI=F"]
MACROS  = ["DX-Y.NYB", "^VIX"]
PERIOD  = "8y"
INTERVAL = "1d"


def download(ticker: str) -> pd.DataFrame:
    print(f"  ↓  {ticker} …", end="", flush=True)
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"  {len(df)} rows")
    return df


def build_metal_frame(metal_df: pd.DataFrame,
                      dxy_df:   pd.DataFrame,
                      vix_df:   pd.DataFrame) -> pd.DataFrame:
    """Merge metal OHLCV with macro columns, forward-fill, drop NaNs."""
    df = metal_df[["Close", "Volume"]].copy()
    df.columns = ["Close_Price", "Volume"]

    df["DXY_Close"] = dxy_df["Close"].reindex(df.index, method="ffill")
    df["VIX_Close"] = vix_df["Close"].reindex(df.index, method="ffill")

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def main():
    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Multivariate Harvester")
    print("  Universe : GC=F  SI=F  |  Macro : DXY  VIX")
    print(f"  Timeframe: {PERIOD} daily")
    print("━" * 60)

    print("\n[1/3] Downloading macro indicators …")
    dxy = download("DX-Y.NYB")
    vix = download("^VIX")

    print("\n[2/3] Downloading metals …")
    metal_data = {ticker: download(ticker) for ticker in METALS}

    print("\n[3/3] Merging & cleaning …")
    for ticker in METALS:
        out_path = DATA / f"{ticker}_8yr_macro.csv"
        df = build_metal_frame(metal_data[ticker], dxy, vix)
        df.to_csv(out_path)
        print(f"  ✅  {ticker}  →  {out_path.name}  ({len(df)} rows, "
              f"{df.index[0].date()} → {df.index[-1].date()})")

    print("\n" + "━" * 60)
    print("  ✅  Harvest complete.  Mac is ready for the GPU Forge.")
    print("  Next step:  python3 scripts/metals_trainer.py")
    print("━" * 60 + "\n")


if __name__ == "__main__":
    main()
