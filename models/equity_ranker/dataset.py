#!/usr/bin/env python3
"""
models/equity_ranker/dataset.py
===============================
Historical snapshot builder for the equity ranker.

For each snapshot date t in a grid, we:
  1. Slice the bulk OHLCV panel to rows ≤ t
  2. Reuse Stage 1's vectorised feature engine (compute_price_features +
     compute_market_model) to emit one feature row per ticker, as-of t
  3. Compute the forward label from t → t + horizon:
        • forward_return          — raw % return
        • label_beats_median      — 1 if ticker beats cross-sectional median

Training on PRICE features only (yfinance exposes no reliable PIT
fundamentals; using current fundamentals in historical labels would
inject heavy look-ahead bias). Fundamentals are joined at INFERENCE
time for screening and UI display.

Also provides a purged walk-forward CV splitter. Standard TimeSeriesSplit
leaks because labels span `horizon` days, so we purge training samples
whose as_of falls inside [test_min − embargo, test_max + embargo].
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_pipelines.equity_features import (  # noqa: E402
    MARKET_PROXY,
    MIN_HISTORY,
    compute_market_model,
    compute_price_features,
)

logger = logging.getLogger("RNKDATA")

# Minimum rows we need before a snapshot is usable. compute_price_features
# references .iloc[-252:] for 52w levels, so 253 is the floor.
_MIN_ROWS = 260


# ══════════════════════════════════════════════════════════════════════════════
# OHLCV download with explicit date range
# ══════════════════════════════════════════════════════════════════════════════

def download_range(universe: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    tickers = sorted(set(universe) | {MARKET_PROXY})
    logger.info("Downloading OHLCV %s → %s for %d tickers…", start, end, len(tickers))
    t0 = time.time()

    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, threads=True, group_by="column",
    )
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected single-level columns from yfinance")

    frames: dict[str, pd.DataFrame] = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        frames[field.lower()] = raw[field].dropna(how="all").copy()

    valid_cols = frames["close"].columns[frames["close"].notna().sum() >= MIN_HISTORY]
    for k in frames:
        frames[k] = frames[k][valid_cols]

    logger.info("Panel: %d rows × %d tickers  (%.1fs)",
                frames["close"].shape[0], frames["close"].shape[1], time.time() - t0)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot generator
# ══════════════════════════════════════════════════════════════════════════════

def _slice_as_of(panel: dict[str, pd.DataFrame], end_i: int) -> dict[str, pd.DataFrame]:
    return {k: v.iloc[: end_i + 1] for k, v in panel.items()}


def build_snapshots(
    universe: list[str],
    start: str,
    end: str,
    step_days: int = 5,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Returns long-format DataFrame with columns:
        as_of, ticker, <features…>, forward_return, label_beats_median
    Each row = one (date, ticker) observation.
    """
    panel = download_range(universe, start, end)
    close = panel["close"]
    idx = close.index

    first_i = _MIN_ROWS
    last_i  = len(idx) - horizon
    if last_i <= first_i:
        raise RuntimeError(
            f"Panel too short: need ≥{_MIN_ROWS + horizon} rows, have {len(idx)}"
        )

    snapshot_idxs = list(range(first_i, last_i, step_days))
    logger.info("Snapshots: %d dates from %s to %s (step=%dd, horizon=%dd)",
                len(snapshot_idxs), idx[first_i].date(), idx[last_i - 1].date(),
                step_days, horizon)

    rows: list[pd.DataFrame] = []
    t0 = time.time()
    for n, i in enumerate(snapshot_idxs, 1):
        as_of = idx[i]
        sliced = _slice_as_of(panel, i)

        feats   = compute_price_features(sliced)
        marketf = compute_market_model(sliced["close"])
        df = feats.join(marketf, how="left").drop(index=MARKET_PROXY, errors="ignore")

        p_now = close.iloc[i]
        p_fwd = close.iloc[i + horizon]
        df["forward_return"] = (p_fwd / p_now) - 1.0
        df = df.dropna(subset=["forward_return"])
        if df.empty:
            continue

        # Cross-sectional label — robust to market-wide drift
        median = df["forward_return"].median()
        df["label_beats_median"] = (df["forward_return"] > median).astype(int)
        df["as_of"] = as_of

        rows.append(df.reset_index())

        if n % 10 == 0 or n == len(snapshot_idxs):
            logger.info("  snapshot %d/%d  (%s)  %.0fs",
                        n, len(snapshot_idxs), as_of.date(), time.time() - t0)

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.set_index(["as_of", "ticker"]).sort_index()
    logger.info("Dataset built: %d rows × %d cols", *dataset.shape)
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# Purged CV  —  Phase-6 overhaul (López de Prado, AFML chapter 7)
# ══════════════════════════════════════════════════════════════════════════════
#
# Two splitters live here:
#
#   (A) purged_splits()   walk-forward. Each fold's TRAIN set is strictly
#                          before its VAL set. Kept for legacy paths + as the
#                          conservative evaluator for the final production
#                          model (no peeking at the future, period).
#
#   (B) PurgedKFold       k-fold where the VAL block sits in the middle of
#                          the calendar and the TRAIN block wraps around it
#                          on BOTH sides. Used as the INNER cross-validator
#                          during Optuna tuning + SHAP feature selection, so
#                          every snapshot date gets held out exactly once.
#                          Purge + embargo are applied on BOTH boundaries
#                          of the val block to eliminate label leakage from
#                          the overlapping forward-return windows.
#
# Both splitters operate on UNIQUE snapshot dates, never on row indices, so
# the purge is done in calendar space — each snapshot t carries a forward
# return that covers [t, t+horizon], and the embargo guarantees that any
# train sample whose label overlaps a val sample is dropped.
# ══════════════════════════════════════════════════════════════════════════════


def purged_splits(
    times: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 10,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward split with embargo. `times` is a pandas Series of snapshot
    timestamps aligned with the feature matrix rows (positional). Yields
    (train_positions, val_positions).

    Use this for the FINAL OOS evaluation of the production model — the
    train set strictly precedes the val set, so there is no chance of
    future data influencing a past prediction.
    """
    times = pd.Series(pd.to_datetime(times.values))
    unique_times = np.array(sorted(times.unique()))
    n = len(unique_times)
    fold = n // (n_splits + 1)
    if fold < 1:
        raise ValueError("Not enough unique snapshot dates for requested splits")

    embargo = pd.Timedelta(days=embargo_days)
    for k in range(n_splits):
        train_end   = fold * (k + 1)
        val_start   = train_end
        val_end     = min(train_end + fold, n)
        train_times = unique_times[:train_end]
        val_times   = unique_times[val_start:val_end]
        if len(val_times) == 0:
            continue

        val_min = pd.Timestamp(val_times[0])
        keep_mask = pd.Series(train_times) <= (val_min - embargo)
        train_times = train_times[keep_mask.values]

        train_idx = np.where(times.isin(train_times))[0]
        val_idx   = np.where(times.isin(val_times))[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


class PurgedKFold:
    """Purged K-Fold with embargo (López de Prado, AFML §7.4).

    Unlike walk-forward, each calendar block takes a turn as the val fold,
    with the train set drawn from BOTH sides of the val block. To eliminate
    leakage from overlapping label windows, we:

      1. PURGE train dates within `horizon_days` of the val block boundaries
         (a train label ending inside [val_min, val_max] would leak).
      2. EMBARGO an additional `embargo_days` AFTER the val block — forward
         label windows from the embargo zone could carry information that
         was partially observable during val.

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be ≥2.
    embargo_days : int
        Calendar-day buffer after each val block. Defaults to the 10d
        forward-return horizon used in dataset construction.
    horizon_days : int
        Label horizon. Train dates with labels ending inside the val block
        are PURGED. Defaults to 10.

    Yields
    ------
    (train_positions, val_positions) : tuple of np.ndarray
        Positional indices into the supplied `times` Series.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 10,
                 horizon_days: int = 10):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo_days < 0 or horizon_days < 0:
            raise ValueError("embargo_days and horizon_days must be >= 0")
        self.n_splits      = int(n_splits)
        self.embargo_days  = int(embargo_days)
        self.horizon_days  = int(horizon_days)

    def split(self, times: pd.Series, X=None, y=None,
              ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t = pd.Series(pd.to_datetime(times.values)).reset_index(drop=True)
        unique_times = np.array(sorted(t.unique()))
        n = len(unique_times)
        if n < self.n_splits + 1:
            raise ValueError(
                f"Not enough unique dates ({n}) for {self.n_splits} folds"
            )

        # Contiguous calendar blocks of roughly equal size — use np.array_split
        # so the remainder is distributed across the early folds.
        fold_blocks = np.array_split(np.arange(n), self.n_splits)
        embargo = pd.Timedelta(days=self.embargo_days)
        horizon = pd.Timedelta(days=self.horizon_days)

        for k, val_block in enumerate(fold_blocks):
            if len(val_block) == 0:
                continue
            val_times = unique_times[val_block]
            val_min = pd.Timestamp(val_times[0])
            val_max = pd.Timestamp(val_times[-1])

            # A train date s is KEPT iff its forward-label window
            # [s, s + horizon] does NOT overlap the purge-or-embargo zone
            # around the val block.
            purge_lo = val_min - horizon           # labels ending before val start can stay
            purge_hi = val_max + embargo + horizon # labels starting after embargo can stay

            train_mask = (unique_times < purge_lo) | (unique_times > purge_hi)
            train_times = unique_times[train_mask]

            train_idx = np.where(t.isin(train_times))[0]
            val_idx   = np.where(t.isin(val_times))[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ══════════════════════════════════════════════════════════════════════════════
# Feature column selection
# ══════════════════════════════════════════════════════════════════════════════

# Columns produced by compute_price_features + compute_market_model.
# Everything else (forward_return, label_*, metadata) is excluded from X.
_EXCLUDE_FROM_X = {
    "forward_return", "label_beats_median",
    # identity / meta (if joined at inference)
    "name", "sector", "industry", "country", "currency",
    "last_close", "last_price_date",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _EXCLUDE_FROM_X
            and pd.api.types.is_numeric_dtype(df[c])]
