#!/usr/bin/env python3
"""
Walk-Forward Backtester
========================
Institutional-grade validation for the gold trading system.
Rolling window train/test with realistic transaction costs.

The gold standard for quant system validation: at each step, retrain
the LSTM on the training window, generate predictions over the test
window, simulate trades with realistic costs, and track cumulative
performance metrics.

Usage:
    python3 scripts/walk_forward_backtest.py
    python3 scripts/walk_forward_backtest.py --train-years 3 --test-months 3
    python3 scripts/walk_forward_backtest.py --train-years 5 --test-months 6 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yfinance as yf

# ── Project root & imports ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.lstm_predictor import (
    GoldLSTM,
    SequenceDataset,
    TwoPartScaler,
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 5       # 5 basis points per trade (one way)
COMMISSION_PER_TRADE = 2.50    # flat commission in USD per trade
RISK_FREE_RATE       = 0.0     # annualized risk-free rate for Sharpe
TRADING_DAYS_YEAR    = 252
RESULTS_PATH         = ROOT / "data" / "backtest_results.json"


# ==============================================================================
# Metrics
# ==============================================================================

@dataclass
class BacktestMetrics:
    """Complete performance summary for walk-forward evaluation."""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0           # annualized, rf=0
    sortino_ratio: float = 0.0          # downside deviation only
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0           # annual_return / max_drawdown
    win_rate_pct: float = 0.0           # % of profitable trades
    profit_factor: float = 0.0          # gross_profits / gross_losses
    avg_trade_return: float = 0.0
    num_trades: int = 0
    avg_holding_days: float = 0.0
    exposure_pct: float = 0.0           # % of time in market

    # Per-window breakdown
    window_returns: list[float] = field(default_factory=list)
    window_sharpes: list[float] = field(default_factory=list)

    # Metadata
    train_years: int = 0
    test_months: int = 0
    num_windows: int = 0
    total_days_tested: int = 0
    data_start: str = ""
    data_end: str = ""


@dataclass
class BenchmarkMetrics:
    """Buy-and-hold baseline for the same period."""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0


# ==============================================================================
# Hardware
# ==============================================================================

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# Data pipeline
# ==============================================================================

def fetch_full_history() -> pd.DataFrame:
    """
    Download maximum available GC=F history from yfinance.
    Returns a cleaned DataFrame with all features needed for the LSTM.
    """
    logger.info("Downloading GC=F full history (max available)...")
    raw = yf.download("GC=F", period="max", interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no data for GC=F")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Drop rows with NaN in OHLC (keep Volume NaN as 0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    # Daily return
    df["ret1"] = df["Close"].pct_change().fillna(0.0)

    # Volatility ratio: 5d/21d realized vol (expansion/contraction signal)
    vol5 = df["Close"].pct_change().rolling(5).std()
    vol21 = df["Close"].pct_change().rolling(21).std()
    df["vol_ratio"] = (vol5 / vol21.replace(0, np.nan)).fillna(1.0)

    # Momentum quality: sign(ret5)*sign(ret21) confirmation signal
    ret5 = df["Close"].pct_change(5)
    ret21 = df["Close"].pct_change(21)
    df["mom_quality"] = (np.sign(ret5) * np.sign(ret21)).fillna(0.0)

    # Perplexity columns: zero for historical backtest (no forward-looking bias)
    for col in ["pplx_fed", "pplx_geo_risk", "pplx_phys_demand", "pplx_macro"]:
        df[col] = 0.0

    # Alternative data: attach from CSV if available, else zero
    alt_cols = ["real_yield_10y", "copper_gold_ratio_zscore", "cot_gold_mm_net_zscore"]
    alt_path = ROOT / "data" / "alt_data.csv"

    if alt_path.exists():
        try:
            alt = pd.read_csv(alt_path, index_col="date", parse_dates=True)
            alt.index = pd.to_datetime(alt.index).tz_localize(None)
            present = [c for c in alt_cols if c in alt.columns]
            if present:
                df = df.join(alt[present], how="left")
                df[present] = df[present].ffill().bfill().fillna(0.0)
                for col in alt_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                logger.info(f"Alt data attached: {len(present)} columns")
            else:
                for col in alt_cols:
                    df[col] = 0.0
        except Exception as exc:
            logger.warning(f"Could not load alt_data.csv: {exc}")
            for col in alt_cols:
                df[col] = 0.0
    else:
        logger.info("No alt_data.csv found -- alt features set to 0.0")
        for col in alt_cols:
            df[col] = 0.0

    # Ensure feature column order matches the model expectation
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns after data prep: {missing}")

    logger.info(
        f"Data ready: {len(df)} rows, "
        f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    )
    return df


# ==============================================================================
# Rolling windows
# ==============================================================================

def generate_windows(
    df: pd.DataFrame,
    train_years: int = 3,
    test_months: int = 3,
) -> list[dict]:
    """
    Generate (train_start, train_end, test_start, test_end) windows.

    Slides forward by test_months at each step. Each window dict also carries
    the integer row indices into df for efficient slicing.
    """
    windows = []
    dates = df.index

    first_date = dates[0]
    last_date = dates[-1]

    # First test window starts after the first train_years of data
    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(months=test_months)
    step_delta = test_delta

    train_start = first_date
    train_end = train_start + train_delta

    while True:
        test_start = train_end
        test_end = test_start + test_delta

        # Clip to available data
        if test_start >= last_date:
            break

        test_end_clipped = min(test_end, last_date)

        # Find row indices
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates <= test_end_clipped)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        # Need at least SEQ_LEN + 1 training samples to form sequences,
        # and at least 1 test day
        if n_train < SEQ_LEN + 10:
            logger.warning(
                f"Skipping window: train has only {n_train} rows "
                f"(need >= {SEQ_LEN + 10})"
            )
            train_start = train_start + step_delta
            train_end = train_start + train_delta
            continue

        if n_test < 1:
            break

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end_clipped,
            "n_train": int(n_train),
            "n_test": int(n_test),
        })

        # Slide forward
        train_start = train_start + step_delta
        train_end = train_start + train_delta

    logger.info(f"Generated {len(windows)} walk-forward windows")
    return windows


# ==============================================================================
# Lightweight training loop (per window)
# ==============================================================================

def _train_window(
    df_train: pd.DataFrame,
    epochs: int = 50,
    device: torch.device = None,
    lr: float = 1e-3,
) -> tuple[GoldLSTM, TwoPartScaler]:
    """
    Train a fresh LSTM on the given training window.
    Returns (model, scaler) ready for inference.
    """
    if device is None:
        device = _get_device()

    # Build sequences with a fresh scaler fitted on this window
    arr = df_train[FEATURE_COLS].values
    scaler = TwoPartScaler()
    scaled = scaler.fit_transform(arr)

    close_idx = FEATURE_COLS.index("Close")
    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X.append(scaled[i : i + SEQ_LEN])
        y.append(scaled[i + SEQ_LEN, close_idx])

    if len(X) == 0:
        raise ValueError(
            f"Cannot form sequences: only {len(scaled)} scaled rows, "
            f"need > {SEQ_LEN}"
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 90/10 train/val split (sequential)
    split = max(1, int(len(X) * 0.9))
    train_ds = SequenceDataset(X[:split], y[:split])
    val_ds = SequenceDataset(X[split:], y[split:]) if split < len(X) else None

    train_dl = DataLoader(
        train_ds, batch_size=min(BATCH_SIZE, len(train_ds)),
        shuffle=True, drop_last=len(train_ds) > BATCH_SIZE,
    )

    model = GoldLSTM(
        n_features=N_FEATURES,
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = max(10, epochs // 5)

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_ds)

        # Validate
        if val_ds is not None and len(val_ds) > 0:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    v_loss += criterion(model(xb), yb).item() * len(xb)
            v_loss /= len(val_ds)

            if v_loss < best_val:
                best_val = v_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                break
        else:
            # No validation set -- just save last state
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        scheduler.step()

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    return model, scaler


# ==============================================================================
# Per-window prediction + trade simulation
# ==============================================================================

def _predict_window(
    model: GoldLSTM,
    scaler: TwoPartScaler,
    df_full: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    device: torch.device,
) -> pd.DataFrame:
    """
    Generate daily predictions for the test window.

    For each test day, we use the preceding SEQ_LEN days (which may extend
    into the training window) to form the input sequence and predict the
    next-day close.

    Returns a DataFrame indexed by date with columns:
        actual_close, predicted_close, position, daily_return
    """
    dates = df_full.index
    test_mask = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]

    if len(test_dates) == 0:
        return pd.DataFrame()

    close_idx = FEATURE_COLS.index("Close")
    results = []

    for dt in test_dates:
        # Get the position of this date in the full dataframe
        loc = df_full.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.start

        # Need SEQ_LEN days before this date for the input sequence
        seq_start = loc - SEQ_LEN
        if seq_start < 0:
            continue

        # Extract the input window
        window_df = df_full.iloc[seq_start : loc]
        if len(window_df) < SEQ_LEN:
            continue

        # Scale using the window's scaler (fitted on training data)
        try:
            scaled = scaler.transform(window_df[FEATURE_COLS].values)
        except Exception:
            continue

        seq = scaled[-SEQ_LEN:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(x).item()

        # Inverse transform to get predicted price
        dummy = scaled[-1].copy()
        dummy[close_idx] = pred_scaled
        pred_price = float(
            scaler.inverse_transform(dummy.reshape(1, -1))[0, close_idx]
        )

        actual_close = float(df_full.loc[dt, "Close"])

        # Previous day close for computing signal
        if loc > 0:
            prev_close = float(df_full.iloc[loc - 1]["Close"])
        else:
            prev_close = actual_close

        results.append({
            "date": dt,
            "actual_close": actual_close,
            "predicted_close": pred_price,
            "prev_close": prev_close,
        })

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results).set_index("date")

    # Signal: if predicted > previous close, go long (1); else flat (0)
    res_df["signal"] = (res_df["predicted_close"] > res_df["prev_close"]).astype(int)

    return res_df


def _simulate_trades(
    predictions: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Simulate trading based on predictions with realistic costs.

    Returns a dict with:
        daily_returns, equity_curve, trade_log, final_equity, etc.
    """
    if predictions.empty:
        return {
            "daily_returns": np.array([]),
            "equity_curve": np.array([initial_capital]),
            "num_trades": 0,
            "trade_returns": [],
            "exposure_days": 0,
            "total_days": 0,
        }

    equity = initial_capital
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    entry_date = None
    trade_returns = []
    daily_returns = []
    equity_curve = [equity]
    exposure_days = 0
    holding_days_list = []

    cost_rate = TRANSACTION_COST_BPS / 10_000  # per-trade cost in fraction

    for i in range(len(predictions)):
        row = predictions.iloc[i]
        actual_close = row["actual_close"]
        signal = int(row["signal"])
        prev_close = row["prev_close"]

        # Daily return if we were holding
        if position == 1 and prev_close > 0:
            raw_return = (actual_close - prev_close) / prev_close
        else:
            raw_return = 0.0

        # Position changes
        trade_cost = 0.0
        if signal == 1 and position == 0:
            # Enter long
            trade_cost = equity * cost_rate + COMMISSION_PER_TRADE
            equity -= trade_cost
            position = 1
            entry_price = actual_close
            entry_date = predictions.index[i]
        elif signal == 0 and position == 1:
            # Exit long
            trade_cost = equity * cost_rate + COMMISSION_PER_TRADE
            equity -= trade_cost
            # Record trade return
            if entry_price > 0:
                trade_ret = (actual_close - entry_price) / entry_price
                trade_returns.append(trade_ret)
            if entry_date is not None:
                days_held = (predictions.index[i] - entry_date).days
                holding_days_list.append(max(1, days_held))
            position = 0
            entry_price = 0.0
            entry_date = None

        # Apply daily return to equity
        if position == 1:
            equity *= (1.0 + raw_return)
            exposure_days += 1

        daily_ret = (equity - equity_curve[-1]) / equity_curve[-1] if equity_curve[-1] > 0 else 0.0
        daily_returns.append(daily_ret)
        equity_curve.append(equity)

    # Close any open position at end
    if position == 1 and len(predictions) > 0:
        last_close = predictions.iloc[-1]["actual_close"]
        if entry_price > 0:
            trade_ret = (last_close - entry_price) / entry_price
            trade_returns.append(trade_ret)
        if entry_date is not None:
            days_held = (predictions.index[-1] - entry_date).days
            holding_days_list.append(max(1, days_held))

    return {
        "daily_returns": np.array(daily_returns, dtype=np.float64),
        "equity_curve": np.array(equity_curve, dtype=np.float64),
        "num_trades": len(trade_returns),
        "trade_returns": trade_returns,
        "exposure_days": exposure_days,
        "total_days": len(predictions),
        "holding_days": holding_days_list,
        "final_equity": equity,
    }


# ==============================================================================
# Metrics computation
# ==============================================================================

def _compute_metrics(
    all_daily_returns: np.ndarray,
    all_trade_returns: list[float],
    all_holding_days: list[int],
    total_days: int,
    exposure_days: int,
    window_returns: list[float],
    window_sharpes: list[float],
    initial_equity: float,
    final_equity: float,
    train_years: int,
    test_months: int,
    num_windows: int,
    data_start: str,
    data_end: str,
) -> BacktestMetrics:
    """Compute the full BacktestMetrics from aggregated results."""

    # Total and annualized return
    total_return_pct = ((final_equity / initial_equity) - 1.0) * 100
    n_years = max(total_days / TRADING_DAYS_YEAR, 0.01)
    if final_equity > 0 and initial_equity > 0:
        annual_return_pct = (
            (final_equity / initial_equity) ** (1.0 / n_years) - 1.0
        ) * 100
    else:
        annual_return_pct = 0.0

    # Sharpe ratio (annualized)
    if len(all_daily_returns) > 1:
        mean_daily = np.mean(all_daily_returns)
        std_daily = np.std(all_daily_returns, ddof=1)
        sharpe = (
            (mean_daily - RISK_FREE_RATE / TRADING_DAYS_YEAR)
            / max(std_daily, 1e-10)
            * math.sqrt(TRADING_DAYS_YEAR)
        )
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation)
    if len(all_daily_returns) > 1:
        downside = all_daily_returns[all_daily_returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
        sortino = (
            (np.mean(all_daily_returns) - RISK_FREE_RATE / TRADING_DAYS_YEAR)
            / max(downside_std, 1e-10)
            * math.sqrt(TRADING_DAYS_YEAR)
        )
    else:
        sortino = 0.0

    # Max drawdown
    if len(all_daily_returns) > 0:
        cum = np.cumprod(1.0 + all_daily_returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
        max_drawdown_pct = abs(float(np.min(dd))) * 100
    else:
        max_drawdown_pct = 0.0

    # Calmar
    calmar = (
        annual_return_pct / max(max_drawdown_pct, 0.01)
        if max_drawdown_pct > 0 else 0.0
    )

    # Win rate
    if all_trade_returns:
        wins = sum(1 for r in all_trade_returns if r > 0)
        win_rate_pct = (wins / len(all_trade_returns)) * 100
    else:
        win_rate_pct = 0.0

    # Profit factor
    gross_profits = sum(r for r in all_trade_returns if r > 0)
    gross_losses = abs(sum(r for r in all_trade_returns if r < 0))
    profit_factor = (
        gross_profits / max(gross_losses, 1e-10)
        if gross_losses > 0 else (float("inf") if gross_profits > 0 else 0.0)
    )

    # Average trade return
    avg_trade_return = (
        float(np.mean(all_trade_returns)) if all_trade_returns else 0.0
    )

    # Average holding days
    avg_holding_days = (
        float(np.mean(all_holding_days)) if all_holding_days else 0.0
    )

    # Exposure
    exposure_pct = (exposure_days / max(total_days, 1)) * 100

    return BacktestMetrics(
        total_return_pct=round(total_return_pct, 4),
        annual_return_pct=round(annual_return_pct, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        calmar_ratio=round(calmar, 4),
        win_rate_pct=round(win_rate_pct, 2),
        profit_factor=round(profit_factor, 4),
        avg_trade_return=round(avg_trade_return, 6),
        num_trades=len(all_trade_returns),
        avg_holding_days=round(avg_holding_days, 1),
        exposure_pct=round(exposure_pct, 2),
        window_returns=[round(r, 4) for r in window_returns],
        window_sharpes=[round(s, 4) for s in window_sharpes],
        train_years=train_years,
        test_months=test_months,
        num_windows=num_windows,
        total_days_tested=total_days,
        data_start=data_start,
        data_end=data_end,
    )


def _compute_benchmark(
    df: pd.DataFrame,
    test_dates_all: list[pd.Timestamp],
) -> BenchmarkMetrics:
    """Compute buy-and-hold metrics over the same test dates."""
    if len(test_dates_all) < 2:
        return BenchmarkMetrics()

    first_date = min(test_dates_all)
    last_date = max(test_dates_all)

    bh_slice = df.loc[first_date:last_date, "Close"]
    if len(bh_slice) < 2:
        return BenchmarkMetrics()

    daily_returns = bh_slice.pct_change().dropna().values
    total_ret = (float(bh_slice.iloc[-1]) / float(bh_slice.iloc[0]) - 1.0) * 100
    n_years = max(len(daily_returns) / TRADING_DAYS_YEAR, 0.01)
    annual_ret = ((1 + total_ret / 100) ** (1.0 / n_years) - 1.0) * 100

    if len(daily_returns) > 1:
        mean_d = np.mean(daily_returns)
        std_d = np.std(daily_returns, ddof=1)
        sharpe = (mean_d / max(std_d, 1e-10)) * math.sqrt(TRADING_DAYS_YEAR)
    else:
        sharpe = 0.0

    cum = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = abs(float(np.min(dd))) * 100 if len(dd) > 0 else 0.0

    calmar = annual_ret / max(max_dd, 0.01) if max_dd > 0 else 0.0

    return BenchmarkMetrics(
        total_return_pct=round(total_ret, 4),
        annual_return_pct=round(annual_ret, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown_pct=round(max_dd, 4),
        calmar_ratio=round(calmar, 4),
    )


# ==============================================================================
# Per-window Sharpe helper
# ==============================================================================

def _window_sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe for a single test window."""
    if len(daily_returns) < 2:
        return 0.0
    mean_d = np.mean(daily_returns)
    std_d = np.std(daily_returns, ddof=1)
    if std_d < 1e-10:
        return 0.0
    return float(mean_d / std_d * math.sqrt(TRADING_DAYS_YEAR))


# ==============================================================================
# Sparkline
# ==============================================================================

_SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

def _sparkline(values: list[float]) -> str:
    """Generate a Unicode sparkline from a list of values."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng < 1e-10:
        return _SPARK_CHARS[3] * len(values)
    chars = []
    for v in values:
        idx = int((v - mn) / rng * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


# ==============================================================================
# Report printer
# ==============================================================================

def _print_report(
    metrics: BacktestMetrics,
    benchmark: BenchmarkMetrics,
) -> None:
    """Print a clean terminal report."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  WALK-FORWARD BACKTEST RESULTS")
    print(f"  {metrics.data_start} to {metrics.data_end}")
    print(f"  Windows: {metrics.num_windows}  |  "
          f"Train: {metrics.train_years}yr  |  "
          f"Test: {metrics.test_months}mo  |  "
          f"Days tested: {metrics.total_days_tested}")
    print(sep)

    print(f"\n  {'STRATEGY':38s} {'BUY & HOLD':>14s} {'ALPHA':>12s}")
    print(f"  {thin}")

    alpha_total = metrics.total_return_pct - benchmark.total_return_pct
    alpha_annual = metrics.annual_return_pct - benchmark.annual_return_pct
    alpha_sharpe = metrics.sharpe_ratio - benchmark.sharpe_ratio

    rows = [
        ("Total Return %",
         f"{metrics.total_return_pct:+.2f}%",
         f"{benchmark.total_return_pct:+.2f}%",
         f"{alpha_total:+.2f}%"),
        ("Annual Return %",
         f"{metrics.annual_return_pct:+.2f}%",
         f"{benchmark.annual_return_pct:+.2f}%",
         f"{alpha_annual:+.2f}%"),
        ("Sharpe Ratio",
         f"{metrics.sharpe_ratio:.3f}",
         f"{benchmark.sharpe_ratio:.3f}",
         f"{alpha_sharpe:+.3f}"),
        ("Sortino Ratio",
         f"{metrics.sortino_ratio:.3f}",
         "--",
         ""),
        ("Max Drawdown %",
         f"{metrics.max_drawdown_pct:.2f}%",
         f"{benchmark.max_drawdown_pct:.2f}%",
         ""),
        ("Calmar Ratio",
         f"{metrics.calmar_ratio:.3f}",
         f"{benchmark.calmar_ratio:.3f}",
         ""),
    ]
    for label, strat, bh, alpha in rows:
        print(f"  {label:38s} {strat:>14s} {bh:>14s} {alpha:>12s}")

    print(f"\n  {thin}")
    print(f"  {'TRADE STATISTICS':38s}")
    print(f"  {thin}")
    print(f"  {'Win Rate':38s} {metrics.win_rate_pct:.1f}%")
    print(f"  {'Profit Factor':38s} {metrics.profit_factor:.3f}")
    print(f"  {'Avg Trade Return':38s} {metrics.avg_trade_return*100:.4f}%")
    print(f"  {'Total Trades':38s} {metrics.num_trades}")
    print(f"  {'Avg Holding Days':38s} {metrics.avg_holding_days:.1f}")
    print(f"  {'Market Exposure':38s} {metrics.exposure_pct:.1f}%")

    # Per-window sparkline
    print(f"\n  {thin}")
    print(f"  PER-WINDOW RETURNS (sparkline)")
    print(f"  {thin}")
    if metrics.window_returns:
        spark = _sparkline(metrics.window_returns)
        print(f"  {spark}")
        print(f"  min={min(metrics.window_returns):+.2f}%  "
              f"max={max(metrics.window_returns):+.2f}%  "
              f"mean={np.mean(metrics.window_returns):+.2f}%  "
              f"std={np.std(metrics.window_returns):.2f}%")

        positive_windows = sum(1 for r in metrics.window_returns if r > 0)
        print(f"  Positive windows: {positive_windows}/{len(metrics.window_returns)} "
              f"({positive_windows/len(metrics.window_returns)*100:.0f}%)")

    print(f"\n  PER-WINDOW SHARPES (sparkline)")
    if metrics.window_sharpes:
        spark = _sparkline(metrics.window_sharpes)
        print(f"  {spark}")
        print(f"  min={min(metrics.window_sharpes):.3f}  "
              f"max={max(metrics.window_sharpes):.3f}  "
              f"mean={np.mean(metrics.window_sharpes):.3f}")

    print(f"\n{sep}\n")


# ==============================================================================
# Results saver
# ==============================================================================

def _save_results(
    metrics: BacktestMetrics,
    benchmark: BenchmarkMetrics,
    per_window_detail: list[dict],
) -> None:
    """Write full results to data/backtest_results.json."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "strategy": asdict(metrics),
        "benchmark": asdict(benchmark),
        "per_window_detail": per_window_detail,
        "config": {
            "train_years": metrics.train_years,
            "test_months": metrics.test_months,
            "seq_len": SEQ_LEN,
            "n_features": N_FEATURES,
            "feature_cols": FEATURE_COLS,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "commission_per_trade": COMMISSION_PER_TRADE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
        },
    }

    # Convert numpy/pandas types to JSON-safe
    def _sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def _deep_sanitize(obj):
        if isinstance(obj, dict):
            return {k: _deep_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_sanitize(v) for v in obj]
        return _sanitize(obj)

    payload = _deep_sanitize(payload)

    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_PATH}")


# ==============================================================================
# Main engine
# ==============================================================================

def run_walk_forward(
    train_years: int = 3,
    test_months: int = 3,
    epochs: int = 50,
    initial_capital: float = 100_000.0,
) -> tuple[BacktestMetrics, BenchmarkMetrics]:
    """
    Execute the full walk-forward backtest.

    Parameters
    ----------
    train_years : int
        Number of years per training window.
    test_months : int
        Number of months per test window (also the step size).
    epochs : int
        Max epochs for LSTM training per window.
    initial_capital : float
        Starting portfolio value in USD.

    Returns
    -------
    (BacktestMetrics, BenchmarkMetrics)
    """
    device = _get_device()
    logger.info(f"Device: {device}")
    logger.info(
        f"Config: train={train_years}yr, test={test_months}mo, "
        f"epochs={epochs}, capital=${initial_capital:,.0f}"
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    df = fetch_full_history()

    # ── Generate windows ──────────────────────────────────────────────────────
    windows = generate_windows(df, train_years=train_years, test_months=test_months)

    if not windows:
        logger.error(
            "No valid walk-forward windows could be generated. "
            f"Data spans {df.index[0]} to {df.index[-1]} "
            f"but need at least {train_years} years + {test_months} months."
        )
        raise RuntimeError("Insufficient data for walk-forward windows")

    # ── Walk forward ──────────────────────────────────────────────────────────
    all_daily_returns = []
    all_trade_returns = []
    all_holding_days = []
    window_returns = []
    window_sharpes = []
    per_window_detail = []
    test_dates_all = []

    equity = initial_capital
    total_exposure_days = 0
    total_test_days = 0
    run_start = time.time()

    for i, w in enumerate(windows):
        w_start = time.time()
        label = (
            f"[{i+1}/{len(windows)}] "
            f"Train: {w['train_start'].strftime('%Y-%m-%d')} to "
            f"{w['train_end'].strftime('%Y-%m-%d')}  |  "
            f"Test: {w['test_start'].strftime('%Y-%m-%d')} to "
            f"{w['test_end'].strftime('%Y-%m-%d')}  "
            f"(train={w['n_train']} test={w['n_test']})"
        )
        logger.info(label)

        # Extract training data
        train_mask = (
            (df.index >= w["train_start"]) & (df.index < w["train_end"])
        )
        df_train = df.loc[train_mask].copy()

        if len(df_train) < SEQ_LEN + 10:
            logger.warning(
                f"  Window {i+1}: insufficient training data "
                f"({len(df_train)} rows) -- skipping"
            )
            per_window_detail.append({
                "window": i + 1,
                "train_start": w["train_start"].isoformat(),
                "train_end": w["train_end"].isoformat(),
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "status": "skipped_insufficient_data",
            })
            continue

        # Train LSTM on this window
        try:
            model, scaler = _train_window(
                df_train, epochs=epochs, device=device
            )
        except Exception as exc:
            logger.warning(f"  Window {i+1}: training failed -- {exc}")
            per_window_detail.append({
                "window": i + 1,
                "train_start": w["train_start"].isoformat(),
                "train_end": w["train_end"].isoformat(),
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "status": f"training_failed: {exc}",
            })
            continue

        # Generate predictions
        predictions = _predict_window(
            model, scaler, df, w["test_start"], w["test_end"], device
        )

        if predictions.empty:
            logger.warning(f"  Window {i+1}: no predictions generated")
            per_window_detail.append({
                "window": i + 1,
                "train_start": w["train_start"].isoformat(),
                "train_end": w["train_end"].isoformat(),
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "status": "no_predictions",
            })
            continue

        # Simulate trades
        sim = _simulate_trades(predictions, initial_capital=equity)

        # Accumulate results
        w_daily = sim["daily_returns"]
        all_daily_returns.extend(w_daily.tolist())
        all_trade_returns.extend(sim["trade_returns"])
        all_holding_days.extend(sim.get("holding_days", []))
        total_exposure_days += sim["exposure_days"]
        total_test_days += sim["total_days"]

        # Collect test dates for benchmark
        test_dates_all.extend(predictions.index.tolist())

        # Per-window metrics
        w_return_pct = ((sim["final_equity"] / equity) - 1.0) * 100
        w_sharpe = _window_sharpe(w_daily)
        window_returns.append(w_return_pct)
        window_sharpes.append(w_sharpe)

        # Update running equity for next window
        equity = sim["final_equity"]

        w_elapsed = time.time() - w_start
        logger.info(
            f"  -> Return: {w_return_pct:+.2f}%  "
            f"Sharpe: {w_sharpe:.3f}  "
            f"Trades: {sim['num_trades']}  "
            f"Equity: ${equity:,.0f}  "
            f"({w_elapsed:.1f}s)"
        )

        per_window_detail.append({
            "window": i + 1,
            "train_start": w["train_start"].isoformat(),
            "train_end": w["train_end"].isoformat(),
            "test_start": w["test_start"].isoformat(),
            "test_end": w["test_end"].isoformat(),
            "status": "ok",
            "return_pct": round(w_return_pct, 4),
            "sharpe": round(w_sharpe, 4),
            "num_trades": sim["num_trades"],
            "equity_after": round(equity, 2),
        })

        # Free GPU memory between windows
        del model
        if device.type in ("mps", "cuda"):
            torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

    total_elapsed = time.time() - run_start

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_daily_returns_arr = np.array(all_daily_returns, dtype=np.float64)

    data_start = df.index[0].strftime("%Y-%m-%d")
    data_end = df.index[-1].strftime("%Y-%m-%d")

    # Use first test window start as the actual tested period start
    if test_dates_all:
        tested_start = min(test_dates_all).strftime("%Y-%m-%d")
        tested_end = max(test_dates_all).strftime("%Y-%m-%d")
    else:
        tested_start = data_start
        tested_end = data_end

    metrics = _compute_metrics(
        all_daily_returns=all_daily_returns_arr,
        all_trade_returns=all_trade_returns,
        all_holding_days=all_holding_days,
        total_days=total_test_days,
        exposure_days=total_exposure_days,
        window_returns=window_returns,
        window_sharpes=window_sharpes,
        initial_equity=initial_capital,
        final_equity=equity,
        train_years=train_years,
        test_months=test_months,
        num_windows=len([d for d in per_window_detail if d.get("status") == "ok"]),
        data_start=tested_start,
        data_end=tested_end,
    )

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark = _compute_benchmark(df, test_dates_all)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_report(metrics, benchmark)
    logger.info(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_results(metrics, benchmark, per_window_detail)

    return metrics, benchmark


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for the gold LSTM trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-years", type=int, default=3,
        help="Years of data per training window (default: 3)",
    )
    parser.add_argument(
        "--test-months", type=int, default=3,
        help="Months per test window / step size (default: 3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs per window (default: 50)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital in USD (default: 100000)",
    )

    args = parser.parse_args()

    if args.train_years < 1:
        parser.error("--train-years must be >= 1")
    if args.test_months < 1:
        parser.error("--test-months must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    print("\n" + "=" * 70)
    print("  WALK-FORWARD BACKTESTER")
    print("  Gold LSTM Trading System Validation")
    print("=" * 70 + "\n")

    try:
        metrics, benchmark = run_walk_forward(
            train_years=args.train_years,
            test_months=args.test_months,
            epochs=args.epochs,
            initial_capital=args.capital,
        )
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
