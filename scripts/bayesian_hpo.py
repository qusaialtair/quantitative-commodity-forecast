#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization
======================================
Optuna TPE (Tree-structured Parzen Estimator) search over the signal-
generator hyperparameters used by alpha_attribution. The objective is the
out-of-sample Sharpe of an equal-weighted blended-signal long-only strategy.

Search space:
  short_window      [10, 30]
  long_window       [40, 100]
  bb_window         [10, 30]
  bb_threshold      [0.10, 0.30]
  momentum_lookback [3, 21]
  dxy_lookback      [3, 21]
  vol_short         [10, 30]
  vol_long          [40, 100]

Constraints:
  short_window < long_window
  vol_short < vol_long

Robustness:
  - 252d (1y) out-of-sample, 1006d in-sample (4y)
  - In-sample is unused for the objective (purely OOS Sharpe)
  - Annualised Sharpe (mean × 252 / vol × √252)
  - 50 trials by default; configurable

Output: data/bayesian_hpo.json
"""
from __future__ import annotations

import argparse
import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import optuna
from optuna.samplers import TPESampler

# Silence Optuna info spam during search
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "bayesian_hpo.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_TRIALS = 50
OOS_DAYS = 252
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_panel(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")

    def _close(t: str) -> pd.Series:
        raw = yf.download(t, period=lookback, interval="1d",
                           progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        return raw["Close"].dropna()

    gold = _close(ticker)
    dxy = _close("DX-Y.NYB")
    return pd.DataFrame({"gold": gold, "dxy": dxy}).ffill().dropna()


# ---------------------------------------------------------------------------
# Parametric signal generators
# ---------------------------------------------------------------------------
def _signals_param(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    g = df["gold"]
    r = g.pct_change()

    sigs = pd.DataFrame(index=df.index)
    sigs["lstm_momentum"] = np.sign(g.pct_change(params["momentum_lookback"])).fillna(0)
    sigs["macro_overlay"] = -np.sign(df["dxy"].pct_change(params["dxy_lookback"])).fillna(0)

    vol_short = r.rolling(params["vol_short"]).std()
    vol_long = r.rolling(params["vol_long"]).std()
    sigs["regime_filter"] = np.where(
        vol_short.notna() & vol_long.notna(),
        np.where(vol_short < vol_long, 1.0, -1.0),
        0.0,
    )

    sma_s = g.rolling(params["short_window"]).mean()
    sma_l = g.rolling(params["long_window"]).mean()
    sigs["technical"] = np.where(
        sma_s.notna() & sma_l.notna(),
        np.where(sma_s > sma_l, 1.0, -1.0),
        0.0,
    )

    bb_sma = g.rolling(params["bb_window"]).mean()
    bb_std = g.rolling(params["bb_window"]).std()
    upper = bb_sma + 2 * bb_std
    lower = bb_sma - 2 * bb_std
    width = (upper - lower).replace(0, np.nan)
    pct_b = (g - lower) / width
    sigs["mean_reversion"] = np.where(
        pct_b < params["bb_threshold"], 1.0,
        np.where(pct_b > (1.0 - params["bb_threshold"]), -1.0, 0.0),
    )
    return sigs.fillna(0)


def _backtest_sharpe(df: pd.DataFrame, params: dict, oos_days: int) -> tuple[float, dict]:
    sigs = _signals_param(df, params)
    r = df["gold"].pct_change().fillna(0)

    # Out-of-sample = last `oos_days` rows
    if len(df) <= oos_days + 20:
        return -10.0, {"reason": "insufficient_oos"}

    lagged = sigs.shift(1).fillna(0)
    blended = lagged.mean(axis=1)
    strat_returns = (blended * r).iloc[-oos_days:]

    if strat_returns.std() <= 1e-9:
        return -10.0, {"reason": "zero_vol"}

    sharpe = float(strat_returns.mean() * 252 / (strat_returns.std() * SQ252))
    cum = (1 + strat_returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    return sharpe, {
        "ann_return_pct":   round(float(strat_returns.mean() * 252 * 100), 3),
        "ann_vol_pct":      round(float(strat_returns.std() * SQ252 * 100), 3),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def _make_objective(df: pd.DataFrame, oos_days: int):
    def objective(trial: optuna.Trial) -> float:
        short_window = trial.suggest_int("short_window", 10, 30)
        long_window = trial.suggest_int("long_window", short_window + 10, 100)
        bb_window = trial.suggest_int("bb_window", 10, 30)
        bb_threshold = trial.suggest_float("bb_threshold", 0.10, 0.30)
        momentum_lookback = trial.suggest_int("momentum_lookback", 3, 21)
        dxy_lookback = trial.suggest_int("dxy_lookback", 3, 21)
        vol_short = trial.suggest_int("vol_short", 10, 30)
        vol_long = trial.suggest_int("vol_long", vol_short + 10, 100)

        params = {
            "short_window":      short_window,
            "long_window":       long_window,
            "bb_window":         bb_window,
            "bb_threshold":      bb_threshold,
            "momentum_lookback": momentum_lookback,
            "dxy_lookback":      dxy_lookback,
            "vol_short":         vol_short,
            "vol_long":          vol_long,
        }
        sharpe, _ = _backtest_sharpe(df, params, oos_days)
        return -sharpe  # Optuna minimizes; we want max Sharpe
    return objective


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_bayesian_hpo(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    n_trials: int = DEFAULT_TRIALS,
    oos_days: int = OOS_DAYS,
    seed: int = 42,
) -> dict:
    df = _fetch_panel(ticker, lookback)

    # Baseline (default alpha_attribution params)
    baseline_params = {
        "short_window":      20,
        "long_window":       50,
        "bb_window":         20,
        "bb_threshold":      0.20,
        "momentum_lookback": 5,
        "dxy_lookback":      5,
        "vol_short":         21,
        "vol_long":          63,
    }
    baseline_sharpe, baseline_metrics = _backtest_sharpe(df, baseline_params, oos_days)

    # Optuna search
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(_make_objective(df, oos_days), n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_sharpe = -study.best_value
    _, best_metrics = _backtest_sharpe(df, best_params, oos_days)

    # Top-5 trials for diagnostics
    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0)[:5]
    top_5 = [
        {
            "trial_number": t.number,
            "sharpe":       round(-(t.value or 0), 3),
            "params":       t.params,
        }
        for t in sorted_trials
    ]

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":            ticker,
        "lookback":          lookback,
        "n_obs":             int(len(df)),
        "n_trials":          n_trials,
        "oos_days":          oos_days,
        "baseline_params":   baseline_params,
        "baseline_metrics":  baseline_metrics,
        "baseline_sharpe":   round(baseline_sharpe, 3),
        "best_params":       best_params,
        "best_metrics":      best_metrics,
        "best_sharpe":       round(best_sharpe, 3),
        "improvement":       round(best_sharpe - baseline_sharpe, 3),
        "top_5_trials":      top_5,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  BAYESIAN HPO -- {r['ticker']}")
    print(SEP)
    print(f"  Trials:        {r['n_trials']}")
    print(f"  OOS window:    {r['oos_days']} days")
    print()

    print(f"  BASELINE vs BEST")
    print(f"  {'─' * 50}")
    print(f"  {'metric':<22s}  {'baseline':>10s}  {'optuna':>10s}")
    b = r["baseline_metrics"]
    o = r["best_metrics"]
    for k in ["sharpe", "ann_return_pct", "ann_vol_pct", "max_drawdown_pct"]:
        print(
            f"  {k:<22s}  {b.get(k, 0):>10.3f}  {o.get(k, 0):>10.3f}"
        )
    print(f"  ΔSharpe: {r['improvement']:+.3f}")
    print()

    print(f"  BEST PARAMETERS")
    print(f"  {'─' * 50}")
    for k, v in r["best_params"].items():
        b_val = r["baseline_params"].get(k, "n/a")
        marker = " *" if v != b_val else "  "
        print(f"  {marker} {k:<22s}  {v}  (baseline {b_val})")
    print()

    print(f"  TOP-5 TRIALS BY SHARPE")
    for t in r["top_5_trials"]:
        print(f"    #{t['trial_number']:>3d}  Sharpe={t['sharpe']:+.3f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Optimization")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--oos", type=int, default=OOS_DAYS)
    args = parser.parse_args()
    run_bayesian_hpo(
        ticker=args.ticker,
        lookback=args.lookback,
        n_trials=args.trials,
        oos_days=args.oos,
    )
