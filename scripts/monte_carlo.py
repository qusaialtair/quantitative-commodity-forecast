#!/usr/bin/env python3
"""
Monte Carlo Forward Simulation Engine
=======================================
Generates probabilistic price scenarios using GARCH(1,1) volatility
and fat-tailed innovations (Student-t).

Outputs:
  - 10,000 simulated paths over configurable horizon
  - Probability cones (5th / 25th / 50th / 75th / 95th percentiles)
  - Simulated VaR and CVaR at multiple confidence levels
  - Probability of reaching target price levels
  - Expected return distribution

Usage:
    python3 scripts/monte_carlo.py
    python3 scripts/monte_carlo.py --horizon 21 --paths 10000
    python3 scripts/monte_carlo.py --ticker SI=F --horizon 63
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "monte_carlo_simulation.json"

# GARCH(1,1) parameters calibrated on gold returns
GARCH_OMEGA = 0.000002
GARCH_ALPHA = 0.06
GARCH_BETA = 0.93

# Student-t degrees of freedom (captures fat tails in gold returns)
T_DOF = 5

N_PATHS = 10_000
DEFAULT_HORIZON = 21  # trading days

LINE_W = 62
SEP = "\u2501" * LINE_W


def _fetch_returns(ticker: str, lookback: str = "2y") -> tuple[pd.Series, float]:
    if yf is None:
        raise ImportError("yfinance required")
    raw = yf.download(ticker, period=lookback, interval="1d",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    close = raw["Close"].dropna()
    returns = close.pct_change().dropna()
    current_price = float(close.iloc[-1])
    return returns, current_price


def _garch_variance_forecast(returns: pd.Series, horizon: int) -> np.ndarray:
    """
    Forecast conditional variance path using GARCH(1,1).
    Returns array of shape (horizon,) with daily variance forecasts.
    """
    r = returns.values
    n = len(r)

    # Initialize with unconditional variance
    long_run_var = GARCH_OMEGA / (1.0 - GARCH_ALPHA - GARCH_BETA)
    sigma2 = np.full(n, long_run_var)

    # Fit historical conditional variances
    for t in range(1, n):
        sigma2[t] = GARCH_OMEGA + GARCH_ALPHA * r[t-1]**2 + GARCH_BETA * sigma2[t-1]

    # Last fitted variance as starting point
    last_var = sigma2[-1]
    last_ret = r[-1]

    # Multi-step forecast
    var_forecast = np.zeros(horizon)
    var_forecast[0] = GARCH_OMEGA + GARCH_ALPHA * last_ret**2 + GARCH_BETA * last_var
    for h in range(1, horizon):
        var_forecast[h] = GARCH_OMEGA + (GARCH_ALPHA + GARCH_BETA) * var_forecast[h-1]

    return var_forecast


def simulate_paths(
    returns: pd.Series,
    current_price: float,
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = N_PATHS,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo simulation with GARCH volatility and Student-t innovations.

    Returns array of shape (n_paths, horizon+1) with price paths.
    Column 0 = current_price for all paths.
    """
    rng = np.random.default_rng(seed)

    # GARCH variance forecast for mean path
    var_forecast = _garch_variance_forecast(returns, horizon)

    # Drift estimate (annualized mean return → daily)
    mu_daily = float(returns.mean())

    # Generate Student-t innovations (fatter tails than normal)
    innovations = rng.standard_t(df=T_DOF, size=(n_paths, horizon))
    # Normalize to unit variance (Student-t with df=v has var = v/(v-2))
    innovations = innovations / np.sqrt(T_DOF / (T_DOF - 2))

    # Build price paths
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = current_price

    for t in range(horizon):
        sigma_t = np.sqrt(var_forecast[t])
        daily_returns = mu_daily + sigma_t * innovations[:, t]
        paths[:, t + 1] = paths[:, t] * (1.0 + daily_returns)

    return paths


def compute_statistics(paths: np.ndarray, current_price: float) -> dict:
    """Compute comprehensive statistics from simulated paths."""
    final_prices = paths[:, -1]
    final_returns = (final_prices / current_price - 1.0) * 100

    # Percentile cones at each timestep
    percentiles = [5, 25, 50, 75, 95]
    cones = {}
    for p in percentiles:
        cones[f"p{p}"] = [round(float(np.percentile(paths[:, t], p)), 2)
                          for t in range(paths.shape[1])]

    # Terminal distribution
    terminal = {
        "mean_price": round(float(final_prices.mean()), 2),
        "median_price": round(float(np.median(final_prices)), 2),
        "std_price": round(float(final_prices.std()), 2),
        "mean_return_pct": round(float(final_returns.mean()), 3),
        "median_return_pct": round(float(np.median(final_returns)), 3),
        "std_return_pct": round(float(final_returns.std()), 3),
        "skewness": round(float(pd.Series(final_returns).skew()), 3),
        "kurtosis": round(float(pd.Series(final_returns).kurt()), 3),
    }

    # VaR and CVaR from simulation (more accurate than parametric)
    risk_metrics = {}
    for conf in [0.90, 0.95, 0.99]:
        var_pct = float(np.percentile(final_returns, (1 - conf) * 100))
        cvar_pct = float(final_returns[final_returns <= var_pct].mean())
        risk_metrics[f"var_{int(conf*100)}_pct"] = round(var_pct, 3)
        risk_metrics[f"cvar_{int(conf*100)}_pct"] = round(cvar_pct, 3)

    # Probability of various outcomes
    prob_positive = float((final_returns > 0).mean())
    prob_up_2pct = float((final_returns > 2.0).mean())
    prob_up_5pct = float((final_returns > 5.0).mean())
    prob_down_2pct = float((final_returns < -2.0).mean())
    prob_down_5pct = float((final_returns < -5.0).mean())

    probabilities = {
        "positive_return": round(prob_positive, 4),
        "up_2pct": round(prob_up_2pct, 4),
        "up_5pct": round(prob_up_5pct, 4),
        "down_2pct": round(prob_down_2pct, 4),
        "down_5pct": round(prob_down_5pct, 4),
    }

    # Maximum drawdown across paths (average and worst)
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max * 100
    max_dd_per_path = drawdowns.min(axis=1)
    dd_stats = {
        "avg_max_drawdown_pct": round(float(max_dd_per_path.mean()), 3),
        "worst_drawdown_pct": round(float(max_dd_per_path.min()), 3),
        "p5_max_drawdown_pct": round(float(np.percentile(max_dd_per_path, 5)), 3),
    }

    return {
        "cones": cones,
        "terminal": terminal,
        "risk": risk_metrics,
        "probabilities": probabilities,
        "drawdown": dd_stats,
    }


def run_simulation(
    ticker: str = "GC=F",
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = N_PATHS,
) -> dict:
    """Full simulation pipeline. Returns result dict and saves to disk."""
    returns, current_price = _fetch_returns(ticker)

    paths = simulate_paths(returns, current_price, horizon, n_paths)
    stats = compute_statistics(paths, current_price)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "horizon_days": horizon,
        "n_paths": n_paths,
        "garch_params": {"omega": GARCH_OMEGA, "alpha": GARCH_ALPHA, "beta": GARCH_BETA},
        "innovation_dist": f"Student-t(df={T_DOF})",
        **stats,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))

    _print_report(result)
    return result


def _print_report(result: dict) -> None:
    """Print formatted simulation report."""
    print(f"\n{SEP}")
    print(f"  MONTE CARLO SIMULATION -- {result['ticker']}")
    print(SEP)
    print(f"  Current Price:  ${result['current_price']:,.2f}")
    print(f"  Horizon:        {result['horizon_days']} trading days")
    print(f"  Paths:          {result['n_paths']:,}")
    print(f"  Vol Model:      GARCH(1,1) + Student-t({T_DOF})")
    print()

    t = result["terminal"]
    print(f"  TERMINAL DISTRIBUTION")
    print(f"  {'─' * 40}")
    print(f"  Mean Price:     ${t['mean_price']:,.2f}  ({t['mean_return_pct']:+.3f}%)")
    print(f"  Median Price:   ${t['median_price']:,.2f}  ({t['median_return_pct']:+.3f}%)")
    print(f"  Std Dev:        ${t['std_price']:,.2f}  ({t['std_return_pct']:.3f}%)")
    print(f"  Skewness:       {t['skewness']:+.3f}")
    print(f"  Kurtosis:       {t['kurtosis']:+.3f}")
    print()

    r = result["risk"]
    print(f"  RISK METRICS (from simulation)")
    print(f"  {'─' * 40}")
    print(f"  VaR 90%:        {r['var_90_pct']:+.3f}%")
    print(f"  VaR 95%:        {r['var_95_pct']:+.3f}%")
    print(f"  VaR 99%:        {r['var_99_pct']:+.3f}%")
    print(f"  CVaR 95%:       {r['cvar_95_pct']:+.3f}%")
    print(f"  CVaR 99%:       {r['cvar_99_pct']:+.3f}%")
    print()

    p = result["probabilities"]
    print(f"  PROBABILITIES")
    print(f"  {'─' * 40}")
    print(f"  P(positive):    {p['positive_return']:.1%}")
    print(f"  P(>+2%):        {p['up_2pct']:.1%}")
    print(f"  P(>+5%):        {p['up_5pct']:.1%}")
    print(f"  P(<-2%):        {p['down_2pct']:.1%}")
    print(f"  P(<-5%):        {p['down_5pct']:.1%}")
    print()

    dd = result["drawdown"]
    print(f"  DRAWDOWN ANALYSIS")
    print(f"  {'─' * 40}")
    print(f"  Avg Max DD:     {dd['avg_max_drawdown_pct']:.3f}%")
    print(f"  Worst Path DD:  {dd['worst_drawdown_pct']:.3f}%")
    print(f"  P5 Max DD:      {dd['p5_max_drawdown_pct']:.3f}%")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Forward Simulation")
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    args = parser.parse_args()

    run_simulation(ticker=args.ticker, horizon=args.horizon, n_paths=args.paths)
