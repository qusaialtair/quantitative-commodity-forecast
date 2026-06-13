#!/usr/bin/env python3
"""
Mean-CVaR Linear Program Optimizer (Rockafellar-Uryasev 2000)
==============================================================
Replaces Markowitz mean-variance with mean-CVaR. Result is a tail-aware
allocation that minimises the conditional expected loss in the worst α-tail
scenarios, optionally subject to a minimum required mean return.

LP formulation (long-only):
    min  η + (1 / (m·(1-α))) · Σ z_i
    s.t. z_i + rᵢᵀ w + η ≥ 0           ∀ scenario i
         z_i ≥ 0                       ∀ i
         w_j ≥ 0                       ∀ j
         Σ w_j = 1
         (optional) μᵀw ≥ R_target

Where {r_i} are daily return scenarios (rows of the returns matrix).
At optimum, η equals the (negative) VaR_α and the objective equals CVaR_α.

The engine produces three portfolios:
  - min_cvar:       LP without return constraint (pure tail minimization)
  - mean_cvar:      LP with R_target = average asset mean return
  - equal_weight:   benchmark

Output: data/mean_cvar.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "mean_cvar.json"

DEFAULT_TICKERS = ["GC=F", "SI=F", "SPY", "TLT", "DX-Y.NYB"]
DEFAULT_LOOKBACK = "5y"
DEFAULT_ALPHA = 0.95

SQ252 = float(np.sqrt(252))
LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_returns(tickers: list[str], lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        tickers, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers[:1]
    close = close.dropna(how="all").ffill().dropna()
    return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# LP optimizer
# ---------------------------------------------------------------------------
def mean_cvar_optimal(
    returns: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    target_return_ann_pct: float | None = None,
) -> dict:
    """
    Solve the Rockafellar-Uryasev LP for one asset universe.

    target_return_ann_pct : annualised return target in percent.
        Set to None for the unconstrained min-CVaR portfolio.
    """
    R = returns.values
    m, n = R.shape
    mean_r_daily = R.mean(axis=0)

    n_vars = n + m + 1  # [w (n), z (m), η (1)]

    # Objective coefficients
    c = np.zeros(n_vars)
    c[n: n + m] = 1.0 / (m * (1.0 - alpha))
    c[-1] = 1.0

    # Inequality constraints: -r_iᵀ w - z_i - η ≤ 0
    A_ub = np.zeros((m, n_vars))
    A_ub[:, :n] = -R
    A_ub[np.arange(m), n + np.arange(m)] = -1.0
    A_ub[:, -1] = -1.0
    b_ub = np.zeros(m)

    # Optional return constraint: -μᵀ w ≤ -R_target/252/100 (daily)
    if target_return_ann_pct is not None:
        daily_target = target_return_ann_pct / 252.0 / 100.0
        A_ret = np.zeros((1, n_vars))
        A_ret[0, :n] = -mean_r_daily
        A_ub = np.vstack([A_ub, A_ret])
        b_ub = np.append(b_ub, -daily_target)

    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds
    bounds = [(0.0, 1.0)] * n + [(0.0, None)] * m + [(None, None)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "weights": pd.Series(np.full(n, 1.0 / n), index=returns.columns),
        }

    w = res.x[:n]
    eta = float(res.x[-1])
    cvar_daily = float(res.fun)

    return {
        "success":         True,
        "weights":         pd.Series(w, index=returns.columns),
        "cvar_daily_pct":  round(cvar_daily * 100, 4),
        "var_daily_pct":   round(-eta * 100, 4),  # -η is VaR threshold
        "alpha":           alpha,
        "target_return_ann_pct": target_return_ann_pct,
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def equal_weights(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------
def _portfolio_metrics(
    returns: pd.DataFrame, weights: pd.Series, alpha: float = DEFAULT_ALPHA,
) -> dict:
    port_ret = returns @ weights
    ann_ret = float(port_ret.mean() * 252)
    ann_vol = float(port_ret.std() * SQ252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = (1 + port_ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    # Realized VaR / CVaR from the in-sample distribution
    losses = -port_ret
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    return {
        "ann_return_pct":   round(ann_ret * 100, 3),
        "ann_vol_pct":      round(ann_vol * 100, 3),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "var_pct":          round(var * 100, 3),
        "cvar_pct":         round(cvar * 100, 3),
        "weights":          {k: round(float(v), 4) for k, v in weights.items()},
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_mean_cvar(
    tickers: list[str] = None,
    lookback: str = DEFAULT_LOOKBACK,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    returns = _fetch_returns(tickers, lookback)
    available = list(returns.columns)
    if len(available) < 2:
        raise RuntimeError(f"Need at least 2 assets, got {available}")

    # Asset-mean target for the constrained version (annualised)
    avg_asset_ann = float(returns.mean().mean() * 252 * 100)

    # Solve unconstrained min-CVaR
    sol_min = mean_cvar_optimal(returns, alpha=alpha, target_return_ann_pct=None)

    # Solve constrained mean-CVaR
    sol_mean = mean_cvar_optimal(
        returns, alpha=alpha, target_return_ann_pct=avg_asset_ann,
    )

    eq_w = equal_weights(returns)

    metrics = {
        "min_cvar":     _portfolio_metrics(returns, sol_min["weights"], alpha),
        "mean_cvar":    _portfolio_metrics(returns, sol_mean["weights"], alpha),
        "equal_weight": _portfolio_metrics(returns, eq_w, alpha),
    }

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers":        available,
        "lookback":       lookback,
        "n_obs":          int(len(returns)),
        "alpha":          alpha,
        "avg_asset_target_ann_pct": round(avg_asset_ann, 3),
        "lp_diagnostics": {
            "min_cvar": {
                "success":       sol_min["success"],
                "var_daily_pct": sol_min.get("var_daily_pct"),
                "cvar_daily_pct":sol_min.get("cvar_daily_pct"),
            },
            "mean_cvar": {
                "success":       sol_mean["success"],
                "var_daily_pct": sol_mean.get("var_daily_pct"),
                "cvar_daily_pct":sol_mean.get("cvar_daily_pct"),
            },
        },
        "metrics":        metrics,
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
    print(f"  MEAN-CVaR LP OPTIMIZER  (Rockafellar-Uryasev)")
    print(SEP)
    print(f"  Universe:   {', '.join(r['tickers'])}")
    print(f"  Lookback:   {r['lookback']}  ({r['n_obs']} obs)")
    print(f"  α:          {r['alpha']:.2f}")
    print(f"  μ-target:   {r['avg_asset_target_ann_pct']:+.2f}% (avg asset mean)")
    print()

    print(f"  WEIGHTS COMPARISON")
    print(f"  {'─' * 58}")
    cols = [("min_cvar", "MinCVaR"), ("mean_cvar", "MeanCVaR"), ("equal_weight", "EqualW")]
    print(f"  {'ticker':<10s}  " + "  ".join(f"{name:>9s}" for _, name in cols))
    for tick in r["tickers"]:
        ws = []
        for k, _ in cols:
            ws.append(r["metrics"][k]["weights"].get(tick, 0))
        print(f"  {tick:<10s}  " + "  ".join(f"{w:>8.2%}" for w in ws))
    print()

    print(f"  PORTFOLIO METRICS (in-sample)")
    print(f"  {'─' * 58}")
    print(f"  {'metric':<22s}  " + "  ".join(f"{n:>9s}" for _, n in cols))
    for fld, label in [
        ("ann_return_pct",   "Ann Return (%)"),
        ("ann_vol_pct",      "Ann Vol (%)"),
        ("sharpe",           "Sharpe"),
        ("max_drawdown_pct", "Max DD (%)"),
        ("var_pct",          f"VaR {int(r['alpha']*100)}% daily"),
        ("cvar_pct",         f"CVaR {int(r['alpha']*100)}% daily"),
    ]:
        print(
            f"  {label:<22s}  " + "  ".join(
                f"{r['metrics'][k][fld]:>9.3f}" for k, _ in cols
            )
        )
    print()

    print(f"  LP DIAGNOSTICS")
    print(f"  {'─' * 58}")
    for k in ["min_cvar", "mean_cvar"]:
        d = r["lp_diagnostics"][k]
        success = "ok" if d["success"] else "FAILED"
        print(
            f"  {k:<14s}  status={success:<6s}  "
            f"VaR={d.get('var_daily_pct', 0):.3f}%  "
            f"CVaR={d.get('cvar_daily_pct', 0):.3f}%  (daily)"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean-CVaR LP Optimizer")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    tlist = [t.strip() for t in args.tickers.split(",") if t.strip()]
    run_mean_cvar(tickers=tlist, lookback=args.lookback, alpha=args.alpha)
