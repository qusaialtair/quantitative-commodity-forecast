#!/usr/bin/env python3
"""
DCC-GARCH Dynamic Conditional Correlations  (Engle 2002)
==========================================================
Two-stage estimation of the time-varying correlation matrix across a multi-
asset universe — pure NumPy / SciPy, no `arch` or `statsmodels` deps.

Stage 1: For each asset run a GARCH(1,1) conditional-variance recursion
         (using the omega/alpha/beta calibrated in monte_carlo.py).
Stage 2: Standardise residuals η_i(t) = r_i(t) / σ_i(t), then estimate the
         DCC parameters (a, b) by maximising the conditional log-likelihood:

             Q(t) = (1 - a - b) Q̄ + a η(t-1) η(t-1)ᵀ + b Q(t-1)
             R(t) = diag(Q(t))^(-½) Q(t) diag(Q(t))^(-½)
             LL  = -½ Σ [ log|R(t)| + η(t)ᵀ R(t)⁻¹ η(t) - η(t)ᵀ η(t) ]

Outputs:
  - Fitted (a, b) and log-likelihood
  - Current conditional correlation matrix
  - Long-run correlation matrix Q̄
  - Per-pair time series stats: current ρ, mean, std, current z-score
  - Crisis-correlation flag — pairs whose current ρ exceeds historical mean
    by more than 2σ are tagged as "STRESSED"

File: data/dcc_garch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "dcc_garch.json"

DEFAULT_TICKERS = ["GC=F", "SI=F", "SPY", "TLT", "DX-Y.NYB"]
DEFAULT_LOOKBACK = "5y"

# GARCH(1,1) — shared params calibrated on gold returns (from monte_carlo.py)
GARCH_OMEGA = 0.000002
GARCH_ALPHA = 0.06
GARCH_BETA = 0.93

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
# Stage 1: per-asset GARCH(1,1) variance recursion
# ---------------------------------------------------------------------------
def garch_variance_series(returns: np.ndarray) -> np.ndarray:
    """GARCH(1,1) with shared params; returns conditional variance per t."""
    n = len(returns)
    long_run = GARCH_OMEGA / max(1.0 - GARCH_ALPHA - GARCH_BETA, 1e-9)
    sigma2 = np.full(n, long_run)
    for t in range(1, n):
        sigma2[t] = (
            GARCH_OMEGA
            + GARCH_ALPHA * returns[t - 1] ** 2
            + GARCH_BETA * sigma2[t - 1]
        )
    return sigma2


def standardize_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Each column scaled by its GARCH conditional vol → η(t)."""
    eta = np.empty_like(returns_df.values)
    for j, col in enumerate(returns_df.columns):
        r = returns_df[col].values
        sigma2 = garch_variance_series(r)
        eta[:, j] = r / np.sqrt(np.maximum(sigma2, 1e-12))
    return pd.DataFrame(eta, index=returns_df.index, columns=returns_df.columns)


# ---------------------------------------------------------------------------
# Stage 2: DCC log-likelihood and fit
# ---------------------------------------------------------------------------
def _dcc_neg_log_likelihood(ab: np.ndarray, eta: np.ndarray, q_bar: np.ndarray) -> float:
    a, b = float(ab[0]), float(ab[1])
    if a < 0 or b < 0 or a + b >= 0.9995:
        return 1e10

    n, k = eta.shape
    Q = q_bar.copy()
    ll = 0.0
    one_ab = 1.0 - a - b

    for t in range(n):
        d = np.sqrt(np.diag(Q))
        d_outer = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = Q / d_outer
        # Numerical floor
        np.fill_diagonal(R, 1.0)

        eta_t = eta[t]
        try:
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0 or not np.isfinite(logdet):
                return 1e10
            R_inv_eta = np.linalg.solve(R, eta_t)
        except np.linalg.LinAlgError:
            return 1e10

        ll -= 0.5 * (logdet + eta_t @ R_inv_eta - eta_t @ eta_t)

        Q = one_ab * q_bar + a * np.outer(eta_t, eta_t) + b * Q

    return -ll


def fit_dcc(eta_df: pd.DataFrame) -> tuple[float, float, float]:
    """Estimate (a, b) by L-BFGS-B; returns (a, b, log_likelihood)."""
    eta = eta_df.values
    # Use corrcoef so Q̄ has unit diagonal even when shared GARCH params
    # imperfectly normalise some assets' residuals.
    q_bar = np.corrcoef(eta.T)
    res = minimize(
        _dcc_neg_log_likelihood,
        x0=np.array([0.05, 0.93]),
        args=(eta, q_bar),
        method="L-BFGS-B",
        bounds=[(1e-4, 0.30), (0.50, 0.999)],
        options={"maxiter": 100, "ftol": 1e-7},
    )
    a, b = float(res.x[0]), float(res.x[1])
    return a, b, float(-res.fun)


# ---------------------------------------------------------------------------
# Build the conditional correlation series
# ---------------------------------------------------------------------------
def correlation_series(
    eta_df: pd.DataFrame, a: float, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - R_history : array of shape (n, k, k) — correlation matrices per t
      - q_bar     : (k, k) unconditional correlation
    """
    eta = eta_df.values
    # Use corrcoef so Q̄ has unit diagonal even when shared GARCH params
    # imperfectly normalise some assets' residuals.
    q_bar = np.corrcoef(eta.T)
    n, k = eta.shape
    one_ab = 1.0 - a - b
    Q = q_bar.copy()
    R_history = np.empty((n, k, k))

    for t in range(n):
        d = np.sqrt(np.diag(Q))
        R_history[t] = Q / np.outer(d, d)
        np.fill_diagonal(R_history[t], 1.0)
        eta_t = eta[t]
        Q = one_ab * q_bar + a * np.outer(eta_t, eta_t) + b * Q

    return R_history, q_bar


# ---------------------------------------------------------------------------
# Per-pair diagnostics + stress flags
# ---------------------------------------------------------------------------
def _pair_diagnostics(R_hist: np.ndarray, tickers: list[str]) -> list[dict]:
    n, k, _ = R_hist.shape
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            series = R_hist[:, i, j]
            cur = float(series[-1])
            mean = float(series.mean())
            std = float(series.std())
            z = (cur - mean) / std if std > 1e-9 else 0.0
            stressed = abs(z) > 2.0
            out.append({
                "pair":          f"{tickers[i]}__{tickers[j]}",
                "asset_a":       tickers[i],
                "asset_b":       tickers[j],
                "current_corr":  round(cur, 4),
                "mean_corr":     round(mean, 4),
                "std_corr":      round(std, 4),
                "min_corr":      round(float(series.min()), 4),
                "max_corr":      round(float(series.max()), 4),
                "z_score":       round(z, 3),
                "stressed":      bool(stressed),
            })
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_dcc_garch(
    tickers: list[str] = None,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    returns = _fetch_returns(tickers, lookback)
    available = list(returns.columns)
    if len(available) < 2:
        raise RuntimeError(f"Need at least 2 assets, got {available}")

    eta = standardize_returns(returns)
    a, b, ll = fit_dcc(eta)

    R_hist, q_bar = correlation_series(eta, a, b)
    R_now = R_hist[-1]
    pair_diag = _pair_diagnostics(R_hist, available)
    stressed = [p for p in pair_diag if p["stressed"]]

    # Convert to readable correlation matrices
    def _matrix_to_dict(M: np.ndarray) -> dict:
        return {
            available[i]: {
                available[j]: round(float(M[i, j]), 4) for j in range(len(available))
            } for i in range(len(available))
        }

    avg_corr_now = float(
        (R_now[np.triu_indices_from(R_now, k=1)]).mean()
    )
    avg_corr_long_run = float(
        (q_bar[np.triu_indices_from(q_bar, k=1)]).mean()
    )

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers":          available,
        "lookback":         lookback,
        "n_obs":            int(len(returns)),
        "garch_params": {
            "omega": GARCH_OMEGA,
            "alpha": GARCH_ALPHA,
            "beta":  GARCH_BETA,
        },
        "dcc_params": {
            "a":              round(a, 5),
            "b":              round(b, 5),
            "a_plus_b":       round(a + b, 5),
            "log_likelihood": round(ll, 3),
        },
        "current_correlation":   _matrix_to_dict(R_now),
        "long_run_correlation":  _matrix_to_dict(q_bar),
        "avg_pairwise_corr_now":     round(avg_corr_now, 4),
        "avg_pairwise_corr_long_run":round(avg_corr_long_run, 4),
        "pairs":           pair_diag,
        "stressed_pairs":  [p["pair"] for p in stressed],
        "n_stressed":      len(stressed),
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
    print(f"  DCC-GARCH DYNAMIC CONDITIONAL CORRELATIONS")
    print(SEP)
    print(f"  Universe:   {', '.join(r['tickers'])}")
    print(f"  Lookback:   {r['lookback']}  ({r['n_obs']} obs)")
    p = r["dcc_params"]
    print(f"  DCC fit:    a={p['a']:.4f}  b={p['b']:.4f}  "
          f"a+b={p['a_plus_b']:.4f}  LL={p['log_likelihood']:.1f}")
    print()

    tk = r["tickers"]
    print(f"  CURRENT CORRELATION (last day)")
    print(f"  {'─' * 58}")
    hdr = "  " + " " * 12 + "  ".join(f"{t[:8]:>8s}" for t in tk)
    print(hdr)
    cur = r["current_correlation"]
    for ti in tk:
        row = "  " + f"{ti[:10]:<10s}  " + "  ".join(
            f"{cur[ti][tj]:>+8.3f}" for tj in tk
        )
        print(row)
    print()

    print(f"  LONG-RUN CORRELATION  (Q̄)")
    print(f"  {'─' * 58}")
    print(hdr)
    lr = r["long_run_correlation"]
    for ti in tk:
        row = "  " + f"{ti[:10]:<10s}  " + "  ".join(
            f"{lr[ti][tj]:>+8.3f}" for tj in tk
        )
        print(row)
    print()

    print(f"  PER-PAIR DIAGNOSTICS")
    print(f"  {'─' * 58}")
    print(
        f"  {'pair':<24s}  {'cur':>7s}  {'mean':>7s}  "
        f"{'min':>7s}  {'max':>7s}  {'z':>6s}"
    )
    for pr in r["pairs"]:
        flag = "  STRESS" if pr["stressed"] else ""
        print(
            f"  {pr['pair'][:23]:<24s}  "
            f"{pr['current_corr']:>+7.3f}  {pr['mean_corr']:>+7.3f}  "
            f"{pr['min_corr']:>+7.3f}  {pr['max_corr']:>+7.3f}  "
            f"{pr['z_score']:>+6.2f}{flag}"
        )
    print()

    print(f"  AVG PAIRWISE CORR — current: {r['avg_pairwise_corr_now']:+.3f}  "
          f"long-run: {r['avg_pairwise_corr_long_run']:+.3f}")
    if r["n_stressed"]:
        print(f"  ⚠ STRESSED PAIRS: {', '.join(r['stressed_pairs'])}")
    else:
        print(f"  All pairs within ±2σ of long-run correlation")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCC-GARCH Dynamic Correlations")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    tlist = [t.strip() for t in args.tickers.split(",") if t.strip()]
    run_dcc_garch(tickers=tlist, lookback=args.lookback)
