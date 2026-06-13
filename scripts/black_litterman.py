#!/usr/bin/env python3
"""
Black-Litterman Bayesian Portfolio
====================================
Combines a CAPM equilibrium prior with investor views to produce
posterior expected returns and an optimal portfolio.

Steps:
  1. Equilibrium implied returns:       Π = δ Σ w_mkt
  2. Build view picking matrix P, view returns Q, view-confidence Ω
  3. Posterior expected returns:        μ = M (τΣ)⁻¹ Π + M Pᵀ Ω⁻¹ Q
                                        M = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
  4. Optimal weights:                   w = (δΣ)⁻¹ μ      then normalise

View sources wired into this engine:
  - Macro oracle (latest score from oracle_engine; bullish → positive gold view)
  - LSTM directional signal (predicted 5d move on gold)
  - Cointegration / mean-reversion (tilts silver via gold-silver pair)
  - Equity momentum (21d SPY momentum becomes SPY view)
  - Bonds carry (negative DXY momentum becomes TLT view, very low confidence)

Each view has a confidence in [0, 1] which scales its Ω diagonal element.

Output: data/black_litterman.json
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
OUTPUT_FILE = DATA_DIR / "black_litterman.json"

DEFAULT_TICKERS = ["GC=F", "SI=F", "SPY", "TLT", "DX-Y.NYB"]
DEFAULT_LOOKBACK = "5y"

# Approximate ETF AUM proxy weights for the equilibrium prior
DEFAULT_MARKET_WEIGHTS = {
    "GC=F":     0.13,   # GLD ~ 12.5%
    "SI=F":     0.02,   # SLV ~ 2%
    "SPY":      0.78,   # SPY ~ 78%
    "TLT":      0.06,   # TLT ~ 6%
    "DX-Y.NYB": 0.01,   # UUP ~ 1%
}

DELTA = 2.5    # risk aversion (Sharpe-implied)
TAU = 0.025    # uncertainty scalar on the prior
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
# Views — sourced from live signals if available, else heuristic
# ---------------------------------------------------------------------------
def _load_oracle_score(ticker: str) -> float | None:
    try:
        from scripts.oracle_engine import get_latest_scores
        return get_latest_scores([ticker]).get(ticker)
    except Exception:
        return None


def _lstm_view() -> tuple[float, float]:
    """Return (annualised view, confidence) from latest LSTM prediction."""
    try:
        from models.lstm_predictor import predict_next
        result = predict_next()
        if "error" in result:
            return 0.0, 0.0
        pct_5d = float(result.get("pct_move", 0.0))
        # Annualise the 5d implied move
        annualised = pct_5d * (252.0 / 5.0) / 100.0
        # Confidence proportional to absolute size, capped at 0.6
        conf = min(0.6, abs(pct_5d) / 5.0)
        return float(annualised), float(conf)
    except Exception:
        return 0.0, 0.0


def _build_views(
    tickers: list[str],
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build (P, Q, confidences, descriptions). Each row of P is one view:
    a vector of weights summing to ±1 indicating which assets the view
    references and how, with Q[i] giving the view's annualised return.
    """
    idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    rows = []
    qs = []
    cs = []
    descs = []

    # View 1: Macro oracle on gold
    g_score = _load_oracle_score("GC=F")
    if g_score is not None and "GC=F" in idx:
        # Score in [0..1]; map to ±10% annualised view
        gold_view = (float(g_score) - 0.5) * 0.20
        gold_conf = min(0.7, abs(float(g_score) - 0.5) * 1.4)
        v = np.zeros(n); v[idx["GC=F"]] = 1.0
        rows.append(v); qs.append(gold_view); cs.append(gold_conf)
        descs.append(f"GC=F oracle: {gold_view*100:+.1f}% (score={g_score:.2f})")

    # View 2: LSTM 5d on gold
    lstm_v, lstm_c = _lstm_view()
    if abs(lstm_v) > 1e-4 and "GC=F" in idx:
        v = np.zeros(n); v[idx["GC=F"]] = 1.0
        rows.append(v); qs.append(lstm_v); cs.append(lstm_c)
        descs.append(f"GC=F LSTM 5d→ann: {lstm_v*100:+.1f}% (conf={lstm_c:.2f})")

    # View 3: Mean-reversion gold-silver pair (silver tilts toward gold)
    if "GC=F" in idx and "SI=F" in idx and len(returns) >= 252:
        recent = returns.tail(63)
        gs_diff = float((recent["GC=F"] - recent["SI=F"]).sum())
        # If gold beat silver substantially, expect silver to catch up
        sv_view = -gs_diff * 0.5  # mean-reversion factor
        sv_view = float(np.clip(sv_view * (252.0 / 63.0), -0.15, 0.15))
        if abs(sv_view) > 0.005:
            v = np.zeros(n); v[idx["SI=F"]] = 1.0; v[idx["GC=F"]] = -0.5
            rows.append(v); qs.append(sv_view); cs.append(0.35)
            descs.append(f"SI-GC pair MR: {sv_view*100:+.1f}%")

    # View 4: Equity momentum (21d SPY)
    if "SPY" in idx and len(returns) >= 21:
        spy_mom = float(returns["SPY"].tail(21).sum() * (252.0 / 21.0))
        spy_mom = float(np.clip(spy_mom, -0.15, 0.20))
        if abs(spy_mom) > 0.01:
            v = np.zeros(n); v[idx["SPY"]] = 1.0
            rows.append(v); qs.append(spy_mom); cs.append(0.40)
            descs.append(f"SPY 21d-mom: {spy_mom*100:+.1f}%")

    # View 5: Duration carry vs DXY
    if "TLT" in idx and "DX-Y.NYB" in idx and len(returns) >= 21:
        dxy_mom = float(returns["DX-Y.NYB"].tail(21).sum() * (252.0 / 21.0))
        # Strong dollar usually pressures TLT (negative correlation in mid-2020s)
        tlt_view = -dxy_mom * 0.4
        tlt_view = float(np.clip(tlt_view, -0.10, 0.10))
        if abs(tlt_view) > 0.005:
            v = np.zeros(n); v[idx["TLT"]] = 1.0
            rows.append(v); qs.append(tlt_view); cs.append(0.25)
            descs.append(f"TLT carry from DXY mom: {tlt_view*100:+.1f}%")

    if not rows:
        # No views → return zero matrix; posterior will equal prior
        return np.zeros((0, n)), np.zeros(0), np.zeros(0), []

    P = np.vstack(rows)
    Q = np.array(qs, dtype=float)
    C = np.array(cs, dtype=float)
    return P, Q, C, descs


# ---------------------------------------------------------------------------
# Black-Litterman core
# ---------------------------------------------------------------------------
def black_litterman(
    cov_ann: np.ndarray,
    market_weights: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    confidences: np.ndarray,
    delta: float = DELTA,
    tau: float = TAU,
) -> dict:
    n = cov_ann.shape[0]

    # Equilibrium implied returns
    pi = delta * (cov_ann @ market_weights)

    if P.shape[0] == 0:
        # No views — posterior is just the prior
        posterior = pi.copy()
        omega = np.zeros((0, 0))
    else:
        k = P.shape[0]
        # Idzorek-style omega: ω_i = τ Pᵢ Σ Pᵢᵀ / c_i
        omega = np.zeros((k, k))
        for i in range(k):
            var_i = float(P[i] @ cov_ann @ P[i].T)
            c = float(max(confidences[i], 1e-3))
            omega[i, i] = tau * var_i / c

        tau_sig_inv = np.linalg.inv(tau * cov_ann)
        omega_inv = np.linalg.inv(omega)
        M_inv = tau_sig_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(M_inv)
        posterior = M @ (tau_sig_inv @ pi + P.T @ omega_inv @ Q)

    # Optimal mean-variance weights
    raw_w = np.linalg.solve(delta * cov_ann, posterior)
    # Normalise so weights sum to 1; clip tiny negative weights to 0 first to keep long-only
    w_clip = np.clip(raw_w, 0.0, None)
    if w_clip.sum() > 1e-6:
        normalized = w_clip / w_clip.sum()
    else:
        normalized = np.ones(n) / n

    return {
        "equilibrium_returns": pi,
        "posterior_returns":   posterior,
        "raw_weights":         raw_w,
        "normalized_weights":  normalized,
        "omega":               omega,
    }


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------
def _portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    w = pd.Series(weights, index=returns.columns)
    port_ret = returns @ w
    ann_ret = float(port_ret.mean() * 252)
    ann_vol = float(port_ret.std() * SQ252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = (1 + port_ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    return {
        "ann_return_pct":  round(ann_ret * 100, 3),
        "ann_vol_pct":     round(ann_vol * 100, 3),
        "sharpe":          round(sharpe, 3),
        "max_drawdown_pct":round(max_dd * 100, 3),
        "weights":         {k: round(float(v), 4) for k, v in w.items()},
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_black_litterman(
    tickers: list[str] = None,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    returns = _fetch_returns(tickers, lookback)
    available = list(returns.columns)
    if len(available) < 2:
        raise RuntimeError(f"Need at least 2 assets, got {available}")

    # Annualised covariance
    cov_ann = (returns.cov() * 252).values

    # Market weights (renormalised to match available tickers)
    raw_mw = np.array(
        [DEFAULT_MARKET_WEIGHTS.get(t, 1.0 / len(available)) for t in available],
        dtype=float,
    )
    raw_mw = raw_mw / raw_mw.sum()

    P, Q, C, descs = _build_views(available, returns)

    bl = black_litterman(cov_ann, raw_mw, P, Q, C)
    posterior_w = bl["normalized_weights"]
    market_w = raw_mw

    # Equal-weight benchmark
    eq_w = np.ones(len(available)) / len(available)

    # Per-asset metrics
    asset_table = []
    for i, tick in enumerate(available):
        asset_table.append({
            "ticker":               tick,
            "market_weight":        round(float(market_w[i]), 4),
            "equilibrium_ret_pct":  round(float(bl["equilibrium_returns"][i]) * 100, 3),
            "posterior_ret_pct":    round(float(bl["posterior_returns"][i]) * 100, 3),
            "view_tilt_pct":        round(
                float(bl["posterior_returns"][i] - bl["equilibrium_returns"][i]) * 100, 3
            ),
            "bl_weight":            round(float(posterior_w[i]), 4),
        })

    metrics = {
        "black_litterman": _portfolio_metrics(returns, posterior_w),
        "market_weights":  _portfolio_metrics(returns, market_w),
        "equal_weight":    _portfolio_metrics(returns, eq_w),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers":      available,
        "lookback":     lookback,
        "n_obs":        int(len(returns)),
        "n_views":      int(P.shape[0]),
        "view_descriptions": descs,
        "delta":        DELTA,
        "tau":          TAU,
        "asset_table":  asset_table,
        "metrics":      metrics,
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
    print(f"  BLACK-LITTERMAN BAYESIAN PORTFOLIO")
    print(SEP)
    print(f"  Universe:   {', '.join(r['tickers'])}")
    print(f"  Views:      {r['n_views']}")
    print(f"  δ={r['delta']}  τ={r['tau']}")
    print()

    if r["view_descriptions"]:
        print(f"  ACTIVE VIEWS")
        print(f"  {'─' * 58}")
        for d in r["view_descriptions"]:
            print(f"    • {d}")
        print()

    print(f"  ASSET TABLE")
    print(
        f"  {'ticker':<10s}  {'mkt-w':>8s}  {'Π %':>8s}  "
        f"{'μ %':>8s}  {'tilt %':>8s}  {'BL w':>8s}"
    )
    print(f"  {'─' * 58}")
    for row in r["asset_table"]:
        print(
            f"  {row['ticker']:<10s}  {row['market_weight']:>8.2%}  "
            f"{row['equilibrium_ret_pct']:>+8.2f}  "
            f"{row['posterior_ret_pct']:>+8.2f}  "
            f"{row['view_tilt_pct']:>+8.2f}  "
            f"{row['bl_weight']:>8.2%}"
        )
    print()

    print(f"  PORTFOLIO METRICS (in-sample backtest)")
    print(f"  {'─' * 58}")
    cols = [("black_litterman", "BL"), ("market_weights", "Mkt"), ("equal_weight", "EqW")]
    print(f"  {'metric':<22s}  " + "  ".join(f"{name:>8s}" for _, name in cols))
    for fld, label in [
        ("ann_return_pct",   "Ann Return (%)"),
        ("ann_vol_pct",      "Ann Vol (%)"),
        ("sharpe",           "Sharpe"),
        ("max_drawdown_pct", "Max DD (%)"),
    ]:
        row = f"  {label:<22s}  " + "  ".join(
            f"{r['metrics'][k][fld]:>8.3f}" for k, _ in cols
        )
        print(row)
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-Litterman Bayesian Portfolio")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    tlist = [t.strip() for t in args.tickers.split(",") if t.strip()]
    run_black_litterman(tickers=tlist, lookback=args.lookback)
