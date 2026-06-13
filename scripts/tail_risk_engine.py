#!/usr/bin/env python3
"""
scripts/tail_risk_engine.py
============================
Institutional-grade tail risk + factor attribution engine.

Three deliverables on every run:

  1. EVT peaks-over-threshold (POT) CVaR / Expected Shortfall
     Fits a generalized Pareto distribution to losses beyond the 90th-percentile
     threshold (Pickands-Balkema-de Haan theorem). Produces 95 / 99 / 99.5
     Expected Shortfall — the BCBS / FRTB standard. Reports historical and
     parametric Gaussian estimates side by side so the tail-fatness premium
     versus a Gaussian assumption is explicit.

  2. Multi-factor return attribution (BlackRock Aladdin-style)
     OLS regression of gold returns on five systematic factors:
         DXY        — US Dollar Index (dominant inverse driver)
         TLT        — long-duration Treasuries (real-yield proxy)
         VIX        — risk-off / safe-haven premium
         HG/GC      — copper-gold ratio (growth proxy)
         MOM21      — 21-day price momentum (carry trend)
     Reports betas with t-statistics, R-squared, residual alpha (annualised),
     factor contributions to today's return, and information ratio.

  3. Tail-aware Kelly fraction
     Standard Kelly assumes a Gaussian return distribution. We apply an
     EVT haircut: f* = f_kelly * (1 - tail_premium), where tail_premium is
     the EVT-CVaR-99 / Gaussian-CVaR-99 ratio capped at 0.5.

Outputs:
    data/tail_risk_engine.json

Usage:
    python3 scripts/tail_risk_engine.py
    python3 scripts/tail_risk_engine.py --ticker GC=F --lookback 1260
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from scipy import stats
    from scipy.stats import genpareto, norm, t as student_t
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "tail_risk_engine.json"

LINE_W = 76
SEP = "━" * LINE_W

# Factor universe — yfinance tickers and human-readable labels
FACTOR_UNIVERSE = [
    ("DX-Y.NYB", "DXY",   "US Dollar Index"),
    ("TLT",      "TLT",   "20Y Treasuries (real-yield proxy)"),
    ("^VIX",     "VIX",   "Volatility Index (risk-off)"),
    ("HG=F",     "COPPER","Copper futures (growth proxy)"),
    # MOM21 is constructed from the target itself, not fetched
]

CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]


# ── Data layer ────────────────────────────────────────────────────────────────

def _fetch_panel(ticker: str, lookback: int) -> pd.DataFrame:
    """Fetch ticker + factor panel as a single aligned DataFrame of returns."""
    if yf is None:
        raise ImportError("yfinance is required for tail_risk_engine.py")

    # Pull a generous buffer so the lookback survives weekends/holidays
    period_days = int(lookback * 1.55) + 30
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=period_days)

    tickers = [ticker] + [f[0] for f in FACTOR_UNIVERSE]
    closes: dict[str, pd.Series] = {}

    for t in tickers:
        try:
            raw = yf.download(
                t, start=start, end=end + pd.Timedelta(days=1),
                progress=False, auto_adjust=True, threads=False,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if "Close" not in raw.columns:
                continue
            closes[t] = raw["Close"].dropna()
        except Exception:
            continue

    if ticker not in closes or len(closes[ticker]) < 60:
        raise RuntimeError(f"Insufficient data for {ticker}")

    panel = pd.concat(closes, axis=1).ffill().dropna(how="all")
    return panel


def _build_returns(panel: pd.DataFrame, ticker: str, lookback: int) -> pd.DataFrame:
    """Convert close prices into aligned daily log-returns + MOM21 carry factor."""
    rets = np.log(panel / panel.shift(1)).dropna(how="all")

    target = rets[ticker]
    # 21-day momentum: mean of last 21 daily log-returns, lagged 1 day so it is
    # known at the start of the period it predicts (no look-ahead).
    mom21 = target.rolling(21).mean().shift(1)
    rets = rets.assign(MOM21=mom21)

    rets = rets.tail(lookback).dropna()
    return rets


# ── EVT block: peaks-over-threshold CVaR ──────────────────────────────────────

@dataclass
class TailRiskMetrics:
    method: str
    var_95_pct: float
    cvar_95_pct: float
    var_99_pct: float
    cvar_99_pct: float
    var_995_pct: float
    cvar_995_pct: float


def _historical_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR/CVaR from the loss distribution."""
    if len(losses) == 0:
        return 0.0, 0.0
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def _gaussian_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Closed-form VaR/CVaR under a Gaussian assumption."""
    mu = float(losses.mean())
    sigma = float(losses.std(ddof=1))
    z = norm.ppf(alpha)
    var = mu + sigma * z
    # CVaR closed form: μ + σ * φ(z) / (1-α)
    cvar = mu + sigma * norm.pdf(z) / (1.0 - alpha)
    return float(var), float(cvar)


def _evt_pot_cvar(losses: np.ndarray, alpha: float, threshold_q: float = 0.90) -> tuple[float, float, dict]:
    """
    Generalized Pareto (peaks-over-threshold) CVaR.

    Pickands-Balkema-de Haan: for high enough threshold u, the conditional
    excess distribution F(x|X>u) converges to a generalized Pareto GPD(ξ,σ).

    For confidence α and tail ratio Nu/N (fraction of obs above u):
        VaR_α  = u + (σ/ξ) * [((1-α) * N/Nu)^(-ξ) - 1]
        CVaR_α = (VaR_α + σ - ξ*u) / (1-ξ)            ξ < 1
    """
    if len(losses) < 100:
        return 0.0, 0.0, {"fit_ok": False, "reason": "insufficient sample"}

    u = float(np.quantile(losses, threshold_q))
    excesses = losses[losses > u] - u
    if len(excesses) < 30:
        return 0.0, 0.0, {"fit_ok": False, "reason": "too few exceedances"}

    try:
        # floc=0 — anchor location at the threshold; scipy returns (shape, loc, scale)
        shape, _loc, scale = genpareto.fit(excesses, floc=0.0)
    except Exception as exc:
        return 0.0, 0.0, {"fit_ok": False, "reason": f"GPD fit failed: {exc}"}

    n = len(losses)
    nu = len(excesses)
    tail_prob = 1.0 - alpha       # P(X > VaR_α)
    ratio = (tail_prob * n / nu)

    if shape != 0:
        var = u + (scale / shape) * (ratio ** (-shape) - 1.0)
    else:
        var = u + scale * (-np.log(ratio))

    if shape < 1.0:
        cvar = (var + scale - shape * u) / (1.0 - shape)
    else:
        # Heavy-tailed (ξ ≥ 1) — infinite mean. Fall back to historical.
        cvar = float(losses[losses >= var].mean()) if (losses >= var).any() else var

    fit_info = {
        "fit_ok": True,
        "threshold_u": round(u, 6),
        "n_exceedances": int(nu),
        "shape_xi": round(float(shape), 4),
        "scale_sigma": round(float(scale), 6),
        "tail_index_interpretation": _xi_interpretation(float(shape)),
    }
    return float(var), float(cvar), fit_info


def _xi_interpretation(xi: float) -> str:
    if xi > 0.5:
        return "very heavy tail (infinite variance)"
    if xi > 0.25:
        return "heavy tail (Pareto-like)"
    if xi > 0:
        return "moderately heavy tail"
    if xi > -0.25:
        return "thin tail (near-Gaussian)"
    return "bounded tail (short)"


def compute_tail_risk(returns: np.ndarray) -> dict:
    """Run all three CVaR methods and return a comparable dict."""
    losses = -returns  # losses are positive numbers

    methods = {}
    for label, fn in [
        ("historical", lambda a: _historical_cvar(losses, a)),
        ("gaussian",   lambda a: _gaussian_cvar(losses, a)),
    ]:
        m: dict = {}
        for a in CONFIDENCE_LEVELS:
            var, cvar = fn(a)
            m[f"var_{int(a*1000)}"] = round(var * 100, 3)    # in %
            m[f"cvar_{int(a*1000)}"] = round(cvar * 100, 3)
        methods[label] = m

    evt: dict = {}
    evt_diag = {}
    for a in CONFIDENCE_LEVELS:
        var, cvar, info = _evt_pot_cvar(losses, a)
        evt[f"var_{int(a*1000)}"] = round(var * 100, 3)
        evt[f"cvar_{int(a*1000)}"] = round(cvar * 100, 3)
        if a == 0.99:
            evt_diag = info
    methods["evt_pot"] = evt

    # Tail-fatness premium: how much worse is reality vs. a Gaussian assumption
    g99 = methods["gaussian"]["cvar_990"]
    e99 = methods["evt_pot"]["cvar_990"]
    if g99 > 0:
        tail_premium = max(0.0, (e99 - g99) / g99)
    else:
        tail_premium = 0.0

    return {
        "n_observations": int(len(returns)),
        "methods": methods,
        "evt_diagnostics": evt_diag,
        "tail_fatness_premium_pct": round(tail_premium * 100, 2),
    }


# ── Factor attribution ────────────────────────────────────────────────────────

def _ols_with_tstats(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS regression with intercept; returns betas, t-stats, R², residuals."""
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n), X])     # intercept first
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ beta
    resid = y - y_hat

    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors via (X'X)^-1 * sigma^2
    dof = max(n - (k + 1), 1)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k + 1, np.nan)

    t_stats = beta / np.where(se > 0, se, np.nan)
    # Two-sided p-values (Student-t with dof degrees of freedom)
    p_values = 2.0 * (1.0 - student_t.cdf(np.abs(t_stats), df=dof))

    return {
        "beta": beta,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": float(r_sq),
        "adj_r_squared": float(1.0 - (1.0 - r_sq) * (n - 1) / dof),
        "residuals": resid,
        "n_obs": n,
        "dof": dof,
    }


def compute_factor_attribution(rets: pd.DataFrame, ticker: str) -> dict:
    """Regress target returns on factor returns, report institutional summary."""
    factor_cols = []
    factor_meta = []
    for tkr, label, desc in FACTOR_UNIVERSE:
        if tkr in rets.columns:
            factor_cols.append(tkr)
            factor_meta.append((label, desc))
    if "MOM21" in rets.columns:
        factor_cols.append("MOM21")
        factor_meta.append(("MOM21", "21-day price momentum (carry trend)"))

    df = rets[[ticker] + factor_cols].dropna()
    if len(df) < 60:
        return {"error": "insufficient data for factor regression"}

    y = df[ticker].values
    X = df[factor_cols].values

    fit = _ols_with_tstats(X, y)

    # Build per-factor report
    intercept = float(fit["beta"][0])
    intercept_t = float(fit["t_stats"][0])
    intercept_p = float(fit["p_values"][0])

    factors = []
    for i, (fcol, (label, desc)) in enumerate(zip(factor_cols, factor_meta)):
        b = float(fit["beta"][i + 1])
        t = float(fit["t_stats"][i + 1])
        p = float(fit["p_values"][i + 1])
        contrib_today = b * float(df[fcol].iloc[-1])
        factors.append({
            "factor": label,
            "description": desc,
            "beta": round(b, 4),
            "t_stat": round(t, 3),
            "p_value": round(p, 4),
            "significant": bool(p < 0.05),
            "contribution_today_bps": round(contrib_today * 1e4, 2),
        })

    # Annualised residual alpha (intercept * 252)
    alpha_ann_pct = intercept * 252 * 100

    # Information ratio: alpha / residual std (annualised)
    resid = fit["residuals"]
    info_ratio = (intercept * 252) / (resid.std(ddof=1) * np.sqrt(252)) if resid.std() > 0 else 0.0

    # Today's decomposition
    today_total = float(df[ticker].iloc[-1])
    today_beta_contrib = sum(f["contribution_today_bps"] / 1e4 for f in factors)
    today_alpha = today_total - today_beta_contrib

    return {
        "n_observations": int(fit["n_obs"]),
        "r_squared": round(fit["r_squared"], 4),
        "adj_r_squared": round(fit["adj_r_squared"], 4),
        "alpha_daily_bps": round(intercept * 1e4, 3),
        "alpha_annualised_pct": round(alpha_ann_pct, 3),
        "alpha_t_stat": round(intercept_t, 3),
        "alpha_p_value": round(intercept_p, 4),
        "alpha_significant": bool(intercept_p < 0.05),
        "information_ratio": round(float(info_ratio), 3),
        "factors": factors,
        "today_decomposition": {
            "total_return_bps": round(today_total * 1e4, 2),
            "factor_explained_bps": round(today_beta_contrib * 1e4, 2),
            "residual_alpha_bps": round(today_alpha * 1e4, 2),
        },
    }


# ── Tail-aware Kelly haircut ──────────────────────────────────────────────────

def compute_tail_aware_kelly(returns: np.ndarray, tail: dict) -> dict:
    """
    Apply an EVT haircut to the Gaussian Kelly fraction.

      f_kelly = μ / σ²   (Gaussian)
      f*      = f_kelly * (1 - tail_premium)   capped at 50% reduction
    """
    mu = float(returns.mean())
    var = float(returns.var(ddof=1))
    if var <= 0:
        return {"f_kelly_gaussian": 0.0, "f_tail_aware": 0.0, "haircut_pct": 0.0}

    f_kelly = mu / var

    tail_premium = tail.get("tail_fatness_premium_pct", 0.0) / 100.0
    haircut = min(tail_premium, 0.5)
    f_adjusted = f_kelly * (1.0 - haircut)

    return {
        "f_kelly_gaussian": round(f_kelly, 4),
        "f_tail_aware": round(f_adjusted, 4),
        "haircut_pct": round(haircut * 100, 2),
        "interpretation": (
            f"Gaussian Kelly suggests {f_kelly*100:.1f}% deployment; "
            f"after EVT haircut for fat tails: {f_adjusted*100:.1f}%"
        ),
    }


# ── Performance metrics for completeness ──────────────────────────────────────

def compute_performance_metrics(returns: np.ndarray) -> dict:
    """Sortino, Calmar, downside dev, drawdown stats — institutional staples."""
    if len(returns) == 0:
        return {}
    daily_mu = float(returns.mean())
    daily_sigma = float(returns.std(ddof=1))
    ann_return = daily_mu * 252
    ann_vol = daily_sigma * np.sqrt(252)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = (ann_return / downside_dev) if downside_dev > 0 else 0.0

    # Max drawdown on the cumulative log-return path
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Skew, kurtosis
    skew = float(stats.skew(returns)) if SCIPY_OK else 0.0
    kurt = float(stats.kurtosis(returns)) if SCIPY_OK else 0.0    # excess kurtosis

    return {
        "annualised_return_pct": round(ann_return * 100, 2),
        "annualised_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "fat_tail_warning": bool(kurt > 3.0),
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_tail_risk_engine(ticker: str = "GC=F", lookback: int = 1260) -> dict:
    """Pull data, run all three blocks, write JSON, return the dict."""
    if not SCIPY_OK:
        raise ImportError("scipy is required for tail_risk_engine.py")

    panel = _fetch_panel(ticker, lookback)
    rets = _build_returns(panel, ticker, lookback)
    target = rets[ticker].values

    tail = compute_tail_risk(target)
    factors = compute_factor_attribution(rets, ticker)
    kelly = compute_tail_aware_kelly(target, tail)
    perf = compute_performance_metrics(target)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "lookback_days": int(lookback),
        "actual_observations": int(len(target)),
        "performance": perf,
        "tail_risk": tail,
        "factor_attribution": factors,
        "tail_aware_kelly": kelly,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    _print_report(output)
    return output


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(out: dict) -> None:
    perf = out["performance"]
    tail = out["tail_risk"]
    fac = out["factor_attribution"]
    kel = out["tail_aware_kelly"]

    print(f"\n{SEP}")
    print(f"  TAIL RISK + FACTOR ATTRIBUTION ENGINE")
    print(f"  Ticker: {out['ticker']}  |  Lookback: {out['actual_observations']} obs")
    print(SEP)

    print(f"  PERFORMANCE STATISTICS")
    print(f"  {'─' * 60}")
    print(f"  Annualised Return:    {perf['annualised_return_pct']:+7.2f}%")
    print(f"  Annualised Vol:       {perf['annualised_vol_pct']:7.2f}%")
    print(f"  Sharpe Ratio:         {perf['sharpe_ratio']:7.3f}")
    print(f"  Sortino Ratio:        {perf['sortino_ratio']:7.3f}")
    print(f"  Calmar Ratio:         {perf['calmar_ratio']:7.3f}")
    print(f"  Max Drawdown:         {perf['max_drawdown_pct']:+7.2f}%")
    print(f"  Skewness:             {perf['skewness']:+7.3f}")
    print(f"  Excess Kurtosis:      {perf['excess_kurtosis']:+7.3f}"
          f"  {'[FAT TAILS]' if perf['fat_tail_warning'] else ''}")

    print()
    print(f"  TAIL RISK — VaR / CVaR comparison (% of capital)")
    print(f"  {'─' * 60}")
    print(f"  {'Method':<14s} {'VaR 95':>8s} {'CVaR 95':>8s} "
          f"{'VaR 99':>8s} {'CVaR 99':>8s} {'CVaR 99.5':>10s}")
    for label, key in [("Historical", "historical"), ("Gaussian", "gaussian"), ("EVT-POT", "evt_pot")]:
        m = tail["methods"].get(key, {})
        print(f"  {label:<14s} "
              f"{m.get('var_950', 0):7.2f}% {m.get('cvar_950', 0):7.2f}% "
              f"{m.get('var_990', 0):7.2f}% {m.get('cvar_990', 0):7.2f}% "
              f"{m.get('cvar_995', 0):9.2f}%")

    diag = tail.get("evt_diagnostics", {})
    if diag.get("fit_ok"):
        print()
        print(f"  EVT GPD Fit:  ξ = {diag['shape_xi']:+.4f}  σ = {diag['scale_sigma']:.4f}  "
              f"u = {diag['threshold_u']*100:+.2f}%  N_u = {diag['n_exceedances']}")
        print(f"  Tail shape:   {diag['tail_index_interpretation']}")
    print(f"  Tail-fatness premium (EVT vs Gaussian, CVaR 99): "
          f"{tail['tail_fatness_premium_pct']:+.2f}%")

    print()
    print(f"  FACTOR ATTRIBUTION  (R² = {fac.get('r_squared', 0):.3f}, "
          f"alpha t = {fac.get('alpha_t_stat', 0):+.2f})")
    print(f"  {'─' * 60}")
    print(f"  {'Factor':<10s} {'Beta':>8s} {'t-stat':>8s} {'p-value':>8s} "
          f"{'Today (bps)':>12s}  Sig")
    for f in fac.get("factors", []):
        sig_marker = "***" if f["p_value"] < 0.01 else ("**" if f["p_value"] < 0.05 else "")
        print(f"  {f['factor']:<10s} {f['beta']:+8.4f} {f['t_stat']:+8.2f} "
              f"{f['p_value']:8.4f} {f['contribution_today_bps']:+11.2f}   {sig_marker}")
    print()
    print(f"  Annualised Alpha:     {fac.get('alpha_annualised_pct', 0):+.3f}%   "
          f"(p = {fac.get('alpha_p_value', 1):.4f})")
    print(f"  Information Ratio:    {fac.get('information_ratio', 0):+.3f}")
    td = fac.get("today_decomposition", {})
    print(f"  Today: total {td.get('total_return_bps', 0):+.1f} bps = "
          f"factors {td.get('factor_explained_bps', 0):+.1f} bps + "
          f"alpha {td.get('residual_alpha_bps', 0):+.1f} bps")

    print()
    print(f"  TAIL-AWARE KELLY")
    print(f"  {'─' * 60}")
    print(f"  Gaussian Kelly:       {kel['f_kelly_gaussian']*100:+.2f}%")
    print(f"  EVT-haircut Kelly:    {kel['f_tail_aware']*100:+.2f}%   "
          f"(haircut: {kel['haircut_pct']:.1f}%)")
    print()
    print(f"  Saved: {OUTPUT_FILE}")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail risk + factor attribution engine")
    parser.add_argument("--ticker", default="GC=F",
                        help="Target ticker (default: GC=F)")
    parser.add_argument("--lookback", type=int, default=1260,
                        help="Lookback in trading days (default: 1260 ≈ 5y)")
    args = parser.parse_args()
    run_tail_risk_engine(ticker=args.ticker, lookback=args.lookback)
