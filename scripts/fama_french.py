#!/usr/bin/env python3
"""
Fama-French-Carhart Factor Loadings
=====================================
Regresses strategy returns on standard risk factors to decompose returns
into systematic exposures vs alpha. Without access to Ken French's data
library directly, we approximate the factors using widely-available ETFs:

  Mkt-Rf   SPY return minus T-bill yield     (SPY − ^IRX)
  SMB      Small minus Big (IWM − OEF)        (Russell 2000 − S&P 100)
  HML      High minus Low book/market (IVE − IVW)  (S&P value − growth)
  Mom      MTUM minus IVW                     (momentum ETF − growth)
  USD      DXY return                          (dollar factor; metals-relevant)

Regression: r_strat = α + β_mkt·Mkt + β_smb·SMB + β_hml·HML + β_mom·Mom + β_usd·USD + ε

Reports:
  - Per-factor β and t-statistic
  - α annualised
  - R², residual vol
  - Significant factors (|t| > 2)

Output: data/fama_french.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "fama_french.json"

DEFAULT_LOOKBACK = "5y"
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Factor construction
# ---------------------------------------------------------------------------
def _fetch_close(ticker: str, lookback: str) -> pd.Series:
    raw = yf.download(ticker, period=lookback, interval="1d",
                       progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].dropna()


def _build_factors(lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    # Pull ETF closes
    spy = _fetch_close("SPY", lookback)
    iwm = _fetch_close("IWM", lookback)
    oef = _fetch_close("OEF", lookback)
    ive = _fetch_close("IVE", lookback)
    ivw = _fetch_close("IVW", lookback)
    mtum = _fetch_close("MTUM", lookback)
    dxy = _fetch_close("DX-Y.NYB", lookback)
    irx = _fetch_close("^IRX", lookback)  # 13-week T-bill

    # Convert IRX (annualised %) to daily rate
    rf_daily = (irx / 100.0 / 252.0).reindex(spy.index).ffill()

    df = pd.DataFrame({
        "spy":  spy.pct_change(),
        "iwm":  iwm.pct_change(),
        "oef":  oef.pct_change(),
        "ive":  ive.pct_change(),
        "ivw":  ivw.pct_change(),
        "mtum": mtum.pct_change(),
        "dxy":  dxy.pct_change(),
    }).dropna()
    df["rf"] = rf_daily.reindex(df.index).ffill().fillna(0)

    factors = pd.DataFrame(index=df.index)
    factors["Mkt-Rf"] = df["spy"] - df["rf"]
    factors["SMB"] = df["iwm"] - df["oef"]
    factors["HML"] = df["ive"] - df["ivw"]
    factors["Mom"] = df["mtum"] - df["ivw"]
    factors["USD"] = df["dxy"]
    factors["RF"] = df["rf"]
    return factors


# ---------------------------------------------------------------------------
# Strategy return proxy
# ---------------------------------------------------------------------------
def _strategy_returns(lookback: str) -> pd.Series:
    """
    Use gold returns as a proxy for "strategy". In production, this would
    be the realized P&L of the live agent. Until live, gold acts as the
    representative asset whose factor exposures we want to know.
    """
    gold = _fetch_close("GC=F", lookback)
    return gold.pct_change().dropna()


# ---------------------------------------------------------------------------
# OLS regression  (pure numpy)
# ---------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    n, k = X.shape
    # Add intercept
    X1 = np.hstack([np.ones((n, 1)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    y_hat = X1 @ beta
    resid = y - y_hat
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    sigma2 = sse / max(n - k - 1, 1)
    var_beta = sigma2 * np.linalg.pinv(X1.T @ X1).diagonal()
    se_beta = np.sqrt(np.maximum(var_beta, 0))
    t_stats = beta / np.where(se_beta > 1e-9, se_beta, 1.0)
    return {
        "alpha":         float(beta[0]),
        "alpha_t":       float(t_stats[0]),
        "alpha_se":      float(se_beta[0]),
        "betas":         beta[1:].tolist(),
        "betas_t":       t_stats[1:].tolist(),
        "betas_se":      se_beta[1:].tolist(),
        "r_squared":     float(r2),
        "residual_vol":  float(resid.std()),
        "n":             int(n),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_fama_french(lookback: str = DEFAULT_LOOKBACK) -> dict:
    factors = _build_factors(lookback)
    strat = _strategy_returns(lookback)

    # Align
    aligned = pd.concat([strat.rename("strat"), factors], axis=1).dropna()
    factor_names = ["Mkt-Rf", "SMB", "HML", "Mom", "USD"]
    y = (aligned["strat"] - aligned["RF"]).values
    X = aligned[factor_names].values

    fit = _ols(y, X)

    factor_summary = []
    for i, fn in enumerate(factor_names):
        beta = fit["betas"][i]
        t = fit["betas_t"][i]
        factor_summary.append({
            "factor":  fn,
            "beta":    round(beta, 4),
            "t_stat":  round(t, 3),
            "se":      round(fit["betas_se"][i], 4),
            "significant": bool(abs(t) > 2.0),
        })

    alpha_annualised = float(fit["alpha"] * 252)
    alpha_ann_pct = alpha_annualised * 100
    resid_vol_ann = float(fit["residual_vol"] * SQ252 * 100)
    info_ratio = alpha_annualised / max(fit["residual_vol"] * SQ252, 1e-9)

    significant_factors = [f for f in factor_summary if f["significant"]]

    result = {
        "generated_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback":           lookback,
        "n_obs":              int(fit["n"]),
        "factor_names":       factor_names,
        "factor_summary":     factor_summary,
        "alpha_daily":        round(fit["alpha"], 6),
        "alpha_annualised_pct": round(alpha_ann_pct, 3),
        "alpha_t_stat":       round(fit["alpha_t"], 3),
        "alpha_significant":  bool(abs(fit["alpha_t"]) > 2.0),
        "r_squared":          round(fit["r_squared"], 4),
        "residual_vol_ann_pct":round(resid_vol_ann, 3),
        "information_ratio":  round(float(info_ratio), 3),
        "n_significant_factors": len(significant_factors),
        "dominant_factor":    max(
            factor_summary, key=lambda f: abs(f["t_stat"])
        )["factor"] if factor_summary else None,
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
    print(f"  FAMA-FRENCH-CARHART FACTOR LOADINGS")
    print(SEP)
    print(f"  Observations:    {r['n_obs']}")
    print(f"  R²:              {r['r_squared']:.4f}")
    print(f"  Residual vol:    {r['residual_vol_ann_pct']:.2f}% / yr")
    print()

    print(f"  FACTOR BETAS")
    print(f"  {'─' * 54}")
    print(f"  {'factor':<10s}  {'β':>8s}  {'SE':>7s}  {'t-stat':>7s}  {'signif':>7s}")
    for f in r["factor_summary"]:
        marker = " *" if f["significant"] else "  "
        print(
            f"  {marker} {f['factor']:<8s}  "
            f"{f['beta']:>+8.4f}  "
            f"{f['se']:>7.4f}  "
            f"{f['t_stat']:>+7.2f}  "
            f"{'YES' if f['significant'] else 'no':>7s}"
        )
    print()

    print(f"  ALPHA")
    print(f"  {'─' * 40}")
    alpha_color = "\033[32m" if r["alpha_significant"] and r["alpha_annualised_pct"] > 0 else "\033[33m"
    print(f"  α (annualised):  {alpha_color}{r['alpha_annualised_pct']:+.3f}%\033[0m")
    print(f"  α t-stat:        {r['alpha_t_stat']:+.2f}  "
          f"(significant: {r['alpha_significant']})")
    print(f"  Information IR:  {r['information_ratio']:+.3f}")
    print()
    print(f"  Dominant factor: {r['dominant_factor']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fama-French Factor Loadings")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_fama_french(lookback=args.lookback)
