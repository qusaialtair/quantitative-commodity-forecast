#!/usr/bin/env python3
"""
scripts/cointegration_engine.py
================================
Statistical-arbitrage cointegration & mean-reversion engine.

For each cross-asset pair we run the Engle-Granger two-step procedure:

  1. OLS:  log(Y_t) = α + β · log(X_t) + ε_t            (cointegrating vector)
  2. ADF on ε_t — if residual is stationary, the pair is cointegrated
  3. Fit Ornstein-Uhlenbeck:  Δε_t = -θ · ε_{t-1} + η_t
     Half-life = ln(2) / θ                              (mean-reversion speed)
  4. Current z-score = (ε_T - μ) / σ
  5. Signal — entry at |z| > 2.0, exit at |z| < 0.5

Pairs analysed:
    GC=F  vs  SI=F     gold-silver (classic)
    GC=F  vs  PL=F     gold-platinum
    GC=F  vs  HG=F     gold-copper
    GC=F  vs  DX-Y.NYB  gold-DXY (inverse driver)
    GC=F  vs  TLT       gold-real-yields proxy

The ADF test is implemented manually with MacKinnon (1996) critical values so
we have no dependency on statsmodels. Critical values for the τ-statistic
under the constant-only specification:
        1%:  -3.43        5%:  -2.86        10%:  -2.57

Output:
    data/cointegration_engine.json

Usage:
    python3 scripts/cointegration_engine.py
    python3 scripts/cointegration_engine.py --lookback 1260
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
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "cointegration_engine.json"

LINE_W = 84
SEP = "━" * LINE_W

# Pairs definition (target_ticker, hedge_ticker, friendly_name, description)
PAIRS = [
    ("GC=F", "SI=F",     "GOLD-SILVER",   "Classic precious-metals ratio"),
    ("GC=F", "PL=F",     "GOLD-PLATINUM", "Gold-platinum complex"),
    ("GC=F", "HG=F",     "GOLD-COPPER",   "Risk-on growth proxy"),
    ("GC=F", "DX-Y.NYB", "GOLD-DXY",      "Dominant inverse driver"),
    ("GC=F", "TLT",      "GOLD-TLT",      "Real-yield proxy"),
]

# MacKinnon (1996) critical values for ADF τ-statistic (constant-only specification)
ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}

ENTRY_Z = 2.0       # |z| > 2.0 to enter
EXIT_Z  = 0.5       # |z| < 0.5 to exit
STOP_Z  = 4.0       # |z| > 4.0 = bail (cointegration may have broken)


# ── Data layer ────────────────────────────────────────────────────────────────

def _fetch_pair(t1: str, t2: str, lookback: int) -> pd.DataFrame:
    """Fetch two tickers, return aligned log-price DataFrame."""
    if yf is None:
        raise ImportError("yfinance is required")

    period_days = int(lookback * 1.55) + 30
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=period_days)

    series = {}
    for t in (t1, t2):
        try:
            raw = yf.download(t, start=start, end=end + pd.Timedelta(days=1),
                              progress=False, auto_adjust=True, threads=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if "Close" not in raw.columns:
                continue
            series[t] = np.log(raw["Close"].dropna())
        except Exception:
            continue

    if t1 not in series or t2 not in series:
        raise RuntimeError(f"Insufficient data for {t1} / {t2}")

    df = pd.concat({t1: series[t1], t2: series[t2]}, axis=1).dropna().tail(lookback)
    return df


# ── Engle-Granger step 1: cointegrating regression ────────────────────────────

@dataclass
class CointResult:
    pair: str
    description: str
    n_obs: int
    alpha: float                 # intercept
    beta: float                  # cointegrating coefficient
    r_squared: float
    adf_stat: float              # τ statistic on residuals
    adf_pvalue_approx: float
    cointegrated_5pct: bool
    cointegrated_1pct: bool
    half_life_days: float        # OU mean-reversion half-life (days)
    spread_mean: float
    spread_std: float
    current_spread: float
    z_score: float
    signal: str                  # LONG_SPREAD / SHORT_SPREAD / FLAT / EXIT_NOW / STOP
    signal_rationale: str


def _ols_simple(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Simple OLS y = α + β·x; returns (α, β, R², residuals)."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_hat = float(beta[0])
    beta_hat = float(beta[1])
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha_hat, beta_hat, r2, resid


# ── ADF test (manual implementation, no statsmodels) ──────────────────────────

def _adf_test(series: np.ndarray, max_lag: int = 12) -> tuple[float, float]:
    """
    Augmented Dickey-Fuller test on a residual series, constant-only.

        Δy_t = α + γ·y_{t-1} + Σ δ_i · Δy_{t-i} + ε_t

    Lag length chosen by minimum-AIC over {0..max_lag}.
    Returns (τ-statistic, approximate p-value via MacKinnon surface fit).
    """
    y = np.asarray(series, dtype=float)
    if len(y) < 30:
        return 0.0, 1.0

    dy = np.diff(y)              # Δy_t
    ylag = y[:-1]                # y_{t-1}, length = n-1
    n = len(dy)

    best_aic = np.inf
    best_tau = 0.0
    for lag in range(min(max_lag, n // 5) + 1):
        # Build regression Δy_t = α + γ·y_{t-1} + Σ δ_i·Δy_{t-i}
        if lag >= n - 2:
            break
        Y = dy[lag:]
        n_eff = len(Y)
        cols = [np.ones(n_eff), ylag[lag:]]
        for i in range(1, lag + 1):
            cols.append(dy[lag - i:lag - i + n_eff])
        X = np.column_stack(cols)

        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ coef
            ss_res = float((resid ** 2).sum())
            sigma2 = ss_res / max(n_eff - X.shape[1], 1)
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se_gamma = float(np.sqrt(cov[1, 1]))
            gamma = float(coef[1])
            tau = gamma / se_gamma if se_gamma > 0 else 0.0
            # Schwarz / AIC; using AIC = n*ln(SSR/n) + 2*k
            k = X.shape[1]
            aic = n_eff * np.log(ss_res / n_eff) + 2 * k if ss_res > 0 else np.inf
            if aic < best_aic:
                best_aic = aic
                best_tau = tau
        except (np.linalg.LinAlgError, ValueError):
            continue

    # Approximate p-value via piecewise-linear interpolation between MacKinnon CVs.
    # This is rough; sufficient for a binary "is it cointegrated at 5%" decision.
    p = _approx_adf_pvalue(best_tau)
    return best_tau, p


def _approx_adf_pvalue(tau: float) -> float:
    """
    Linear-interpolated p-value from MacKinnon (1996) τ critical values.
    Constant-only specification; not exact but good enough for a 1%/5%/10%
    classification.
    """
    # Anchor points (cv at p, ordered most-negative to least-negative)
    pts = [(-3.96, 0.001), (-3.43, 0.01), (-2.86, 0.05),
           (-2.57, 0.10),  (-1.95, 0.25), (-1.62, 0.50),
           (0.0,   0.75),  (1.0,   0.95)]
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
        if x0 <= tau <= x1:
            w = (tau - x0) / (x1 - x0)
            return p0 + w * (p1 - p0)
    return 0.5


# ── OU half-life ──────────────────────────────────────────────────────────────

def _ou_half_life(spread: np.ndarray) -> float:
    """
    Fit Δs_t = α - θ · s_{t-1} + ε; return half-life = ln(2) / θ.
    Returns ∞ for non-mean-reverting series, capped at 9999 for serialisation.
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 30:
        return 9999.0
    ds = np.diff(s)
    s_lag = s[:-1]
    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    try:
        coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
        theta = -float(coef[1])
        if theta <= 0:
            return 9999.0
        return float(np.log(2) / theta)
    except np.linalg.LinAlgError:
        return 9999.0


# ── Per-pair analysis ─────────────────────────────────────────────────────────

def _analyse_pair(t1: str, t2: str, name: str, description: str,
                  lookback: int) -> CointResult:
    df = _fetch_pair(t1, t2, lookback)
    y = df[t1].values
    x = df[t2].values
    n = len(y)

    alpha, beta, r2, resid = _ols_simple(x, y)

    # Step 2: ADF on residuals
    tau, p_value = _adf_test(resid)
    coint_5 = bool(tau < ADF_CRITICAL[0.05])
    coint_1 = bool(tau < ADF_CRITICAL[0.01])

    # Step 3: OU half-life
    half_life = _ou_half_life(resid)

    # Step 4: spread statistics + z-score
    mu = float(resid.mean())
    sigma = float(resid.std(ddof=1))
    cur = float(resid[-1])
    z = (cur - mu) / sigma if sigma > 0 else 0.0

    # Step 5: signal logic
    signal, rationale = _generate_signal(z, coint_5, half_life, t1, t2)

    return CointResult(
        pair=f"{t1}/{t2}",
        description=f"{name} — {description}",
        n_obs=int(n),
        alpha=round(alpha, 6),
        beta=round(beta, 6),
        r_squared=round(r2, 4),
        adf_stat=round(float(tau), 4),
        adf_pvalue_approx=round(float(p_value), 4),
        cointegrated_5pct=coint_5,
        cointegrated_1pct=coint_1,
        half_life_days=round(half_life, 2),
        spread_mean=round(mu, 6),
        spread_std=round(sigma, 6),
        current_spread=round(cur, 6),
        z_score=round(float(z), 4),
        signal=signal,
        signal_rationale=rationale,
    )


def _generate_signal(z: float, coint_5: bool, half_life: float,
                     t1: str, t2: str) -> tuple[str, str]:
    """
    Mean-reversion signal logic:
        z > +ENTRY  → SHORT_SPREAD (sell t1, buy β·t2)   spread is too high
        z < -ENTRY  → LONG_SPREAD  (buy t1, sell β·t2)   spread is too low
        |z| < EXIT  → FLAT / EXIT
        |z| > STOP  → STOP — cointegration likely broken
    Suppressed if not cointegrated at 5% or half-life > 90 days.
    """
    if not coint_5:
        return "DISABLED", f"Pair not cointegrated at 5% (ADF fails)"
    if half_life > 90 or half_life < 1:
        return "DISABLED", f"Half-life {half_life:.0f}d out of [1, 90] tradable range"
    if abs(z) > STOP_Z:
        return "STOP", f"|z|={abs(z):.2f} > {STOP_Z} — possible structural break"
    if z > ENTRY_Z:
        return "SHORT_SPREAD", f"z={z:+.2f} — spread overshoot, sell {t1} / buy β·{t2}"
    if z < -ENTRY_Z:
        return "LONG_SPREAD",  f"z={z:+.2f} — spread undershoot, buy {t1} / sell β·{t2}"
    if abs(z) < EXIT_Z:
        return "FLAT", f"z={z:+.2f} near mean — no edge"
    return "WATCH", f"z={z:+.2f} — building toward entry threshold"


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_cointegration_engine(lookback: int = 1260) -> dict:
    results = []
    actionable = []
    cointegrated_count = 0

    for t1, t2, name, desc in PAIRS:
        try:
            r = _analyse_pair(t1, t2, name, desc, lookback)
            results.append(asdict(r))
            if r.cointegrated_5pct:
                cointegrated_count += 1
            if r.signal in ("LONG_SPREAD", "SHORT_SPREAD"):
                actionable.append({
                    "pair": r.pair, "name": name, "signal": r.signal,
                    "z_score": r.z_score, "half_life_days": r.half_life_days,
                    "rationale": r.signal_rationale,
                })
        except Exception as exc:
            results.append({
                "pair": f"{t1}/{t2}", "description": f"{name} — {desc}",
                "error": str(exc),
            })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_days": int(lookback),
        "n_pairs": len(PAIRS),
        "n_cointegrated_5pct": int(cointegrated_count),
        "n_actionable": len(actionable),
        "actionable_signals": actionable,
        "pairs": results,
        "thresholds": {
            "entry_z": ENTRY_Z, "exit_z": EXIT_Z, "stop_z": STOP_Z,
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    _print_report(output)
    return output


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(out: dict) -> None:
    print(f"\n{SEP}")
    print(f"  COINTEGRATION & MEAN-REVERSION ENGINE")
    print(f"  Pairs: {out['n_pairs']}  |  Cointegrated (5%): {out['n_cointegrated_5pct']}  "
          f"|  Actionable: {out['n_actionable']}")
    print(SEP)
    print(f"  {'Pair':<14s} {'β':>8s} {'R²':>6s} {'ADF τ':>8s} {'p-val':>7s} "
          f"{'½-life':>7s} {'z':>7s}  Signal")
    print(f"  {'─' * (LINE_W - 4)}")

    for p in out["pairs"]:
        if "error" in p:
            print(f"  {p['pair']:<14s}  {p['error'][:50]}")
            continue
        sig_color = {
            "LONG_SPREAD":  "\033[32m",  "SHORT_SPREAD": "\033[31m",
            "STOP":         "\033[31;1m", "FLAT":        "\033[2m",
            "WATCH":        "\033[33m",  "DISABLED":    "\033[2m",
        }.get(p["signal"], "\033[0m")
        coint_marker = "✓" if p.get("cointegrated_1pct") else ("·" if p.get("cointegrated_5pct") else " ")
        hl = p["half_life_days"]
        hl_str = f"{hl:.0f}d" if hl < 9999 else "∞"
        print(f"  {coint_marker} {p['pair']:<12s} "
              f"{p['beta']:+8.4f} "
              f"{p['r_squared']:6.3f} "
              f"{p['adf_stat']:+8.3f} "
              f"{p['adf_pvalue_approx']:7.3f} "
              f"{hl_str:>7s} "
              f"{p['z_score']:+7.2f}  "
              f"{sig_color}{p['signal']}\033[0m")

    print()
    if out["actionable_signals"]:
        print(f"  ACTIONABLE SIGNALS")
        print(f"  {'─' * 60}")
        for a in out["actionable_signals"]:
            sig_color = "\033[32m" if a["signal"] == "LONG_SPREAD" else "\033[31m"
            print(f"  {sig_color}{a['signal']:<14s}\033[0m {a['name']:<14s} "
                  f"z={a['z_score']:+.2f}  ½-life={a['half_life_days']:.0f}d")
            print(f"    {a['rationale']}")
    else:
        print(f"  No actionable mean-reversion signals at this time.")

    print(f"\n  Saved: {OUTPUT_FILE}")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cointegration & mean-reversion engine")
    parser.add_argument("--lookback", type=int, default=1260,
                        help="Lookback in trading days (default: 1260 ≈ 5y)")
    args = parser.parse_args()
    run_cointegration_engine(lookback=args.lookback)
