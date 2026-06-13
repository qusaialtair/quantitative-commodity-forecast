#!/usr/bin/env python3
"""
Split Conformal Prediction Intervals
=====================================
Distribution-free prediction intervals around a 5-day-forward gold return
forecast. Provides finite-sample coverage guarantees under exchangeability:
    P(y ∈ [ŷ − q, ŷ + q]) ≥ 1 − α

Pipeline:
  1. Train a Ridge regression on the technical + macro feature set used
     by ensemble_stacking, targeting the 5d-forward return.
  2. Hold out a calibration set (chronologically after training).
  3. Compute non-conformity scores  s_i = |y_i − ŷ_i|  on the calibration set.
  4. q_{1−α} = the ⌈(1−α)(n+1)⌉ / n quantile of the calibration scores.
  5. For any new x, the interval is  ŷ(x) ± q_{1−α}.

Reports:
  - 90% and 95% interval widths (in percent move)
  - Empirical coverage on the test set (chronologically after calibration)
  - Latest live prediction and interval

Output: data/conformal_intervals.json
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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import yfinance as yf
except ImportError:
    yf = None

from scripts.ensemble_stacking import _fetch_panel, _build_technical_features, _build_macro_features

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "conformal_intervals.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON = 5
DEFAULT_TRAIN_PCT = 0.70
DEFAULT_CALIB_PCT = 0.15
DEFAULT_ALPHA_LEVELS = [0.10, 0.05]  # 90%, 95%

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_conformal_intervals(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    train_pct: float = DEFAULT_TRAIN_PCT,
    calib_pct: float = DEFAULT_CALIB_PCT,
    alpha_levels: list = None,
) -> dict:
    alpha_levels = alpha_levels or DEFAULT_ALPHA_LEVELS

    df = _fetch_panel(ticker, lookback)
    tech = _build_technical_features(df)
    macro = _build_macro_features(df)
    X = pd.concat([tech, macro], axis=1)
    y = df["gold"].pct_change(horizon).shift(-horizon)

    aligned = pd.concat([X, y.rename("y")], axis=1).dropna()
    Xf = aligned.drop(columns="y").values
    yf_arr = aligned["y"].values

    n = len(aligned)
    train_end = int(n * train_pct)
    calib_end = int(n * (train_pct + calib_pct))

    X_tr, y_tr = Xf[:train_end], yf_arr[:train_end]
    X_ca, y_ca = Xf[train_end:calib_end], yf_arr[train_end:calib_end]
    X_te, y_te = Xf[calib_end:], yf_arr[calib_end:]

    if min(len(X_tr), len(X_ca), len(X_te)) < 30:
        raise RuntimeError(
            f"Splits too small: train={len(X_tr)} calib={len(X_ca)} test={len(X_te)}"
        )

    # Fit Ridge with standardised features
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_ca_s = scaler.transform(X_ca)
    X_te_s = scaler.transform(X_te)

    model = Ridge(alpha=1.0).fit(X_tr_s, y_tr)

    # Calibration residuals (absolute)
    pred_ca = model.predict(X_ca_s)
    residuals = np.abs(y_ca - pred_ca)

    n_ca = len(residuals)
    coverage_results = {}
    intervals = {}
    for alpha in alpha_levels:
        rank = int(np.ceil((1 - alpha) * (n_ca + 1))) - 1
        rank = min(max(rank, 0), n_ca - 1)
        q = float(np.sort(residuals)[rank])

        # Empirical coverage on test set
        pred_te = model.predict(X_te_s)
        in_interval = (np.abs(y_te - pred_te) <= q).mean()

        # Interval widths
        width_pct = q * 2 * 100  # full width as % return

        intervals[f"alpha_{int(alpha*100):02d}"] = {
            "nominal_coverage": round(1 - alpha, 4),
            "interval_radius_pct": round(q * 100, 4),
            "interval_width_pct": round(width_pct, 4),
            "empirical_coverage": round(float(in_interval), 4),
            "valid":              bool(in_interval >= (1 - alpha) - 0.05),
        }

    # Live prediction (latest row from the full dataset)
    latest_X = Xf[-1:].copy()
    latest_X_s = scaler.transform(latest_X)
    latest_pred = float(model.predict(latest_X_s)[0])

    live_intervals = {}
    for alpha in alpha_levels:
        rank = int(np.ceil((1 - alpha) * (n_ca + 1))) - 1
        rank = min(max(rank, 0), n_ca - 1)
        q = float(np.sort(residuals)[rank])
        live_intervals[f"alpha_{int(alpha*100):02d}"] = {
            "point_forecast_pct":  round(latest_pred * 100, 4),
            "lower_pct":           round((latest_pred - q) * 100, 4),
            "upper_pct":           round((latest_pred + q) * 100, 4),
            "interval_width_pct":  round(q * 200, 4),
        }

    # In-sample model fit diagnostics
    train_r2 = float(model.score(X_tr_s, y_tr))
    test_r2 = float(model.score(X_te_s, y_te))

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":           ticker,
        "lookback":         lookback,
        "horizon":          horizon,
        "n_obs":            int(n),
        "splits": {
            "train": int(len(X_tr)),
            "calib": int(len(X_ca)),
            "test":  int(len(X_te)),
        },
        "model": "Ridge(alpha=1.0)",
        "model_r2": {
            "train": round(train_r2, 4),
            "test":  round(test_r2, 4),
        },
        "intervals":          intervals,
        "live_intervals":     live_intervals,
        "latest_forecast_pct":round(latest_pred * 100, 4),
        "calibration_n":      int(n_ca),
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
    print(f"  SPLIT CONFORMAL INTERVALS -- {r['ticker']}")
    print(SEP)
    print(f"  Observations:    {r['n_obs']}")
    print(f"  Splits:          "
          f"train={r['splits']['train']}  calib={r['splits']['calib']}  test={r['splits']['test']}")
    print(f"  Model:           {r['model']}")
    print(f"  R² (train/test): {r['model_r2']['train']:+.4f}  /  {r['model_r2']['test']:+.4f}")
    print()

    print(f"  CALIBRATED INTERVALS")
    print(f"  {'─' * 58}")
    print(f"  {'level':<10s}  {'radius %':>9s}  {'width %':>9s}  "
          f"{'emp cov':>9s}  {'valid':>8s}")
    for k, v in r["intervals"].items():
        valid_str = "yes" if v["valid"] else "BREACH"
        print(
            f"  {k:<10s}  {v['interval_radius_pct']:>9.3f}  "
            f"{v['interval_width_pct']:>9.3f}  "
            f"{v['empirical_coverage']:>9.4f}  "
            f"{valid_str:>8s}"
        )
    print()

    print(f"  LIVE PREDICTION (next {r['horizon']}d return %)")
    print(f"  {'─' * 58}")
    print(f"  Point forecast:  {r['latest_forecast_pct']:+.3f}%")
    for k, v in r["live_intervals"].items():
        print(
            f"  {k}:    "
            f"[{v['lower_pct']:>+7.2f}%, {v['upper_pct']:>+7.2f}%]  "
            f"width {v['interval_width_pct']:.2f}%"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Conformal Prediction Intervals")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    run_conformal_intervals(
        ticker=args.ticker,
        lookback=args.lookback,
        horizon=args.horizon,
    )
