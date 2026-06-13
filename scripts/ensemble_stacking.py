#!/usr/bin/env python3
"""
Ensemble Stacking
====================
Trains three diverse base learners on overlapping feature sets, gets
out-of-fold predictions from each via purged K-fold, then trains a
logistic-regression meta-learner on the OOF base predictions.

Base models:
  1. Logistic regression  →  technical features (returns / RSI / Bollinger %B)
  2. Random forest        →  macro features    (real yields / DXY / copper-gold / COT)
  3. Gradient boost       →  combined features

Target: sign of 5-day forward gold return  (binary, +1 / 0)

Meta-learner: logistic regression on the three base OOF probabilities.

Reports:
  - per-base accuracy and AUC
  - meta-learner accuracy and AUC
  - stacking lift over the best single base
  - feature importance (where models expose it)

Pure scikit-learn; no XGBoost.

Output: data/ensemble_stacking.json
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import yfinance as yf
except ImportError:
    yf = None

from scripts.purged_kfold import PurgedKFold

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "ensemble_stacking.json"
ALT_CSV = DATA_DIR / "alt_data.csv"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON = 5
DEFAULT_FOLDS = 5

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Features + label
# ---------------------------------------------------------------------------
def _fetch_panel(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")

    def _c(t: str) -> pd.Series:
        raw = yf.download(t, period=lookback, interval="1d",
                           progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        return raw["Close"].dropna()

    gold = _c(ticker)
    dxy = _c("DX-Y.NYB")
    return pd.DataFrame({"gold": gold, "dxy": dxy}).ffill().dropna()


def _build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df["gold"]
    r = g.pct_change()
    feats = pd.DataFrame(index=df.index)
    feats["ret_1d"] = r
    feats["ret_5d"] = g.pct_change(5)
    feats["ret_21d"] = g.pct_change(21)
    sma_s = g.rolling(20).mean()
    sma_l = g.rolling(50).mean()
    feats["sma_cross"] = (sma_s > sma_l).astype(float)
    bb_sma = g.rolling(20).mean()
    bb_std = g.rolling(20).std()
    upper = bb_sma + 2 * bb_std
    lower = bb_sma - 2 * bb_std
    feats["bb_pct_b"] = ((g - lower) / (upper - lower).replace(0, np.nan)).clip(-2, 3)
    # RSI(14)
    delta = g.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - 100 / (1 + rs)
    feats["vol_21d"] = r.rolling(21).std()
    return feats


def _build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["dxy_ret_5d"] = df["dxy"].pct_change(5)
    feats["dxy_ret_21d"] = df["dxy"].pct_change(21)
    # Load alt_data and join
    if ALT_CSV.exists():
        try:
            alt = pd.read_csv(ALT_CSV, index_col=0, parse_dates=True)
            for col in ["real_yield_10y", "copper_gold_ratio_zscore",
                         "cot_gold_mm_net_zscore"]:
                if col in alt.columns:
                    feats[col] = alt[col].reindex(feats.index).ffill()
        except Exception:
            pass
    return feats


def _make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd_return = df["gold"].pct_change(horizon).shift(-horizon)
    return (fwd_return > 0).astype(int)


# ---------------------------------------------------------------------------
# OOF prediction
# ---------------------------------------------------------------------------
def _oof_predict(
    model_factory, X: pd.DataFrame, y: pd.Series, folds: int, horizon: int,
) -> np.ndarray:
    """Out-of-fold predicted probabilities, generated via PurgedKFold."""
    n = len(X)
    oof = np.full(n, 0.5)
    splitter = PurgedKFold(n_splits=folds, label_horizon_days=horizon, embargo_pct=0.02)
    for train_idx, test_idx in splitter.split(n):
        X_tr = X.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_te = X.iloc[test_idx].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = model_factory()
        model.fit(X_tr_s, y_tr)
        oof[test_idx] = model.predict_proba(X_te_s)[:, 1]
    return oof


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_ensemble_stacking(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    folds: int = DEFAULT_FOLDS,
) -> dict:
    df = _fetch_panel(ticker, lookback)

    tech_feats = _build_technical_features(df)
    macro_feats = _build_macro_features(df)
    combined_feats = pd.concat([tech_feats, macro_feats], axis=1)
    y = _make_target(df, horizon)

    # Drop rows with any NaN in features OR label
    aligned = pd.concat([combined_feats, y.rename("y"), tech_feats.add_prefix("tech_"),
                         macro_feats.add_prefix("macro_")], axis=1).dropna()

    if len(aligned) < 200:
        raise RuntimeError(f"Only {len(aligned)} clean rows; need more history")

    X_tech = aligned[[c for c in aligned.columns if c.startswith("tech_")]]
    X_macro = aligned[[c for c in aligned.columns if c.startswith("macro_")]]
    X_comb = aligned[[c for c in aligned.columns
                       if not c.startswith("tech_") and not c.startswith("macro_") and c != "y"]]
    y_clean = aligned["y"]

    # OOF predictions per base
    oof_tech = _oof_predict(
        lambda: LogisticRegression(max_iter=500, C=1.0),
        X_tech, y_clean, folds, horizon,
    )
    oof_macro = _oof_predict(
        lambda: RandomForestClassifier(n_estimators=100, max_depth=4,
                                       random_state=42, n_jobs=-1),
        X_macro, y_clean, folds, horizon,
    )
    oof_comb = _oof_predict(
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.05, random_state=42),
        X_comb, y_clean, folds, horizon,
    )

    base_oof = pd.DataFrame({
        "logit_tech": oof_tech,
        "rf_macro":   oof_macro,
        "gb_comb":    oof_comb,
    }, index=aligned.index)

    # Base metrics
    base_metrics = {}
    for col in base_oof.columns:
        preds = (base_oof[col] > 0.5).astype(int)
        acc = accuracy_score(y_clean, preds)
        try:
            auc = roc_auc_score(y_clean, base_oof[col])
        except Exception:
            auc = 0.5
        base_metrics[col] = {
            "accuracy": round(float(acc), 4),
            "auc":      round(float(auc), 4),
        }

    # Meta-learner OOF
    meta_oof = _oof_predict(
        lambda: LogisticRegression(max_iter=500, C=1.0),
        base_oof, y_clean, folds, horizon,
    )
    meta_preds = (meta_oof > 0.5).astype(int)
    meta_acc = float(accuracy_score(y_clean, meta_preds))
    try:
        meta_auc = float(roc_auc_score(y_clean, meta_oof))
    except Exception:
        meta_auc = 0.5

    best_base_acc = max(m["accuracy"] for m in base_metrics.values())
    best_base_auc = max(m["auc"] for m in base_metrics.values())

    # Feature importances (where available)
    feature_importance = {}
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        scaler = StandardScaler()
        rf.fit(scaler.fit_transform(X_macro.values), y_clean.values)
        feature_importance["rf_macro"] = {
            col: round(float(imp), 4)
            for col, imp in zip(X_macro.columns, rf.feature_importances_)
        }
    except Exception:
        pass

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "lookback":       lookback,
        "label_horizon":  horizon,
        "folds":          folds,
        "n_obs":          int(len(aligned)),
        "n_tech_features":  int(X_tech.shape[1]),
        "n_macro_features": int(X_macro.shape[1]),
        "base_metrics":   base_metrics,
        "meta_metrics": {
            "accuracy": round(meta_acc, 4),
            "auc":      round(meta_auc, 4),
        },
        "stacking_lift": {
            "accuracy": round(meta_acc - best_base_acc, 4),
            "auc":      round(meta_auc - best_base_auc, 4),
        },
        "best_single_base": {
            "accuracy": round(best_base_acc, 4),
            "auc":      round(best_base_auc, 4),
        },
        "feature_importance": feature_importance,
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
    print(f"  ENSEMBLE STACKING -- {r['ticker']}")
    print(SEP)
    print(f"  Observations:    {r['n_obs']}")
    print(f"  Target horizon:  {r['label_horizon']}d  (binary up/down)")
    print(f"  Folds:           {r['folds']}  (purged + embargoed)")
    print()

    print(f"  BASE LEARNERS (OOF)")
    print(f"  {'─' * 50}")
    print(f"  {'model':<14s}  {'accuracy':>10s}  {'AUC':>8s}")
    for name, m in r["base_metrics"].items():
        print(f"  {name:<14s}  {m['accuracy']:>10.4f}  {m['auc']:>8.4f}")
    print()

    meta = r["meta_metrics"]
    lift = r["stacking_lift"]
    best = r["best_single_base"]
    print(f"  META LEARNER  (logistic on base OOF)")
    print(f"  {'─' * 50}")
    print(f"  Accuracy:       {meta['accuracy']:.4f}  "
          f"(best base {best['accuracy']:.4f}, lift {lift['accuracy']:+.4f})")
    print(f"  AUC:            {meta['auc']:.4f}  "
          f"(best base {best['auc']:.4f}, lift {lift['auc']:+.4f})")
    print()

    if r["feature_importance"]:
        print(f"  RF MACRO FEATURE IMPORTANCES")
        items = sorted(r["feature_importance"]["rf_macro"].items(),
                       key=lambda kv: kv[1], reverse=True)
        for k, v in items[:5]:
            print(f"    {k:<28s}  {v:.4f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Stacking")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    args = parser.parse_args()
    run_ensemble_stacking(
        ticker=args.ticker,
        lookback=args.lookback,
        horizon=args.horizon,
        folds=args.folds,
    )
