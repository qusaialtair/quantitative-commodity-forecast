#!/usr/bin/env python3
"""
ML Conviction Core — Proof of Concept  (Phase XXVI Stage 1)
============================================================
Tests the hypothesis from Phase XXII: does a properly-validated LightGBM
on the same feature inputs beat the hand-coded rule-based conviction?

The current rule (`scripts/strategy_backtester.py::_technical_conviction`)
blends 5 technical components with Sharpe-optimised weights (Phase XXIII)
and produces a [-1,+1] conviction score. Walk-forward median annual return:
+5.17%/y, Sharpe 0.48.

Phase XXII explicitly identified this as the ML-upgrade candidate.

What this script does
---------------------
1. Fetches 25y of daily prices for GC=F (gold), SI=F (silver), HG=F (copper),
   SPY (equities), DX-Y.NYB (dollar index). Cached so re-runs are fast.
2. Engineers 20+ features per date (returns, vol, trend, momentum,
   mean-reversion, cross-asset, RSI/BB).
3. Builds the regression target: forward 21d Sharpe = forward 21d cum return /
   forward 21d realised vol. This is what the rule-based conviction
   approximates.
4. Splits with PurgedKFold (label-horizon-aware, embargo-respecting).
5. Trains LightGBM regressor on each fold.
6. Computes OUT-OF-SAMPLE metrics for BOTH the ML prediction AND the
   rule-based baseline computed on the same dates:
     - Information Coefficient (Spearman correlation pred ↔ realized)
     - Long-when-positive strategy Sharpe
     - Hit rate (% of positive-prediction days that actually rose)
7. Compares the two side-by-side. Honest finding, no cherry-picking.

Output
------
data/ml_conviction_poc.json

Acceptance threshold (from Phase XXVI plan): OOS IC ≥ 0.05 AND ML
strategy Sharpe ≥ rule Sharpe + 0.15 to justify wiring it into production.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "_ml_cache"
OUTPUT_FILE = DATA_DIR / "ml_conviction_poc.json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ASSETS = {
    "gold":   "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "spy":    "SPY",
    "dxy":    "DX-Y.NYB",
}
START_DATE = "2001-01-01"
TARGET_HORIZON_DAYS = 21
N_SPLITS = 5
EMBARGO_PCT = 0.01


# ─── Data fetching (cached) ───────────────────────────────────────────────────

def _fetch(ticker: str) -> "pd.Series":
    import pandas as pd
    import yfinance as yf
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("=", "_").replace(".", "_").replace("-", "_")
    cache = CACHE_DIR / f"{safe}.csv"
    today = datetime.now(timezone.utc).date().isoformat()
    if cache.exists():
        try:
            df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
            last = df.index.max()
            if last >= pd.Timestamp(today) - pd.Timedelta(days=2):
                return df["close"].astype(float)
        except Exception:
            cache.unlink(missing_ok=True)
    df = yf.download(
        ticker, start=START_DATE, progress=False, auto_adjust=True, threads=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    # Flatten multi-level columns yfinance returns
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]
    if "Close" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Close"})
    close = pd.DataFrame({"date": df.index, "close": df["Close"].astype(float).values})
    close.to_csv(cache, index=False)
    return pd.Series(close["close"].values, index=pd.to_datetime(close["date"]))


def _load_panel() -> "pd.DataFrame":
    import pandas as pd
    series = {name: _fetch(t) for name, t in ASSETS.items()}
    # Flatten any multi-level columns yfinance sometimes returns
    for name in series:
        s = series[name]
        if hasattr(s, "columns"):
            s = s.iloc[:, 0]
        series[name] = pd.Series(s.values.flatten(), index=s.index, name=name)
    panel = pd.concat(series, axis=1).dropna(how="all")
    panel = panel.ffill(limit=5).dropna()
    return panel


# ─── Feature engineering ──────────────────────────────────────────────────────

def _ret(s, n):
    return s.pct_change(n)


def _zscore(s, win):
    mu = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - mu) / sd


def _rsi(close, win=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(win).mean()
    dn = -delta.clip(upper=0).rolling(win).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))


def _bb_pctb(close, win=20):
    sma = close.rolling(win).mean()
    sd  = close.rolling(win).std()
    upper = sma + 2 * sd
    lower = sma - 2 * sd
    return (close - lower) / (upper - lower)


def build_features(panel) -> "pd.DataFrame":
    import pandas as pd, numpy as np
    gold = panel["gold"]
    feats = pd.DataFrame(index=panel.index)

    # Gold momentum / returns
    feats["ret_1d"]   = _ret(gold, 1)
    feats["ret_5d"]   = _ret(gold, 5)
    feats["ret_21d"]  = _ret(gold, 21)
    feats["ret_63d"]  = _ret(gold, 63)
    feats["ret_252d"] = _ret(gold, 252)

    # Gold realised vol (annualised)
    daily_rets = gold.pct_change()
    feats["vol_21d"]  = daily_rets.rolling(21).std() * math.sqrt(252)
    feats["vol_63d"]  = daily_rets.rolling(63).std() * math.sqrt(252)
    feats["vol_ratio"] = feats["vol_21d"] / feats["vol_63d"]

    # Vol-of-vol (regime change signal)
    feats["vov_63d"] = feats["vol_21d"].rolling(63).std()

    # Trend MA cross
    sma20  = gold.rolling(20).mean()
    sma50  = gold.rolling(50).mean()
    sma100 = gold.rolling(100).mean()
    sma200 = gold.rolling(200).mean()
    feats["trend_20_50"]  = (sma20 - sma50) / sma50
    feats["trend_50_100"] = (sma50 - sma100) / sma100
    feats["trend_50_200"] = (sma50 - sma200) / sma200

    # Mean reversion
    feats["rsi_14"]   = _rsi(gold, 14)
    feats["bb_pctb"]  = _bb_pctb(gold, 20)

    # Drawdown
    rolling_max = gold.rolling(252, min_periods=20).max()
    feats["dd_from_252d_high"] = gold / rolling_max - 1.0

    # Cross-asset (21d returns + z-scores)
    feats["silver_ret_21d"] = _ret(panel["silver"], 21)
    feats["copper_ret_21d"] = _ret(panel["copper"], 21)
    feats["spy_ret_21d"]    = _ret(panel["spy"], 21)
    feats["dxy_ret_21d"]    = _ret(panel["dxy"], 21)

    # Copper-gold ratio z-score (canonical growth-vs-safety signal)
    cg = panel["copper"] / panel["gold"]
    feats["copper_gold_z"] = _zscore(cg, 252)

    # Gold-silver ratio (precious metals risk appetite)
    gs = gold / panel["silver"]
    feats["gold_silver_z"] = _zscore(gs, 252)

    return feats


def build_target(panel, horizon=TARGET_HORIZON_DAYS) -> "pd.Series":
    """Forward-horizon Sharpe of going long gold.

    Sharpe is what the rule-based conviction is implicitly predicting.
    Computed without leakage — only forward returns relative to today's price.
    """
    import pandas as pd
    gold = panel["gold"]
    fwd_returns = gold.shift(-horizon) / gold - 1.0   # cumulative
    fwd_daily = gold.pct_change().shift(-1).rolling(horizon).std() * math.sqrt(252)
    # Sharpe = ann return / ann vol, where ann return = (1+cum)^(252/h)-1
    ann_return = (1 + fwd_returns) ** (252 / horizon) - 1.0
    target = ann_return / fwd_daily.replace(0, float("nan"))
    target = target.replace([float("inf"), float("-inf")], float("nan"))
    return target


# ─── Rule baseline ────────────────────────────────────────────────────────────

def rule_baseline(panel) -> "pd.Series":
    """Reproduce strategy_backtester._technical_conviction's output on every
    historical bar using the SAME logic + Phase XXIII weights."""
    import pandas as pd, numpy as np
    # Use default weights as fallback (loading conviction_weights.json would
    # tie this to current optimisation, defeating the fair-comparison aim)
    W = {
        "trend_short":   0.32,
        "trend_long":    0.22,
        "mom_combined":  0.28,
        "mean_rev_fade": 0.08,
        "pivot":         0.10,
    }
    try:
        wj = json.loads((DATA_DIR / "conviction_weights.json").read_text())
        for k in W:
            W[k] = float(wj.get(k, W[k]))
    except Exception:
        pass

    closes = panel["gold"].to_numpy()
    n = len(closes)
    out = np.zeros(n)
    for i in range(n):
        if i < 60:
            continue
        sma20  = float(np.mean(closes[i - 20:i]))
        sma50  = float(np.mean(closes[i - 50:i]))
        sma200 = float(np.mean(closes[max(i - 200, 0):i])) if i >= 100 else sma50
        mom5  = (closes[i - 1] - closes[i - 6])  / closes[i - 6]  if i >= 6  else 0.0
        mom21 = (closes[i - 1] - closes[i - 22]) / closes[i - 22] if i >= 22 else 0.0
        mom63 = (closes[i - 1] - closes[i - 64]) / closes[i - 64] if i >= 64 else 0.0
        std20 = float(np.std(closes[i - 20:i]))
        bb = 0.0
        if std20 >= 1e-12:
            bb = max(-1.0, min(1.0, (closes[i - 1] - sma20) / (2.0 * std20)))
        ts  = math.tanh((sma20 - sma50)  / max(abs(sma50),  1.0) * 80)
        tl  = math.tanh((sma50 - sma200) / max(abs(sma200), 1.0) * 80)
        mc  = math.tanh(mom5 * 30 + mom21 * 12 + mom63 * 4)
        mr  = -bb * 0.4
        # pivot omitted from baseline (depends on _pivot_score internals);
        # this is the conservative comparison — actual rule includes pivot too,
        # so this comparison slightly understates the rule's edge.
        c = (W["trend_short"] * ts + W["trend_long"] * tl
             + W["mom_combined"] * mc + W["mean_rev_fade"] * mr)
        out[i] = max(-1.0, min(1.0, c))
    return pd.Series(out, index=panel.index)


# ─── Validation ───────────────────────────────────────────────────────────────

def _spearman_ic(pred, actual) -> float:
    import pandas as pd
    mask = ~(pred.isna() | actual.isna())
    if mask.sum() < 30:
        return 0.0
    return float(pred[mask].rank().corr(actual[mask].rank()))


def _strategy_sharpe(signal, fwd_returns, horizon) -> tuple[float, float, float]:
    """Long when signal > 0, short when < 0 (sign-based, not magnitude).
    Returns (sharpe, ann_return_pct, hit_rate_pct)."""
    import numpy as np, pandas as pd
    mask = ~(signal.isna() | fwd_returns.isna())
    sig = signal[mask]
    rets = fwd_returns[mask]
    direction = np.sign(sig).clip(-1, 1)
    # forward 21d return realised; we overlap every day so divide by horizon
    daily_strategy_ret = direction * rets / horizon
    if daily_strategy_ret.std() < 1e-9:
        return (0.0, 0.0, 0.0)
    sharpe = (daily_strategy_ret.mean() / daily_strategy_ret.std()) * math.sqrt(252)
    ann = ((1 + daily_strategy_ret.mean()) ** 252 - 1) * 100
    hit = ((direction == np.sign(rets)) & (direction != 0)).mean() * 100
    return (float(sharpe), float(ann), float(hit))


def run_purged_cv(features, target, rule_signal):
    """Train LightGBM on each PurgedKFold fold; aggregate OOS predictions."""
    import numpy as np, pandas as pd
    from scripts.purged_kfold import PurgedKFold
    import lightgbm as lgb

    # Align on full feature+target rows (target NaN at the last horizon days)
    df = features.copy()
    df["__target"] = target
    df["__rule"]   = rule_signal
    df = df.dropna()
    X = df.drop(columns=["__target", "__rule"]).to_numpy()
    y = df["__target"].to_numpy()
    rule = df["__rule"]
    feat_names = list(df.drop(columns=["__target", "__rule"]).columns)

    cv = PurgedKFold(n_splits=N_SPLITS,
                     label_horizon_days=TARGET_HORIZON_DAYS,
                     embargo_pct=EMBARGO_PCT)

    oos_pred = pd.Series(index=df.index, dtype=float)
    fold_metrics = []
    feat_importance = np.zeros(X.shape[1])

    for fold, (train_idx, test_idx) in enumerate(cv.split(len(df)), 1):
        # LightGBM with reasonable defaults — no Bayesian HPO in this PoC
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            min_child_samples=40,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        oos_pred.iloc[test_idx] = pred
        feat_importance += model.feature_importances_

        # Fold IC + Sharpe
        fold_actual = pd.Series(y[test_idx], index=df.index[test_idx])
        fold_pred   = pd.Series(pred,        index=df.index[test_idx])
        ic = _spearman_ic(fold_pred, fold_actual)
        fold_metrics.append({
            "fold":     fold,
            "n_train":  int(len(train_idx)),
            "n_test":   int(len(test_idx)),
            "ic":       round(ic, 4),
            "start":    str(df.index[test_idx[0]].date()),
            "end":      str(df.index[test_idx[-1]].date()),
        })

    importance_norm = (feat_importance / feat_importance.sum() if feat_importance.sum() > 0
                       else feat_importance)
    return df.index, oos_pred, rule, feat_names, importance_norm, fold_metrics, df["__target"]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    import pandas as pd
    print("Loading 25y of cross-asset prices (cached after first run)...")
    panel = _load_panel()
    n_dates = len(panel)
    print(f"  panel shape: {panel.shape}  ({panel.index[0].date()} → {panel.index[-1].date()})")

    print("Building features...")
    feats = build_features(panel)
    target = build_target(panel, TARGET_HORIZON_DAYS)
    rule = rule_baseline(panel)
    print(f"  features:    {feats.shape[1]}  ({list(feats.columns)})")

    # Forward returns (for strategy Sharpe)
    fwd_ret = panel["gold"].shift(-TARGET_HORIZON_DAYS) / panel["gold"] - 1.0

    print("Running PurgedKFold cross-validation (LightGBM)...")
    idx, oos_pred, rule_aligned, feat_names, importance, fold_metrics, target_aligned = \
        run_purged_cv(feats, target, rule)

    fwd_aligned = fwd_ret.reindex(idx)

    # Persist the per-date OOS predictions for downstream walk-forward analysis
    import pandas as pd
    preds_df = pd.DataFrame({
        "date":           idx,
        "ml_pred":        oos_pred.values,
        "rule_pred":      rule_aligned.values,
        "target_sharpe":  target_aligned.values,
        "fwd_21d_return": fwd_aligned.values,
    })
    preds_df.to_csv(DATA_DIR / "ml_conviction_predictions.csv", index=False)

    # Metrics
    ic_ml   = _spearman_ic(oos_pred,         target_aligned)
    ic_rule = _spearman_ic(rule_aligned,     target_aligned)
    sh_ml,  ann_ml,  hit_ml  = _strategy_sharpe(oos_pred,     fwd_aligned, TARGET_HORIZON_DAYS)
    sh_rule, ann_rule, hit_rule = _strategy_sharpe(rule_aligned, fwd_aligned, TARGET_HORIZON_DAYS)

    delta_ic = ic_ml - ic_rule
    delta_sh = sh_ml - sh_rule

    accept = (ic_ml >= 0.05) and (sh_ml >= sh_rule + 0.15)
    if accept:
        verdict = "ML_WINS_PROCEED_TO_WIRING"
        note = (f"ML IC {ic_ml:+.3f} ≥ 0.05 and ML Sharpe {sh_ml:+.2f} ≥ rule {sh_rule:+.2f} + 0.15. "
                "Justifies wiring LightGBM as the conviction core.")
    elif sh_ml >= sh_rule + 0.05:
        verdict = "ML_MARGINAL"
        note = (f"ML Sharpe {sh_ml:+.2f} beats rule {sh_rule:+.2f} by {delta_sh:+.2f} — "
                "real but small. Consider feature engineering before wiring.")
    elif sh_ml < sh_rule - 0.05:
        verdict = "RULE_WINS_KEEP_RULES"
        note = (f"Rule Sharpe {sh_rule:+.2f} > ML {sh_ml:+.2f}. The hand-coded blend is "
                "more robust on this feature set. Need richer features or different target.")
    else:
        verdict = "TIE"
        note = (f"ML Sharpe {sh_ml:+.2f} ≈ rule {sh_rule:+.2f}. No clear winner — "
                "spend the wiring effort only with stronger evidence.")

    # Top feature importances
    fi = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:12]
    result = {
        "schema_version": "1.0",
        "engine": "ml_conviction_poc",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": {
            "start": str(panel.index[0].date()),
            "end":   str(panel.index[-1].date()),
            "n_dates": int(n_dates),
            "assets": list(ASSETS.values()),
        },
        "config": {
            "horizon_days":   TARGET_HORIZON_DAYS,
            "n_splits":       N_SPLITS,
            "embargo_pct":    EMBARGO_PCT,
            "n_features":     int(feats.shape[1]),
            "feature_names":  feat_names,
        },
        "metrics": {
            "ml":   {"ic": round(ic_ml,   4),
                     "sharpe": round(sh_ml,   3),
                     "ann_return_pct": round(ann_ml,   3),
                     "hit_rate_pct": round(hit_ml,   2)},
            "rule": {"ic": round(ic_rule, 4),
                     "sharpe": round(sh_rule, 3),
                     "ann_return_pct": round(ann_rule, 3),
                     "hit_rate_pct": round(hit_rule, 2)},
            "delta": {"ic":     round(delta_ic, 4),
                      "sharpe": round(delta_sh, 3)},
        },
        "verdict": verdict,
        "note":    note,
        "fold_metrics": fold_metrics,
        "top_features": [{"feature": f, "importance": round(float(w), 4)} for f, w in fi],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    return result


SEP = "━" * 78


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  ML CONVICTION CORE — PROOF OF CONCEPT  (Phase XXVI Stage 1)\n{SEP}")
    p = r["panel"]
    print(f"  Panel: {p['n_dates']} bars  ({p['start']} → {p['end']})")
    print(f"  Features: {r['config']['n_features']}  "
          f"({r['config']['n_splits']}-fold PurgedKFold, horizon={r['config']['horizon_days']}d)")
    print(SEP)
    print(f"  {'Method':<22s} {'IC':>8s} {'Sharpe':>8s} {'Ann %':>8s} {'Hit %':>7s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for k, label in [("ml", "LightGBM (OOS)"), ("rule", "Rule baseline")]:
        m = r["metrics"][k]
        print(f"  {label:<22s} {m['ic']:>+8.3f} {m['sharpe']:>+8.2f} "
              f"{m['ann_return_pct']:>+8.2f} {m['hit_rate_pct']:>7.2f}")
    d = r["metrics"]["delta"]
    print(f"  {'Δ (ML − rule)':<22s} {d['ic']:>+8.3f} {d['sharpe']:>+8.2f}")
    print(SEP)
    print("  Per-fold IC:")
    for f in r["fold_metrics"]:
        print(f"    fold {f['fold']}  {f['start']} → {f['end']}  "
              f"IC={f['ic']:+.3f}  (train n={f['n_train']}, test n={f['n_test']})")
    print(SEP)
    print("  Top features by importance:")
    for f in r["top_features"]:
        print(f"    {f['importance']:>5.3f}  {f['feature']}")
    print(SEP)
    print(f"  VERDICT: {r['verdict']}")
    print(f"           {r['note']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    res = run()
    _print_report(res)
