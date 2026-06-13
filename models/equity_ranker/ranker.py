#!/usr/bin/env python3
"""
models/equity_ranker/ranker.py
==============================
Stage 2 of the Equity funnel — offline-trained ML ranker (Phase 6 overhaul).

  Classifier   →  LightGBM binary                P(r_10d > cross-sectional median)
  Regressor    →  XGBoost squared-error          E[r_10d]
  Blended score =  0.6 · P_up  +  0.4 · clip(z(E[r]), ±3) / 3
  Attribution   →  SHAP-based |z(feature) × mean|SHAP|| composite (top-3 drivers)

Phase 6 — three-pillar institutional hardening:

  (1) PURGED K-FOLD CV
      PurgedKFold (López de Prado, AFML §7.4) runs the INNER loop during
      hyperparameter tuning and SHAP estimation. Each val block pulls train
      rows from BOTH sides; both boundaries carry a `horizon`-day purge
      plus an `embargo`-day buffer so forward-label windows can never leak.
      The walk-forward `purged_splits()` is retained for the FINAL OOS
      report on the production model, so no future data ever influences
      the shipped score.

  (2) OPTUNA HYPERPARAMETER TUNING (Spearman-IC objective)
      RMSE and AUC don't reward rank-ordering — a model can improve RMSE
      while shuffling the tail and wrecking the book. We optimise on the
      mean Spearman rank-IC across purged folds — the actual metric we
      care about at inference time. Classifier and regressor are tuned
      independently; both targets rank-correlate with realised returns.

  (3) SHAP-BASED FEATURE SELECTION
      Global |SHAP| values are aggregated across every purged val fold,
      the bottom `shap_drop_pct` (default 20%) of features are dropped,
      and both models are REFIT on the pruned feature set before shipping.
      Noise reduction + interpretable attribution in one pass.

Artefact schema (Phase-1 contract — unchanged):
    MODELS_DIR/lgbm_classifier.joblib
    MODELS_DIR/xgb_regressor.joblib
    MODELS_DIR/feature_list.json              ← post-SHAP-prune production list
    MODELS_DIR/feature_importances.json       ← keyed by feature, legacy format
    MODELS_DIR/training_manifest.json         ← extended with tuning/shap blocks
New auxiliary artefacts (non-breaking additions):
    MODELS_DIR/shap_importances.json
    MODELS_DIR/optuna_study.json

Requires libomp at runtime on macOS (installed at /opt/homebrew/opt/libomp).

Subcommands
-----------
  build-dataset    generate historical snapshot+label dataset (slow; 30-60 min)
  train            tune → SHAP-prune → refit → dump artefacts (Phase 6)
  rank             load latest features, score the universe, write ranking
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_pipelines.equity_features import OUT_DIR as FEATURES_DIR  # noqa: E402
from scripts.equity_screener import _STATIC_SEED, build_halal_universe  # noqa: E402

from models.equity_ranker.dataset import (  # noqa: E402
    PurgedKFold,
    build_snapshots,
    feature_columns,
    purged_splits,
)

# Optuna + SHAP are REQUIRED in Phase 6. Hard-fail fast if either is missing —
# the training contract depends on them.
import optuna  # noqa: E402
import shap    # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("RANKER")

DATA_DIR    = ROOT / "data"
RANKER_DIR  = DATA_DIR / "equity_ranker"
MODELS_DIR  = RANKER_DIR / "models"
LOG_DIR     = DATA_DIR / "logs"
for _d in (RANKER_DIR, MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLF_PATH    = MODELS_DIR / "lgbm_classifier.joblib"
REG_PATH    = MODELS_DIR / "xgb_regressor.joblib"
IMP_PATH    = MODELS_DIR / "feature_importances.json"
FEATLIST    = MODELS_DIR / "feature_list.json"
MANIFEST    = MODELS_DIR / "training_manifest.json"
# Phase 6 auxiliary artefacts (non-breaking additions — the inference path
# never depends on these; the core contract above stays intact).
SHAP_PATH    = MODELS_DIR / "shap_importances.json"
OPTUNA_PATH  = MODELS_DIR / "optuna_study.json"


# ══════════════════════════════════════════════════════════════════════════════
# build-dataset
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_universe(kind: str, top: int | None) -> list[str]:
    u = sorted(_STATIC_SEED) if kind == "seed" else build_halal_universe()
    return u[:top] if top else u


def cmd_build_dataset(args: argparse.Namespace) -> None:
    universe = _resolve_universe(args.universe, args.top)
    logger.info("Universe: %d tickers (%s)", len(universe), args.universe)

    df = build_snapshots(
        universe=universe,
        start=args.start, end=args.end,
        step_days=args.step, horizon=args.horizon,
    )

    stamp = date.today().isoformat()
    out = RANKER_DIR / f"dataset_{stamp}.parquet"
    latest = RANKER_DIR / "dataset_latest.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, compression="snappy")
    tmp.rename(out)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(out.name)
    except OSError:
        import shutil; shutil.copyfile(out, latest)

    logger.info("Dataset → %s  (%d × %d)", out, *df.shape)


# ══════════════════════════════════════════════════════════════════════════════
# train
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Fixed params (shared across the cascade; tunable knobs are handled by Optuna).
# ──────────────────────────────────────────────────────────────────────────────

_LGBM_FIXED = dict(
    objective     = "binary",
    n_estimators  = 600,
    n_jobs        = -1,
    verbosity     = -1,
    random_state  = 42,
)

_XGB_FIXED = dict(
    objective    = "reg:squarederror",
    tree_method  = "hist",
    n_estimators = 600,
    n_jobs       = -1,
    verbosity    = 0,
    random_state = 42,
)


def _clean_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return df[feats].replace([np.inf, -np.inf], np.nan)


def _spearman_ic(y_true, y_pred) -> float:
    """Safe Spearman rank-IC. Returns 0.0 if either side is degenerate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return 0.0
    if np.nanstd(y_pred[mask]) == 0 or np.nanstd(y_true[mask]) == 0:
        return 0.0
    rho, _ = spearmanr(y_pred[mask], y_true[mask])
    return float(rho) if np.isfinite(rho) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# (1) PURGED-KFOLD EVALUATORS  — reused by both Optuna + SHAP stages
# ══════════════════════════════════════════════════════════════════════════════

def _cv_evaluate_clf(params: dict, X: pd.DataFrame, y_cls: pd.Series,
                     y_reg: pd.Series, splitter: PurgedKFold,
                     times: pd.Series) -> tuple[float, list[float]]:
    """Fit LGBM classifier across purged folds; return (mean IC vs raw fwd
    return, per-fold IC list). Classifier IC is measured against the
    CONTINUOUS forward return, not the binary label — a probability that
    rank-orders realised returns is the trading-useful signal."""
    fold_ic: list[float] = []
    for tr, va in splitter.split(times):
        m = lgb.LGBMClassifier(**{**_LGBM_FIXED, **params})
        m.fit(X.iloc[tr], y_cls.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        fold_ic.append(_spearman_ic(y_reg.iloc[va].values, p))
    if not fold_ic:
        return 0.0, []
    return float(np.mean(fold_ic)), fold_ic


def _cv_evaluate_reg(params: dict, X: pd.DataFrame, y_reg: pd.Series,
                     splitter: PurgedKFold, times: pd.Series
                     ) -> tuple[float, list[float]]:
    """Fit XGB regressor across purged folds; return (mean Spearman IC,
    per-fold IC list) — the canonical ranking-quality metric."""
    fold_ic: list[float] = []
    for tr, va in splitter.split(times):
        m = xgb.XGBRegressor(**{**_XGB_FIXED, **params})
        m.fit(X.iloc[tr], y_reg.iloc[tr])
        p = m.predict(X.iloc[va])
        fold_ic.append(_spearman_ic(y_reg.iloc[va].values, p))
    if not fold_ic:
        return 0.0, []
    return float(np.mean(fold_ic)), fold_ic


# ══════════════════════════════════════════════════════════════════════════════
# (2) OPTUNA TUNING  — objective is mean Spearman IC across Purged-KFold
# ══════════════════════════════════════════════════════════════════════════════

def _lgbm_search_space(trial: "optuna.Trial") -> dict:
    return {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":       trial.suggest_int  ("num_leaves",      15, 255),
        "max_depth":        trial.suggest_int  ("max_depth",        3,  12),
        "min_data_in_leaf": trial.suggest_int  ("min_data_in_leaf", 20, 400),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":     trial.suggest_int  ("bagging_freq",     1,  10),
        "lambda_l2":        trial.suggest_float("lambda_l2",      1e-3,  10.0, log=True),
    }


def _xgb_search_space(trial: "optuna.Trial") -> dict:
    return {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int  ("max_depth",        3,  12),
        "min_child_weight": trial.suggest_int  ("min_child_weight", 1,  50),
        "subsample":        trial.suggest_float("subsample",       0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6, 1.0),
        "gamma":            trial.suggest_float("gamma",          1e-4, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",     1e-3, 10.0, log=True),
    }


def _tune_with_optuna(kind: str, X: pd.DataFrame, y_cls: pd.Series,
                      y_reg: pd.Series, splitter: PurgedKFold,
                      times: pd.Series, n_trials: int,
                      seed: int = 42) -> dict:
    """Run an Optuna study whose objective is mean Spearman IC across the
    Purged-KFold. Returns best-params dict. Empty dict if n_trials == 0."""
    if n_trials <= 0:
        return {}
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _obj(trial: "optuna.Trial") -> float:
        try:
            if kind == "clf":
                params = _lgbm_search_space(trial)
                ic, _ = _cv_evaluate_clf(params, X, y_cls, y_reg, splitter, times)
            else:
                params = _xgb_search_space(trial)
                ic, _ = _cv_evaluate_reg(params, X, y_reg, splitter, times)
        except Exception as exc:   # pragma: no cover — guard degenerate folds
            logger.debug("trial crashed (%s) — pruning", exc)
            raise optuna.TrialPruned() from exc
        return float(ic) if np.isfinite(ic) else -1.0

    logger.info("Optuna [%s]: %d trials  objective=mean_IC_over_purged_folds",
                kind, n_trials)
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna [%s] best: IC=%.4f  params=%s",
                kind, study.best_value, study.best_params)
    return {
        "best_params": dict(study.best_params),
        "best_ic":     float(study.best_value),
        "n_trials":    int(len(study.trials)),
        "trials":      [
            {"number": t.number, "value": float(t.value) if t.value is not None else None,
             "params": dict(t.params), "state": str(t.state)}
            for t in study.trials
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# (3) SHAP FEATURE SELECTION  — mean |SHAP| across purged val folds
# ══════════════════════════════════════════════════════════════════════════════

def _shap_importance_across_folds(
    model_ctor, X: pd.DataFrame, y: pd.Series,
    splitter: PurgedKFold, times: pd.Series,
    max_sample: int = 4000,
) -> dict[str, float]:
    """For each purged fold: fit `model_ctor()`, compute |SHAP| on val rows,
    accumulate per-feature means weighted by fold size. Returns a dict
    {feature: mean |SHAP|} normalised to sum to 1.0.
    """
    rng = np.random.default_rng(42)
    agg = np.zeros(X.shape[1], dtype=float)
    total = 0
    for fold, (tr, va) in enumerate(splitter.split(times), 1):
        m = model_ctor()
        m.fit(X.iloc[tr], y.iloc[tr])
        if len(va) > max_sample:
            va_sampled = rng.choice(va, size=max_sample, replace=False)
        else:
            va_sampled = va
        X_val = X.iloc[va_sampled]
        try:
            explainer = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
            sv = explainer.shap_values(X_val, check_additivity=False)
        except Exception as exc:
            logger.warning("SHAP fold %d failed (%s) — falling back to model importances", fold, exc)
            imp = getattr(m, "feature_importances_", None)
            if imp is None:
                continue
            agg += np.asarray(imp, dtype=float) * len(va_sampled)
            total += len(va_sampled)
            continue

        # LightGBM binary returns [class0, class1] list — take the positive-class frame.
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv, dtype=float)
        # Handle shap>=0.44 3D output (n, f, classes) for binary/multiclass.
        if sv.ndim == 3:
            sv = sv[..., -1]
        fold_mean = np.abs(sv).mean(axis=0)
        agg += fold_mean * len(va_sampled)
        total += len(va_sampled)
        logger.info("  [SHAP fold %d]  n_val=%d  Σ|SHAP|=%.3f",
                    fold, len(va_sampled), float(fold_mean.sum()))

    if total == 0:
        return {c: 0.0 for c in X.columns}
    mean = agg / float(total)
    s = float(mean.sum())
    if s > 0:
        mean = mean / s
    return {c: float(v) for c, v in zip(X.columns, mean)}


def _select_top_features(shap_imp: dict[str, float], keep_pct: float) -> list[str]:
    """Keep the top `keep_pct` features by |SHAP|. keep_pct ∈ (0,1]."""
    ranked = sorted(shap_imp.items(), key=lambda kv: -kv[1])
    k = max(1, int(round(len(ranked) * keep_pct)))
    return [f for f, _ in ranked[:k]]


# ══════════════════════════════════════════════════════════════════════════════
# FINAL OOS REPORT  — walk-forward on the SHIPPED model
# ══════════════════════════════════════════════════════════════════════════════

def _final_oos_report(
    clf, reg, X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series,
    times: pd.Series, n_splits: int, embargo: int,
) -> tuple[dict, dict]:
    """Walk-forward OOS evaluation of the PRODUCTION models (post-tuning,
    post-SHAP-prune). Never peeks at the future — this is the honest headline."""
    clf_oof = np.full(len(y_cls), np.nan)
    reg_oof = np.full(len(y_reg), np.nan)
    clf_aucs, clf_ics = [], []
    reg_rmses, reg_ics = [], []

    for fold, (tr, va) in enumerate(purged_splits(times, n_splits, embargo), 1):
        cf = lgb.LGBMClassifier(**{**_LGBM_FIXED, **getattr(clf, "_tuned_params_", {})})
        cf.fit(X.iloc[tr], y_cls.iloc[tr])
        p = cf.predict_proba(X.iloc[va])[:, 1]
        clf_oof[va] = p
        clf_aucs.append(float(roc_auc_score(y_cls.iloc[va], p)))
        clf_ics.append(_spearman_ic(y_reg.iloc[va].values, p))

        rf = xgb.XGBRegressor(**{**_XGB_FIXED, **getattr(reg, "_tuned_params_", {})})
        rf.fit(X.iloc[tr], y_reg.iloc[tr])
        q = rf.predict(X.iloc[va])
        reg_oof[va] = q
        reg_rmses.append(float(np.sqrt(np.mean((y_reg.iloc[va].values - q) ** 2))))
        reg_ics.append(_spearman_ic(y_reg.iloc[va].values, q))

        logger.info("  [oos fold %d]  AUC=%.4f  clf_IC=%.4f  "
                    "RMSE=%.4f  reg_IC=%.4f",
                    fold, clf_aucs[-1], clf_ics[-1], reg_rmses[-1], reg_ics[-1])

    cmask = ~np.isnan(clf_oof); rmask = ~np.isnan(reg_oof)
    clf_report = {
        "fold_aucs":  [round(a, 4) for a in clf_aucs],
        "mean_auc":   round(float(np.mean(clf_aucs)), 4) if clf_aucs else None,
        "fold_ic":    [round(c, 4) for c in clf_ics],
        "mean_ic":    round(float(np.mean(clf_ics)), 4) if clf_ics else None,
        "oof_auc":    (round(float(roc_auc_score(y_cls[cmask], clf_oof[cmask])), 4)
                       if cmask.any() else None),
        "oof_ic":     (round(_spearman_ic(y_reg[cmask].values, clf_oof[cmask]), 4)
                       if cmask.any() else None),
        "n_estimators": int(getattr(clf, "n_estimators", 0) or 0),
    }
    reg_report = {
        "fold_rmse":  [round(r, 4) for r in reg_rmses],
        "mean_rmse":  round(float(np.mean(reg_rmses)), 4) if reg_rmses else None,
        "fold_ic":    [round(c, 4) for c in reg_ics],
        "mean_ic":    round(float(np.mean(reg_ics)), 4) if reg_ics else None,
        "oof_ic":     (round(_spearman_ic(y_reg[rmask].values, reg_oof[rmask]), 4)
                       if rmask.any() else None),
        "n_estimators": int(getattr(reg, "n_estimators", 0) or 0),
    }
    return clf_report, reg_report


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN COMMAND — Phase 6 pipeline: tune → SHAP-prune → refit → audit
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset) if args.dataset else RANKER_DIR / "dataset_latest.parquet"
    if not dataset_path.exists():
        logger.error("Dataset not found at %s — run build-dataset first.", dataset_path)
        sys.exit(2)

    df = pd.read_parquet(dataset_path).reset_index()
    logger.info("Loaded dataset %s  (%d × %d)", dataset_path.name, *df.shape)

    feats = feature_columns(df)
    logger.info("Feature count (pre-clean): %d", len(feats))

    df_model = df.dropna(subset=["label_beats_median", "forward_return"]).copy()
    X_full = _clean_matrix(df_model, feats)
    y_cls  = df_model["label_beats_median"].astype(int)
    y_reg  = df_model["forward_return"].astype(float)
    times  = df_model["as_of"]

    # Drop all-NaN feature columns.
    good = X_full.columns[X_full.notna().any()]
    X_full = X_full[good]
    feats  = list(good)
    logger.info("After NaN-column filter: %d features · %d rows",
                len(feats), len(X_full))

    splitter = PurgedKFold(
        n_splits     = args.splits,
        embargo_days = args.embargo,
        horizon_days = args.horizon,
    )

    # ── (1) Optuna tuning on Spearman IC ─────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Phase 6 · Stage A — Optuna IC tuning")
    logger.info("━" * 60)
    clf_study = _tune_with_optuna("clf", X_full, y_cls, y_reg,
                                  splitter, times, args.n_trials)
    reg_study = _tune_with_optuna("reg", X_full, y_cls, y_reg,
                                  splitter, times, args.n_trials)
    clf_best = clf_study.get("best_params", {})
    reg_best = reg_study.get("best_params", {})

    # ── (2) SHAP importance across folds → select top (1 - drop_pct) ─────────
    logger.info("━" * 60)
    logger.info("Phase 6 · Stage B — SHAP feature selection")
    logger.info("━" * 60)
    keep_pct = max(0.1, min(1.0, 1.0 - args.shap_drop_pct))

    def _clf_ctor():
        return lgb.LGBMClassifier(**{**_LGBM_FIXED, **clf_best})
    def _reg_ctor():
        return xgb.XGBRegressor(**{**_XGB_FIXED, **reg_best})

    shap_clf = _shap_importance_across_folds(_clf_ctor, X_full, y_cls, splitter, times)
    shap_reg = _shap_importance_across_folds(_reg_ctor, X_full, y_reg, splitter, times)

    # Combine the two SHAP maps: sum of normalised importances. A feature only
    # has to be useful for ONE head to survive — avoids amputating signals
    # that are strong regressors but weak classifiers (and vice versa).
    shap_combined: dict[str, float] = {}
    for c in feats:
        shap_combined[c] = float(shap_clf.get(c, 0.0) + shap_reg.get(c, 0.0))

    selected = _select_top_features(shap_combined, keep_pct)
    dropped  = [f for f in feats if f not in selected]
    logger.info("SHAP selection: keep %d / drop %d  (keep_pct=%.0f%%)",
                len(selected), len(dropped), keep_pct * 100)
    logger.info("Top-15 surviving features (by combined |SHAP|): %s",
                [f for f, _ in sorted(shap_combined.items(), key=lambda kv: -kv[1])[:15]])
    if dropped:
        logger.info("Dropped features: %s", dropped)

    X = X_full[selected]

    # ── (3) Refit production models on pruned feature set ────────────────────
    logger.info("━" * 60)
    logger.info("Phase 6 · Stage C — refit on pruned feature set (%d feats)",
                len(selected))
    logger.info("━" * 60)
    clf = lgb.LGBMClassifier(**{**_LGBM_FIXED, **clf_best}).fit(X, y_cls)
    reg = xgb.XGBRegressor (**{**_XGB_FIXED, **reg_best}).fit(X, y_reg)
    # Stash tuned params on the estimator so the walk-forward auditor can
    # reconstruct identical configurations for each OOS fold.
    clf._tuned_params_ = clf_best
    reg._tuned_params_ = reg_best

    # ── (4) Honest walk-forward OOS audit of the SHIPPED pair ────────────────
    logger.info("── Walk-forward OOS audit (purged, no-lookahead) ──")
    clf_report, reg_report = _final_oos_report(
        clf, reg, X, y_cls, y_reg, times, args.splits, args.embargo)
    logger.info("Classifier OOS · mean_AUC=%.4f  mean_IC=%.4f  OOF_AUC=%.4f",
                clf_report["mean_auc"] or 0, clf_report["mean_ic"] or 0,
                clf_report["oof_auc"] or 0)
    logger.info("Regressor  OOS · mean_RMSE=%.4f  mean_IC=%.4f  OOF_IC=%.4f",
                reg_report["mean_rmse"] or 0, reg_report["mean_ic"] or 0,
                reg_report["oof_ic"]    or 0)

    # ── Persist artefacts — Phase-1 schema intact + Phase-6 extras ───────────
    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)
    FEATLIST.write_text(json.dumps(selected, indent=2))
    # Legacy importance file — keep the filename, serve SHAP values keyed by
    # feature so downstream `_top_contributors` (attribution) works unchanged.
    IMP_PATH.write_text(json.dumps(
        {k: shap_combined[k] for k in selected}, indent=2))
    SHAP_PATH.write_text(json.dumps({
        "classifier": shap_clf,
        "regressor":  shap_reg,
        "combined":   shap_combined,
        "selected":   selected,
        "dropped":    dropped,
        "keep_pct":   keep_pct,
    }, indent=2))
    OPTUNA_PATH.write_text(json.dumps({
        "classifier": clf_study,
        "regressor":  reg_study,
    }, indent=2))

    manifest = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "dataset":      dataset_path.name,
        "rows":         int(len(df_model)),
        "n_features_in":  int(len(feats)),
        "n_features_out": int(len(selected)),
        "phase":        "Phase-6 (purged-kfold · optuna-IC · shap-prune)",
        "cv": {
            "inner_kfold_n_splits": int(args.splits),
            "inner_kfold_embargo_days": int(args.embargo),
            "inner_kfold_horizon_days": int(args.horizon),
            "outer_walk_forward_splits": int(args.splits),
        },
        "tuning": {
            "n_trials_per_model": int(args.n_trials),
            "objective":         "mean Spearman IC over PurgedKFold folds",
            "classifier_best":   clf_best,
            "classifier_best_ic": clf_study.get("best_ic"),
            "regressor_best":    reg_best,
            "regressor_best_ic": reg_study.get("best_ic"),
        },
        "shap": {
            "drop_pct":  float(args.shap_drop_pct),
            "kept":      int(len(selected)),
            "dropped":   int(len(dropped)),
            "top_10":    dict(sorted(
                {k: shap_combined[k] for k in selected}.items(),
                key=lambda kv: -kv[1])[:10]),
        },
        "classifier":   clf_report,
        "regressor":    reg_report,
        "top_10_importance": dict(sorted(
            {k: shap_combined[k] for k in selected}.items(),
            key=lambda kv: -kv[1])[:10]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("Artefacts → %s", MODELS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# rank (inference)
# ══════════════════════════════════════════════════════════════════════════════

def _load_models():
    if not (CLF_PATH.exists() and REG_PATH.exists() and FEATLIST.exists()):
        logger.error("Trained models not found under %s — run `train` first.", MODELS_DIR)
        sys.exit(2)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    feats = json.loads(FEATLIST.read_text())
    importances = json.loads(IMP_PATH.read_text()) if IMP_PATH.exists() else {}
    return clf, reg, feats, importances


def _top_contributors(X: pd.DataFrame, importances: dict[str, float],
                      k: int = 3) -> list[list[str]]:
    """
    SHAP-weighted attribution (Phase 6):
       driver_score[i, f] = |z_cross-section(X[f])[i]| × mean|SHAP|[f]
    Top-k features per row by this composite. Computed vectorised.

    `importances` is the map persisted in feature_importances.json; post-
    Phase-6 it carries mean |SHAP| values keyed by surviving features.
    """
    names = X.columns.to_numpy()
    imp_vec = np.array([importances.get(c, 0.0) for c in names], dtype=float)
    Xn = X.to_numpy(dtype=float)
    mu = np.nanmean(Xn, axis=0)
    sd = np.nanstd(Xn, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    z = np.abs((Xn - mu) / sd)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    composite = z * imp_vec  # broadcast
    idx = np.argsort(-composite, axis=1)[:, :k]
    return [names[row].tolist() for row in idx]


def cmd_rank(args: argparse.Namespace) -> None:
    features_path = Path(args.features) if args.features else FEATURES_DIR / "latest.parquet"
    if not features_path.exists():
        logger.error("Feature matrix not found at %s — run Stage 1 first.", features_path)
        sys.exit(2)

    feats_df = pd.read_parquet(features_path)
    logger.info("Loaded features %s  (%d × %d)", features_path.name, *feats_df.shape)

    clf, reg, feat_list, importances = _load_models()
    missing = [c for c in feat_list if c not in feats_df.columns]
    if missing:
        logger.warning("Missing %d trained features at inference: %s",
                       len(missing), missing[:5])
        for c in missing:
            feats_df[c] = np.nan

    X = feats_df[feat_list].replace([np.inf, -np.inf], np.nan)
    p_up = clf.predict_proba(X)[:, 1]
    e_r  = reg.predict(X)

    # Cross-sectional z-score of expected return → score stays rank-invariant
    er = pd.Series(e_r, index=feats_df.index)
    z_er = (er - er.mean()) / (er.std() or 1.0)

    score = 0.6 * p_up + 0.4 * z_er.clip(-3, 3) / 3
    top_feats = _top_contributors(X, importances, k=3)

    out = feats_df.copy()
    out["prob_beats_median"] = p_up
    out["expected_return_10d"] = e_r
    out["ai_score"] = score.values
    out["ai_drivers_top3"] = top_feats
    out = out.sort_values("ai_score", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    out["top_50"] = out["rank"] <= 50

    stamp = date.today().isoformat()
    ranking_path = RANKER_DIR / f"ranking_{stamp}.parquet"
    latest = RANKER_DIR / "ranking_latest.parquet"
    tmp = ranking_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, compression="snappy"); tmp.rename(ranking_path)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(ranking_path.name)
    except OSError:
        import shutil; shutil.copyfile(ranking_path, latest)

    logger.info("Ranking → %s", ranking_path)
    preview_cols = [c for c in ("name", "sector", "ai_score",
                                "prob_beats_median", "expected_return_10d")
                    if c in out.columns]
    logger.info("Top 10:\n%s", out.head(10)[preview_cols].round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Equity ML Ranker  (Stage 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="generate historical (features, label) snapshots")
    b.add_argument("--universe", choices=("live", "seed"), default="live")
    b.add_argument("--top",   type=int, default=None)
    b.add_argument("--start", type=str, default="2021-01-01")
    b.add_argument("--end",   type=str, default=date.today().isoformat())
    b.add_argument("--step",  type=int, default=5,  help="trading-day stride between snapshots")
    b.add_argument("--horizon", type=int, default=10, help="forward-return window in trading days")
    b.set_defaults(func=cmd_build_dataset)

    t = sub.add_parser("train",
        help="Phase 6: Optuna-IC tune → SHAP-prune → refit → walk-forward audit")
    t.add_argument("--dataset",  type=str, default=None, help="path to dataset parquet")
    t.add_argument("--splits",   type=int, default=5,
                   help="n folds for PurgedKFold (inner) and walk-forward audit (outer)")
    t.add_argument("--embargo",  type=int, default=10,
                   help="calendar-day embargo around each val block")
    t.add_argument("--horizon",  type=int, default=10,
                   help="label horizon in days — used for fold-boundary purging")
    t.add_argument("--n-trials", type=int, default=40,
                   help="Optuna trials per model (Spearman-IC objective); 0 skips tuning")
    t.add_argument("--shap-drop-pct", type=float, default=0.20,
                   help="fraction of features to drop by |SHAP| (e.g. 0.2 = bottom 20%%)")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("rank", help="score the universe, emit ranking parquet")
    r.add_argument("--features", type=str, default=None, help="path to Stage 1 features parquet")
    r.set_defaults(func=cmd_rank)

    args = ap.parse_args()

    log_file = LOG_DIR / f"equity_ranker_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-7s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    wall = time.time()
    args.func(args)
    logger.info("Done  %.0fs", time.time() - wall)


if __name__ == "__main__":
    main()
