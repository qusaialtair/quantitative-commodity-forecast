#!/usr/bin/env python3
"""
models/macro_regime.py
======================
Phase 5 — The Crash Shield.

A Gaussian Hidden Markov Model that classifies the broad US equity
market into one of three latent states — RISK_ON / NEUTRAL / RISK_OFF —
and emits a target gross exposure that the CIO Swarm is contractually
obliged to respect.

Features (daily, at close)
--------------------------
  f1  = SPY 21-day log-return              (price momentum)
  f2  = log(VIX)                           (fear gauge; VIX is lognormal)
  f3  = HYG / IEF 21-day log-change        (credit risk appetite proxy:
                                            HYG underperforming IEF
                                            signals credit stress).

All three are standardised (sklearn StandardScaler, fit on the training
slice, persisted with the model) before being fed to hmmlearn's
GaussianHMM. Full-covariance emissions — SPY momentum, vol, and credit
co-move strongly, so we let the model learn the joint Gaussian.

Regime labelling
----------------
HMM latent states are unordered. After fit we score each state by

    score(i) = mean(spy_mom | s=i)  −  mean(log_vix | s=i)
               + mean(hyg_ief_mom  | s=i)

using RAW (unscaled) features so the interpretation is concrete: high
SPY momentum, low vol, positive credit trend → risk appetite. We then
rank: lowest score → RISK_OFF, middle → NEUTRAL, highest → RISK_ON.
This mapping is persisted alongside the model, so the same state
integer always maps to the same regime label across runs.

Output contract
---------------
data/macro_state.json is atomically rewritten on every --infer. Shape:

{
  "generated_at":          "2026-04-14T22:30:00+00:00",
  "as_of_date":            "2026-04-11",
  "regime":                "RISK_ON" | "NEUTRAL" | "RISK_OFF",
  "state_probs":           {"RISK_ON": 0.93, "NEUTRAL": 0.06, "RISK_OFF": 0.01},
  "target_gross_exposure": 1.00,
  "features":              {spy_21d_log_return, log_vix, vix_spot,
                            hyg_ief_21d_log_return},
  "model_meta":            {trained_at, train_start, train_end, train_samples,
                            converged}
}

The CIO Swarm (agents/equity_swarm/swarm.py) is to read this file at
the top of every run; if target_gross_exposure < current gross exposure,
it MUST de-risk before sizing any new entry. This is a hard gate, not
a soft suggestion — RISK_OFF days are when the fund survives.

CLI
---
  python3 -m models.macro_regime --train       # refit from 15y of data
  python3 -m models.macro_regime --infer       # classify today, write JSON
  python3 -m models.macro_regime               # auto: train if no artefact,
                                               #       then infer.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
#  Paths — absolute, anchored at the project root (cron/launchd-safe).
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(ROOT))

DATA_DIR      = ROOT / "data"
ARTIFACT_DIR  = ROOT / "models" / "artifacts"
STATE_PATH    = DATA_DIR / "macro_state.json"
MODEL_PATH    = ARTIFACT_DIR / "macro_hmm.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Model hyperparameters (frozen — any change is a retraining event).
# ══════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "SPY": "SPY",     # S&P 500 proxy — momentum
    "VIX": "^VIX",    # VIX index    — implied vol / fear
    "HYG": "HYG",     # US high-yield credit ETF
    "IEF": "IEF",     # 7-10Y Treasury ETF — duration-matched risk-free leg
}

MOM_WINDOW     = 21        # 21 trading days ≈ 1 calendar month
N_STATES       = 3         # RISK_ON / NEUTRAL / RISK_OFF
TRAIN_YEARS    = 15        # ~15 years of daily data → ~3,800 samples
HMM_MAX_ITER   = 200
HMM_TOL        = 1e-4
RANDOM_STATE   = 42        # deterministic EM

# Regime → fund target gross exposure. The CIO Swarm is forbidden from
# exceeding this cap on any given day, regardless of how bullish the
# bottom-up ranker is.
REGIME_TARGET_EXPOSURE = {
    "RISK_ON":  1.00,   # fully invested
    "NEUTRAL":  0.65,   # two-thirds invested
    "RISK_OFF": 0.30,   # defensive: 30% equity, 70% cash / sukuk
}

logger = logging.getLogger("MACRO_HMM")


# ══════════════════════════════════════════════════════════════════════════════
#  Data acquisition — yfinance with defensive post-processing.
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_closes(start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted closes for SPY / VIX / HYG / IEF.

    Returns a tz-naive DataFrame indexed on trading days with one
    column per alias. Drops rows where any of the four series is
    missing (ensures all features align on the same calendar).
    """
    raw = yf.download(
        list(TICKERS.values()),
        start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned empty frame for macro tickers "
            f"(start={start}, end={end}). Network / symbol problem."
        )

    closes = pd.DataFrame(index=raw.index if not isinstance(raw.columns, pd.MultiIndex)
                                          else raw.index)
    has_multi = isinstance(raw.columns, pd.MultiIndex)

    for alias, t in TICKERS.items():
        try:
            col = raw[t]["Close"] if has_multi else raw["Close"]
        except Exception as exc:
            raise RuntimeError(f"Missing close column for {t}: {exc}")
        closes[alias] = col

    closes = closes.dropna(how="any")
    if closes.empty:
        raise RuntimeError("No overlapping data across SPY / VIX / HYG / IEF.")
    # Make index tz-naive for clean pickling.
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    return closes


def _build_features(closes: pd.DataFrame) -> pd.DataFrame:
    """Turn adjusted closes into the HMM feature matrix."""
    out = pd.DataFrame(index=closes.index)
    out["spy_mom"]     = np.log(closes["SPY"] / closes["SPY"].shift(MOM_WINDOW))
    out["log_vix"]     = np.log(closes["VIX"])
    ratio              = closes["HYG"] / closes["IEF"]
    out["hyg_ief_mom"] = np.log(ratio / ratio.shift(MOM_WINDOW))
    return out.dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  State labelling — map unordered HMM indices → human regimes.
# ══════════════════════════════════════════════════════════════════════════════

def _label_states(model, states: np.ndarray,
                  feats_raw: pd.DataFrame) -> dict[int, str]:
    """Rank latent states by a composite 'risk appetite' score."""
    score_per_state: dict[int, float] = {}
    for s in range(model.n_components):
        mask = states == s
        if not mask.any():
            # A state with zero posterior mass — penalise so it falls to
            # RISK_OFF; it shouldn't matter in practice.
            score_per_state[s] = -np.inf
            continue
        mom = float(feats_raw.iloc[mask]["spy_mom"].mean())
        vix = float(feats_raw.iloc[mask]["log_vix"].mean())
        cre = float(feats_raw.iloc[mask]["hyg_ief_mom"].mean())
        score_per_state[s] = mom - vix + cre

    # Lowest score → RISK_OFF, middle → NEUTRAL, highest → RISK_ON.
    ranked = sorted(score_per_state.items(), key=lambda kv: kv[1])
    labels = ["RISK_OFF", "NEUTRAL", "RISK_ON"]
    return {ranked[i][0]: labels[i] for i in range(len(ranked))}


# ══════════════════════════════════════════════════════════════════════════════
#  Training — fit once from ~15y of history, persist with scaler + labels.
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_save() -> dict:
    """Refit the HMM from a rolling 15-year window and persist to disk."""
    # Local imports so `--infer` doesn't force a training-only dep path.
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    today = date.today()
    start = today.replace(year=today.year - TRAIN_YEARS).isoformat()
    end   = today.isoformat()

    logger.info("Training macro HMM · window %s → %s", start, end)
    closes = _fetch_closes(start, end)
    feats  = _build_features(closes)
    logger.info("Feature matrix: %d rows × %d cols", len(feats), feats.shape[1])

    scaler = StandardScaler().fit(feats.values)
    X      = scaler.transform(feats.values)

    model = GaussianHMM(
        n_components   = N_STATES,
        covariance_type= "full",
        n_iter         = HMM_MAX_ITER,
        tol            = HMM_TOL,
        random_state   = RANDOM_STATE,
        init_params    = "stmc",
    )
    model.fit(X)

    converged = bool(getattr(model.monitor_, "converged", False))
    if not converged:
        logger.warning("HMM did NOT converge in %d iters (tol=%.1e).",
                       HMM_MAX_ITER, HMM_TOL)

    # Freeze the regime labels on the training sample.
    states          = model.predict(X)
    state_to_regime = _label_states(model, states, feats)
    logger.info("State → regime map: %s", state_to_regime)

    # Sanity: report per-regime prevalence in-sample.
    for s, name in state_to_regime.items():
        frac = float((states == s).mean())
        logger.info("  %-8s  state=%d  prevalence=%.1f%%", name, s, frac * 100)

    bundle = {
        "model":           model,
        "scaler":          scaler,
        "state_to_regime": state_to_regime,
        "train_start":     start,
        "train_end":       end,
        "train_samples":   int(len(feats)),
        "converged":       converged,
        "trained_at":      datetime.now(timezone.utc)
                                   .isoformat(timespec="seconds"),
        "feature_cols":    list(feats.columns),
        "hyperparams": {
            "mom_window":      MOM_WINDOW,
            "n_states":        N_STATES,
            "covariance_type": "full",
            "random_state":    RANDOM_STATE,
        },
    }
    # Atomic write: tmp → replace, so a crash mid-pickle can't brick the model.
    tmp = MODEL_PATH.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(bundle, fh)
    tmp.replace(MODEL_PATH)
    logger.info("HMM bundle persisted → %s", MODEL_PATH.relative_to(ROOT))
    return bundle


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Macro HMM artefact missing at {MODEL_PATH}. "
            f"Run `python3 -m models.macro_regime --train` first."
        )
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
#  Inference — classify today's market, emit data/macro_state.json.
# ══════════════════════════════════════════════════════════════════════════════

def infer_today() -> dict:
    """Load the persisted HMM, pull recent prices, classify today."""
    bundle          = _load_bundle()
    model           = bundle["model"]
    scaler          = bundle["scaler"]
    state_to_regime = bundle["state_to_regime"]

    # ~1y is enough for a stable 21-day momentum read; widen if the
    # market was closed for an extended period.
    today = date.today()
    start = today.replace(year=today.year - 1).isoformat()
    closes = _fetch_closes(start)
    feats  = _build_features(closes)
    if feats.empty:
        raise RuntimeError("Macro feature matrix empty — yfinance pull failed.")

    X     = scaler.transform(feats.values)
    probs = model.predict_proba(X)[-1]         # posterior on the last obs
    hard  = int(model.predict(X)[-1])
    regime = state_to_regime[hard]

    # Normalise the probability dict under the HUMAN labels so downstream
    # code never has to remember which integer means what.
    probs_named = {state_to_regime[s]: float(probs[s])
                   for s in range(model.n_components)}
    # Ensure all three labels are present even if one has zero mass.
    for lbl in ("RISK_ON", "NEUTRAL", "RISK_OFF"):
        probs_named.setdefault(lbl, 0.0)

    last = feats.iloc[-1]
    state = {
        "generated_at":          datetime.now(timezone.utc)
                                         .isoformat(timespec="seconds"),
        "as_of_date":            feats.index[-1].strftime("%Y-%m-%d"),
        "regime":                regime,
        "state_probs":           probs_named,
        "target_gross_exposure": REGIME_TARGET_EXPOSURE[regime],
        "features": {
            "spy_21d_log_return":     float(last["spy_mom"]),
            "log_vix":                float(last["log_vix"]),
            "vix_spot":               float(np.exp(last["log_vix"])),
            "hyg_ief_21d_log_return": float(last["hyg_ief_mom"]),
        },
        "regime_target_exposure_table": REGIME_TARGET_EXPOSURE,
        "model_meta": {
            "trained_at":    bundle.get("trained_at"),
            "train_start":   bundle.get("train_start"),
            "train_end":     bundle.get("train_end"),
            "train_samples": bundle.get("train_samples"),
            "converged":     bundle.get("converged"),
        },
    }
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)
    logger.info("Macro state written · %s · target_gross=%.0f%%",
                regime, state["target_gross_exposure"] * 100)
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Public helper — other modules call this instead of re-reading the file.
# ══════════════════════════════════════════════════════════════════════════════

def load_macro_state() -> dict:
    """Read data/macro_state.json. Returns {} if the file is absent."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Macro regime HMM — crash shield (Phase 5)."
    )
    ap.add_argument("--train", action="store_true",
                    help="Refit the HMM from 15y history and persist.")
    ap.add_argument("--infer", action="store_true",
                    help="Classify today and write data/macro_state.json.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-11s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Default path: bootstrap if needed, then infer.
    if not (args.train or args.infer):
        if not MODEL_PATH.exists():
            logger.info("No persisted HMM — auto-training from 15y of history.")
            fit_and_save()
        state = infer_today()
        print(json.dumps(state, indent=2, default=str))
        return

    if args.train:
        fit_and_save()
    if args.infer:
        state = infer_today()
        print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    main()
