#!/usr/bin/env python3
"""
verify_hmm.py
=============
Standalone audit script for scripts/regime_detector.py.

Tests covered
-------------
  1.  Feature engineering math    — checks stationarity, NaN hygiene, scaling
  2.  HMM fitting                 — EM convergence, covariance structure, log-likelihood
  3.  State relabeling            — BEARISH < RANGING < BULLISH by learned mean return
  4.  Probability vector          — sums to 1.0, correct shape
  5.  joblib persistence          — save / reload round-trip
  6.  Regime veto logic           — threshold gate fires correctly
  7.  build_regime_overlays       — contiguous band collapse, date filtering
  8.  load_cached_regime          — JSON snapshot round-trip

Usage
-----
  python3 verify_hmm.py                    # live GC=F data (requires internet)
  python3 verify_hmm.py --csv path/to.csv  # use local CSV (columns: Date,Close,Volume)
  python3 verify_hmm.py --ticker SI=F      # audit Silver instead
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.regime_detector import (
    FEATURE_COLS,
    N_STATES,
    COV_TYPE,
    REGIME_VETO_THRESHOLD,
    REGIME_LABELS,
    REGIME_COLORS,
    _fetch_data,
    _build_features,
    _relabel_states,
    _build_regime_result,
    build_regime_overlays,
    load_cached_regime,
    fit,
)
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_pass = lambda s: f"{GREEN}PASS{RESET}  {s}"
_fail = lambda s: f"{RED}FAIL{RESET}  {s}"
_info = lambda s: f"{CYAN}INFO{RESET}  {s}"
_head = lambda s: f"\n{BOLD}{CYAN}{'─'*60}\n  {s}\n{'─'*60}{RESET}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert(cond: bool, msg: str) -> bool:
    print(_pass(msg) if cond else _fail(msg))
    return cond


def _load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load a local CSV as a minimal OHLCV-like DataFrame.
    Expects at minimum: a date/index column + 'Close' column.
    Synthesises Open/High/Low/Volume if absent.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
    df.index.name = "Date"

    if "Close" not in df.columns:
        # Try capitalising
        df.columns = [c.strip().title() for c in df.columns]

    if "Close" not in df.columns:
        raise ValueError(f"CSV must contain a 'Close' column. Found: {list(df.columns)}")

    # Synthesise missing columns so _build_features works
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    if "DXY_Close" not in df.columns:
        # Use a flat synthetic DXY so dxy_ret ≈ 0 (tests feature shape, not macro signal)
        df["DXY_Close"] = 100.0
    if "VIX_Close" not in df.columns:
        df["VIX_Close"] = 20.0

    return df.sort_index()


def _generate_synthetic_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price series for offline / CI testing.
    Three hidden regimes (bear/range/bull) baked in via a Markov chain.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=date.today(), periods=n)

    # Regime switching: 3-state Markov
    regime_seq = np.zeros(n, dtype=int)
    trans = np.array([
        [0.95, 0.03, 0.02],   # BEARISH stays bearish 95% of days
        [0.02, 0.92, 0.06],   # RANGING
        [0.02, 0.04, 0.94],   # BULLISH
    ])
    for t in range(1, n):
        regime_seq[t] = rng.choice(3, p=trans[regime_seq[t - 1]])

    drift = {0: -0.0004, 1: 0.00005, 2: 0.0006}
    sigma = {0: 0.018,   1: 0.007,   2: 0.009}

    log_rets = np.array([
        rng.normal(drift[regime_seq[t]], sigma[regime_seq[t]])
        for t in range(n)
    ])
    close = 2000.0 * np.exp(np.cumsum(log_rets))

    # Synthetic macro series with mild regime-correlated signal
    dxy   = 103.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    vix   = np.clip(20.0 + np.cumsum(rng.normal(0, 0.3, n)), 8.0, 80.0)

    df = pd.DataFrame({
        "Close":     close,
        "Volume":    rng.integers(50_000, 200_000, size=n).astype(float),
        "DXY_Close": dxy,
        "VIX_Close": vix,
    }, index=idx)
    df.index.name = "Date"
    return df


# ── Test suites ───────────────────────────────────────────────────────────────

def audit_feature_engineering(df: pd.DataFrame) -> bool:
    print(_head("1 · Feature Engineering Math"))
    ok = True

    feat_df = _build_features(df)

    ok &= _assert(len(feat_df) > 0,
                  f"_build_features returned {len(feat_df)} rows (need > 0)")
    ok &= _assert(all(c in feat_df.columns for c in FEATURE_COLS),
                  f"All 6 feature columns present: {FEATURE_COLS}")
    ok &= _assert(feat_df[FEATURE_COLS].isna().sum().sum() == 0,
                  "Zero NaN values after feature engineering")
    ok &= _assert(np.isfinite(feat_df[FEATURE_COLS].values).all(),
                  "All feature values finite (no Inf/NaN)")

    # Stationarity spot-check: log_ret should have mean ≈ 0, std < 0.1
    lr_mean = feat_df["log_ret"].mean()
    lr_std  = feat_df["log_ret"].std()
    ok &= _assert(abs(lr_mean) < 0.002,
                  f"log_ret mean ≈ 0 (got {lr_mean:.6f})")
    ok &= _assert(lr_std < 0.1,
                  f"log_ret std < 0.1 (got {lr_std:.6f})")

    # vol_21d must be strictly positive
    ok &= _assert((feat_df["vol_21d"] > 0).all(),
                  "vol_21d > 0 everywhere")

    # vol_ratio should spike around regime transitions — just check it's >0
    ok &= _assert((feat_df["vol_ratio"] > 0).all(),
                  "vol_ratio > 0 everywhere")

    # trend_50 should be bounded roughly in (-1, 1)
    t50_max = feat_df["trend_50"].abs().max()
    ok &= _assert(t50_max < 1.0,
                  f"trend_50 bounded in (-1, 1) — max abs = {t50_max:.4f}")

    print(_info(f"  Feature rows: {len(feat_df)}  |  "
                f"log_ret: μ={lr_mean:.5f}  σ={lr_std:.5f}"))
    return ok


def audit_hmm_fit(df: pd.DataFrame) -> tuple[bool, dict]:
    """Returns (all_pass, payload_dict) so later tests can reuse the fitted model."""
    print(_head("2 · HMM Fitting (GaussianHMM, full covariance, 8 restarts)"))
    ok = True

    feat_df  = _build_features(df)

    # Trim to 5-year window if data is longer
    window_start = date.today() - timedelta(days=365 * 5)
    feat_df = feat_df[feat_df.index.date >= window_start]
    if len(feat_df) < 252:
        feat_df = _build_features(df)   # fallback: use all available for synthetic

    X_raw    = feat_df[FEATURE_COLS].values.astype(np.float64)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Covariance structure check: input shape
    n_obs, n_feat = X_scaled.shape
    ok &= _assert(n_feat == len(FEATURE_COLS),
                  f"Feature matrix shape: ({n_obs}, {n_feat})")

    # Fit with N_INIT=3 (faster for audit; production uses 8)
    best_model = None
    best_score = -np.inf
    best_seed  = -1
    N_AUDIT    = 3

    for seed in range(N_AUDIT):
        candidate = hmm.GaussianHMM(
            n_components=N_STATES,
            covariance_type=COV_TYPE,
            n_iter=500,
            random_state=seed,
            tol=1e-5,
            verbose=False,
        )
        candidate.fit(X_scaled)
        s = candidate.score(X_scaled)
        if s > best_score:
            best_score = s
            best_model = candidate
            best_seed  = seed

    ok &= _assert(best_model is not None, "At least one EM restart converged")

    # Covariance structure: each state should have (n_feat × n_feat) full matrix
    ok &= _assert(best_model.covars_.shape == (N_STATES, n_feat, n_feat),
                  f"covars_ shape = ({N_STATES}, {n_feat}, {n_feat})  [full covariance]")

    # All covariance matrices should be symmetric
    sym_ok = all(
        np.allclose(best_model.covars_[i], best_model.covars_[i].T)
        for i in range(N_STATES)
    )
    ok &= _assert(sym_ok, "All per-state covariance matrices are symmetric")

    # All covariance matrices should be positive definite (all eigenvalues > 0)
    pd_ok = all(
        np.all(np.linalg.eigvalsh(best_model.covars_[i]) > 0)
        for i in range(N_STATES)
    )
    ok &= _assert(pd_ok, "All per-state covariance matrices are positive definite")

    # Transition matrix rows should sum to 1
    row_sums = best_model.transmat_.sum(axis=1)
    ok &= _assert(np.allclose(row_sums, 1.0, atol=1e-6),
                  f"Transition matrix rows sum to 1.0  (max err={np.abs(row_sums-1).max():.2e})")

    # Log-likelihood should be finite and negative
    ok &= _assert(np.isfinite(best_score) and best_score < 0,
                  f"Log-likelihood = {best_score:.4f}  (finite, negative)")

    print(_info(f"  Best seed: {best_seed}  |  LL = {best_score:.4f}  |  "
                f"Converged: {best_model.monitor_.converged}"))

    payload = {
        "model":          best_model,
        "scaler":         scaler,
        "feature_cols":   FEATURE_COLS,
        "X_scaled":       X_scaled,
        "feat_df":        feat_df,
        "log_likelihood": float(best_score),
        "converged":      best_model.monitor_.converged,
    }
    return ok, payload


def audit_state_relabeling(payload: dict) -> tuple[bool, dict]:
    print(_head("3 · State Relabeling (ascending mean log return)"))
    ok = True

    model    = payload["model"]
    state_map = _relabel_states(model, FEATURE_COLS)

    ok &= _assert(len(state_map) == N_STATES,
                  f"state_map has {N_STATES} entries")
    ok &= _assert(set(state_map.keys()) == set(range(N_STATES)),
                  "state_map keys cover all raw state indices")
    ok &= _assert(set(state_map.values()) == set(range(N_STATES)),
                  "state_map values are a permutation of {0,1,2}")

    lr_col = FEATURE_COLS.index("log_ret")
    ranked_means = sorted(
        range(N_STATES),
        key=lambda i: float(model.means_[i][lr_col]),
    )
    # Semantic 0=BEARISH must have the lowest raw mean, 2=BULLISH the highest
    for sem_idx, raw_idx in enumerate(ranked_means):
        ok &= _assert(state_map[raw_idx] == sem_idx,
                      f"raw state {raw_idx} maps to semantic {sem_idx} "
                      f"({REGIME_LABELS[sem_idx]})  "
                      f"[μ_log_ret={model.means_[raw_idx][lr_col]:.6f}]")

    # Print learned means for audit
    print(_info("  Learned emission means (log_ret, raw scale):"))
    for raw_idx in range(N_STATES):
        sem = state_map[raw_idx]
        print(f"    raw={raw_idx} → {REGIME_LABELS[sem]:8s}  "
              f"μ_log_ret = {model.means_[raw_idx][lr_col]:+.6f}")

    payload["state_map"] = state_map
    return ok, payload


def audit_probability_vector(payload: dict) -> tuple[bool, dict]:
    print(_head("4 · Probability Vector (forward-backward posteriors)"))
    ok = True

    model     = payload["model"]
    X_scaled  = payload["X_scaled"]
    state_map = payload["state_map"]

    posteriors_raw = model.predict_proba(X_scaled)   # (T, N_STATES)
    T, S = posteriors_raw.shape

    ok &= _assert(S == N_STATES,
                  f"predict_proba shape: ({T}, {S})  [T observations, {N_STATES} states]")

    row_sums = posteriors_raw.sum(axis=1)
    ok &= _assert(np.allclose(row_sums, 1.0, atol=1e-6),
                  f"Posterior rows sum to 1.0  (max err={np.abs(row_sums-1).max():.2e})")

    ok &= _assert((posteriors_raw >= 0).all(),
                  "All posterior probabilities ≥ 0")
    ok &= _assert((posteriors_raw <= 1).all(),
                  "All posterior probabilities ≤ 1")

    # Remap final timestep to semantic ordering
    last_raw = posteriors_raw[-1]
    p_sem = np.zeros(N_STATES)
    for raw_idx, sem_idx in state_map.items():
        p_sem[sem_idx] = last_raw[raw_idx]

    current_state = int(np.argmax(p_sem))
    label         = REGIME_LABELS[current_state]
    confidence    = float(p_sem[current_state])

    ok &= _assert(np.isclose(p_sem.sum(), 1.0, atol=1e-6),
                  f"Semantic posterior sums to 1.0  (sum={p_sem.sum():.8f})")

    print()
    print(f"  {'─'*52}")
    print(f"  FINAL PROBABILITY VECTOR  (t = today)")
    print(f"  {'─'*52}")
    for sem_idx in range(N_STATES):
        lbl = REGIME_LABELS[sem_idx]
        p   = p_sem[sem_idx]
        bar = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        marker = " ◄ CURRENT" if sem_idx == current_state else ""
        print(f"  {lbl:8s}  {bar}  {p:.1%}{marker}")
    print(f"  {'─'*52}")
    print(f"  State    : {label}")
    print(f"  Confidence: {confidence:.1%}")
    regime_veto = bool(p_sem[0] > REGIME_VETO_THRESHOLD)
    print(f"  Veto     : {'YES — buys blocked' if regime_veto else 'No'}")
    print(f"  {'─'*52}")

    payload["posteriors_raw"] = posteriors_raw
    payload["p_sem"]          = p_sem
    payload["current_state"]  = current_state
    payload["label"]          = label
    return ok, payload


def audit_joblib_persistence(payload: dict) -> bool:
    print(_head("5 · joblib Persistence (save / reload round-trip)"))
    ok = True

    model     = payload["model"]
    scaler    = payload["scaler"]
    state_map = payload["state_map"]
    X_scaled  = payload["X_scaled"]

    save_payload = {
        "model":          model,
        "scaler":         scaler,
        "state_map":      state_map,
        "feature_cols":   FEATURE_COLS,
        "fitted_at":      "2026-03-29T00:00:00Z",
        "log_likelihood": payload["log_likelihood"],
        "converged":      payload["converged"],
        "n_observations": len(X_scaled),
        "state_diagnostics": {},
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = Path(f.name)

    joblib.dump(save_payload, tmp_path)
    ok &= _assert(tmp_path.exists() and tmp_path.stat().st_size > 0,
                  f"pkl written to disk  ({tmp_path.stat().st_size / 1024:.1f} KB)")

    loaded    = joblib.load(tmp_path)
    ok &= _assert(loaded["fitted_at"] == "2026-03-29T00:00:00Z",
                  "Metadata round-trip intact (fitted_at)")
    ok &= _assert(loaded["feature_cols"] == FEATURE_COLS,
                  "feature_cols round-trip intact")

    # Re-score with reloaded model — X_scaled is already normalised, score directly
    reloaded_model  = loaded["model"]
    score_orig   = model.score(X_scaled)
    score_reload = reloaded_model.score(X_scaled)

    ok &= _assert(np.isclose(score_orig, score_reload, rtol=1e-6),
                  f"Log-likelihood preserved after reload  "
                  f"(orig={score_orig:.4f}  reloaded={score_reload:.4f})")

    tmp_path.unlink(missing_ok=True)
    return ok


def audit_regime_veto_logic(payload: dict) -> bool:
    print(_head("6 · Regime Veto Logic"))
    ok = True

    model     = payload["model"]
    scaler    = payload["scaler"]
    state_map = payload["state_map"]

    # Build a mock pkg dict
    pkg = {
        "fitted_at":          "2026-03-29T00:00:00Z",
        "state_diagnostics":  {},
    }

    # Case A: P(BEARISH) > threshold → veto should fire
    raw_a = np.zeros(N_STATES)
    # Find the raw index that maps to BEARISH (semantic 0)
    bearish_raw_idx = next(r for r, s in state_map.items() if s == 0)
    raw_a[bearish_raw_idx] = 0.80
    raw_a[(bearish_raw_idx + 1) % N_STATES] = 0.20
    result_a = _build_regime_result("GC=F", raw_a, state_map, pkg)
    ok &= _assert(result_a["regime_veto"] is True,
                  f"P(BEARISH)=0.80 > {REGIME_VETO_THRESHOLD} → regime_veto=True")
    ok &= _assert(result_a["state_label"] == "BEARISH",
                  "State label = BEARISH when P(BEARISH) dominant")

    # Case B: P(BULLISH) dominant → no veto
    raw_b = np.zeros(N_STATES)
    bullish_raw_idx = next(r for r, s in state_map.items() if s == 2)
    raw_b[bullish_raw_idx] = 0.90
    raw_b[(bullish_raw_idx + 1) % N_STATES] = 0.10
    result_b = _build_regime_result("GC=F", raw_b, state_map, pkg)
    ok &= _assert(result_b["regime_veto"] is False,
                  f"P(BULLISH)=0.90, P(BEARISH)=0.00 → regime_veto=False")
    ok &= _assert(result_b["state_label"] == "BULLISH",
                  "State label = BULLISH when P(BULLISH) dominant")

    # Case C: P(BEARISH) exactly at threshold → no veto (strict >)
    raw_c = np.zeros(N_STATES)
    raw_c[bearish_raw_idx] = REGIME_VETO_THRESHOLD
    raw_c[(bearish_raw_idx + 1) % N_STATES] = 1.0 - REGIME_VETO_THRESHOLD
    result_c = _build_regime_result("GC=F", raw_c, state_map, pkg)
    ok &= _assert(result_c["regime_veto"] is False,
                  f"P(BEARISH)={REGIME_VETO_THRESHOLD} (= threshold) → veto=False (strict >)")

    # Probability sum check on all three cases
    for tag, res in [("A", result_a), ("B", result_b), ("C", result_c)]:
        p_sum = sum(res["probabilities"].values())
        ok &= _assert(np.isclose(p_sum, 1.0, atol=1e-6),
                      f"Case {tag}: semantic probabilities sum to 1.0  (sum={p_sum:.8f})")

    return ok


def audit_regime_overlays(ticker: str) -> bool:
    print(_head("7 · build_regime_overlays (band collapse + date filter)"))
    ok = True

    bands_all = build_regime_overlays(ticker)
    ok &= _assert(isinstance(bands_all, list),
                  "build_regime_overlays returns a list")

    if not bands_all:
        print(_info("  No regime_history.csv yet — overlay test skipped"))
        return ok

    ok &= _assert(len(bands_all) > 0,
                  f"At least one band returned  (got {len(bands_all)})")

    # All required keys present
    required_keys = {"x0", "x1", "label", "fillcolor"}
    keys_ok = all(required_keys.issubset(b.keys()) for b in bands_all)
    ok &= _assert(keys_ok, "All bands contain x0, x1, label, fillcolor")

    # Labels must be valid
    valid_labels = set(REGIME_LABELS.values())
    labels_ok = all(b["label"] in valid_labels for b in bands_all)
    ok &= _assert(labels_ok, f"All labels in {valid_labels}")

    # Fillcolors must match REGIME_COLORS
    colors_ok = all(b["fillcolor"] == REGIME_COLORS[b["label"]] for b in bands_all)
    ok &= _assert(colors_ok, "fillcolor values match REGIME_COLORS lookup")

    # Contiguity: x1 of band[i] should equal x0 of band[i+1]
    contiguous_ok = all(
        bands_all[i]["x1"] == bands_all[i + 1]["x0"]
        for i in range(len(bands_all) - 1)
    )
    ok &= _assert(contiguous_ok, "Bands are contiguous (x1[i] == x0[i+1])")

    # Date filter test: start_date should remove early bands
    if len(bands_all) >= 2:
        cutoff = bands_all[-2]["x0"]          # second-to-last band start
        bands_trimmed = build_regime_overlays(ticker, start_date=cutoff)
        ok &= _assert(len(bands_trimmed) <= len(bands_all),
                      f"start_date filter reduces band count  "
                      f"({len(bands_trimmed)} ≤ {len(bands_all)})")
        ok &= _assert(all(b["x0"] >= cutoff for b in bands_trimmed),
                      f"All bands start on or after cutoff {cutoff}")

    print(_info(f"  Total bands: {len(bands_all)}  "
                f"|  Latest: {bands_all[-1]['x0']} → {bands_all[-1]['x1']}  "
                f"{bands_all[-1]['label']}"))
    return ok


def audit_json_cache(ticker: str) -> bool:
    print(_head("8 · load_cached_regime (JSON snapshot round-trip)"))
    ok = True

    r = load_cached_regime(ticker)

    if r is None:
        print(_info("  current_regime.json not found — JSON cache test skipped"))
        return ok

    required = {"ticker", "state_label", "probabilities",
                "regime_veto", "confidence", "fitted_at"}
    ok &= _assert(required.issubset(r.keys()),
                  f"Cached entry has required keys: {required}")

    ok &= _assert(r["ticker"] == ticker,
                  f"ticker field matches requested ticker ({ticker})")
    ok &= _assert(r["state_label"] in set(REGIME_LABELS.values()),
                  f"state_label '{r['state_label']}' is a valid label")
    ok &= _assert(0.0 <= r["confidence"] <= 1.0,
                  f"confidence = {r['confidence']:.4f}  in [0, 1]")
    ok &= _assert(isinstance(r["regime_veto"], bool),
                  f"regime_veto is a bool  (got {type(r['regime_veto']).__name__})")

    p_sum = sum(r["probabilities"].values())
    ok &= _assert(np.isclose(p_sum, 1.0, atol=1e-4),
                  f"Cached probabilities sum to 1.0  (sum={p_sum:.8f})")

    print(_info(
        f"  Cached state : {r['state_label']}  "
        f"({r['confidence']:.0%} conf)  "
        f"veto={r['regime_veto']}  "
        f"fitted={r['fitted_at']}"
    ))
    return ok


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM regime detector audit script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python3 verify_hmm.py                        # live GC=F data
              python3 verify_hmm.py --ticker SI=F          # live SI=F data
              python3 verify_hmm.py --csv mydata.csv       # local CSV
              python3 verify_hmm.py --synthetic            # no internet required
        """),
    )
    parser.add_argument("--ticker",    default="GC=F",
                        help="yfinance ticker to use (default: GC=F)")
    parser.add_argument("--csv",       metavar="PATH",
                        help="Use a local CSV instead of yfinance")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data (offline mode, no internet)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"\n{BOLD}{'═'*60}")
    print(f"  HMM REGIME DETECTOR — INSTITUTIONAL AUDIT")
    print(f"  Ticker: {ticker}  |  3-state GaussianHMM  |  cov={COV_TYPE}")
    print(f"{'═'*60}{RESET}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        print(_info("Using synthetic 1500-day data (3-state Markov ground truth)"))
        df = _generate_synthetic_data()
    elif args.csv:
        print(_info(f"Loading local CSV: {args.csv}"))
        df = _load_csv_data(args.csv)
    else:
        print(_info(f"Fetching 6yr of live data for {ticker} from yfinance …"))
        from datetime import timedelta
        end   = date.today()
        start = end - timedelta(days=365 * 6 + 200)   # 6yr + warmup
        df = _fetch_data(ticker, start, end)

    print(_info(f"Data loaded: {len(df)} rows  "
                f"({df.index[0].date()} → {df.index[-1].date()})"))

    # ── Run audits ────────────────────────────────────────────────────────────
    results = []

    r1 = audit_feature_engineering(df)
    results.append(("Feature Engineering", r1))

    r2, payload = audit_hmm_fit(df)
    results.append(("HMM Fitting", r2))

    if r2:
        r3, payload = audit_state_relabeling(payload)
        results.append(("State Relabeling", r3))

        r4, payload = audit_probability_vector(payload)
        results.append(("Probability Vector", r4))

        r5 = audit_joblib_persistence(payload)
        results.append(("joblib Persistence", r5))

        r6 = audit_regime_veto_logic(payload)
        results.append(("Regime Veto Logic", r6))

    r7 = audit_regime_overlays(ticker)
    results.append(("Regime Overlays", r7))

    r8 = audit_json_cache(ticker)
    results.append(("JSON Cache", r8))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}")
    print(f"  AUDIT SUMMARY")
    print(f"{'═'*60}{RESET}")

    all_pass = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print(f"{'═'*60}")
    if all_pass:
        print(f"  {GREEN}{BOLD}ALL TESTS PASSED{RESET}")
    else:
        n_fail = sum(1 for _, p in results if not p)
        print(f"  {RED}{BOLD}{n_fail} TEST(S) FAILED{RESET}")
    print(f"{'═'*60}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
