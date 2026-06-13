#!/usr/bin/env python3
"""
scripts/regime_detector.py
==========================
Institutional-grade Hidden Markov Model regime engine for precious metals.

Architecture
------------
  Separate 3-state GaussianHMM (full covariance) per metal: GC=F, SI=F
  Rolling 5-year window (~1,250 trading days)  —  pre-COVID ZIRP era discarded
  N_INIT=8 random restarts  —  best log-likelihood wins
  Baum-Welch EM fitting  →  Viterbi decoding (chart bands)  +  forward-backward
  (posterior probabilities for live UI display)

Automatic state relabeling by learned mean log return (ascending):
    State 0  →  BEARISH   (negative drift, risk-off liquidation)
    State 1  →  VOLATILE  (near-zero drift, indecision / macro shock)
    State 2  →  BULLISH   (positive drift, safe-haven accumulation)

Labels are re-derived after every daily refit, so semantics stay consistent
regardless of how the EM algorithm initialises.

Feature vector (5 stationary features — no price levels, only returns/diffs):
    log_ret    :  log(Close_t / Close_{t-1})
    vol_ratio  :  vol_5d / vol_21d   (spikes before regime transitions; relative vol)
    dxy_ret    :  log(DXY_t / DXY_{t-1})   (gold's primary macro driver)
    vix_chg    :  VIX_t − VIX_{t-1}         (fear / risk-off signal)
    rsi_14     :  Wilder RSI(14) − 0.5      (mean-reverting momentum; replaces
                  trend_50 which locked near ±1 during sustained trends)

NOTE: vol_21d (absolute 21-day vol) was removed because it dominated state separation.
When gold enters a sustained high-vol macro regime (tariff shock, Ukraine, SVB), vol_21d
drifts 2-3σ above its training mean and all observations lock into the high-vol state
regardless of direction. Replacing it with vol_ratio (relative vol) lets the model
distinguish direction (bull vs bear) within any vol environment.

Outputs
-------
  models/regime/{ticker}_hmm.pkl    — GaussianHMM + StandardScaler + metadata
  data/regime_history.csv           — full Viterbi sequence  (chart overlay)
  data/current_regime.json          — latest inference result (fast UI load)

Integration
-----------
  from scripts.regime_detector import load_cached_regime, build_regime_overlays
  r = load_cached_regime("GC=F")     # instant — reads JSON, no inference
  bands = build_regime_overlays("GC=F", start_date="2024-01-01")

CLI
---
  python3 scripts/regime_detector.py --fit GC=F
  python3 scripts/regime_detector.py --fit-all
  python3 scripts/regime_detector.py --regime GC=F
  python3 scripts/regime_detector.py --history GC=F
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")
warnings.filterwarnings("ignore", message=".*converge.*", category=UserWarning)

try:
    from hmmlearn import hmm
except ImportError:
    raise ImportError("pip install hmmlearn>=0.3.2")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = ROOT / "models" / "regime"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = ROOT / "data" / "regime_history.csv"
REGIME_JSON = ROOT / "data" / "current_regime.json"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_STATES      = 3          # BEARISH / RANGING / BULLISH
COV_TYPE      = "full"     # Full cross-feature covariance per state
N_ITER        = 2000       # Max EM iterations per restart
N_INIT        = 8          # Random restarts — keep highest log-likelihood
TOL           = 1e-5       # EM convergence tolerance
MIN_COVAR     = 1e-2       # Covariance floor — prevents diagonal collapse to zero.
                            # Default in hmmlearn is 1e-3; 1e-2 (10×) is needed
                            # after switching to RSI which can lock in narrow bands
                            # during sustained trends, causing near-zero variance.

WINDOW_YEARS  = 5          # Rolling window length (discards pre-COVID ZIRP era)
WARMUP_DAYS   = 150        # Buffer before window start — needed for RSI(14),
                            # vol_21d (21d), vol_ratio (5d+21d)
INFERENCE_DAYS = 252       # Context for live forward-backward pass (1yr).
                            # Reduced from 504: shorter backward-pass memory lets
                            # the filter detect regime recoveries ~2x faster.

# P(BEARISH) above this threshold → regime_veto = True → block all buy signals
REGIME_VETO_THRESHOLD = 0.65

# Semantic labels — indices must match the relabeling sort order (ascending return)
REGIME_LABELS = {0: "BEARISH", 1: "VOLATILE", 2: "BULLISH"}

# Plotly vrect fill colours (low opacity — overlaid on candlestick chart)
REGIME_COLORS = {
    "BEARISH":  "rgba(239,68,68,0.11)",
    "VOLATILE": "rgba(245,158,11,0.07)",
    "BULLISH":  "rgba(34,197,94,0.09)",
}

# Display colours for UI signal panel
REGIME_TEXT_COLORS = {
    "BEARISH":  "#ef4444",
    "VOLATILE": "#f59e0b",
    "BULLISH":  "#22c55e",
}

FEATURE_COLS = [
    "log_ret",
    # vol_21d REMOVED — absolute vol level dominated Mahalanobis distance and
    # caused the model to lock into the high-vol state for entire macro regimes
    # (tariff shock, Ukraine, SVB) regardless of price direction.  vol_ratio
    # (5d/21d) is kept as the relative vol signal.
    "vol_ratio",
    "dxy_ret",
    "vix_chg",
    "rsi_14",      # Replaces trend_50 — RSI(14) normalised to [-0.5, +0.5].
                   # Mean-reverts naturally: oversold = -0.2, neutral = 0.0,
                   # overbought = +0.2. Prevents state lock-in during trends.
]

VALID_TICKERS = {"GC=F", "SI=F"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_ticker(ticker: str) -> str:
    """Filesystem-safe ticker name (GC=F → GC_F)."""
    return ticker.replace("=", "_").replace("^", "")


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily OHLCV for `ticker` + DXY + VIX.
    All three series are aligned on the metal's trading calendar.
    DXY and VIX are forward-filled to handle holiday misalignment.
    """
    start_str = str(start)
    end_str   = str(end + timedelta(days=2))   # yfinance end is exclusive + weekend buffer

    def _dl(sym: str) -> pd.DataFrame:
        import time
        for attempt in range(3):
            try:
                df = yf.download(
                    sym, start=start_str, end=end_str,
                    interval="1d", progress=False, auto_adjust=True,
                )
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    return df
            except Exception as e:
                logger.debug(f"  [{sym}] download attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)
        return pd.DataFrame()

    metal_raw = _dl(ticker)
    if metal_raw.empty:
        raise RuntimeError(f"[{ticker}] Failed to download price data after 3 attempts.")
    metal = metal_raw[["Close", "Volume"]].copy()

    dxy_raw = _dl("DX-Y.NYB")
    if dxy_raw.empty:
        raise RuntimeError(f"[{ticker}] Failed to download DXY data after 3 attempts.")
    dxy = dxy_raw[["Close"]].rename(columns={"Close": "DXY_Close"})

    # VIX: try primary symbol, then fallback; synthesize neutral value if unavailable
    vix_raw = _dl("^VIX")
    if vix_raw.empty:
        logger.warning(f"[{ticker}] ^VIX download failed — trying VIXY as fallback")
        vix_raw = _dl("VIXY")
    if not vix_raw.empty:
        vix = vix_raw[["Close"]].rename(columns={"Close": "VIX_Close"})
        df = metal.join(dxy, how="left").join(vix, how="left")
    else:
        logger.warning(
            f"[{ticker}] VIX unavailable — synthesizing neutral VIX=20 (vix_chg=0)"
        )
        df = metal.join(dxy, how="left")
        df["VIX_Close"] = 20.0

    df["DXY_Close"] = df["DXY_Close"].ffill()
    df["VIX_Close"] = df["VIX_Close"].ffill().fillna(20.0)
    df = df.dropna(subset=["Close", "DXY_Close", "VIX_Close"])

    logger.debug(f"  [{ticker}] fetched {len(df)} rows  {start_str} → {end_str}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 6 stationary features from raw OHLCV + macro data.
    Returns a copy of df with the feature columns appended.
    NaN rows (from rolling windows) are dropped.
    """
    out = df.copy()

    # 1. Log return  —  stationarity ensured by differencing
    out["log_ret"]  = np.log(out["Close"] / out["Close"].shift(1))

    # 2. 21-day realised volatility  —  level of vol regime
    out["vol_21d"]  = out["log_ret"].rolling(21).std()

    # 3. Volatility ratio  —  early-warning signal for regime transitions
    #    Spikes sharply when short-term vol expands faster than long-term vol
    vol_5d           = out["log_ret"].rolling(5).std()
    out["vol_ratio"] = (vol_5d / out["vol_21d"]).replace([np.inf, -np.inf], np.nan)

    # 4. DXY log return  —  gold's primary negative-correlation macro driver
    out["dxy_ret"]  = np.log(out["DXY_Close"] / out["DXY_Close"].shift(1))

    # 5. VIX daily change  —  raw diff (not log) to avoid distortion near 0
    out["vix_chg"]  = out["VIX_Close"].diff()

    # 6. RSI(14) normalised to [-0.5, +0.5]  — replaces trend_50.
    #    Wilder's smoothed RSI: RSI=50 → 0.0, RSI=30 (oversold) → -0.2,
    #    RSI=70 (overbought) → +0.2.  Mean-reverts strongly; never locks
    #    near ±1 during sustained trends the way (Close-MA50)/Close did.
    delta          = out["Close"].diff()
    gain           = delta.clip(lower=0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss           = (-delta.clip(upper=0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_raw        = 1.0 - 1.0 / (1.0 + gain / loss.replace(0, np.nan))  # [0, 1]
    out["rsi_14"]  = rsi_raw - 0.5                                          # [-0.5, +0.5]

    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)


# ── State relabeling ──────────────────────────────────────────────────────────

def _relabel_states(
    model: hmm.GaussianHMM,
    feature_cols: list[str],
) -> dict[int, int]:
    """
    Map raw HMM state indices (arbitrary from EM) to semantic labels:
        0 = BEARISH  (lowest mean log return → most negative drift)
        1 = RANGING  (median mean log return)
        2 = BULLISH  (highest mean log return → strongest positive drift)

    Deterministic and consistent across daily refits.  The sort key is the
    learned emission mean of the 'log_ret' feature — not the realised data mean,
    so it is unaffected by the rolling window shift.

    Returns
    -------
    dict  {raw_state_index: semantic_index}
    """
    log_ret_col = feature_cols.index("log_ret")

    # Learned mean log return per state from emission distribution
    ranked = sorted(
        range(model.n_components),
        key=lambda i: float(model.means_[i][log_ret_col]),
    )
    # ranked[0] → most bearish raw idx, ranked[2] → most bullish raw idx
    return {raw_idx: sem_idx for sem_idx, raw_idx in enumerate(ranked)}


# ── Regime history I/O ────────────────────────────────────────────────────────

def _write_regime_history(
    ticker:          str,
    feat_index:      pd.DatetimeIndex,
    viterbi_raw:     np.ndarray,
    state_map:       dict[int, int],
    posteriors_raw:  np.ndarray,
) -> None:
    """
    Write the full Viterbi state sequence to regime_history.csv.
    Existing rows for this ticker are replaced (idempotent on daily refit).
    """
    rows = []
    for i, (dt, raw_state) in enumerate(zip(feat_index, viterbi_raw)):
        sem = state_map[int(raw_state)]

        # Remap posterior probabilities from raw to semantic ordering
        p = np.zeros(N_STATES)
        for r_idx, s_idx in state_map.items():
            p[s_idx] = posteriors_raw[i, r_idx]

        rows.append({
            "date":         dt.date().isoformat(),
            "ticker":       ticker,
            "regime_raw":   int(raw_state),
            "regime_label": REGIME_LABELS[sem],
            "p_bearish":    round(float(p[0]), 6),
            "p_volatile":   round(float(p[1]), 6),
            "p_bullish":    round(float(p[2]), 6),
        })

    new_df = pd.DataFrame(rows)

    if HISTORY_CSV.exists():
        existing = pd.read_csv(HISTORY_CSV)
        existing["date"] = pd.to_datetime(existing["date"], format="mixed").dt.date.astype(str)
        existing = existing[existing["ticker"] != ticker]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.sort_values(["ticker", "date"]).to_csv(HISTORY_CSV, index=False)
    logger.info(f"  [{ticker}] regime_history.csv updated  ({len(new_df)} rows)")


# ── Core fitting ──────────────────────────────────────────────────────────────

def fit(ticker: str) -> dict:
    """
    Fit a 3-state GaussianHMM on the rolling 5-year window for `ticker`.

    Pipeline
    --------
    1.  Fetch 5yr + 150-day warmup from yfinance (metal + DXY + VIX)
    2.  Engineer 6 stationary features
    3.  Trim to exact 5-year rolling window (discard warmup rows)
    4.  StandardScaler normalisation
    5.  N_INIT=8 random-restart EM — best log-likelihood wins
    6.  Automatic state relabeling (ascending mean return)
    7.  Viterbi decode + posterior probabilities
    8.  Persist model + metadata to models/regime/{ticker}_hmm.pkl
    9.  Write full Viterbi sequence to data/regime_history.csv
    10. Write current regime snapshot to data/current_regime.json

    Returns
    -------
    dict  —  model metadata payload (same structure as the .pkl)
    """
    if ticker not in VALID_TICKERS:
        raise ValueError(f"Unknown ticker '{ticker}'. Valid: {VALID_TICKERS}")

    # ── Window boundaries ─────────────────────────────────────────────────────
    window_end   = date.today()
    window_start = window_end  - timedelta(days=365 * WINDOW_YEARS)
    fetch_start  = window_start - timedelta(days=WARMUP_DAYS)

    logger.info(f"[{ticker}] Fetching data  {fetch_start} → {window_end} …")
    raw     = _fetch_data(ticker, fetch_start, window_end)
    feat_df = _build_features(raw)

    # Drop warmup rows — enforce rolling 5-year window
    feat_df = feat_df[feat_df.index.date >= window_start]  # type: ignore[attr-defined]

    n_obs = len(feat_df)
    if n_obs < 252:
        raise RuntimeError(
            f"[{ticker}] Only {n_obs} observations after feature engineering "
            f"(need ≥252). Check yfinance connectivity."
        )

    logger.info(f"[{ticker}] {n_obs} observations × {len(FEATURE_COLS)} features")

    # ── Normalise ─────────────────────────────────────────────────────────────
    X_raw    = feat_df[FEATURE_COLS].values.astype(np.float64)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Multi-start EM ────────────────────────────────────────────────────────
    best_model: Optional[hmm.GaussianHMM] = None
    best_score  = -np.inf
    best_seed   = -1

    for seed in range(N_INIT):
        try:
            candidate = hmm.GaussianHMM(
                n_components=N_STATES,
                covariance_type=COV_TYPE,
                n_iter=N_ITER,
                random_state=seed,
                tol=TOL,
                min_covar=MIN_COVAR,   # prevents covariance collapse (see MIN_COVAR)
                verbose=False,
            )
            candidate.fit(X_scaled)
            score = candidate.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = candidate
                best_seed  = seed
        except Exception as exc:
            logger.debug(f"  seed={seed} failed: {exc}")

    if best_model is None:
        raise RuntimeError(f"[{ticker}] All {N_INIT} HMM restarts failed")

    converged = best_model.monitor_.converged
    logger.info(
        f"[{ticker}] EM complete  "
        f"seed={best_seed}  ll={best_score:.4f}  converged={converged}"
    )
    if not converged:
        logger.warning(f"[{ticker}] Did not converge — consider raising N_ITER")

    # ── Relabel states ────────────────────────────────────────────────────────
    state_map = _relabel_states(best_model, FEATURE_COLS)

    # ── Decode ────────────────────────────────────────────────────────────────
    viterbi_raw    = best_model.predict(X_scaled)          # (T,)       hard labels
    posteriors_raw = best_model.predict_proba(X_scaled)   # (T, N)     soft probs

    # Renormalise each row (float precision can cause rows to not sum exactly to 1)
    row_sums = posteriors_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    posteriors_raw = posteriors_raw / row_sums

    # ── Per-state diagnostics ─────────────────────────────────────────────────
    lr_col  = FEATURE_COLS.index("log_ret")
    # vol_21d removed from features — use log_ret covariance diagonal for σ diagnostic
    vr_col  = FEATURE_COLS.index("vol_ratio") if "vol_ratio" in FEATURE_COLS else None

    state_diagnostics: dict[str, dict] = {}
    for raw_idx, sem_idx in state_map.items():
        label     = REGIME_LABELS[sem_idx]
        occupancy = float((viterbi_raw == raw_idx).mean())

        # Emission distribution statistics (from learned parameters, not data)
        mean_ret   = float(best_model.means_[raw_idx][lr_col])
        sigma_ret  = float(np.sqrt(best_model.covars_[raw_idx][lr_col, lr_col]))

        state_diagnostics[label] = {
            "mean_log_ret_bps": round(mean_ret * 1e4, 2),   # basis points/day
            "vol_daily":        round(sigma_ret, 6),          # σ of daily return (scaled space)
            "ann_vol_approx":   round(sigma_ret * np.sqrt(252) * 100, 2),  # approx % pa
            "occupancy_pct":    round(occupancy * 100, 2),
        }
        logger.info(
            f"  {label:8s}: drift={mean_ret*1e4:+.1f} bps/d  "
            f"σ_ann≈{sigma_ret*np.sqrt(252)*100:.1f}%  "
            f"occ={occupancy*100:.1f}%"
        )

    # ── Persist model ─────────────────────────────────────────────────────────
    payload = {
        "model":             best_model,
        "scaler":            scaler,
        "state_map":         state_map,           # {raw_idx: semantic_idx}
        "feature_cols":      FEATURE_COLS,
        "fitted_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_start":      str(feat_df.index[0].date()),
        "window_end":        str(feat_df.index[-1].date()),
        "n_observations":    n_obs,
        "log_likelihood":    float(best_score),
        "converged":         converged,
        "state_diagnostics": state_diagnostics,
    }
    model_path = MODEL_DIR / f"{_safe_ticker(ticker)}_hmm.pkl"
    joblib.dump(payload, model_path)
    logger.info(f"[{ticker}] Saved → {model_path}")

    # ── Update regime history CSV ──────────────────────────────────────────────
    _write_regime_history(ticker, feat_df.index, viterbi_raw, state_map, posteriors_raw)

    # ── Write live snapshot ───────────────────────────────────────────────────
    _write_current_regime_snapshot(ticker, posteriors_raw[-1], state_map, payload)

    return payload


# ── Live inference ────────────────────────────────────────────────────────────

def get_current_regime(ticker: str) -> dict:
    """
    Load the persisted model and run the forward-backward algorithm on the
    last INFERENCE_DAYS of data.

    At the final timestep, the smoothed posterior equals the filtered
    posterior (no future data leakage) — safe for live signal generation.

    Returns
    -------
    {
        "ticker":        "GC=F",
        "current_state": 2,
        "state_label":   "BULLISH",
        "probabilities": {"BEARISH": 0.05, "VOLATILE": 0.10, "BULLISH": 0.85},
        "regime_veto":   False,   # True when P(BEARISH) > REGIME_VETO_THRESHOLD
        "confidence":    0.85,    # probability of the winning state
        "fitted_at":     "2026-03-29T22:30:00Z",
        "state_diagnostics": { ... }
    }
    """
    model_path = MODEL_DIR / f"{_safe_ticker(ticker)}_hmm.pkl"
    if not model_path.exists():
        return {
            "error": (
                f"No fitted model for {ticker}. "
                f"Run: python3 scripts/regime_detector.py --fit {ticker}"
            )
        }

    pkg       = joblib.load(model_path)
    model     = pkg["model"]
    scaler    = pkg["scaler"]
    state_map = pkg["state_map"]

    # Fetch inference context window
    end   = date.today()
    start = end - timedelta(days=int(INFERENCE_DAYS * 1.6) + WARMUP_DAYS)

    try:
        raw     = _fetch_data(ticker, start, end)
        feat_df = _build_features(raw).tail(INFERENCE_DAYS)
    except Exception as exc:
        return {"error": f"Data fetch failed during inference: {exc}"}

    if len(feat_df) < 50:
        return {"error": f"Insufficient recent data: {len(feat_df)} rows"}

    X_scaled       = scaler.transform(feat_df[FEATURE_COLS].values.astype(np.float64))
    posteriors_raw = model.predict_proba(X_scaled)   # (T, N_STATES)

    # Log degenerate posteriors before epsilon smoothing (raw values are informative).
    last_raw = posteriors_raw[-1].astype(np.float64)
    if float(last_raw.max()) / max(last_raw.sum(), 1e-300) > 0.9995:
        logger.warning(
            f"[{ticker}] Degenerate posterior (forward-backward collapse): "
            f"{last_raw.tolist()!r} — epsilon floor applied in _build_regime_result."
        )

    result = _build_regime_result(ticker, last_raw, state_map, pkg)

    # Persist fresh snapshot to JSON
    _write_json_snapshot(ticker, result)

    return result


def _build_regime_result(
    ticker:    str,
    last_raw:  np.ndarray,
    state_map: dict[int, int],
    pkg:       dict,
) -> dict:
    """Convert raw posterior vector to a semantic regime result dict."""
    # Epsilon-smooth the raw posterior before remapping.
    # The forward-backward algorithm can produce numerically degenerate vectors
    # (e.g. [1e-24, 1e-21, 1.0]) when the current feature point is very unlikely
    # under two states.  A floor of 1e-3 ensures no state shows exactly 0% / 100%
    # in the UI while leaving the dominant state at ≥99.8% — signal is preserved.
    last_raw = np.maximum(last_raw.astype(np.float64), 1e-3)
    last_raw = last_raw / last_raw.sum()

    # Remap probabilities from raw → semantic ordering
    p_sem = np.zeros(N_STATES)
    for raw_idx, sem_idx in state_map.items():
        p_sem[sem_idx] = last_raw[raw_idx]

    current_state = int(np.argmax(p_sem))

    return {
        "ticker":        ticker,
        "current_state": current_state,
        "state_label":   REGIME_LABELS[current_state],
        "probabilities": {
            REGIME_LABELS[i]: round(float(p_sem[i]), 4)
            for i in range(N_STATES)
        },
        "regime_veto":   bool(p_sem[0] > REGIME_VETO_THRESHOLD),
        "confidence":    round(float(p_sem[current_state]), 4),
        "fitted_at":     pkg.get("fitted_at", "unknown"),
        "state_diagnostics": pkg.get("state_diagnostics", {}),
    }


def _write_current_regime_snapshot(
    ticker:    str,
    last_raw:  np.ndarray,
    state_map: dict[int, int],
    pkg:       dict,
) -> None:
    """Write the current regime snapshot into current_regime.json (fast UI load)."""
    result = _build_regime_result(ticker, last_raw, state_map, pkg)
    _write_json_snapshot(ticker, result)


def _write_json_snapshot(ticker: str, result: dict) -> None:
    existing: dict = {}
    if REGIME_JSON.exists():
        try:
            existing = json.loads(REGIME_JSON.read_text())
        except Exception:
            pass
    existing[ticker] = result
    REGIME_JSON.parent.mkdir(parents=True, exist_ok=True)
    REGIME_JSON.write_text(json.dumps(existing, indent=2))


# ── Fast cache read (for UI — no inference, no network) ──────────────────────

def load_cached_regime(ticker: str) -> Optional[dict]:
    """
    Instant read from current_regime.json.  Used by the Streamlit UI on every
    page render — zero latency, zero API calls.

    Falls back to None if the cache is missing or contains an error entry.
    Call get_current_regime(ticker) to force a fresh inference pass.
    """
    if not REGIME_JSON.exists():
        return None
    try:
        data = json.loads(REGIME_JSON.read_text())
        entry = data.get(ticker, {})
        return entry if ("error" not in entry and "state_label" in entry) else None
    except Exception:
        return None


# ── Chart overlay builder ─────────────────────────────────────────────────────

def build_regime_overlays(
    ticker:     str,
    start_date: Optional[str] = None,
) -> list[dict]:
    """
    Load regime_history.csv and collapse the Viterbi sequence into a list of
    contiguous regime *bands* ready to be passed to Plotly fig.add_vrect().

    Parameters
    ----------
    ticker      : "GC=F" or "SI=F"
    start_date  : ISO date string  (e.g. "2025-01-01") to trim old history

    Returns
    -------
    List of dicts, each representing one contiguous regime band:
        {
            "x0":        "2025-01-15",
            "x1":        "2025-03-02",
            "label":     "BULLISH",
            "fillcolor": "rgba(34,197,94,0.09)",
        }
    """
    if not HISTORY_CSV.exists():
        return []

    try:
        df = pd.read_csv(HISTORY_CSV)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
    except Exception:
        return []

    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if df.empty:
        return []

    bands: list[dict] = []
    current_label = str(df.at[0, "regime_label"])
    band_start    = df.at[0, "date"]

    for i in range(1, len(df)):
        row_label = str(df.at[i, "regime_label"])
        if row_label != current_label:
            bands.append({
                "x0":        str(band_start.date()),
                "x1":        str(df.at[i, "date"].date()),
                "label":     current_label,
                "fillcolor": REGIME_COLORS.get(current_label, "rgba(128,128,128,0.07)"),
            })
            current_label = row_label
            band_start    = df.at[i, "date"]

    # Close final band at last available date
    bands.append({
        "x0":        str(band_start.date()),
        "x1":        str(df.iloc[-1]["date"].date()),
        "label":     current_label,
        "fillcolor": REGIME_COLORS.get(current_label, "rgba(128,128,128,0.07)"),
    })

    return bands


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli_fit(ticker: str) -> None:
    import time
    print(f"\n{'━'*64}")
    print(f"  REGIME DETECTOR — Fitting {ticker}")
    print(f"  Window: {WINDOW_YEARS}yr rolling  |  States: {N_STATES}  |  Cov: {COV_TYPE}")
    print(f"  Restarts: {N_INIT}  |  Max EM iter: {N_ITER}  |  Tol: {TOL}")
    print(f"{'━'*64}")

    t0      = time.time()
    payload = fit(ticker)
    elapsed = time.time() - t0

    print(f"\n  Fitted in {elapsed:.1f}s")
    print(f"  Observations : {payload['n_observations']}")
    print(f"  Window       : {payload['window_start']}  →  {payload['window_end']}")
    print(f"  Log-likelihood: {payload['log_likelihood']:.4f}")
    print(f"  Converged    : {payload['converged']}")
    print(f"\n  State diagnostics:")
    for label in ["BEARISH", "VOLATILE", "BULLISH"]:
        d = payload["state_diagnostics"].get(label, {})
        if d:
            print(
                f"    {label:8s}: drift={d['mean_log_ret_bps']:+6.1f} bps/d  "
                f"σ_ann≈{d['ann_vol_approx']:5.1f}%  "
                f"occupancy={d['occupancy_pct']:5.1f}%"
            )
    print(f"\n{'━'*64}\n")


def _cli_regime(ticker: str) -> None:
    print(f"\n{'━'*64}")
    print(f"  REGIME DETECTOR — Current Regime: {ticker}")
    print(f"{'━'*64}")

    r = get_current_regime(ticker)
    if "error" in r:
        print(f"\n  Error: {r['error']}\n")
        return

    print(f"\n  State       : {r['state_label']}  (confidence: {r['confidence']:.1%})")
    print(f"  Regime Veto : {'YES — all buy signals blocked' if r['regime_veto'] else 'No'}")
    print(f"\n  Probability vector:")
    for label in ["BEARISH", "VOLATILE", "BULLISH"]:
        p   = r["probabilities"].get(label, 0.0)
        bar = "█" * int(p * 32) + "░" * (32 - int(p * 32))
        print(f"    {label:8s}  {bar}  {p:.1%}")
    print(f"\n  Model fitted : {r['fitted_at']}")
    print(f"{'━'*64}\n")


def _cli_history(ticker: str) -> None:
    bands = build_regime_overlays(ticker)
    print(f"\n  {ticker} — {len(bands)} regime bands (most recent 30):")
    for b in bands[-30:]:
        print(f"    {b['x0']}  →  {b['x1']}  {b['label']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precious metals regime detector — 3-state Gaussian HMM"
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--fit",      metavar="TICKER", help="Fit HMM for one ticker")
    g.add_argument("--fit-all",  action="store_true", help="Fit HMM for GC=F and SI=F")
    g.add_argument("--regime",   metavar="TICKER", help="Print current regime")
    g.add_argument("--history",  metavar="TICKER", help="Print regime band summary")
    args = parser.parse_args()

    # Default: fit all tickers when called with no arguments
    if not any([args.fit, args.fit_all, args.regime, args.history]):
        args.fit_all = True

    if args.fit:
        _cli_fit(args.fit.upper())
    elif args.fit_all:
        for t in sorted(VALID_TICKERS):
            _cli_fit(t)
    elif args.regime:
        _cli_regime(args.regime.upper())
    elif args.history:
        _cli_history(args.history.upper())
