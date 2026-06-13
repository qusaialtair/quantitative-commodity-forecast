#!/usr/bin/env python3
"""
Model Performance Tracker
==========================
Tracks prediction accuracy across all models and computes
dynamic ensemble weights based on recent performance.

Runs daily after the pipeline. Reads prediction logs, compares
against realized prices, and outputs adaptive weights.

Usage:
    python3 scripts/model_performance_tracker.py
    python3 scripts/model_performance_tracker.py --lookback 60
    python3 scripts/model_performance_tracker.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PREDICTION_LOG  = ROOT / "data" / "prediction_log.csv"
ENSEMBLE_WEIGHTS = ROOT / "data" / "ensemble_weights.json"
DATA_DIR        = ROOT / "data"

# ── CSV schema ───────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "model", "ticker", "pred_date", "target_date",
    "predicted_pct", "predicted_price",
    "actual_price", "actual_pct", "direction_correct",
]

# ── Tuning constants ─────────────────────────────────────────────────────────
DEFAULT_LOOKBACK   = 30        # days of history for accuracy evaluation
WEIGHT_FLOOR       = 0.10      # no model drops below 10% weight
SOFTMAX_TEMP       = 1.0       # softmax temperature for weight computation
HIT_BAND_PCT       = 2.0       # "within 2%" hit-rate band
DEGRADATION_FACTOR = 2.0       # MAE_10 > 2 * MAE_90 => DEGRADING
STALENESS_DAYS     = 7         # no predictions in 7 days => STALE

# ── ANSI helpers ─────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _TTY else t


RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
GREEN  = lambda t: _c("32;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)


# ==============================================================================
#  1. PREDICTION LOGGER
# ==============================================================================

def _ensure_log_exists() -> None:
    """Create prediction_log.csv with headers if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_LOG.exists():
        with open(PREDICTION_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def log_prediction(
    model_name: str,
    ticker: str,
    pred_date: str,
    target_date: str,
    predicted_pct: float,
    predicted_price: float,
) -> None:
    """
    Append a new prediction row to data/prediction_log.csv.

    Called by predict_next and daily_trainer after each prediction.
    actual_price, actual_pct, and direction_correct are left blank --
    they are filled in later by evaluate_accuracy().

    Parameters
    ----------
    model_name      : e.g. "GoldLSTM-v1", "GCF_tactical", "GCF_strategic"
    ticker          : e.g. "GC=F", "SI=F"
    pred_date       : ISO date when the prediction was made, e.g. "2026-05-01"
    target_date     : ISO date the prediction targets, e.g. "2026-05-06"
    predicted_pct   : predicted percentage change from pred_date close
    predicted_price : predicted target-date price in USD
    """
    _ensure_log_exists()
    with open(PREDICTION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            ticker,
            pred_date,
            target_date,
            round(predicted_pct, 4),
            round(predicted_price, 2),
            "",    # actual_price  -- filled later
            "",    # actual_pct    -- filled later
            "",    # direction_correct -- filled later
        ])


def _load_prediction_log() -> pd.DataFrame:
    """Load the prediction log CSV into a DataFrame, handling empty/missing files."""
    _ensure_log_exists()
    df = pd.read_csv(PREDICTION_LOG, dtype=str)
    if df.empty:
        return df

    # Parse dates
    df["pred_date"]   = pd.to_datetime(df["pred_date"], errors="coerce").dt.date
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date

    # Parse numeric columns (blanks become NaN)
    for col in ["predicted_pct", "predicted_price", "actual_price", "actual_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["direction_correct"] = pd.to_numeric(df["direction_correct"], errors="coerce")

    return df


def _save_prediction_log(df: pd.DataFrame) -> None:
    """Write the DataFrame back to prediction_log.csv."""
    _ensure_log_exists()
    df.to_csv(PREDICTION_LOG, index=False)


# ==============================================================================
#  2. ACCURACY EVALUATOR
# ==============================================================================

def _fetch_close_price(ticker: str, target_date: date) -> float | None:
    """
    Fetch closing price for ticker on or near target_date via yfinance.
    Returns None on failure. Searches up to +3 calendar days to handle
    weekends and holidays.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    for offset in range(4):
        check = target_date + timedelta(days=offset)
        end   = check + timedelta(days=1)
        try:
            hist = yf.Ticker(ticker).history(
                start=check.isoformat(), end=end.isoformat()
            )
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return None


def _backfill_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where target_date has passed but actual_price is still NaN,
    fetch the realized price and compute actual_pct and direction_correct.
    """
    today = date.today()
    mask = (
        df["actual_price"].isna()
        & df["target_date"].notna()
        & (df["target_date"] <= today)
    )
    if not mask.any():
        return df

    updated = 0
    for idx in df[mask].index:
        ticker      = df.at[idx, "ticker"]
        target_dt   = df.at[idx, "target_date"]
        pred_pct    = df.at[idx, "predicted_pct"]
        pred_price  = df.at[idx, "predicted_price"]

        if pd.isna(target_dt) or pd.isna(pred_pct):
            continue

        actual = _fetch_close_price(ticker, target_dt)
        if actual is None:
            continue

        # Compute the baseline price from predicted_pct and predicted_price
        # predicted_price = base_price * (1 + predicted_pct / 100)
        # So base_price = predicted_price / (1 + predicted_pct / 100)
        if abs(pred_pct) < 1e-9:
            base_price = pred_price
        else:
            base_price = pred_price / (1.0 + pred_pct / 100.0)

        if base_price <= 0:
            continue

        actual_pct = (actual - base_price) / base_price * 100.0

        # Direction: predicted UP (>0) and went UP (>0) => correct (1)
        dir_correct = int(
            (pred_pct > 0 and actual_pct > 0)
            or (pred_pct < 0 and actual_pct < 0)
            or (abs(pred_pct) < 0.01 and abs(actual_pct) < 0.01)
        )

        df.at[idx, "actual_price"]       = round(actual, 2)
        df.at[idx, "actual_pct"]         = round(actual_pct, 4)
        df.at[idx, "direction_correct"]  = dir_correct
        updated += 1

    if updated > 0:
        print(f"  Backfilled {updated} realized prices.")

    return df


def evaluate_accuracy(
    model_name: str,
    lookback_days: int = DEFAULT_LOOKBACK,
) -> dict:
    """
    Evaluate a single model's accuracy over the last `lookback_days`.

    Returns
    -------
    dict with keys:
        direction_accuracy  : float   fraction of correct direction calls
        mae                 : float   mean absolute error of predicted_pct vs actual_pct
        hit_rate_2pct       : float   fraction where |predicted_pct - actual_pct| <= 2%
        n_evaluated         : int     number of predictions with realized outcomes
        last_prediction     : str     ISO date of last prediction (or None)
    """
    df = _load_prediction_log()
    if df.empty:
        return _empty_metrics()

    # Backfill any newly-realized predictions
    df = _backfill_actuals(df)
    _save_prediction_log(df)

    # Filter to this model and lookback window
    cutoff = date.today() - timedelta(days=lookback_days)
    mask = (
        (df["model"] == model_name)
        & df["actual_price"].notna()
        & (df["pred_date"] >= cutoff)
    )
    subset = df[mask].copy()

    if subset.empty:
        # Still return last_prediction even if no evaluated rows
        model_rows = df[df["model"] == model_name]
        last_pred = None
        if not model_rows.empty and model_rows["pred_date"].notna().any():
            last_pred = str(model_rows["pred_date"].max())
        metrics = _empty_metrics()
        metrics["last_prediction"] = last_pred
        return metrics

    n = len(subset)
    direction_acc = float(subset["direction_correct"].sum()) / n if n > 0 else 0.0

    errors = (subset["predicted_pct"] - subset["actual_pct"]).abs()
    mae = float(errors.mean())

    hit_count = int((errors <= HIT_BAND_PCT).sum())
    hit_rate = hit_count / n if n > 0 else 0.0

    last_pred = str(subset["pred_date"].max()) if not subset.empty else None

    return {
        "direction_accuracy": round(direction_acc, 4),
        "mae":                round(mae, 4),
        "hit_rate_2pct":      round(hit_rate, 4),
        "n_evaluated":        n,
        "last_prediction":    last_pred,
    }


def _empty_metrics() -> dict:
    return {
        "direction_accuracy": 0.0,
        "mae":                0.0,
        "hit_rate_2pct":      0.0,
        "n_evaluated":        0,
        "last_prediction":    None,
    }


# ==============================================================================
#  3. DYNAMIC WEIGHT CALCULATOR
# ==============================================================================

def _discover_models() -> list[str]:
    """
    Discover all model names present in the prediction log.
    Falls back to known defaults if the log is empty.
    """
    df = _load_prediction_log()
    if df.empty or "model" not in df.columns:
        return ["GoldLSTM-v1", "GCF_tactical", "GCF_strategic"]
    models = df["model"].dropna().unique().tolist()
    return sorted(models) if models else ["GoldLSTM-v1", "GCF_tactical", "GCF_strategic"]


def compute_ensemble_weights(
    lookback_days: int = DEFAULT_LOOKBACK,
) -> dict:
    """
    Evaluate each model's recent accuracy and assign ensemble weights
    proportional to hit rates via softmax, with a floor of WEIGHT_FLOOR.

    Returns
    -------
    dict with 'weights' and 'metrics' sub-dicts.
    Also writes data/ensemble_weights.json.
    """
    models = _discover_models()
    metrics_map: dict[str, dict] = {}

    for m in models:
        metrics_map[m] = evaluate_accuracy(m, lookback_days)

    # --- Softmax of hit rates ---
    hit_rates = np.array([
        metrics_map[m]["hit_rate_2pct"] for m in models
    ], dtype=np.float64)

    # If all models have 0 evaluations, equal weight
    total_evals = sum(metrics_map[m]["n_evaluated"] for m in models)
    if total_evals == 0:
        raw_weights = np.ones(len(models)) / len(models)
    else:
        # Softmax: exp(h / T) / sum(exp(h / T))
        shifted = hit_rates - hit_rates.max()  # numerical stability
        exps = np.exp(shifted / SOFTMAX_TEMP)
        raw_weights = exps / exps.sum()

    # Apply floor
    weights = np.maximum(raw_weights, WEIGHT_FLOOR)
    # Re-normalize to sum to 1.0
    weights = weights / weights.sum()

    weight_map = {m: round(float(w), 4) for m, w in zip(models, weights)}

    # Add health status to metrics
    health_map = check_model_health()
    for m in models:
        metrics_map[m]["health"] = health_map.get(m, {}).get("status", "UNKNOWN")

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_days": lookback_days,
        "weights": weight_map,
        "metrics": metrics_map,
    }

    return payload


def save_ensemble_weights(payload: dict) -> None:
    """Write the ensemble weights payload to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_WEIGHTS.write_text(json.dumps(payload, indent=2, default=str))


def load_ensemble_weights() -> dict | None:
    """Load ensemble weights from disk, or return None if not found."""
    if not ENSEMBLE_WEIGHTS.exists():
        return None
    try:
        return json.loads(ENSEMBLE_WEIGHTS.read_text())
    except Exception:
        return None


# ==============================================================================
#  4. STALENESS DETECTOR / MODEL HEALTH
# ==============================================================================

def check_model_health() -> dict[str, dict]:
    """
    For each model in the prediction log, compute rolling MAE over the
    last 10, 30, 90 predictions and detect degradation or staleness.

    Returns
    -------
    dict[model_name] -> {
        "status":  "HEALTHY" | "DEGRADING" | "STALE" | "INSUFFICIENT_DATA",
        "mae_10":  float | None,
        "mae_30":  float | None,
        "mae_90":  float | None,
        "days_since_last": int | None,
        "reason":  str,
    }
    """
    df = _load_prediction_log()
    models = _discover_models()
    today = date.today()
    results: dict[str, dict] = {}

    for m in models:
        subset = df[(df["model"] == m) & df["actual_pct"].notna()].copy()

        if subset.empty:
            # Check if there are any predictions at all (even without actuals)
            all_model = df[df["model"] == m]
            if all_model.empty:
                results[m] = {
                    "status": "STALE",
                    "mae_10": None, "mae_30": None, "mae_90": None,
                    "days_since_last": None,
                    "reason": "No predictions found.",
                }
            else:
                last_pred = all_model["pred_date"].max()
                days_since = (today - last_pred).days if last_pred else None
                if days_since is not None and days_since > STALENESS_DAYS:
                    results[m] = {
                        "status": "STALE",
                        "mae_10": None, "mae_30": None, "mae_90": None,
                        "days_since_last": days_since,
                        "reason": f"No predictions in {days_since} days.",
                    }
                else:
                    results[m] = {
                        "status": "INSUFFICIENT_DATA",
                        "mae_10": None, "mae_30": None, "mae_90": None,
                        "days_since_last": days_since,
                        "reason": "Predictions exist but no realized outcomes yet.",
                    }
            continue

        # Sort by pred_date descending for rolling windows
        subset = subset.sort_values("pred_date", ascending=False)
        errors = (subset["predicted_pct"] - subset["actual_pct"]).abs()

        mae_10 = float(errors.iloc[:10].mean()) if len(errors) >= 10 else None
        mae_30 = float(errors.iloc[:30].mean()) if len(errors) >= 30 else None
        mae_90 = float(errors.iloc[:90].mean()) if len(errors) >= 90 else None

        # Days since last prediction
        last_pred = subset["pred_date"].max()
        days_since = (today - last_pred).days if last_pred else None

        # Determine health status
        status = "HEALTHY"
        reason = "OK"

        if days_since is not None and days_since > STALENESS_DAYS:
            status = "STALE"
            reason = f"No predictions in {days_since} days."
        elif mae_10 is not None and mae_90 is not None:
            if mae_10 > DEGRADATION_FACTOR * mae_90:
                status = "DEGRADING"
                reason = (
                    f"Recent MAE ({mae_10:.2f}%) is {mae_10/mae_90:.1f}x "
                    f"the long-term MAE ({mae_90:.2f}%)."
                )
        elif mae_10 is not None and mae_30 is not None:
            # Fallback: compare 10 vs 30 if we don't have 90 yet
            if mae_10 > DEGRADATION_FACTOR * mae_30:
                status = "DEGRADING"
                reason = (
                    f"Recent MAE ({mae_10:.2f}%) is {mae_10/mae_30:.1f}x "
                    f"the 30-pred MAE ({mae_30:.2f}%)."
                )
        elif len(errors) < 10:
            status = "INSUFFICIENT_DATA"
            reason = f"Only {len(errors)} evaluated predictions."

        results[m] = {
            "status": status,
            "mae_10": round(mae_10, 4) if mae_10 is not None else None,
            "mae_30": round(mae_30, 4) if mae_30 is not None else None,
            "mae_90": round(mae_90, 4) if mae_90 is not None else None,
            "days_since_last": days_since,
            "reason": reason,
        }

    return results


# ==============================================================================
#  5. REPORT AND OUTPUT
# ==============================================================================

def _health_color(status: str) -> str:
    """Color-code health status for terminal display."""
    if status == "HEALTHY":
        return GREEN(status)
    elif status == "DEGRADING":
        return RED(status)
    elif status == "STALE":
        return YELLOW(status)
    return DIM(status)


def print_report(payload: dict) -> None:
    """Print the formatted performance report to stdout."""
    today_str = date.today().isoformat()
    weights = payload.get("weights", {})
    metrics = payload.get("metrics", {})

    print()
    line = "\u2501" * 64
    print(BOLD(line))
    print(BOLD(f"  MODEL PERFORMANCE TRACKER -- {today_str}"))
    print(BOLD(line))
    print()

    # Header
    hdr = f"  {'Model':<20s} {'Dir.Acc':>8s} {'MAE':>7s} {'Hit+/-2%':>9s} {'Weight':>8s}   {'Health'}"
    print(hdr)
    print("  " + "\u2500" * 62)

    for model in sorted(weights.keys()):
        m = metrics.get(model, {})
        n_eval   = m.get("n_evaluated", 0)
        dir_acc  = m.get("direction_accuracy", 0.0)
        mae      = m.get("mae", 0.0)
        hit_rate = m.get("hit_rate_2pct", 0.0)
        weight   = weights.get(model, 0.0)
        health   = m.get("health", "UNKNOWN")

        if n_eval == 0:
            dir_str  = "   --  "
            mae_str  = "  --  "
            hit_str  = "   --   "
        else:
            dir_str  = f"{dir_acc*100:6.1f}%"
            mae_str  = f"{mae:5.2f}%"
            hit_str  = f"{hit_rate*100:6.1f}%"

        health_str = _health_color(health)
        print(f"  {model:<20s} {dir_str:>8s} {mae_str:>7s} {hit_str:>9s} {weight:8.2f}   {health_str}")

    print()
    print(f"  Ensemble weights updated -> {ENSEMBLE_WEIGHTS.relative_to(ROOT)}")
    print(BOLD(line))
    print()


# ==============================================================================
#  CLI
# ==============================================================================

def _check_retrain_needed(payload: dict) -> list[str]:
    """
    Check if any model needs retraining based on health status.
    Returns list of model names that should be retrained.

    Trigger conditions:
      - Status DEGRADING (MAE spiking vs long-term)
      - Status STALE for more than 14 days
    """
    retrain_list = []
    metrics = payload.get("metrics", {})
    for model, m in metrics.items():
        health = m.get("health", "UNKNOWN")
        if health == "DEGRADING":
            retrain_list.append(model)
        elif health == "STALE":
            # Only retrain main model if stale (proving ground retrains separately)
            if model == "GoldLSTM-v1":
                retrain_list.append(model)
    return retrain_list


def _auto_retrain(models_to_retrain: list[str]) -> dict:
    """
    Trigger automatic retraining for degraded models.
    Returns status dict.
    """
    results = {}
    for model in models_to_retrain:
        if model == "GoldLSTM-v1":
            try:
                from models.lstm_predictor import train as lstm_train
                print(f"\n  {YELLOW('AUTO-RETRAIN')}: {model} -- DEGRADING detected")
                print(f"  Triggering full retrain...")
                lstm_train(period="5y", save=True)
                results[model] = "RETRAINED"
                print(f"  {GREEN('SUCCESS')}: {model} retrained")
            except Exception as e:
                results[model] = f"FAILED: {e}"
                print(f"  {RED('FAILED')}: {model} retrain -- {e}")
        elif model.startswith("GCF_"):
            try:
                from scripts.metals_trainer import main as trainer_main
                print(f"\n  {YELLOW('AUTO-RETRAIN')}: {model} -- DEGRADING detected")
                print(f"  Triggering proving ground retrain...")
                trainer_main()
                results[model] = "RETRAINED"
                print(f"  {GREEN('SUCCESS')}: {model} retrained")
            except Exception as e:
                results[model] = f"FAILED: {e}"
                print(f"  {RED('FAILED')}: {model} retrain -- {e}")
    return results


def run(lookback_days: int = DEFAULT_LOOKBACK, dry_run: bool = False) -> dict:
    """
    Full pipeline: backfill actuals, evaluate, compute weights, print report.
    Auto-retrains degraded models when detected.
    Returns the ensemble weights payload.
    """
    _ensure_log_exists()

    # Backfill realized prices for all pending predictions
    print(f"\n  Backfilling realized prices...")
    df = _load_prediction_log()
    if not df.empty:
        df = _backfill_actuals(df)
        _save_prediction_log(df)
    else:
        print("  (prediction log is empty -- nothing to backfill)")

    # Compute ensemble weights
    payload = compute_ensemble_weights(lookback_days)

    # Save (unless dry-run)
    if not dry_run:
        save_ensemble_weights(payload)
    else:
        print(YELLOW("  [DRY-RUN] Weights not written to disk."))

    # Report
    print_report(payload)

    # Auto-retrain check
    if not dry_run:
        retrain_list = _check_retrain_needed(payload)
        if retrain_list:
            retrain_results = _auto_retrain(retrain_list)
            payload["retrain_triggered"] = retrain_results
        else:
            payload["retrain_triggered"] = None

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Performance Tracker -- adaptive ensemble weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lookback", type=int, default=DEFAULT_LOOKBACK,
        help=f"Lookback window in days (default: {DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate and report but do not write files",
    )
    args = parser.parse_args()

    run(lookback_days=args.lookback, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
