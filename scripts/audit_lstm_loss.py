"""
scripts/audit_lstm_loss.py
==========================
Diagnostic script that audits the GoldLSTM-v1 model's learning progress.

Reads training_log.json and gold_lstm_meta.json, runs a live eval on the
current model weights (no training), prints a clean report, and saves
data/lstm_audit.json.

Usage:
    python3 scripts/audit_lstm_loss.py
"""

import json
import sys
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAINING_LOG_PATH = ROOT / "data" / "training_log.json"
META_PATH         = ROOT / "data" / "gold_lstm_meta.json"
MODEL_PATH        = ROOT / "data" / "gold_lstm_brain.pth"
SCALER_PATH       = ROOT / "data" / "gold_lstm_scaler.pkl"
AUDIT_OUT_PATH    = ROOT / "data" / "lstm_audit.json"

RUN_DATE = str(date.today())

# ── Sparkline helpers ──────────────────────────────────────────────────────────
_BARS = "▁▂▃▄▅▆▇█"

def _sparkline(values: list[float]) -> str:
    """Map a list of floats to a sparkline string using min/max normalisation."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    chars = []
    for v in values:
        if span == 0:
            idx = 0
        else:
            idx = int((v - lo) / span * (len(_BARS) - 1))
        chars.append(_BARS[idx])
    return "".join(chars)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_loss(v) -> str:
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_price(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return str(v)


# ── Step 1: Read training log ──────────────────────────────────────────────────

def load_fine_tune_runs() -> list[dict]:
    """
    Load training_log.json and return entries where:
      fine_tune_ran=True and fine_tune_result.success=True.
    """
    if not TRAINING_LOG_PATH.exists():
        return []
    try:
        raw = json.loads(TRAINING_LOG_PATH.read_text())
    except Exception as exc:
        print(f"  [warn] Could not parse training_log.json: {exc}")
        return []

    runs = []
    for entry in raw:
        if not entry.get("fine_tune_ran", False):
            continue
        result = entry.get("fine_tune_result", {})
        if not result.get("success", False):
            continue
        runs.append({
            "date":       entry.get("date", ""),
            "loss":       result.get("final_loss"),
            "oracle":     entry.get("oracle_score"),
            "gold_close": entry.get("gold_close"),
        })
    return runs


# ── Step 2: Read metadata ──────────────────────────────────────────────────────

def load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except Exception as exc:
        print(f"  [warn] Could not parse gold_lstm_meta.json: {exc}")
        return {}


# ── Step 3: Live evaluation ────────────────────────────────────────────────────

def run_live_eval() -> dict:
    """
    Load model and scaler, fetch 3mo OHLCV + perplexity, build sequences,
    split 80/20, compute HuberLoss on train+val.  Returns dict with keys:
      train_loss, val_loss, train_n, val_n, error (if applicable).
    """
    result = {"train_loss": None, "val_loss": None, "train_n": 0, "val_n": 0, "error": None}

    # Check model files exist
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        result["error"] = "model_missing"
        return result

    # Import torch
    try:
        import torch
        import torch.nn as nn
        import joblib
    except ImportError as exc:
        result["error"] = f"import_error: {exc}"
        return result

    try:
        from models.lstm_predictor import (
            GoldLSTM,
            build_sequences,
            _fetch_ohlcv,
            _attach_perplexity,
            FEATURE_COLS,
            N_FEATURES,
            SEQ_LEN,
            HIDDEN_SIZE,
            NUM_LAYERS,
            TwoPartScaler,
        )
    except Exception as exc:
        result["error"] = f"lstm_import_error: {exc}"
        return result

    # Fetch data
    try:
        df = _fetch_ohlcv(period="1y")
        df = _attach_perplexity(df)
    except ImportError:
        result["error"] = "yfinance_unavailable"
        return result
    except Exception as exc:
        result["error"] = f"data_fetch_error: {exc}"
        return result

    # Load scaler
    try:
        scaler = TwoPartScaler.load(SCALER_PATH)
    except Exception as exc:
        result["error"] = f"scaler_load_error: {exc}"
        return result

    # Build sequences using saved scaler (no fitting)
    try:
        X, y, _, _ = build_sequences(df, scaler=scaler, seq_len=SEQ_LEN)
    except Exception as exc:
        result["error"] = f"sequence_build_error: {exc}"
        return result

    if len(X) < 5:
        result["error"] = "not_enough_data"
        return result

    # Split 80/20 sequentially
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    # Load model
    try:
        device = torch.device("cpu")  # eval only — cpu is fine and always available
        model = GoldLSTM(n_features=N_FEATURES, hidden=HIDDEN_SIZE, layers=NUM_LAYERS)
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    except Exception as exc:
        result["error"] = f"model_load_error: {exc}"
        return result

    criterion = nn.HuberLoss(delta=0.01)

    def _eval_loss(X_arr, y_arr):
        if len(X_arr) == 0:
            return None, 0
        Xt = torch.tensor(X_arr, dtype=torch.float32).to(device)
        yt = torch.tensor(y_arr, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(Xt)
            loss  = criterion(preds, yt)
        return float(loss.item()), len(y_arr)

    train_loss, train_n = _eval_loss(X_train, y_train)
    val_loss,   val_n   = _eval_loss(X_val,   y_val)

    result["train_loss"] = train_loss
    result["val_loss"]   = val_loss
    result["train_n"]    = train_n
    result["val_n"]      = val_n
    return result


# ── Step 4+5: Report + save ────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    meta      = load_meta()
    all_runs  = load_fine_tune_runs()
    live      = run_live_eval()

    # Bail early if model has never been trained
    model_missing = (
        not MODEL_PATH.exists()
        or not SCALER_PATH.exists()
        or live.get("error") == "model_missing"
    )
    if model_missing and not meta:
        print("Model not trained")
        sys.exit(0)

    # ── Derive verdict ────────────────────────────────────────────────────────
    best_val_loss = meta.get("best_val_loss")
    n_runs        = len(all_runs)
    live_val      = live.get("val_loss")

    ft_losses = [r["loss"] for r in all_runs[-10:] if r["loss"] is not None]
    ft_trend_improving = (
        len(ft_losses) >= 3 and ft_losses[-1] < ft_losses[0]
    )

    if n_runs < 3:
        verdict = "FROZEN"
    elif live_val is None:
        verdict = "FROZEN"
    elif best_val_loss is not None:
        ratio = live_val / best_val_loss if best_val_loss > 0 else 1.0
        if ratio <= 1.0:
            verdict = "LEARNING"
        elif ratio <= 1.50:
            verdict = "STABLE"
        elif ft_trend_improving:
            verdict = "STABLE"
        else:
            verdict = "DEGRADING"
    else:
        verdict = "STABLE"

    # ── Last 10 fine-tune runs ────────────────────────────────────────────────
    last10 = all_runs[-10:]

    # ── Sparkline ─────────────────────────────────────────────────────────────
    spark_losses = [r["loss"] for r in last10 if r["loss"] is not None]
    spark        = _sparkline(spark_losses)

    # ── Print report ──────────────────────────────────────────────────────────
    W = 60
    print("=" * W)
    print("  GoldLSTM-v1 — Learning Audit")
    print(f"  Run date: {RUN_DATE}")
    print("=" * W)

    # TRAINING METADATA
    print()
    print("TRAINING METADATA")
    print(f"  Full train date : {meta.get('trained_at', 'N/A')}")
    best_epoch = meta.get('best_epoch', 'N/A')
    print(f"  Best val loss   : {_fmt_loss(best_val_loss)}  (epoch {best_epoch})")
    print(f"  Last fine-tune  : {meta.get('last_fine_tuned', 'N/A')}")
    print(f"  Last tune loss  : {_fmt_loss(meta.get('last_fine_tune_loss'))}")
    n_feat = meta.get('n_features', 'N/A')
    print(f"  Architecture    : {n_feat} features  |  LSTM(128,2)  |  seq=30")

    # FINE-TUNE HISTORY
    print()
    print(f"FINE-TUNE HISTORY ({n_runs} runs)")
    if all_runs:
        header = f"  {'Date':<10}  {'Loss':<9}  {'Oracle':<7}  {'Gold Close'}"
        sep    = f"  {'----------':<10}  {'---------':<9}  {'-------':<7}  {'----------'}"
        print(header)
        print(sep)
        for r in last10:
            loss_s   = f"{float(r['loss']):.6f}" if r["loss"] is not None else "N/A"
            oracle_s = f"{float(r['oracle']):.2f}" if r["oracle"] is not None else "N/A"
            price_s  = _fmt_price(r["gold_close"])
            print(f"  {r['date']:<10}  {loss_s:<9}  {oracle_s:<7}  {price_s}")
        if n_runs > 10:
            print(f"  [showing last 10 of {n_runs}]")
    else:
        print("  No successful fine-tune runs recorded.")

    # LOSS TREND
    print()
    print("LOSS TREND (last 10 fine-tunes)")
    if len(spark_losses) >= 2:
        lo_label = f"{spark_losses[0]:.6f}"
        hi_label = f"{spark_losses[-1]:.6f}"
        print(f"  {lo_label}  {spark}  {hi_label}")
    elif spark_losses:
        print(f"  {spark_losses[0]:.6f}  {spark}")
    else:
        print("  Insufficient data for trend.")

    # LIVE EVALUATION
    print()
    print("LIVE EVALUATION (current model on 1yr data)")
    if live.get("error") == "yfinance_unavailable":
        print("  [skipped] yfinance unavailable — cannot fetch live data.")
    elif live.get("error") == "model_missing":
        print("  [skipped] Model files not found.")
    elif live.get("error"):
        print(f"  [skipped] Error during live eval: {live['error']}")
    else:
        tl = live.get("train_loss")
        vl = live.get("val_loss")
        tn = live.get("train_n", 0)
        vn = live.get("val_n", 0)
        print(f"  Train loss  : {_fmt_loss(tl)}  ({tn} samples)")
        print(f"  Val loss    : {_fmt_loss(vl)}  ({vn} samples)")
        if tl is not None and vl is not None and tl > 0:
            gap_pct = (vl - tl) / tl * 100
            sign    = "+" if gap_pct >= 0 else ""
            print(f"  Overfit gap : {sign}{gap_pct:.2f}%")

    # VERDICT
    print()
    print("VERDICT")
    descriptions = {
        "LEARNING":  "val loss improving vs initial training",
        "STABLE":    "val loss within range or fine-tune trend improving",
        "DEGRADING": "val loss drifted >50% and fine-tune trend worsening",
        "FROZEN":    "fewer than 3 fine-tune runs, cannot assess trend",
    }
    desc = descriptions.get(verdict, "")
    print(f"  {verdict:<10} — {desc}")
    print()

    # ── Save audit JSON ───────────────────────────────────────────────────────
    audit = {
        "run_date":       RUN_DATE,
        "meta":           meta,
        "fine_tune_runs": all_runs,
        "live_train_loss": live.get("train_loss"),
        "live_val_loss":   live.get("val_loss"),
        "verdict":         verdict,
    }
    try:
        AUDIT_OUT_PATH.write_text(json.dumps(audit, indent=2))
    except Exception as exc:
        print(f"  [warn] Could not save audit JSON: {exc}")


if __name__ == "__main__":
    main()
