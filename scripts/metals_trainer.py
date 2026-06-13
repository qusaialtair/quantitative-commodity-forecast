#!/usr/bin/env python3
"""
PHASE 2 — The Tri-Horizon GPU Forge
=====================================
Trains THREE independent LSTMs per metal on Apple Silicon MPS:

  [TICKER]_tactical.pth    seq=60  target=t+5d   (The Sniper)
  [TICKER]_strategic.pth   seq=60  target=t+21d  (The Swing)
  [TICKER]_structural.pth  seq=120 target=t+252d (The Macro Trend)

Each model gets its own _scaler.pkl.

Usage:
    python3 scripts/metals_trainer.py
    python3 scripts/metals_trainer.py --ticker GC=F   # single metal
    python3 scripts/metals_trainer.py --horizon tactical  # single horizon
"""

import sys, argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
MODELS = ROOT / "models" / "proving_ground"
MODELS.mkdir(parents=True, exist_ok=True)

# ── Hardware ───────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  🖥️   Device: {device}")

# ── Horizon definitions ────────────────────────────────────────────────────────
HORIZONS = {
    "tactical":   {"seq": 60,  "target_days": 5,   "label": "The Sniper  (t+5d)"},
    "strategic":  {"seq": 60,  "target_days": 21,  "label": "The Swing   (t+21d)"},
    "structural": {"seq": 120, "target_days": 252, "label": "The Macro Trend (t+252d)"},
}

FEATURES   = ["Close_Price", "Volume", "DXY_Close", "VIX_Close"]
INPUT_SIZE = len(FEATURES)   # 4

# Training hyper-parameters
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
BATCH_SIZE   = 64
MAX_EPOCHS   = 300
PATIENCE     = 20
LR           = 1e-3
VAL_SPLIT    = 0.15


# ── Model ─────────────────────────────────────────────────────────────────────
class MetalLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(scaled: np.ndarray, raw_close: np.ndarray,
                    seq_len: int, target_days: int):
    X, y = [], []
    for i in range(len(scaled) - seq_len - target_days + 1):
        X.append(scaled[i : i + seq_len])
        base = raw_close[i + seq_len - 1]
        future = raw_close[i + seq_len + target_days - 1]
        pct_ret = (future / base - 1.0) * 100.0 if base > 0 else 0.0
        y.append(pct_ret)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_horizon(df: pd.DataFrame, ticker: str, horizon: str) -> None:
    cfg   = HORIZONS[horizon]
    tag   = ticker.replace("=", "").replace("^", "")
    label = cfg["label"]
    seq   = cfg["seq"]
    tgt   = cfg["target_days"]

    print(f"\n  ┌── {ticker}  ·  {horizon.upper()}  ·  {label}")

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)
    raw_close = df["Close_Price"].values.astype(float)

    # Save scaler
    scaler_path = MODELS / f"{tag}_{horizon}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Sequences — target is % return over horizon, not absolute price
    X, y = build_sequences(scaled, raw_close, seq, tgt)
    if len(X) < 50:
        print(f"  ⚠️  Not enough data ({len(X)} sequences) — skipping.")
        return

    # Train / val split (chronological)
    n_val  = max(1, int(len(X) * VAL_SPLIT))
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    tr_loader  = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    # Model
    model     = MetalLSTM().to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    t0         = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or no_improve == PATIENCE:
            elapsed = time.time() - t0
            print(f"  │   Epoch {epoch:>3}  val_loss={best_val:.6f}  "
                  f"patience={no_improve}/{PATIENCE}  ({elapsed:.0f}s)")

        if no_improve >= PATIENCE:
            print(f"  │   Early stop at epoch {epoch}.")
            break

    # Save best weights
    model.load_state_dict(best_state)
    out_path = MODELS / f"{tag}_{horizon}.pth"
    torch.save({"model_state": best_state,
                "input_size":  INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers":  NUM_LAYERS,
                "seq_len":     seq,
                "target_days": tgt,
                "features":    FEATURES,
                "ticker":      ticker,
                "horizon":     horizon}, out_path)

    elapsed = time.time() - t0
    print(f"  └── ✅  Saved  {out_path.name}  (val_loss={best_val:.6f}  {elapsed:.0f}s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main(tickers=None, horizons=None):
    tickers  = tickers  or ["GC=F", "SI=F"]
    horizons = horizons or list(HORIZONS.keys())

    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Tri-Horizon GPU Forge")
    print(f"  Metals  : {tickers}")
    print(f"  Horizons: {horizons}")
    print(f"  Device  : {device}")
    print("━" * 60)

    for ticker in tickers:
        tag      = ticker.replace("=", "").replace("^", "")
        csv_path = DATA / f"{ticker}_8yr_macro.csv"
        if not csv_path.exists():
            print(f"\n  ❌  {csv_path.name} not found — run metals_harvester.py first.")
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"\n  📂  {ticker}  —  {len(df)} rows  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        for horizon in horizons:
            train_one_horizon(df, ticker, horizon)

    generate_predictions(tickers)
    print("━" * 60)
    print("  ✅  All models trained.  Run the Streamlit dashboard to see results.")
    print("━" * 60 + "\n")


def generate_predictions(tickers=None):
    """Run inference on trained proving ground models and write predictions JSON."""
    import json
    tickers = tickers or ["GC=F", "SI=F"]
    preds = {}

    for ticker in tickers:
        tag = ticker.replace("=", "").replace("^", "")
        csv_path = DATA / f"{ticker}_8yr_macro.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        current_close = float(df["Close_Price"].iloc[-1])

        for horizon, cfg in HORIZONS.items():
            model_path  = MODELS / f"{tag}_{horizon}.pth"
            scaler_path = MODELS / f"{tag}_{horizon}_scaler.pkl"
            if not model_path.exists() or not scaler_path.exists():
                continue

            try:
                scaler = joblib.load(scaler_path)
                ckpt   = torch.load(model_path, map_location=device, weights_only=False)
                model  = MetalLSTM(
                    input_size=ckpt.get("input_size", INPUT_SIZE),
                    hidden=ckpt.get("hidden_size", HIDDEN_SIZE),
                    layers=ckpt.get("num_layers", NUM_LAYERS),
                ).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                seq_len = cfg["seq"]
                scaled = scaler.transform(df[FEATURES].values)
                seq = scaled[-seq_len:]
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_pct = float(model(x).item())

                pred_price = current_close * (1.0 + pred_pct / 100.0)

                key = f"{tag}_{horizon}"
                preds[key] = {
                    "pred_price": round(pred_price, 2),
                    "pred_pct": round(pred_pct, 3),
                    "current_close": round(current_close, 2),
                    "target_days": cfg["target_days"],
                    "horizon": horizon,
                    "ticker": ticker,
                }
            except Exception as e:
                print(f"  ⚠️  Prediction failed for {tag}_{horizon}: {e}")

    out_path = DATA / "proving_ground_predictions.json"
    out_path.write_text(json.dumps(preds, indent=2))
    print(f"\n  📊  Predictions written → {out_path.name}  ({len(preds)} models)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  default=None, help="GC=F or SI=F (default: both)")
    parser.add_argument("--horizon", default=None,
                        choices=["tactical", "strategic", "structural"],
                        help="Single horizon to train (default: all 3)")
    args = parser.parse_args()

    main(
        tickers  = [args.ticker]  if args.ticker  else None,
        horizons = [args.horizon] if args.horizon else None,
    )

    # ─────────────────────────────────────────────────────────────────────────
    ### PHASE 9: 5-FEATURE INTEGRATION ###
    # Once 6+ months of oracle_history.csv have accumulated, upgrade to 5-feature
    # tensors by following these steps:
    #
    # 1. Load oracle_history.csv:
    #       oracle = pd.read_csv(DATA / "oracle_history.csv", parse_dates=["date"])
    #       oracle = oracle[oracle["ticker"] == ticker].set_index("date")[["score"]]
    #
    # 2. Merge into the macro CSV:
    #       df = df.join(oracle.rename(columns={"score": "Oracle_Score"}), how="left")
    #       df["Oracle_Score"].fillna(0.5, inplace=True)   # neutral fill for gaps
    #
    # 3. Update the feature list:
    #       FEATURES   = ["Close_Price", "Volume", "DXY_Close", "VIX_Close", "Oracle_Score"]
    #       INPUT_SIZE = 5
    #
    # 4. Retrain all horizons.  The scaler and model architecture auto-adapt
    #    because INPUT_SIZE drives MetalLSTM(input_size=INPUT_SIZE).
    #
    # No other code changes needed — the pipeline is already 5-feature ready.
    # ─────────────────────────────────────────────────────────────────────────
