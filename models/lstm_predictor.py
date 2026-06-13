#!/usr/bin/env python3
"""
Multi-Head LSTM Predictor  (Phase V upgrade)
=============================================
Shared LSTM encoder + two output heads:

  Regression head     -- next-session adjusted price (original goal)
  Classification head -- P(next-day return > 0)  (new: direction confidence)

Multi-asset: gold (GC=F), silver (SI=F), copper (HG=F).  Each asset keeps its
own weight file but the architecture is identical, so representations are
comparable and the daily fine-tuner can target any asset with one call.

Training loss:
  L = MSE(regression) + lambda_cls * BCE(classification)

BCE targets are derived from actual next-day returns at training time -- no
separate labelling step needed.

Backwards-compatible: callers that only read "adjusted_price" keep working.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "lstm_predictions.json"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    "GC=F": {"name": "gold",
              "pth":    DATA_DIR / "gold_lstm_brain.pth",
              "scaler": DATA_DIR / "gold_lstm_scaler.pkl",
              "meta":   DATA_DIR / "gold_lstm_meta.json"},
    "SI=F": {"name": "silver",
              "pth":    DATA_DIR / "silver_lstm_brain.pth",
              "scaler": DATA_DIR / "silver_lstm_scaler.pkl",
              "meta":   DATA_DIR / "silver_lstm_meta.json"},
    "HG=F": {"name": "copper",
              "pth":    DATA_DIR / "copper_lstm_brain.pth",
              "scaler": DATA_DIR / "copper_lstm_scaler.pkl",
              "meta":   DATA_DIR / "copper_lstm_meta.json"},
}

DEFAULT_HIDDEN     = 192
DEFAULT_LAYERS     = 3
DEFAULT_SEQ_LEN    = 60
DEFAULT_N_FEAT     = 8
DEFAULT_LAMBDA_CLS = 0.5

# ---------------------------------------------------------------------------
# Lazy torch helpers
# ---------------------------------------------------------------------------
def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch required: pip install torch")

def _nn():
    try:
        import torch.nn as nn
        return nn
    except ImportError:
        raise ImportError("PyTorch required: pip install torch")

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MultiHeadLSTM:
    """Lazy wrapper so the module is importable without PyTorch installed."""
    _cls = None

    @classmethod
    def _module_cls(cls):
        if cls._cls is not None:
            return cls._cls
        torch = _torch()
        nn = _nn()

        class _Module(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
                super().__init__()
                self.input_size  = input_size
                self.hidden_size = hidden_size
                self.num_layers  = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.regression_head = nn.Sequential(
                    nn.Linear(hidden_size, 64), nn.ReLU(),
                    nn.Dropout(0.1), nn.Linear(64, 1))
                self.classification_head = nn.Sequential(
                    nn.Linear(hidden_size, 32), nn.ReLU(),
                    nn.Linear(32, 1), nn.Sigmoid())

            def forward(self, x):
                out, _ = self.lstm(x)
                h = out[:, -1, :]
                return (self.regression_head(h).squeeze(-1),
                        self.classification_head(h).squeeze(-1))

            def forward_regression(self, x):
                """Single-output forward for old regression-only checkpoints."""
                out, _ = self.lstm(x)
                return self.regression_head(out[:, -1, :]).squeeze(-1)

        cls._cls = _Module
        return _Module

    @classmethod
    def build(cls, input_size, hidden_size, num_layers, dropout=0.2):
        return cls._module_cls()(input_size, hidden_size, num_layers, dropout)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _load_alt_row() -> dict:
    try:
        import pandas as pd
        df = pd.read_csv(DATA_DIR / "alt_data.csv", index_col=0, parse_dates=True)
        row = df.dropna(how="all").iloc[-1]
        return {k: (float(v) if v == v else 0.0) for k, v in row.items()}
    except Exception:
        return {}

def _build_features(ticker, seq_len, feature_cols, period="2y"):
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        return None, False

    raw = yf.download(ticker, period=period, interval="1d",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw = raw.dropna(subset=["Close"])
    if len(raw) < seq_len + 2:
        return None, False

    close = raw["Close"]
    df = pd.DataFrame(index=raw.index)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in feature_cols:
            df[col] = raw.get(col, close)
    if "ret1" in feature_cols:
        df["ret1"] = close.pct_change().fillna(0)
    if "vol_ratio" in feature_cols:
        v21 = close.pct_change().rolling(21).std()
        v63 = close.pct_change().rolling(63).std()
        df["vol_ratio"] = (v21 / v63.replace(0, float("nan"))).fillna(1.0)
    if "mom_quality" in feature_cols:
        r21 = close.pct_change(21)
        v21 = close.pct_change().rolling(21).std() * 252 ** 0.5
        df["mom_quality"] = (r21 / v21.replace(0, float("nan"))).fillna(0.0)

    alt = _load_alt_row()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = float(alt.get(col, 0.0))
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols].ffill().fillna(0.0)
    return df, True

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _load_checkpoint(pth_path, input_size, hidden_size, num_layers):
    torch = _torch()
    if not pth_path.exists():
        return None, False
    try:
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.warning("Checkpoint load failed %s: %s", pth_path.name, e)
        return None, False

    model = MultiHeadLSTM.build(input_size, hidden_size, num_layers)
    state = (ckpt.get("model_state") if isinstance(ckpt, dict) else None)
    if state is None:
        try:
            state = ckpt.state_dict()
        except Exception:
            state = ckpt if isinstance(ckpt, dict) else {}

    try:
        model.load_state_dict(state, strict=True)
        return model, True
    except RuntimeError:
        pass
    try:
        model.load_state_dict(state, strict=False)
        return model, False
    except Exception as e:
        log.warning("Partial load failed %s: %s", pth_path.name, e)
        return None, False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _direction_prob_heuristic(pred_price, current_price):
    if current_price <= 0:
        return 0.5
    ret = (pred_price / current_price) - 1.0
    return 1.0 / (1.0 + math.exp(-ret * 15.0))

def _confidence_label(prob):
    if prob >= 0.75 or prob <= 0.25:
        return "HIGH"
    if prob >= 0.65 or prob <= 0.35:
        return "MEDIUM"
    return "LOW"

# ---------------------------------------------------------------------------
# Public API: predict_next
# ---------------------------------------------------------------------------
def predict_next(ticker="GC=F"):
    """
    Run inference.  Returns dict with keys:
      adjusted_price, direction_prob, direction, ret_pct, confidence, head.
    On failure returns {"error": <reason>}.
    """
    torch = _torch()
    asset  = ASSETS.get(ticker, ASSETS["GC=F"])
    meta_p = asset["meta"]
    pth_p  = asset["pth"]
    sc_p   = asset["scaler"]

    if not meta_p.exists():
        meta: dict[str, Any] = {
            "n_features": DEFAULT_N_FEAT,
            "feature_cols": ["Open","High","Low","Close","Volume",
                             "ret1","vol_ratio","mom_quality"],
            "hidden_size": DEFAULT_HIDDEN,
            "num_layers":  DEFAULT_LAYERS,
            "seq_len":     DEFAULT_SEQ_LEN,
        }
    else:
        try:
            meta = json.loads(meta_p.read_text())
        except Exception as e:
            return {"error": f"meta read failed: {e}"}

    n_feat   = int(meta.get("n_features", DEFAULT_N_FEAT))
    feat_cols = meta.get("feature_cols", [])[:n_feat]
    hidden   = int(meta.get("hidden_size", DEFAULT_HIDDEN))
    layers   = int(meta.get("num_layers",  DEFAULT_LAYERS))
    seq_len  = int(meta.get("seq_len",     DEFAULT_SEQ_LEN))

    df, ok = _build_features(ticker, seq_len, feat_cols)
    if not ok:
        return {"error": "insufficient price history"}

    scaler = None
    if sc_p.exists():
        try:
            import joblib
            scaler = joblib.load(sc_p)
        except Exception:
            pass

    import numpy as np
    raw = df.values[-seq_len:]
    if scaler is not None:
        try:
            raw = scaler.transform(raw)
        except Exception:
            pass
    x = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)

    current_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0

    model, has_cls = _load_checkpoint(pth_p, n_feat, hidden, layers)
    if model is None:
        return {"error": "checkpoint unavailable or corrupted"}

    model.eval()
    with torch.no_grad():
        try:
            p_reg, p_cls = model(x)
            dir_prob = float(p_cls.item())
            head_type = "dual" if has_cls else "heuristic"
        except Exception:
            p_reg = model.forward_regression(x)
            dir_prob = None
            head_type = "heuristic"

    price_raw = float(p_reg.item())
    if scaler is not None:
        try:
            dummy = np.zeros((1, n_feat))
            ci = feat_cols.index("Close") if "Close" in feat_cols else 3
            dummy[0, ci] = price_raw
            price_raw = float(scaler.inverse_transform(dummy)[0, ci])
        except Exception:
            pass

    if dir_prob is None:
        dir_prob = _direction_prob_heuristic(price_raw, current_price)
        head_type = "heuristic"

    ret_pct = round((price_raw / current_price - 1.0) * 100, 3) if current_price > 0 else 0.0
    return {
        "adjusted_price": round(price_raw, 2),
        "current_price":  round(current_price, 2),
        "ret_pct":        ret_pct,
        "direction_prob": round(dir_prob, 4),
        "direction":      "UP" if dir_prob >= 0.5 else "DOWN",
        "confidence":     _confidence_label(dir_prob),
        "head":           head_type,
        "ticker":         ticker,
    }

def predict_all():
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for ticker, asset in ASSETS.items():
        try:
            results[asset["name"]] = predict_next(ticker)
        except Exception as e:
            results[asset["name"]] = {"error": str(e)}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    return results

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(ticker="GC=F", lookback="5y", seq_len=DEFAULT_SEQ_LEN,
          hidden_size=DEFAULT_HIDDEN, num_layers=DEFAULT_LAYERS,
          lr=1e-3, epochs=30, batch_size=32,
          lambda_cls=DEFAULT_LAMBDA_CLS, val_split=0.15,
          device="auto", quiet=False):
    """
    Train MultiHeadLSTM with combined MSE + BCE loss.
    Saves: data/<asset>_lstm_brain.pth  |  data/<asset>_lstm_scaler.pkl
           data/<asset>_lstm_meta.json
    """
    torch = _torch()
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib

    asset  = ASSETS.get(ticker, ASSETS["GC=F"])
    name   = asset["name"]
    pth_p  = asset["pth"]
    sc_p   = asset["scaler"]
    meta_p = asset["meta"]

    feat_cols = [
        "Open","High","Low","Close","Volume",
        "ret1","vol_ratio","mom_quality",
        "pplx_fed","pplx_geo_risk","pplx_phys_demand","pplx_macro",
        "real_yield_10y","copper_gold_ratio_zscore",
        "cot_gold_mm_net_zscore" if ticker == "GC=F" else "cot_silver_mm_net_zscore",
    ]
    n_feat = len(feat_cols)

    df, ok = _build_features(ticker, seq_len, feat_cols, period=lookback)
    if not ok:
        return {"error": "insufficient data"}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)

    X, y_reg, y_cls = [], [], []
    close_v = df["Close"].values
    ci = feat_cols.index("Close")
    for i in range(len(scaled) - seq_len - 1):
        X.append(scaled[i: i + seq_len])
        y_reg.append(scaled[i + seq_len, ci])
        fwd = close_v[i + seq_len + 1] - close_v[i + seq_len]
        y_cls.append(1.0 if fwd > 0 else 0.0)

    X     = torch.tensor(np.array(X),     dtype=torch.float32)
    y_reg = torch.tensor(np.array(y_reg), dtype=torch.float32)
    y_cls = torch.tensor(np.array(y_cls), dtype=torch.float32)

    n_val = max(int(len(X) * val_split), 30)
    Xtr, Xv   = X[:-n_val], X[-n_val:]
    yrt, yrv   = y_reg[:-n_val], y_reg[-n_val:]
    yct, ycv   = y_cls[:-n_val], y_cls[-n_val:]

    if device == "auto":
        import torch
        dev = (torch.device("mps")  if torch.backends.mps.is_available() else
               torch.device("cuda") if torch.cuda.is_available() else
               torch.device("cpu"))
    else:
        dev = torch.device(device)

    model = MultiHeadLSTM.build(n_feat, hidden_size, num_layers).to(dev)
    Xtr, Xv = Xtr.to(dev), Xv.to(dev)
    yrt, yrv = yrt.to(dev), yrv.to(dev)
    yct, ycv = yct.to(dev), ycv.to(dev)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse   = torch.nn.MSELoss()
    bce   = torch.nn.BCELoss()

    best_val, best_epoch, best_state = float("inf"), 0, None
    idx = torch.randperm(len(Xtr))
    n_batch = max(1, len(Xtr) // batch_size)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for b in range(n_batch):
            bi = idx[b * batch_size: b * batch_size + batch_size]
            xb, yrb, ycb = Xtr[bi], yrt[bi], yct[bi]
            optim.zero_grad()
            pr, pc = model(xb)
            loss = mse(pr, yrb) + lambda_cls * bce(pc, ycb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += float(loss.item())
        model.eval()
        with torch.no_grad():
            pv, pc_v = model(Xv)
            val_loss = float(mse(pv, yrv) + lambda_cls * bce(pc_v, ycv))
        if not quiet:
            print(f"  [{ticker}] ep {ep:3d}/{epochs}  tr={tr_loss/n_batch:.6f}  val={val_loss:.6f}")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        idx = torch.randperm(len(Xtr))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "n_features": n_feat,
                "hidden_size": hidden_size, "num_layers": num_layers,
                "seq_len": seq_len, "lambda_cls": lambda_cls}, pth_p)
    joblib.dump(scaler, sc_p)
    meta_p.write_text(json.dumps({
        "n_features": n_feat, "feature_cols": feat_cols,
        "hidden_size": hidden_size, "num_layers": num_layers, "seq_len": seq_len,
        "best_val_loss": round(best_val, 8), "best_epoch": best_epoch,
        "trained_period": lookback,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lambda_cls": lambda_cls, "has_classification_head": True,
    }, indent=2))
    return {"ticker": ticker, "name": name, "best_val_loss": round(best_val, 8),
            "best_epoch": best_epoch, "n_train": int(len(Xtr)), "n_val": n_val}

def train_all(tickers=None, **kw):
    tickers = tickers or list(ASSETS.keys())
    return {t: train(t, **kw) for t in tickers}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Multi-Head LSTM Predictor")
    sub = p.add_subparsers(dest="cmd")
    pr  = sub.add_parser("predict")
    pr.add_argument("--ticker", default=None)
    tr  = sub.add_parser("train")
    tr.add_argument("--ticker", default="GC=F")
    tr.add_argument("--lookback", default="5y")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--lambda-cls", type=float, default=DEFAULT_LAMBDA_CLS)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--quiet", action="store_true")
    ta = sub.add_parser("train-all")
    ta.add_argument("--epochs", type=int, default=30)
    ta.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.cmd == "predict":
        r = predict_next(args.ticker or "GC=F") if args.ticker else predict_all()
    elif args.cmd == "train":
        r = train(args.ticker, lookback=args.lookback, epochs=args.epochs,
                  lr=args.lr, lambda_cls=args.lambda_cls,
                  device=args.device, quiet=args.quiet)
    elif args.cmd == "train-all":
        r = train_all(epochs=args.epochs, quiet=args.quiet)
    else:
        r = predict_all()
    print(json.dumps(r, indent=2, default=str))
