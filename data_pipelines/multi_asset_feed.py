"""
Multi-Asset Market Data Feed
Downloads and aligns 10+ years of data for gold, silver, DXY, Treasury yields,
VIX, S&P 500, and crude oil. Builds a comprehensive ~60-feature macro + technical
matrix with zero lookahead bias (all features strictly lagged by ≥1 day).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

ASSETS = {
    "gold":   "GC=F",     # Gold Futures (primary)
    "silver": "SI=F",     # Silver Futures → gold/silver ratio
    "dxy":    "DX-Y.NYB", # US Dollar Index (inverse gold correlation)
    "tnx":    "^TNX",     # 10-Year Treasury Yield
    "irx":    "^IRX",     # 13-Week T-Bill Rate (for yield curve)
    "vix":    "^VIX",     # CBOE Fear Index
    "spx":    "^GSPC",    # S&P 500 (risk-on/risk-off regime)
    "oil":    "CL=F",     # Crude Oil (commodity complex correlation)
}


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def download_all(period: str = "10y") -> dict:
    """Downloads close + OHLC where needed. Returns dict of aligned Series."""
    data = {}
    for name, ticker in ASSETS.items():
        logger.info(f"  Downloading {name} ({ticker})...")
        try:
            raw = yf.download(ticker, period=period, interval="1d",
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty and "Close" in raw.columns:
                data[name] = raw["Close"].rename(name)
                if name == "gold":
                    data["gold_h"] = raw["High"].rename("gold_h")
                    data["gold_l"] = raw["Low"].rename("gold_l")
        except Exception as exc:
            logger.warning(f"  Failed to download {name}: {exc}")
    return data


def build_features(period: str = "10y") -> tuple:
    """
    Returns (feature_df, feature_cols, gold_close_series).

    feature_df  : rows = trading days, columns = lagged features (NO target, NO lookahead)
    feature_cols: list of feature column names
    gold_close  : raw gold close Series (aligned to same index) — used externally for target
    """
    logger.info(f"Building multi-asset feature matrix ({period} lookback)...")
    data = download_all(period)

    if "gold" not in data:
        raise RuntimeError("Gold data download failed — cannot build features.")

    g = data["gold"]
    gh = data.get("gold_h", g)
    gl = data.get("gold_l", g)

    # Base DataFrame aligned to gold trading calendar
    base = pd.DataFrame({"gold": g, "gold_h": gh, "gold_l": gl})
    for name, series in data.items():
        if name in ("gold", "gold_h", "gold_l"):
            continue
        base[name] = series.reindex(base.index).ffill().bfill()

    base = base.dropna(subset=["gold"])
    g = base["gold"]

    f = pd.DataFrame(index=base.index)

    # ── Gold price lags ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 21, 63]:
        f[f"g_lag{lag}"] = g.shift(lag)

    # ── Moving averages & price/MA ratios ────────────────────────────────────
    for w in [5, 10, 21, 50, 200]:
        ma = g.shift(1).rolling(w).mean()
        f[f"g_ma{w}"]   = ma
        f[f"g_pma{w}"]  = g.shift(1) / ma   # >1 = price above MA

    f["g_ma5_21"]   = g.shift(1).rolling(5).mean()   / g.shift(1).rolling(21).mean()
    f["g_ma21_50"]  = g.shift(1).rolling(21).mean()  / g.shift(1).rolling(50).mean()
    f["g_ma50_200"] = g.shift(1).rolling(50).mean()  / g.shift(1).rolling(200).mean()

    # ── Returns over multiple horizons ───────────────────────────────────────
    for w in [1, 2, 3, 5, 10, 21, 63]:
        f[f"g_ret{w}"] = g.pct_change(w).shift(1)

    # ── RSI ──────────────────────────────────────────────────────────────────
    for p in [7, 14, 21]:
        f[f"g_rsi{p}"] = compute_rsi(g, p).shift(1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = g.ewm(span=12, adjust=False).mean().shift(1)
    ema26 = g.ewm(span=26, adjust=False).mean().shift(1)
    f["g_macd"]      = ema12 - ema26
    f["g_macd_sig"]  = f["g_macd"].ewm(span=9, adjust=False).mean()
    f["g_macd_hist"] = f["g_macd"] - f["g_macd_sig"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_ma  = g.shift(1).rolling(20).mean()
    bb_std = g.shift(1).rolling(20).std()
    bb_rng = 4 * bb_std + 1e-9
    f["g_bb_pos"]   = (g.shift(1) - (bb_ma - 2 * bb_std)) / bb_rng
    f["g_bb_width"] = bb_rng / (bb_ma + 1e-9)

    # ── Volatility regimes ────────────────────────────────────────────────────
    g_ret = g.pct_change()
    for w in [5, 10, 21, 63]:
        f[f"g_vol{w}"] = g_ret.shift(1).rolling(w).std()
    f["g_vol_ratio"] = f["g_vol5"] / (f["g_vol63"] + 1e-9)

    # ── ATR ───────────────────────────────────────────────────────────────────
    prev_c = g.shift(1)
    tr = pd.concat([
        base["gold_h"].shift(1) - base["gold_l"].shift(1),
        (base["gold_h"].shift(1) - prev_c).abs(),
        (base["gold_l"].shift(1) - prev_c).abs(),
    ], axis=1).max(axis=1)
    f["g_atr14"]     = tr.rolling(14).mean()
    f["g_atr_norm"]  = f["g_atr14"] / (g.shift(1) + 1e-9)

    # ── Gold / Silver ratio ───────────────────────────────────────────────────
    if "silver" in base.columns:
        gs = (g / base["silver"].replace(0, np.nan))
        f["gs_ratio"]       = gs.shift(1)
        f["gs_ret1"]        = gs.pct_change(1).shift(1)
        f["gs_ret5"]        = gs.pct_change(5).shift(1)
        f["gs_vs_ma50"]     = gs.shift(1) / gs.shift(1).rolling(50).mean()
        # Ratio > 90 → gold historically expensive; < 55 → historically cheap
        f["gs_high"]        = (gs.shift(1) > 90).astype(float)
        f["gs_low"]         = (gs.shift(1) < 55).astype(float)

    # ── US Dollar Index ───────────────────────────────────────────────────────
    if "dxy" in base.columns:
        dxy = base["dxy"]
        f["dxy_ret1"]    = dxy.pct_change(1).shift(1)
        f["dxy_ret5"]    = dxy.pct_change(5).shift(1)
        f["dxy_ret21"]   = dxy.pct_change(21).shift(1)
        dxy_ma50 = dxy.shift(1).rolling(50).mean()
        f["dxy_vs_ma50"] = dxy.shift(1) / (dxy_ma50 + 1e-9)
        f["dxy_strong"]  = (f["dxy_vs_ma50"] > 1.02).astype(float)
        f["dxy_weak"]    = (f["dxy_vs_ma50"] < 0.98).astype(float)

    # ── 10-Year Treasury Yield ────────────────────────────────────────────────
    if "tnx" in base.columns:
        tnx = base["tnx"]
        f["tnx_level"]  = tnx.shift(1)
        f["tnx_d1"]     = tnx.diff(1).shift(1)    # yield change (bps)
        f["tnx_d5"]     = tnx.diff(5).shift(1)
        f["tnx_d21"]    = tnx.diff(21).shift(1)
        tnx_ma50 = tnx.shift(1).rolling(50).mean()
        f["tnx_vs_ma"]  = tnx.shift(1) - tnx_ma50   # spread to MA
        f["tnx_rising"] = (f["tnx_d5"] > 0.15).astype(float)  # rapid yield rise

    # ── Yield Curve (10Y - 3M spread) ────────────────────────────────────────
    if "tnx" in base.columns and "irx" in base.columns:
        yc = base["tnx"] - base["irx"]
        f["yield_curve"]  = yc.shift(1)
        f["yc_d5"]        = yc.diff(5).shift(1)
        f["inverted"]     = (yc.shift(1) < 0).astype(float)  # recession signal

    # ── Real Yield Proxy (10Y yield minus trailing gold return as inflation proxy) ──
    if "tnx" in base.columns:
        gold_1y_ret_pct = g.pct_change(252) * 100   # gold annualized return
        f["real_yield"]   = (base["tnx"] - gold_1y_ret_pct).shift(1)

    # ── VIX — Fear Index ─────────────────────────────────────────────────────
    if "vix" in base.columns:
        vix = base["vix"]
        f["vix_level"]   = vix.shift(1)
        f["vix_ret1"]    = vix.pct_change(1).shift(1)
        f["vix_ret5"]    = vix.pct_change(5).shift(1)
        vix_ma21 = vix.shift(1).rolling(21).mean()
        f["vix_vs_ma"]   = vix.shift(1) / (vix_ma21 + 1e-9)
        f["vix_high"]    = (vix.shift(1) > 25).astype(float)  # safe-haven demand
        f["vix_crisis"]  = (vix.shift(1) > 35).astype(float)  # crisis / extreme fear

    # ── S&P 500 — Risk-On / Risk-Off ─────────────────────────────────────────
    if "spx" in base.columns:
        spx = base["spx"]
        f["spx_ret1"]    = spx.pct_change(1).shift(1)
        f["spx_ret5"]    = spx.pct_change(5).shift(1)
        f["spx_ret21"]   = spx.pct_change(21).shift(1)
        spx_ma50 = spx.shift(1).rolling(50).mean()
        f["spx_vs_ma50"] = spx.shift(1) / (spx_ma50 + 1e-9)
        f["risk_off"]    = (spx.pct_change(5).shift(1) < -0.04).astype(float)

    # ── Crude Oil ─────────────────────────────────────────────────────────────
    if "oil" in base.columns:
        oil = base["oil"]
        f["oil_ret5"]    = oil.pct_change(5).shift(1)
        f["oil_ret21"]   = oil.pct_change(21).shift(1)
        oil_ma50 = oil.shift(1).rolling(50).mean()
        f["oil_vs_ma50"] = oil.shift(1) / (oil_ma50 + 1e-9)
        # Gold/Oil ratio — high ratio = gold expensive relative to commodities
        f["gold_oil_r"]  = (g.shift(1) / oil.shift(1).replace(0, np.nan))
        f["gold_oil_ma"] = f["gold_oil_r"] / f["gold_oil_r"].rolling(50).mean()

    # ── Seasonality ───────────────────────────────────────────────────────────
    f["month_sin"]   = np.sin(2 * np.pi * base.index.month / 12)
    f["month_cos"]   = np.cos(2 * np.pi * base.index.month / 12)
    f["q4_flag"]     = (base.index.month >= 10).astype(float)  # Q4 physical demand

    feature_cols = list(f.columns)
    result = f.copy()
    result["_gold_close"] = g   # carry raw price for target computation
    result = result.dropna(subset=[c for c in feature_cols if "lag" not in c or "lag1" in c])

    logger.info(f"Feature matrix: {len(result)} rows × {len(feature_cols)} features")
    return result, feature_cols, g.reindex(result.index)
