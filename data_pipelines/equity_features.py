#!/usr/bin/env python3
"""
data_pipelines/equity_features.py
=================================
Stage 1 of the Equity 3-Stage Funnel — Vectorised Feature Engine.

Consumes a Halal equity universe (up to ~3,000 tickers), emits a wide
feature matrix (≥60 columns, one row per ticker) suitable for:
    • LightGBM/XGBoost ranking (Stage 2)
    • Agent swarm context (Stage 3)
    • UI screener grid

Design
------
Price-based features: computed once over a DataFrame whose columns are
tickers, so every indicator is evaluated simultaneously across the whole
universe via pandas/numpy broadcasting (no per-ticker Python loop).

Fundamental features: yfinance.Ticker.info is inherently per-ticker; we
use a bounded ThreadPool (concurrency=8) to keep wall-time linear.

Sharia compliance: AAOIFI financial-ratio gates are computed as floats
AND as a boolean pass flag; Stage 2 decides how strictly to apply them.

CLI
---
    python3 -m data_pipelines.equity_features                # full run
    python3 -m data_pipelines.equity_features --universe seed   # static seed only
    python3 -m data_pipelines.equity_features --no-fund       # skip slow fundamentals
    python3 -m data_pipelines.equity_features --top 500       # limit for dev
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer"):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

# Phase 5 — alternative-data feeds. Signal-only: failures fall through to
# zero-filled columns, never eject tickers from the Sharia-compliant universe.
try:
    from data_pipelines.alt_data.insider  import (
        fetch_insider_signals,
        FEATURE_COLS as INSIDER_COLS,
    )
    from data_pipelines.alt_data.congress import (
        fetch_congress_signals,
        FEATURE_COLS as CONGRESS_COLS,
    )
    _ALT_DATA_AVAILABLE = True
except Exception as _alt_exc:  # pragma: no cover — missing optional deps
    logging.getLogger("EQUITY_FEATURES").warning(
        "Alt-data modules unavailable (%s); features will be zero-filled.",
        _alt_exc,
    )
    INSIDER_COLS  = ["insider_buy_count_90d", "insider_net_usd_90d",
                     "insider_cluster_flag"]
    CONGRESS_COLS = ["congress_buy_count_180d", "congress_net_direction_180d",
                     "congress_committee_flag"]
    _ALT_DATA_AVAILABLE = False

ALT_DATA_COLS = INSIDER_COLS + CONGRESS_COLS

from scripts.equity_screener import (  # noqa: E402
    _STATIC_SEED,
    build_halal_universe,
)

DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "equity_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR  = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("FEATENG")

# ── Pipeline parameters ──────────────────────────────────────────────────────
LOOKBACK       = "2y"   # yfinance period — gives ~504 trading days
MARKET_PROXY   = "SPY"  # for β / idio-vol regression
TRADING_DAYS_Y = 252
FUND_WORKERS   = 4      # concurrent yfinance.info calls (reduced to avoid throttle)
FUND_RETRIES   = 3      # per-ticker retry budget on .info / .fast_info
FUND_BACKOFF   = 0.6    # seconds; doubles per retry, + jitter
FUND_PACING    = 0.15   # floor delay per worker request
MIN_HISTORY    = 252    # require ≥1yr of closes to survive
RF_ANNUAL      = 0.045  # risk-free for Sharpe/Sortino (≈ 3M T-bill)

# AAOIFI Sharia screening thresholds
AAOIFI_DEBT_MAX    = 0.33
AAOIFI_CASH_MAX    = 0.33
AAOIFI_RECV_MAX    = 0.49
AAOIFI_NONHLL_MAX  = 0.05   # non-halal income / total revenue


# ══════════════════════════════════════════════════════════════════════════════
# 1. BULK PRICE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(universe: list[str]) -> dict[str, pd.DataFrame]:
    """
    Single bulk yfinance call returning a dict with Close/Open/High/Low/Volume
    matrices (rows=dates, cols=tickers). Market proxy is appended then split.
    """
    tickers = sorted(set(universe) | {MARKET_PROXY})
    logger.info("Downloading %s OHLCV for %d tickers…", LOOKBACK, len(tickers))
    t0 = time.time()

    raw = yf.download(
        tickers,
        period=LOOKBACK,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )

    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex columns from bulk yfinance download")

    frames: dict[str, pd.DataFrame] = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        df = raw[field].copy()
        df = df.dropna(how="all")
        frames[field.lower()] = df

    valid_cols = frames["close"].columns[frames["close"].notna().sum() >= MIN_HISTORY]
    for k in frames:
        frames[k] = frames[k][valid_cols]

    logger.info(
        "Price matrix: %d rows × %d tickers  (%.1fs)",
        frames["close"].shape[0], frames["close"].shape[1], time.time() - t0,
    )
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# 2. VECTORISED PRICE FEATURES  (all tickers simultaneously)
# ══════════════════════════════════════════════════════════════════════════════

def _wilder_ewm(s: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's smoothing = EMA with alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.DataFrame(
        np.maximum.reduce([tr1.values, tr2.values, tr3.values]),
        index=tr1.index, columns=tr1.columns,
    )


def _pct(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)) - 1.0


def compute_price_features(px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns a (tickers × features) DataFrame. Computed once across the full
    universe via pandas broadcasting — no per-ticker Python loop.
    """
    close  = px["close"]
    high   = px["high"]
    low    = px["low"]
    open_  = px["open"]
    volume = px["volume"]

    rets   = close.pct_change(fill_method=None)
    f: dict[str, pd.Series] = {}

    # ── Returns ──────────────────────────────────────────────────────────────
    p_now = close.iloc[-1]
    for w, name in [(1, "1d"), (5, "5d"), (21, "21d"),
                    (63, "63d"), (126, "126d"), (252, "252d")]:
        if len(close) > w:
            f[f"return_{name}"] = _pct(p_now, close.iloc[-1 - w])
    # Jegadeesh–Titman 12-1 momentum: return from t-252 to t-21
    if len(close) > 252:
        f["momentum_12_1"] = _pct(close.iloc[-21], close.iloc[-252])

    # ── 52-week levels & drawdown ────────────────────────────────────────────
    hi_252 = close.iloc[-252:].max()
    lo_252 = close.iloc[-252:].min()
    f["dist_to_52w_high"] = (p_now / hi_252) - 1.0
    f["dist_to_52w_low"]  = (p_now / lo_252) - 1.0
    run_max = close.iloc[-252:].cummax().iloc[-1]
    f["drawdown_current"] = (p_now / run_max) - 1.0

    # ── Volatility ───────────────────────────────────────────────────────────
    for w, name in [(21, "21d"), (63, "63d"), (126, "126d"), (252, "252d")]:
        if len(rets) > w:
            f[f"vol_{name}_ann"] = rets.iloc[-w:].std() * math.sqrt(TRADING_DAYS_Y)

    # Parkinson (high/low range, more efficient than close-to-close)
    hl_sq = (np.log(high / low) ** 2).iloc[-63:]
    f["parkinson_vol_63d"] = np.sqrt(hl_sq.mean() / (4 * math.log(2))) * math.sqrt(TRADING_DAYS_Y)

    # Garman-Klass (includes open-close)
    logs_hl = np.log(high / low).iloc[-63:]
    logs_co = np.log(close / open_).iloc[-63:]
    gk = 0.5 * logs_hl ** 2 - (2 * math.log(2) - 1) * logs_co ** 2
    f["garman_klass_vol_63d"] = np.sqrt(gk.mean().clip(lower=0)) * math.sqrt(TRADING_DAYS_Y)

    # ── Moving averages & price distance ─────────────────────────────────────
    sma20  = close.rolling(20).mean().iloc[-1]
    sma50  = close.rolling(50).mean().iloc[-1]
    sma100 = close.rolling(100).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    ema12  = close.ewm(span=12,  adjust=False).mean().iloc[-1]
    ema26  = close.ewm(span=26,  adjust=False).mean().iloc[-1]

    f["price_to_sma20"]   = _pct(p_now, sma20)
    f["price_to_sma50"]   = _pct(p_now, sma50)
    f["price_to_sma100"]  = _pct(p_now, sma100)
    f["price_to_sma200"]  = _pct(p_now, sma200)
    f["price_to_ema12"]   = _pct(p_now, ema12)
    f["price_to_ema26"]   = _pct(p_now, ema26)
    f["golden_cross"]     = (sma50 / sma200.replace(0, np.nan)) - 1.0

    # 20-day slope of SMA20, normalised by level
    sma20_past = close.rolling(20).mean().iloc[-21]
    f["sma20_slope_21d"] = (sma20 - sma20_past) / sma20_past.replace(0, np.nan)

    # ── RSI (Wilder, 14) and RSI-2 (Connors mean-rev) ─────────────────────────
    for period, tag in [(14, "14"), (2, "2")]:
        delta = close.diff()
        gain  = _wilder_ewm(delta.clip(lower=0), period)
        loss  = _wilder_ewm(-delta.clip(upper=0), period)
        rs = gain / loss.replace(0, np.nan)
        f[f"rsi_{tag}"] = (100 - 100 / (1 + rs)).iloc[-1]

    # ── MACD(12,26,9) ────────────────────────────────────────────────────────
    macd_line   = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    # Normalise by price so it's comparable across tickers
    f["macd_line_norm"]   = macd_line.iloc[-1]   / p_now.replace(0, np.nan)
    f["macd_signal_norm"] = macd_signal.iloc[-1] / p_now.replace(0, np.nan)
    f["macd_hist_norm"]   = macd_hist.iloc[-1]   / p_now.replace(0, np.nan)

    # ── Bollinger(20, 2) ─────────────────────────────────────────────────────
    bb_ma = close.rolling(20).mean()
    bb_sd = close.rolling(20).std()
    bb_up = bb_ma + 2 * bb_sd
    bb_lo = bb_ma - 2 * bb_sd
    band_range = (bb_up - bb_lo).replace(0, np.nan)
    f["bb_pct_b_20"]     = ((close.iloc[-1] - bb_lo.iloc[-1]) / band_range.iloc[-1])
    f["bb_bandwidth_20"] = band_range.iloc[-1] / bb_ma.iloc[-1].replace(0, np.nan)

    # ── Donchian (20) position within channel ────────────────────────────────
    dc_hi = close.rolling(20).max().iloc[-1]
    dc_lo = close.rolling(20).min().iloc[-1]
    f["donchian_pct_20"] = (p_now - dc_lo) / (dc_hi - dc_lo).replace(0, np.nan)

    # ── Stochastic(14, 3) ────────────────────────────────────────────────────
    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    stoch_k = 100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)
    f["stoch_k_14"] = stoch_k.iloc[-1]
    f["stoch_d_3"]  = stoch_k.rolling(3).mean().iloc[-1]

    # ── ATR(14) absolute + ATR % of price ────────────────────────────────────
    tr   = _true_range(high, low, close)
    atr14 = _wilder_ewm(tr, 14).iloc[-1]
    f["atr_14"]     = atr14
    f["atr_pct_14"] = atr14 / p_now.replace(0, np.nan)

    # ── ADX(14) — simplified vectorised ──────────────────────────────────────
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_full  = _wilder_ewm(tr, 14)
    plus_di   = 100 * _wilder_ewm(plus_dm, 14)  / atr_full.replace(0, np.nan)
    minus_di  = 100 * _wilder_ewm(minus_dm, 14) / atr_full.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["adx_14"] = _wilder_ewm(dx, 14).iloc[-1]

    # ── Volume / liquidity ───────────────────────────────────────────────────
    vol21 = volume.rolling(21).mean().iloc[-1]
    f["volume_21d_mean_m"]  = vol21 / 1e6
    f["volume_ratio_21d"]   = volume.iloc[-1] / vol21.replace(0, np.nan)
    f["dollar_vol_21d_m"]   = (close * volume).rolling(21).mean().iloc[-1] / 1e6

    # OBV slope (normalised)
    sign = np.sign(close.diff()).fillna(0.0)
    obv  = (sign * volume).cumsum()
    obv_21 = obv.iloc[-21] if len(obv) > 21 else obv.iloc[0]
    f["obv_slope_21d"] = (obv.iloc[-1] - obv_21) / volume.rolling(21).mean().iloc[-1].replace(0, np.nan)

    # VWAP (21d) deviation
    typical = (high + low + close) / 3
    vwap21 = (typical * volume).rolling(21).sum() / volume.rolling(21).sum()
    f["vwap_deviation_21d"] = _pct(p_now, vwap21.iloc[-1])

    # Amihud illiquidity:  mean(|ret| / $volume)  — lower = more liquid
    dollar_vol = (close * volume).replace(0, np.nan)
    amihud = (rets.abs() / dollar_vol).iloc[-21:].mean()
    f["amihud_illiq_21d"] = amihud * 1e9  # scale for readability

    # ── Distributional moments (last 63d) ────────────────────────────────────
    r63 = rets.iloc[-63:]
    f["skew_63d"] = r63.skew()
    f["kurt_63d"] = r63.kurt()

    # ── Sharpe / Sortino (last 126d, annualised) ─────────────────────────────
    r126 = rets.iloc[-126:]
    mean_ann = r126.mean() * TRADING_DAYS_Y
    std_ann  = r126.std()  * math.sqrt(TRADING_DAYS_Y)
    f["sharpe_126d"] = (mean_ann - RF_ANNUAL) / std_ann.replace(0, np.nan)
    downside = r126.where(r126 < 0, 0.0)
    d_std = downside.std() * math.sqrt(TRADING_DAYS_Y)
    f["sortino_126d"] = (mean_ann - RF_ANNUAL) / d_std.replace(0, np.nan)

    out = pd.DataFrame(f)
    out.index.name = "ticker"
    return out


def compute_market_model(close: pd.DataFrame) -> pd.DataFrame:
    """
    OLS regression of each ticker's daily returns on SPY over trailing 252d.
    Returns beta, alpha (daily), and idiosyncratic annualised volatility.
    """
    if MARKET_PROXY not in close.columns:
        logger.warning("%s missing; skipping β/idio-vol.", MARKET_PROXY)
        return pd.DataFrame(
            index=close.columns.drop(MARKET_PROXY, errors="ignore"),
            columns=["beta_252d", "alpha_daily", "idio_vol_252d"], dtype=float,
        )

    rets = close.pct_change(fill_method=None).iloc[-252:]
    mkt  = rets[MARKET_PROXY].dropna()
    peers = rets.drop(columns=MARKET_PROXY).loc[mkt.index]

    m_mean = mkt.mean()
    m_var  = mkt.var()
    beta   = peers.apply(lambda col: col.cov(mkt) / m_var)
    alpha  = peers.mean() - beta * m_mean

    # Residual σ: r_i - (α + β·r_m), annualised
    fitted = np.outer(mkt.values, beta.values) + alpha.values
    resid  = peers.values - fitted
    idio   = pd.Series(np.nanstd(resid, axis=0) * math.sqrt(TRADING_DAYS_Y),
                       index=peers.columns)

    out = pd.DataFrame({
        "beta_252d":      beta,
        "alpha_daily":    alpha,
        "idio_vol_252d":  idio,
    })
    out.index.name = "ticker"
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNDAMENTALS  (threaded yfinance.info)
# ══════════════════════════════════════════════════════════════════════════════

_FUND_FIELDS = [
    # identity
    "shortName", "longName", "sector", "industry", "country", "currency",
    "marketCap", "enterpriseValue", "sharesOutstanding", "floatShares",
    # valuation
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "pegRatio",
    # profitability
    "returnOnAssets", "returnOnEquity", "grossMargins", "operatingMargins",
    "profitMargins", "ebitdaMargins",
    # growth
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    # balance sheet
    "totalCash", "totalDebt", "debtToEquity", "currentRatio", "quickRatio",
    "totalRevenue", "ebitda", "freeCashflow", "operatingCashflow",
    # ownership / short
    "heldPercentInsiders", "heldPercentInstitutions",
    "shortPercentOfFloat", "shortRatio",
    # risk
    "beta",
    # dividend (for AAOIFI screening — purification)
    "dividendYield", "payoutRatio",
]


# Balance-sheet line names vary across yfinance versions / SEC concepts.
# We probe in priority order and take the first non-NaN match.
_BS_DEBT_KEYS = [
    "Total Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Long Term Debt",
    "Current Debt",
    "Short Long Term Debt",
]
_BS_CASH_KEYS = [
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash Financial",
    "Total Cash",
]


def _retry_info(tk: "yf.Ticker") -> dict:
    """yfinance .info with exponential backoff + jitter.

    Returns {} if every attempt yields an empty / throttled response. A dict
    is considered populated if ≥1 of {marketCap, sector, shortName} is set.
    """
    import random
    delay = FUND_BACKOFF
    for attempt in range(1, FUND_RETRIES + 1):
        try:
            info = tk.info or {}
            if info and (info.get("marketCap") or info.get("sector") or info.get("shortName")):
                return info
        except Exception as exc:
            logger.debug("info attempt %d failed: %s", attempt, exc)
        time.sleep(delay + random.uniform(0, 0.3))
        delay = min(delay * 2.0, 4.0)
    return {}


def _fast_info_fill(tk: "yf.Ticker") -> dict:
    """Scrape .fast_info for fields .info missed — primarily marketCap."""
    out: dict = {}
    try:
        fi = tk.fast_info
    except Exception:
        return out
    for src, dst in (("market_cap", "marketCap"),
                     ("shares", "sharesOutstanding"),
                     ("currency", "currency")):
        try:
            v = getattr(fi, src, None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[dst] = float(v) if src != "currency" else v
        except Exception:
            continue
    return out


def _balance_sheet_fill(tk: "yf.Ticker") -> dict:
    """Pull Total Debt + Cash from the most recent balance sheet column."""
    try:
        bs = tk.balance_sheet
    except Exception:
        return {}
    if bs is None or not hasattr(bs, "empty") or bs.empty:
        return {}
    latest = bs.iloc[:, 0]  # most recent period

    def pick(keys: list[str]) -> float | None:
        for k in keys:
            if k in latest.index:
                v = latest[k]
                if pd.notna(v):
                    return float(v)
        return None

    out: dict = {}
    debt = pick(_BS_DEBT_KEYS)
    cash = pick(_BS_CASH_KEYS)
    if debt is not None:
        out["totalDebt"] = debt
    if cash is not None:
        out["totalCash"] = cash
    return out


def _fetch_one(ticker: str) -> tuple[str, dict]:
    """Three-source cascade: .info (retried) → .fast_info → .balance_sheet.

    Result is a dict with every key in _FUND_FIELDS. None means "truly
    unknown after all fallbacks" — the strict Sharia gate will reject such
    rows downstream, as required by AAOIFI.
    """
    row: dict = {k: None for k in _FUND_FIELDS}
    try:
        tk = yf.Ticker(ticker)
    except Exception as exc:
        logger.debug("ticker construct fail %s: %s", ticker, exc)
        return ticker, row

    # Pacing floor — gentle on the endpoint, keeps workers from burst-calling.
    time.sleep(FUND_PACING)

    info = _retry_info(tk)
    if info:
        for k in _FUND_FIELDS:
            row[k] = info.get(k)

    # Fallback #1: fast_info for marketCap / shares / currency
    if row.get("marketCap") in (None, 0):
        for k, v in _fast_info_fill(tk).items():
            if row.get(k) in (None, 0):
                row[k] = v

    # Fallback #2: balance sheet for totalDebt / totalCash (Sharia-critical)
    need_debt = row.get("totalDebt") in (None, 0)
    need_cash = row.get("totalCash") in (None, 0)
    if need_debt or need_cash:
        for k, v in _balance_sheet_fill(tk).items():
            if row.get(k) in (None, 0):
                row[k] = v

    # Diagnostic: flag tickers whose Sharia inputs stayed unknown.
    if row.get("marketCap") is None or row.get("totalDebt") is None:
        logger.debug("incomplete fundamentals %s  mcap=%s debt=%s sector=%s",
                     ticker, row.get("marketCap"), row.get("totalDebt"),
                     row.get("sector"))
    return ticker, row


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    tickers = list(tickers)
    logger.info("Fundamentals: %d tickers, %d workers", len(tickers), FUND_WORKERS)
    t0 = time.time()

    rows: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=FUND_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t, row = fut.result()
            rows[t] = row
            done += 1
            if done % 50 == 0:
                logger.info("  fundamentals %d/%d", done, len(tickers))

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    logger.info("Fundamentals done (%.0fs)", time.time() - t0)
    return df


def derive_fundamental_features(fund: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw yfinance.info fields into clean numeric features plus AAOIFI
    Sharia ratios. Conservative NaN handling: missing inputs propagate.
    """
    def _num(col: str) -> pd.Series:
        return pd.to_numeric(fund[col], errors="coerce") if col in fund.columns else pd.Series(np.nan, index=fund.index)

    mcap       = _num("marketCap")
    ev         = _num("enterpriseValue")
    tot_cash   = _num("totalCash")
    tot_debt   = _num("totalDebt")
    revenue    = _num("totalRevenue")
    fcf        = _num("freeCashflow")
    ebitda     = _num("ebitda")
    dte_raw    = _num("debtToEquity")   # yfinance reports in %
    # yfinance returns D/E as percentage (e.g. 85.4 for 0.854). Normalise.
    dte = dte_raw / 100.0

    out = pd.DataFrame(index=fund.index)
    out["mkt_cap"]            = mcap
    out["enterprise_value"]   = ev
    out["pe_trailing"]        = _num("trailingPE")
    out["pe_forward"]         = _num("forwardPE")
    out["pb_ratio"]           = _num("priceToBook")
    out["ps_ratio"]           = _num("priceToSalesTrailing12Months")
    out["ev_ebitda"]          = _num("enterpriseToEbitda")
    out["peg_ratio"]          = _num("pegRatio")
    out["roa"]                = _num("returnOnAssets")
    out["roe"]                = _num("returnOnEquity")
    out["gross_margin"]       = _num("grossMargins")
    out["operating_margin"]   = _num("operatingMargins")
    out["profit_margin"]      = _num("profitMargins")
    out["ebitda_margin"]      = _num("ebitdaMargins")
    out["revenue_growth_yoy"] = _num("revenueGrowth")
    out["earnings_growth_yoy"] = _num("earningsGrowth")
    out["current_ratio"]      = _num("currentRatio")
    out["quick_ratio"]        = _num("quickRatio")
    out["debt_to_equity"]     = dte
    out["fcf_yield"]          = fcf / mcap.replace(0, np.nan)
    out["ocf_yield"]          = _num("operatingCashflow") / mcap.replace(0, np.nan)
    out["ebitda_margin_calc"] = ebitda / revenue.replace(0, np.nan)
    out["inst_ownership"]     = _num("heldPercentInstitutions")
    out["insider_ownership"]  = _num("heldPercentInsiders")
    out["short_pct_float"]    = _num("shortPercentOfFloat")
    out["short_ratio_days"]   = _num("shortRatio")
    out["beta_info"]          = _num("beta")
    out["dividend_yield"]     = _num("dividendYield")

    # ── AAOIFI Sharia financial-ratio gates ──────────────────────────────────
    # Uses market cap as the normaliser (AAOIFI accepts 24-month avg mcap;
    # single-point mcap is a conservative, widely-used simplification).
    out["debt_to_mktcap"] = tot_debt / mcap.replace(0, np.nan)
    out["cash_to_mktcap"] = tot_cash / mcap.replace(0, np.nan)
    # Receivables not in yfinance.info → leave NaN; gate treats NaN as pass.
    out["receivables_to_mktcap"] = np.nan

    # AAOIFI strict compliance: missing/corrupt data is "guilty until proven
    # innocent" — NaN ratios must NOT pass. .fillna(False) ejects any ticker
    # where totalDebt, totalCash, or marketCap failed to come back from yfinance.
    sharia_debt_ok = out["debt_to_mktcap"].le(AAOIFI_DEBT_MAX).fillna(False)
    sharia_cash_ok = out["cash_to_mktcap"].le(AAOIFI_CASH_MAX).fillna(False)
    out["sharia_debt_pass"] = sharia_debt_ok
    out["sharia_cash_pass"] = sharia_cash_ok
    out["sharia_pass"]      = sharia_debt_ok & sharia_cash_ok

    # Identity fields kept for UI
    for col in ("shortName", "longName", "sector", "industry", "country", "currency"):
        out[col] = fund[col] if col in fund.columns else None
    out["name"] = out["shortName"].fillna(out["longName"]).fillna(out.index.to_series())

    out.index.name = "ticker"
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. ASSEMBLY & EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _merge_alt_data(feats: pd.DataFrame) -> pd.DataFrame:
    """Left-join insider + congressional feeds; NaN → 0 per signal-only contract.

    Alt-data is NEVER permitted to alter the Sharia universe — the gate
    lives upstream in derive_fundamental_features(). Any scrape outage
    yields zero-filled columns so the ranker contract stays stable.
    """
    tickers = feats.index.tolist()
    sector_map = (feats["sector"].dropna().astype(str).to_dict()
                  if "sector" in feats.columns else {})

    # Insider (OpenInsider / Form-4)
    if _ALT_DATA_AVAILABLE:
        try:
            ins = fetch_insider_signals(tickers)
        except Exception as exc:
            logger.warning("insider feed crashed (%s) — zero-filling", exc)
            ins = pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                               columns=INSIDER_COLS)
    else:
        ins = pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                           columns=INSIDER_COLS)

    # Congressional (CapitolTrades)
    if _ALT_DATA_AVAILABLE:
        try:
            cong = fetch_congress_signals(tickers, sectors=sector_map)
        except Exception as exc:
            logger.warning("congress feed crashed (%s) — zero-filling", exc)
            cong = pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                                columns=CONGRESS_COLS)
    else:
        cong = pd.DataFrame(0, index=pd.Index(tickers, name="ticker"),
                            columns=CONGRESS_COLS)

    feats = feats.join(ins,  how="left")
    feats = feats.join(cong, how="left")
    for c in ALT_DATA_COLS:
        if c not in feats.columns:
            feats[c] = 0
    feats[ALT_DATA_COLS] = feats[ALT_DATA_COLS].fillna(0)
    logger.info("alt-data merged: %d tickers · insider_active=%d · congress_active=%d",
                len(tickers),
                int((feats["insider_buy_count_90d"]   > 0).sum()),
                int((feats["congress_buy_count_180d"] > 0).sum()))
    return feats


def build_feature_matrix(
    universe: list[str],
    include_fundamentals: bool = True,
    include_alt_data: bool = True,
) -> pd.DataFrame:
    px = download_ohlcv(universe)
    logger.info("Computing %d vectorised price features…", 50)
    price_feats  = compute_price_features(px)
    market_feats = compute_market_model(px["close"])

    feats = price_feats.join(market_feats, how="left")
    feats = feats.drop(index=MARKET_PROXY, errors="ignore")

    if include_fundamentals:
        raw_fund = fetch_fundamentals(feats.index.tolist())
        fund_feats = derive_fundamental_features(raw_fund)
        feats = feats.join(fund_feats, how="left")

    if include_alt_data:
        feats = _merge_alt_data(feats)

    feats["last_close"]     = px["close"].iloc[-1]
    feats["last_price_date"] = px["close"].index[-1].strftime("%Y-%m-%d")

    # Stable column order: identity → technical → risk → fundamental → sharia → alt-data
    ordered = [c for c in (
        "name", "sector", "industry", "country", "currency",
        "last_close", "last_price_date", "mkt_cap",
    ) if c in feats.columns]
    ordered += [c for c in feats.columns if c not in ordered and c not in ALT_DATA_COLS]
    ordered += [c for c in ALT_DATA_COLS if c in feats.columns]
    feats = feats[ordered]

    return feats


def export(feats: pd.DataFrame, tag: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = f"features_{stamp}" + (f"_{tag}" if tag else "")
    parquet_path = OUT_DIR / f"{base}.parquet"
    csv_path     = OUT_DIR / f"{base}.csv"
    latest       = OUT_DIR / "latest.parquet"
    manifest     = OUT_DIR / "latest_manifest.json"

    tmp = parquet_path.with_suffix(".parquet.tmp")
    try:
        feats.to_parquet(tmp, compression="snappy")
        tmp.rename(parquet_path)
    except Exception as exc:
        logger.warning("Parquet write failed (%s); falling back to CSV only.", exc)
        parquet_path = None

    feats.to_csv(csv_path, index=True)

    if parquet_path and parquet_path.exists():
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(parquet_path.name)
        except OSError:
            import shutil
            shutil.copyfile(parquet_path, latest)

    manifest.write_text(json.dumps({
        "run_date":      stamp,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_tickers":     int(feats.shape[0]),
        "n_features":    int(feats.shape[1]),
        "parquet":       parquet_path.name if parquet_path else None,
        "csv":           csv_path.name,
    }, indent=2))
    logger.info("Exported %d×%d → %s", feats.shape[0], feats.shape[1],
                parquet_path or csv_path)
    return parquet_path or csv_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_universe(kind: str, top: int | None) -> list[str]:
    if kind == "seed":
        u = sorted(_STATIC_SEED)
    elif kind == "live":
        u = build_halal_universe()
    else:
        raise ValueError(f"unknown universe kind: {kind}")
    if top and top < len(u):
        u = u[:top]
    return u


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1 Vectorised Feature Engine for Halal Equity universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--universe", choices=("live", "seed"), default="live",
                    help="live = ETF feeds ∪ static seed (default); seed = static only")
    ap.add_argument("--no-fund", action="store_true",
                    help="skip fundamentals fetch (price features only)")
    ap.add_argument("--no-alt-data", action="store_true",
                    help="skip insider + congressional alt-data scrapes")
    ap.add_argument("--top", type=int, default=None,
                    help="limit universe to first N tickers (dev/debug)")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix appended to output filename")
    args = ap.parse_args()

    log_file = LOG_DIR / f"feature_engine_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-8s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    wall = time.time()
    universe = _resolve_universe(args.universe, args.top)
    logger.info("Universe: %d tickers (%s)", len(universe), args.universe)

    feats = build_feature_matrix(
        universe,
        include_fundamentals=not args.no_fund,
        include_alt_data=not args.no_alt_data,
    )
    path = export(feats, tag=args.tag)

    logger.info("─" * 60)
    logger.info("FEATURE ENGINE COMPLETE  %.0fs  |  %d × %d  →  %s",
                time.time() - wall, feats.shape[0], feats.shape[1], path.name)


if __name__ == "__main__":
    main()
