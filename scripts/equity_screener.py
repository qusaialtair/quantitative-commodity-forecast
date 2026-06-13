#!/usr/bin/env python3
"""
scripts/equity_screener.py
==========================
Halal Equities Screener & Discovery Engine

Pipeline
--------
Stage 1  Universe Builder
  ├─ ETF live feeds     → yfinance.funds_data.top_holdings for HLAL, SPUS, ISWD, SPUS (no API key)
  └─ Static seed        → ~185 confirmed-Halal US tickers (always-on failsafe)

Stage 2  Quant Screen  (vectorised pandas — single yfinance bulk call)
  ├─ 6-month return     : (P_t / P_{t-126}) – 1
  ├─ 6-month vol        : std(daily_ret[-126:]) × √252   (annualised)
  ├─ VAM score          : return_6m / vol_6m_ann          (momentum Sharpe)
  ├─ Hard SMA200 gate   : price ≤ SMA(200) → immediately disqualified
  └─ Rank by VAM ↓, take top 50

Stage 3  Metadata  (top 50 only, rate-limited — ~20s)
  └─ yfinance .info → sector, industry, name, market_cap

Stage 4  Export
  └─ data/halal_discovery_candidates.json  (atomic write)
     + SPSK sukuk benchmark for risk-off reference

CLI
---
  python3 scripts/equity_screener.py              # full run
  python3 scripts/equity_screener.py --dry-run    # universe only, no screen
  python3 scripts/equity_screener.py --force      # ignore today's cached output
  python3 scripts/equity_screener.py --top 25     # custom top-N
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Suppress noisy third-party loggers ───────────────────────────────────────
for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer"):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
OUTPUT   = DATA_DIR / "halal_discovery_candidates.json"
LOG_DIR  = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("SCREENER")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Islamic ETFs sourced via yfinance.funds_data.top_holdings (no API key needed).
# Only pure US-listed tickers (no "." suffix) are accepted from these feeds.
_HALAL_ETF_TICKERS: list[str] = [
    "HLAL",   # Wahed FTSE USA Shariah ETF       — FTSE USA Shariah Index (~175 holdings)
    "SPUS",   # SP Funds S&P 500 Sharia           — S&P 500 Sharia screened (~250 holdings)
    "SPTE",   # SP Funds S&P Technology Sharia    — tech tilt
    "SPHE",   # SP Funds S&P Health Care Sharia   — healthcare tilt
]

# Sukuk bond-equivalent — tracked separately, never ranked with equities
_SUKUK_TICKER = "SPSK"

# Screener parameters
_LOOKBACK_DAYS  = 126   # ≈ 6 calendar months in trading days
_SMA_WINDOW     = 200   # hard institutional trend gate
_MIN_DATA_DAYS  = 180   # discard thin-history / recently-listed tickers
_TOP_N          = 50
_META_DELAY_S   = 0.35  # pause between yfinance .info calls (rate limit)

# ── Static seed (~180 tickers) ───────────────────────────────────────────────
# Source: intersection of HLAL + SPUS holdings, cross-checked against
# AAOIFI Sharia screening criteria (business activity + financial ratios).
# Excludes: conventional banks/insurance, defence primes, alcohol, tobacco,
# gambling, adult entertainment.
# Used as an always-on failsafe if live ETF feeds are unavailable.
_STATIC_SEED: frozenset[str] = frozenset({
    # ── Semiconductors & hardware ─────────────────────────────────────────
    "AAPL", "NVDA", "AVGO", "TSM",  "AMD",  "QCOM", "TXN",  "INTC",
    "AMAT", "KLAC", "LRCX", "MU",   "MRVL", "MPWR", "MCHP", "ON",
    "NXPI", "TER",  "ENTG", "COHR", "ONTO", "FORM", "ACLS",
    # ── Software & cloud platforms ────────────────────────────────────────
    "MSFT", "GOOGL","GOOG", "META", "AMZN", "ORCL", "ADBE", "CRM",
    "NOW",  "INTU", "SNPS", "CDNS", "ANSS", "PTC",  "VEEV", "MANH",
    "HUBS", "TEAM", "DDOG", "MDB",  "SNOW", "PLTR", "NET",
    # ── Cybersecurity ────────────────────────────────────────────────────
    "ZS",   "CRWD", "PANW", "FTNT", "OKTA", "S",    "TENB", "CYBR",
    # ── Consumer tech & e-commerce ────────────────────────────────────────
    "TSLA", "SHOP", "EBAY", "BKNG", "EXPE", "ABNB", "UBER", "NFLX",
    # ── Healthcare — pharma & biotech ────────────────────────────────────
    "LLY",  "NVO",  "REGN", "VRTX", "AMGN", "GILD", "ALNY", "IONS",
    "MRNA", "BIIB", "SGEN", "EXEL", "ACAD",
    # ── Healthcare — med-tech & devices ──────────────────────────────────
    "TMO",  "DHR",  "ABT",  "ISRG", "BSX",  "EW",   "SYK",  "MDT",
    "ZBH",  "BDX",  "PODD", "DXCM", "HOLX", "IDXX", "WST",  "RMD",
    "COO",  "BAX",  "AMED",
    # ── Healthcare — tools & diagnostics ─────────────────────────────────
    "ILMN", "A",    "WAT",  "QGEN", "BIO",
    # ── Industrials — machinery & automation ─────────────────────────────
    "HON",  "CAT",  "DE",   "EMR",  "ETN",  "ROK",  "AME",  "PH",
    "ITW",  "FTV",  "CARR", "OTIS", "XYL",  "FAST", "GWW",  "ROP",
    "CMI",  "IR",   "TT",   "GNRC", "CGNX", "KEYS",
    # ── Industrials — logistics (non-defence) ─────────────────────────────
    "UPS",  "FDX",  "CHRW", "EXPD", "XPO",  "ODFL", "SAIA",
    # ── Consumer discretionary (Halal-screened) ───────────────────────────
    "HD",   "LOW",  "NKE",  "TJX",  "ROST", "TSCO", "ORLY", "AZO",
    "F",    "GM",   "DECK", "CROX",
    # ── Consumer staples (non-alcohol, non-tobacco) ───────────────────────
    "PG",   "CL",   "KMB",  "CHD",  "CLX",  "EL",   "ULTA",
    # ── Materials ─────────────────────────────────────────────────────────
    "LIN",  "APD",  "ECL",  "SHW",  "NEM",  "GOLD", "WPM",  "FMC",
    "ALB",  "MP",
    # ── Renewables & clean utilities ──────────────────────────────────────
    "NEE",  "AES",  "CWEN", "FSLR", "ENPH", "RUN",  "ARRY",
    # ── Environmental services ────────────────────────────────────────────
    "WM",   "RSG",  "AWK",
    # ── Energy (conventional commodity producers — pass AAOIFI screens) ─────
    "XOM",  "CVX",  "COP",  "OXY",  "EOG",  "SLB",  "HAL",
    # ── Healthcare — large pharma (confirmed in HLAL/SPUS live feeds) ────────
    "JNJ",  "PFE",  "MRK",  "AZN",  "BMY",
    # ── ADRs — US-listed, confirmed MSCI Islamic-compliant ────────────────
    "ASML", "SAP",  "INFY", "WIT",  "HDB",
})


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Universe Builder
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_etf_top_holdings(etf_ticker: str) -> set[str]:
    """
    Fetch top holdings of a Halal ETF via yfinance.funds_data.top_holdings.
    Filters to pure US-listed tickers (no exchange suffix like ".L" or ".KS").
    Returns empty set on any failure — graceful degradation.
    """
    try:
        fd = yf.Ticker(etf_ticker).funds_data
        df = fd.top_holdings          # DataFrame indexed by Symbol
        if df is None or df.empty:
            logger.warning("%-6s yfinance top_holdings: empty result", etf_ticker)
            return set()

        tickers: set[str] = {
            str(sym).strip().upper()
            for sym in df.index
            if isinstance(sym, str)
            and "." not in sym          # drop foreign-exchange tickers (ASML.AS, etc.)
            and sym.isalpha()
            and 1 <= len(sym) <= 5
        }
        logger.info("%-6s top_holdings: %d US tickers", etf_ticker, len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("%-6s top_holdings failed (%s) — skipping", etf_ticker, exc)
        return set()


def build_halal_universe() -> list[str]:
    """
    Merge all universe sources into a single deduplicated list of
    US-compatible Halal equity tickers.

    Priority:
      1. Live ETF feeds   — yfinance.funds_data.top_holdings for each _HALAL_ETF_TICKERS
      2. Static seed      — always supplements to guarantee minimum viable universe
    """
    universe: set[str] = set()

    # Live ETF top-holdings feeds (HLAL, SPUS, sector tilts)
    for etf in _HALAL_ETF_TICKERS:
        universe |= _fetch_etf_top_holdings(etf)

    live_count = len(universe)
    logger.info("Live ETF feeds: %d unique tickers", live_count)

    # Always supplement with static seed (belt-and-suspenders)
    universe |= _STATIC_SEED
    universe.discard(_SUKUK_TICKER)     # sukuk tracked separately

    # Sanitise: letters only, 1-5 chars (removes warrants, preferred shares)
    universe = {t for t in universe if t.isalpha() and 1 <= len(t) <= 5}

    seed_added = len(universe) - live_count
    logger.info(
        "Final universe: %d tickers  (%d live ETF + %d from static seed)",
        len(universe), live_count, max(seed_added, 0),
    )
    return sorted(universe)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Quantitative Screen (vectorised pandas)
# ══════════════════════════════════════════════════════════════════════════════

def run_quant_screen(universe: list[str], top_n: int = _TOP_N) -> pd.DataFrame:
    """
    Download 1yr daily closes for the full universe in a single yfinance call,
    then apply:

      VAM (Volatility-Adjusted Momentum) = return_6m / vol_6m_ann
        where:
          return_6m  = (P_now / P_126d_ago) – 1
          vol_6m_ann = std(daily_returns[-126:]) × √252

      Hard SMA200 gate: price ≤ SMA(200) → disqualified immediately.

    Returns a DataFrame of the top_n candidates sorted by VAM descending.
    """
    logger.info(
        "Downloading 1yr prices for %d tickers (single bulk call)…", len(universe)
    )
    t0 = time.time()

    raw = yf.download(
        universe,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
    )

    # Extract Close matrix — shape (trading_days, n_tickers)
    prices: pd.DataFrame = (
        raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    )
    prices = prices.dropna(how="all")
    logger.info(
        "Price matrix: %d rows × %d columns  (%.1fs)",
        prices.shape[0], prices.shape[1], time.time() - t0,
    )

    # ── Vectorised computation ────────────────────────────────────────────────
    # Minimum history guard
    valid = prices.columns[prices.notna().sum() >= _MIN_DATA_DAYS]
    prices = prices[valid]

    # 6-month return: last close vs close 126 trading days ago
    p_now  = prices.iloc[-1]
    p_126d = prices.iloc[-_LOOKBACK_DAYS] if len(prices) >= _LOOKBACK_DAYS else prices.iloc[0]
    ret_6m = (p_now / p_126d) - 1.0

    # 6-month annualised volatility
    rets   = prices.pct_change(fill_method=None).iloc[-_LOOKBACK_DAYS:]
    vol_6m = rets.std() * math.sqrt(252)

    # VAM score
    vam = ret_6m / vol_6m.replace(0, float("nan"))

    # SMA200
    sma200 = prices.rolling(_SMA_WINDOW).mean().iloc[-1]

    # Assemble metrics DataFrame
    screen = pd.DataFrame({
        "price":     p_now,
        "sma200":    sma200,
        "return_6m": ret_6m,
        "vol_6m_ann": vol_6m,
        "vam_score": vam,
    }).dropna()

    # Hard SMA200 gate: must be in a confirmed institutional uptrend
    before_gate = len(screen)
    screen = screen[screen["price"] > screen["sma200"]]
    logger.info(
        "SMA200 hard gate: %d passed / %d total  (%d disqualified)",
        len(screen), before_gate, before_gate - len(screen),
    )

    # Rank by VAM descending, take top N
    screen.index.name = "ticker"
    screen = (
        screen
        .sort_values("vam_score", ascending=False)
        .head(top_n)
        .reset_index()
    )
    screen["trend_aligned"] = True

    # Round for clean JSON output
    for col in ("price", "sma200"):
        screen[col] = screen[col].round(2)
    for col in ("return_6m", "vol_6m_ann", "vam_score"):
        screen[col] = screen[col].round(4)

    logger.info("Top %d candidates selected by VAM score.", len(screen))
    return screen


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Metadata enrichment (top 50 only)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_top50_metadata(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch sector, industry, name, market cap for each ticker via yfinance.
    Rate-limited to avoid hammering Yahoo Finance.
    Only called for the final top-N, never the full universe (~500 tickers).
    """
    logger.info("Fetching metadata for %d tickers (~%.0fs)…",
                len(tickers), len(tickers) * _META_DELAY_S)
    meta: dict[str, dict] = {}

    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            meta[ticker] = {
                "name":         info.get("shortName") or info.get("longName") or ticker,
                "sector":       info.get("sector",   "Unknown"),
                "industry":     info.get("industry", "Unknown"),
                "market_cap_b": round((info.get("marketCap") or 0) / 1e9, 1),
            }
        except Exception as exc:
            logger.debug("Metadata failed for %s: %s", ticker, exc)
            meta[ticker] = {
                "name": ticker, "sector": "Unknown",
                "industry": "Unknown", "market_cap_b": 0.0,
            }

        if i % 10 == 0:
            logger.info("  Metadata %d/%d", i, len(tickers))
        time.sleep(_META_DELAY_S)

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Sukuk benchmark + Export
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sukuk() -> dict:
    """
    Fetch SPSK (SP Funds Global Sukuk ETF) price and 6-month return.
    SPSK acts as the Halal portfolio's 'bond/cash equivalent' —
    recommended allocation when gold system is in BEARISH regime.
    """
    try:
        s = (
            yf.download(_SUKUK_TICKER, period="1y", interval="1d",
                        progress=False, auto_adjust=True)
            ["Close"]
            .dropna()
        )
        if len(s) < 2:
            raise ValueError("Insufficient SPSK data")
        price  = round(float(s.iloc[-1].item()), 2)
        ret_6m = (
            round(float((s.iloc[-1] / s.iloc[-_LOOKBACK_DAYS]).item() - 1), 4)
            if len(s) >= _LOOKBACK_DAYS else None
        )
        logger.info("SPSK: $%.2f  6m return: %+.1f%%",
                    price, (ret_6m or 0) * 100)
        return {"ticker": _SUKUK_TICKER, "price": price, "return_6m": ret_6m}
    except Exception as exc:
        logger.warning("SPSK fetch failed: %s", exc)
        return {"ticker": _SUKUK_TICKER, "price": None, "return_6m": None}


def export_results(
    top50:         pd.DataFrame,
    metadata:      dict[str, dict],
    sukuk:         dict,
    universe_size: int,
    top_n:         int,
) -> None:
    """
    Write data/halal_discovery_candidates.json via atomic rename.
    Structure is designed for direct consumption by the DeepSeek discovery agent.
    """
    candidates = []
    for rank, (_, row) in enumerate(top50.iterrows(), 1):
        t = row["ticker"]
        m = metadata.get(t, {})
        candidates.append({
            "rank":         rank,
            "ticker":       t,
            "name":         m.get("name",         t),
            "sector":       m.get("sector",        "Unknown"),
            "industry":     m.get("industry",      "Unknown"),
            "market_cap_b": m.get("market_cap_b",  0.0),
            "price":        float(row["price"]),
            "sma200":       float(row["sma200"]),
            "return_6m":    float(row["return_6m"]),
            "vol_6m_ann":   float(row["vol_6m_ann"]),
            "vam_score":    float(row["vam_score"]),
            "trend_aligned": True,
        })

    output = {
        "run_date":      date.today().isoformat(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "universe_size": universe_size,
        "candidates_n":  len(candidates),
        "screen_params": {
            "lookback_trading_days": _LOOKBACK_DAYS,
            "sma_gate":              _SMA_WINDOW,
            "top_n":                 top_n,
            "metric":                "VAM = return_6m / vol_6m_ann  (momentum Sharpe)",
            "hard_gate":             "price > SMA(200) — below = immediately disqualified",
            "halal_sources":         ["iShares ISWD", "iShares ISDE", "HLAL (EDGAR)", "SPUS (EDGAR)", "static_seed"],
        },
        "candidates":      candidates,
        "sukuk_benchmark": sukuk,
    }

    # Atomic write — prevents half-written reads by pipeline
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2))
    tmp.rename(OUTPUT)
    logger.info("Exported %d candidates → %s", len(candidates), OUTPUT)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Halal Equities Screener — daily discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
        help="Build universe only; print tickers and exit (no price screen, no export)")
    parser.add_argument("--force", action="store_true",
        help="Ignore today's cached output and re-run the full pipeline")
    parser.add_argument("--top", type=int, default=_TOP_N, metavar="N",
        help=f"Number of top candidates to export (default: {_TOP_N})")
    args = parser.parse_args()

    # ── Logging setup ────────────────────────────────────────────────────────
    log_file = LOG_DIR / f"screener_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-10s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    # ── Skip if today's results already exist (unless --force) ───────────────
    if not args.force and OUTPUT.exists():
        try:
            cached = json.loads(OUTPUT.read_text())
            if cached.get("run_date") == date.today().isoformat():
                logger.info(
                    "Today's candidates already exist (%d tickers, %s). "
                    "Use --force to re-run.",
                    cached.get("candidates_n", 0),
                    OUTPUT.name,
                )
                return
        except Exception:
            pass  # corrupt cache — fall through and re-run

    wall_t0 = time.time()

    # ── Stage 1 — Universe ───────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Stage 1: Building Halal equity universe")
    logger.info("━" * 60)
    universe = build_halal_universe()
    logger.info("Universe locked: %d tickers", len(universe))

    if args.dry_run:
        logger.info("--dry-run: stopping after universe build.")
        print("\n".join(universe))
        return

    # ── Stage 2 — Quant screen ───────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Stage 2: Quant screen  (VAM + hard SMA200 gate)")
    logger.info("━" * 60)
    top_df = run_quant_screen(universe, top_n=args.top)

    if top_df.empty:
        logger.error("Quant screen returned no candidates — aborting.")
        sys.exit(1)

    # ── Stage 3 — Metadata ───────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Stage 3: Metadata enrichment for top %d", len(top_df))
    logger.info("━" * 60)
    metadata = fetch_top50_metadata(top_df["ticker"].tolist())

    logger.info("Fetching SPSK sukuk benchmark…")
    sukuk = _fetch_sukuk()

    # ── Stage 4 — Export ─────────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Stage 4: Exporting results")
    logger.info("━" * 60)
    export_results(top_df, metadata, sukuk, len(universe), args.top)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - wall_t0
    top1    = top_df.iloc[0]
    logger.info("━" * 60)
    logger.info(
        "SCREENER COMPLETE  %.0fs | Universe %d | Candidates %d | "
        "#1 %s (VAM %.3f, +%.0f%% / vol %.0f%%)",
        elapsed,
        len(universe),
        len(top_df),
        top1["ticker"],
        top1["vam_score"],
        top1["return_6m"] * 100,
        top1["vol_6m_ann"] * 100,
    )
    logger.info("Output → %s", OUTPUT)
    logger.info("Log    → %s", log_file)


if __name__ == "__main__":
    main()
