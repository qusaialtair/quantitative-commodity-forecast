#!/usr/bin/env python3
"""
scripts/equity_logic.py
=======================
Three-Agent DeepSeek Investment Committee for Halal Equities

Architecture
------------
  Agent 1 — Quant/Fundamental Analyst  (deepseek-chat, parallel)
    Reads halal_discovery_candidates.json, fetches yfinance.Ticker.info for all 50
    tickers, applies AAOIFI debt screen (totalDebt/marketCap > 0.333 → exclude),
    ranks by composite score, selects top 10 with 2-sentence theses.

  Agent 2 — Macro Economist            (deepseek-chat, parallel)
    Fetches VIX (^VIX via yfinance) + HY credit spread (BAMLH0A0HYM2 via FRED API).
    Classifies: RISK_ON / NEUTRAL / RISK_OFF and outputs sector bias JSON.

  Agent 3 — CIO / Portfolio Manager    (deepseek-reasoner, sequential after 1+2)
    Takes top-10 quant picks + macro report. Selects exactly 3 tickers. Weights
    sum to 1.00. SPSK override rule: if RISK_OFF, SPSK gets 50-70% allocation.

Deterministic rules (LLM-independent):
  AAOIFI screen  : totalDebt / marketCap > 0.333  → disqualified (Quant Agent)
  RISK_OFF rule  : VIX > 28 OR HY spread > 550bps → spsk_override = True
  RISK_ON  rule  : VIX < 20 AND HY spread < 400bps → regime = RISK_ON

Output
------
  data/equity_decision.json  (atomic write)

CLI
---
  python3 scripts/equity_logic.py            # run (skip if today's decision exists)
  python3 scripts/equity_logic.py --force    # ignore today's cache, re-run
  python3 scripts/equity_logic.py --dry-run  # agents only, no final write
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# ── Root & env ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
CANDIDATES_FILE  = ROOT / "data" / "halal_discovery_candidates.json"
DECISION_FILE    = ROOT / "data" / "equity_decision.json"
DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── API clients ───────────────────────────────────────────────────────────────
_DS_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
_FRED_KEY = os.getenv("FRED_API_KEY", "")

_client: OpenAI | None = (
    OpenAI(api_key=_DS_KEY, base_url="https://api.deepseek.com") if _DS_KEY else None
)

# ── Constants ─────────────────────────────────────────────────────────────────
_DEBT_RATIO_LIMIT  = 0.333   # AAOIFI Sharia screen: debt/mktcap > 33.3% → exclude
_VIX_RISK_ON       = 20.0
_VIX_RISK_OFF      = 28.0
_SPREAD_RISK_ON    = 400.0   # bps
_SPREAD_RISK_OFF   = 550.0   # bps
_SUKUK_TICKER      = "SPSK"
_SPSK_WEIGHT_MIN   = 0.50
_SPSK_WEIGHT_MAX   = 0.70
_TOP_N_QUANT       = 10      # how many tickers quant agent picks for CIO
_FINAL_PICKS       = 3       # how many final selections CIO makes
_DS_TIMEOUT        = 180     # seconds per DeepSeek call
_INFO_PAUSE        = 0.15    # seconds between yfinance .info calls

# ── Logging ───────────────────────────────────────────────────────────────────
for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer", "openai", "httpx"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("equity_logic")


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.rename(tmp, path)
    logger.info("Wrote %s", path.name)


def _extract_json(text: str) -> dict:
    """
    Pull the first ```json ... ``` block from an LLM response,
    or try to parse the entire text as JSON.
    """
    import re
    m = re.search(r"```json\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    raw = m.group(1) if m else text.strip()
    return json.loads(raw)


def _ds_chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    """Single DeepSeek chat call. Raises on failure (callers catch and log)."""
    if _client is None:
        raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=temperature,
        timeout=_DS_TIMEOUT,
    )
    return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Market data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_vix() -> float | None:
    """Fetch latest VIX close via yfinance."""
    try:
        hist = yf.Ticker("^VIX").history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return None


def _fetch_hy_spread() -> float | None:
    """
    Fetch ICE BofA US High-Yield OAS (BAMLH0A0HYM2) from FRED.
    Returns spread in basis points, or None on failure.

    Fallback chain:
      1. FRED BAMLH0A0HYM2  (primary — retried once on 5xx)
      2. HYG vs IEF price-based proxy  (when FRED is down)
         HYG = iShares HY Corp Bond ETF
         IEI = iShares 3-7yr Treasury ETF
         Proxy spread ≈ 30d annualised underperformance of HYG vs IEI,
         mapped to bps (historical calibration: 1% relative drag ≈ 100bps).
    """
    # ── Primary: FRED ─────────────────────────────────────────────────────────
    if _FRED_KEY:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            "?series_id=BAMLH0A0HYM2"
            f"&api_key={_FRED_KEY}"
            "&sort_order=desc&limit=5&file_type=json"
        )
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code >= 500:
                    logger.warning("FRED returned %d (attempt %d/2) — retrying…",
                                   resp.status_code, attempt + 1)
                    time.sleep(3)
                    continue
                resp.raise_for_status()
                obs = [o for o in resp.json()["observations"] if o["value"] != "."]
                if obs:
                    spread_bps = float(obs[0]["value"]) * 100.0
                    logger.info("HY spread (FRED): %.0f bps", spread_bps)
                    return spread_bps
            except Exception as exc:
                logger.warning("FRED attempt %d failed: %s", attempt + 1, exc)
                time.sleep(3)
        logger.warning("FRED unavailable after 2 attempts — using HYG/IEI proxy")
    else:
        logger.warning("FRED_API_KEY not set — using HYG/IEI proxy")

    # ── Fallback: HYG vs IEI market-based proxy ───────────────────────────────
    try:
        raw = yf.download(["HYG", "IEI"], period="45d", interval="1d",
                          progress=False, auto_adjust=True)
        closes = raw["Close"].dropna()
        if len(closes) < 21:
            return None
        # 30-day total return for each
        hyg_ret = float(closes["HYG"].iloc[-1] / closes["HYG"].iloc[-21] - 1)
        iei_ret = float(closes["IEI"].iloc[-1] / closes["IEI"].iloc[-21] - 1)
        # Relative underperformance of HYG vs IEI → proxy credit stress
        rel_drag = iei_ret - hyg_ret           # positive = HYG lagging (spreads widening)
        proxy_bps = max(200.0, min(900.0, 350.0 + rel_drag * 5000.0))
        logger.info("HY spread (HYG/IEI proxy): %.0f bps  (HYG 30d %+.1f%%  IEI %+.1f%%)",
                    proxy_bps, hyg_ret * 100, iei_ret * 100)
        return proxy_bps
    except Exception as exc:
        logger.warning("HYG/IEI proxy also failed: %s", exc)
        return None


def _classify_regime(vix: float | None, spread_bps: float | None) -> str:
    """
    Deterministic regime classification.
    Returns 'RISK_ON' | 'NEUTRAL' | 'RISK_OFF'.
    """
    if vix is None and spread_bps is None:
        return "NEUTRAL"
    risk_off = (vix is not None and vix > _VIX_RISK_OFF) or \
               (spread_bps is not None and spread_bps > _SPREAD_RISK_OFF)
    risk_on  = (vix is not None and vix < _VIX_RISK_ON) and \
               (spread_bps is None or spread_bps < _SPREAD_RISK_ON)
    if risk_off:
        return "RISK_OFF"
    if risk_on:
        return "RISK_ON"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental data helper
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch yfinance .info for each ticker. Returns dict keyed by ticker.
    Fields extracted: forwardPE, trailingPE, priceToBook, pegRatio,
                      totalDebt, marketCap, sector, industry, longName.
    Rate-limited to avoid 429s.
    """
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            mkt  = info.get("marketCap") or 0
            debt = info.get("totalDebt") or 0
            results[ticker] = {
                "ticker":      ticker,
                "name":        info.get("longName", ticker),
                "sector":      info.get("sector", "Unknown"),
                "industry":    info.get("industry", "Unknown"),
                "market_cap":  mkt,
                "total_debt":  debt,
                "debt_ratio":  (debt / mkt) if mkt > 0 else 1.0,
                "fwd_pe":      info.get("forwardPE"),
                "trailing_pe": info.get("trailingPE"),
                "pb_ratio":    info.get("priceToBook"),
                "peg_ratio":   info.get("pegRatio"),
            }
            if (i + 1) % 10 == 0:
                logger.info("Fundamentals: %d/%d fetched", i + 1, len(tickers))
        except Exception as exc:
            logger.warning("Info fetch failed for %s: %s", ticker, exc)
            results[ticker] = {"ticker": ticker, "debt_ratio": 1.0}
        time.sleep(_INFO_PAUSE)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Quant / Fundamental Analyst
# ─────────────────────────────────────────────────────────────────────────────

_QUANT_SYSTEM = """\
You are a Quantitative Analyst specializing in Sharia-compliant US equities.
You combine momentum signals with fundamental quality screens to identify
the best risk-adjusted opportunities within a pre-screened Halal universe.

STRICT RULES:
1. You will receive a JSON list of up to 50 Halal equity candidates with their
   VAM scores (momentum / annualised vol), price vs SMA200, and fundamentals.
2. Apply the AAOIFI debt screen: any ticker with debt_ratio > 0.333 is
   EXCLUDED — mark it 'excluded' and do NOT include it in your top-10.
3. From the remaining tickers rank by a composite:
     composite = 0.60 × vam_score_norm + 0.25 × quality_norm + 0.15 × valuation_norm
   where:
     quality_norm     = 1 if peg_ratio < 2 AND pb_ratio < 5, else 0.5 or 0 appropriately
     valuation_norm   = simple bucketing on fwd_pe vs sector median
4. Select exactly TOP 10 tickers (or fewer if fewer survive the debt screen).
5. For each selected ticker write a 2-sentence investment thesis covering:
   (a) the momentum/technical picture and (b) the fundamental quality.
6. Return ONLY valid JSON — no prose, no markdown fences outside the block.

Output schema (strict):
{
  "top10": [
    {
      "rank": 1,
      "ticker": "XOM",
      "sector": "Energy",
      "vam_score": 2.47,
      "debt_ratio": 0.15,
      "fwd_pe": 12.3,
      "pb_ratio": 1.8,
      "composite_score": 0.82,
      "thesis": "Strong 6-month momentum (VAM 2.47) with price well above SMA200. \
Low debt ratio and attractive valuation make it high-conviction."
    }
  ],
  "excluded_tickers": ["TSLA", "AMZN"],
  "exclusion_reason": "debt_ratio > 0.333 (AAOIFI screen)",
  "summary": "2-sentence macro-quant synthesis of the top picks."
}
"""

def _run_quant_agent(candidates: list[dict]) -> dict:
    """
    Agent 1 — fetch fundamentals for all candidates, apply AAOIFI screen,
    call DeepSeek-chat with the Quant system prompt, return parsed JSON.
    """
    t0 = time.time()
    logger.info("Agent 1 (Quant): fetching fundamentals for %d tickers…", len(candidates))

    tickers = [c["ticker"] for c in candidates]
    fundamentals = _fetch_fundamentals(tickers)

    # Merge fundamentals into candidates
    enriched = []
    for c in candidates:
        tk   = c["ticker"]
        fund = fundamentals.get(tk, {})
        enriched.append({
            "ticker":      tk,
            "rank_screener": c.get("rank"),
            "sector":      fund.get("sector") or c.get("sector", "Unknown"),
            "industry":    fund.get("industry", "Unknown"),
            "vam_score":   round(c.get("vam_score", 0.0), 4),
            "return_6m":   round(c.get("return_6m", 0.0), 4),
            "vol_6m_ann":  round(c.get("vol_6m_ann", 0.0), 4),
            "price":       round(c.get("price", 0.0), 2),
            "sma200":      round(c.get("sma200", 0.0), 2),
            "trend_aligned": c.get("trend_aligned", True),
            "debt_ratio":  round(fund.get("debt_ratio", 1.0), 4),
            "market_cap":  fund.get("market_cap"),
            "fwd_pe":      fund.get("fwd_pe"),
            "trailing_pe": fund.get("trailing_pe"),
            "pb_ratio":    fund.get("pb_ratio"),
            "peg_ratio":   fund.get("peg_ratio"),
        })

    user_msg = (
        "Here are the Halal equity candidates. Apply the AAOIFI debt screen "
        "(debt_ratio > 0.333 → exclude) then select your TOP 10 by composite score.\n\n"
        f"```json\n{json.dumps(enriched, indent=2)}\n```"
    )

    logger.info("Agent 1 (Quant): calling DeepSeek-chat…")
    try:
        raw = _ds_chat("deepseek-chat", _QUANT_SYSTEM, user_msg)
        result = _extract_json(raw)
        result["status"]     = "OK"
        result["duration_s"] = round(time.time() - t0, 1)
    except Exception as exc:
        logger.error("Agent 1 (Quant) failed: %s", exc)
        result = {
            "status":     "FAILED",
            "error":      str(exc),
            "duration_s": round(time.time() - t0, 1),
            "top10":      [],
        }
        # Fallback: take top-10 by VAM with basic AAOIFI filter
        result["top10"] = [
            {
                "rank":            i + 1,
                "ticker":          c["ticker"],
                "sector":          fundamentals.get(c["ticker"], {}).get("sector", c.get("sector", "Unknown")),
                "vam_score":       c.get("vam_score", 0.0),
                "debt_ratio":      fundamentals.get(c["ticker"], {}).get("debt_ratio", 0.0),
                "composite_score": c.get("vam_score", 0.0),
                "thesis":          "Fallback: top VAM score in Halal universe.",
            }
            for i, c in enumerate(
                [x for x in candidates
                 if fundamentals.get(x["ticker"], {}).get("debt_ratio", 1.0) <= _DEBT_RATIO_LIMIT
                ][:_TOP_N_QUANT]
            )
        ]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Macro Economist
# ─────────────────────────────────────────────────────────────────────────────

_MACRO_SYSTEM = """\
You are a Macro Economist advising an Islamic investment fund.
Your role is to assess the current macro environment and provide
a sector rotation bias for a 4-week holding horizon.

You will receive:
  - VIX level (implied equity volatility)
  - HY credit spread in basis points (ICE BofA US High-Yield OAS)
  - Pre-computed regime classification: RISK_ON | NEUTRAL | RISK_OFF

RULES:
1. Accept the pre-computed regime — do NOT override the classification.
2. For each of the following sectors output a bias: OVERWEIGHT | NEUTRAL | UNDERWEIGHT
   Sectors: Technology, Healthcare, Energy, Industrials, Materials,
            Consumer_Staples, Consumer_Discretionary, Financials, Utilities
3. Add a 'key_risk' field: the single biggest macro tail risk in one sentence.
4. Add a 'narrative' field: 3-sentence macro backdrop (rates, USD, credit cycle).
5. Return ONLY valid JSON — no prose outside the JSON block.

Output schema (strict):
{
  "regime": "NEUTRAL",
  "sector_bias": {
    "Technology":             "OVERWEIGHT",
    "Healthcare":             "NEUTRAL",
    "Energy":                 "NEUTRAL",
    "Industrials":            "NEUTRAL",
    "Materials":              "NEUTRAL",
    "Consumer_Staples":       "UNDERWEIGHT",
    "Consumer_Discretionary": "OVERWEIGHT",
    "Financials":             "UNDERWEIGHT",
    "Utilities":              "UNDERWEIGHT"
  },
  "key_risk": "One-sentence tail risk.",
  "narrative": "Three-sentence macro narrative."
}
"""

def _run_macro_agent(vix: float | None, hy_spread: float | None) -> dict:
    """
    Agent 2 — fetch live VIX + HY spread, classify regime deterministically,
    call DeepSeek-chat for sector bias. Returns parsed JSON.
    """
    t0 = time.time()
    regime = _classify_regime(vix, hy_spread)
    logger.info(
        "Agent 2 (Macro): VIX=%.1f  HY=%.0fbps  Regime=%s",
        vix or 0.0, hy_spread or 0.0, regime,
    )

    user_msg = (
        f"Current market data:\n"
        f"  VIX:           {f'{vix:.2f}' if vix else 'unavailable'}\n"
        f"  HY spread:     {f'{hy_spread:.0f} bps' if hy_spread else 'unavailable'}\n"
        f"  Regime:        {regime}\n\n"
        "Based on this data, provide your sector rotation bias for the next 4 weeks."
    )

    logger.info("Agent 2 (Macro): calling DeepSeek-chat…")
    try:
        raw    = _ds_chat("deepseek-chat", _MACRO_SYSTEM, user_msg)
        result = _extract_json(raw)
        result["regime"]     = regime          # enforce deterministic regime
        result["vix"]        = vix
        result["hy_spread_bps"] = hy_spread
        result["status"]     = "OK"
        result["duration_s"] = round(time.time() - t0, 1)
    except Exception as exc:
        logger.error("Agent 2 (Macro) failed: %s", exc)
        result = {
            "status":     "FAILED",
            "error":      str(exc),
            "duration_s": round(time.time() - t0, 1),
            "regime":     regime,
            "vix":        vix,
            "hy_spread_bps": hy_spread,
            "sector_bias":   {},
            "key_risk":      "Macro agent unavailable.",
            "narrative":     "Macro agent call failed — defaulting to neutral stance.",
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — CIO / Portfolio Manager
# ─────────────────────────────────────────────────────────────────────────────

_CIO_SYSTEM = """\
You are the Chief Investment Officer of a Sharia-compliant equity fund.
You receive a Quant Analyst's top-10 shortlist and a Macro Economist's sector
bias report. Your job is to select exactly 3 positions for the next 4 weeks
and assign portfolio weights that sum to exactly 1.00.

STRICT RULES:
1. You MUST select exactly 3 tickers from the quant top-10 list (plus SPSK
   if the RISK_OFF override is active — see rule 2).
2. RISK_OFF override (spsk_override = true):
   - SPSK (SP Funds Global Sukuk ETF) MUST be included as position #1
   - SPSK weight must be between 0.50 and 0.70
   - The remaining 2 equity positions share the rest (lowest-beta preference)
   - Total weights must still sum to 1.00
3. Weight each position based on conviction, macro alignment, and momentum.
4. For each position provide exactly 3 thesis bullets:
   (a) "Momentum: ..."  (b) "Fundamentals: ..."  (c) "Macro alignment: ..."
5. Include a 'cio_reasoning' field: your 3-4 sentence overall rationale.
6. Return ONLY valid JSON. No prose, no markdown outside the JSON block.

Output schema (strict):
{
  "selections": [
    {
      "rank": 1,
      "ticker": "XOM",
      "action": "BUY",
      "weight": 0.40,
      "sector": "Energy",
      "vam_score": 2.47,
      "thesis_bullets": [
        "Momentum: Price is 18% above SMA200 with VAM score of 2.47.",
        "Fundamentals: Fwd P/E 12.3x, debt ratio 0.15 — well within AAOIFI limits.",
        "Macro alignment: Energy is OVERWEIGHT in a RISK_ON/NEUTRAL environment."
      ]
    }
  ],
  "weights_sum": 1.00,
  "spsk_override": false,
  "cio_reasoning": "3-4 sentence overall rationale for the portfolio."
}
"""

def _run_cio_agent(quant_report: dict, macro_report: dict, spsk_override: bool) -> dict:
    """
    Agent 3 — CIO synthesises quant top-10 + macro report into 3 final picks.
    Uses deepseek-reasoner for chain-of-thought portfolio construction.
    """
    t0 = time.time()
    regime = macro_report.get("regime", "NEUTRAL")
    logger.info("Agent 3 (CIO): spsk_override=%s  regime=%s", spsk_override, regime)

    user_msg = (
        f"RISK_OFF override active: {spsk_override}\n"
        f"Market regime: {regime}\n\n"
        "=== QUANT ANALYST REPORT ===\n"
        f"```json\n{json.dumps(quant_report, indent=2)}\n```\n\n"
        "=== MACRO ECONOMIST REPORT ===\n"
        f"```json\n{json.dumps(macro_report, indent=2)}\n```\n\n"
        "Select exactly 3 positions (SPSK mandatory if spsk_override=true). "
        "Weights must sum to 1.00. Return JSON only."
    )

    logger.info("Agent 3 (CIO): calling deepseek-reasoner…")
    try:
        raw    = _ds_chat("deepseek-reasoner", _CIO_SYSTEM, user_msg, temperature=0.0)
        result = _extract_json(raw)
        result["spsk_override"] = spsk_override   # enforce deterministic override flag
        result["status"]     = "OK"
        result["duration_s"] = round(time.time() - t0, 1)

        # Validate weights sum
        picks  = result.get("selections", [])
        w_sum  = round(sum(p.get("weight", 0.0) for p in picks), 4)
        result["weights_sum"] = w_sum
        if abs(w_sum - 1.0) > 0.02:
            logger.warning("CIO weights sum %.4f — expected ~1.00", w_sum)

        # ── Hard SPSK enforcement (Finding 4 — audit patch) ───────────────────
        # If the LLM ignored spsk_override=True, force-inject SPSK at 50 % and
        # re-normalise the remaining two equity picks to 25 % each.
        if spsk_override:
            spsk_picks = [p for p in picks if p.get("ticker") == _SUKUK_TICKER]
            spsk_ok    = spsk_picks and spsk_picks[0].get("weight", 0.0) >= _SPSK_WEIGHT_MIN

            if not spsk_ok:
                logger.warning(
                    "SPSK enforcement: CIO did not include SPSK with ≥%.0f%% weight "
                    "(spsk_override=True) — force-injecting.",
                    _SPSK_WEIGHT_MIN * 100,
                )
                # Keep up to 2 non-SPSK equity picks, drop any rogue SPSK entry
                equity_picks = [p for p in picks if p.get("ticker") != _SUKUK_TICKER][:2]
                eq_weight    = round((1.0 - _SPSK_WEIGHT_MIN) / max(len(equity_picks), 1), 4)

                spsk_entry = {
                    "rank":       1,
                    "ticker":     _SUKUK_TICKER,
                    "action":     "BUY",
                    "weight":     _SPSK_WEIGHT_MIN,
                    "sector":     "Fixed Income",
                    "vam_score":  0.0,
                    "thesis_bullets": [
                        "Momentum: Sukuk ETF provides capital preservation in RISK_OFF.",
                        "Fundamentals: Investment-grade sukuk — Sharia-compliant bond proxy.",
                        "Macro alignment: RISK_OFF override — force-injected by hard enforcement.",
                    ],
                }
                for rank_i, ep in enumerate(equity_picks, 2):
                    ep["rank"]   = rank_i
                    ep["weight"] = eq_weight

                picks = [spsk_entry] + equity_picks
                result["selections"]    = picks
                result["weights_sum"]   = round(sum(p["weight"] for p in picks), 4)
                result["cio_reasoning"] = (
                    result.get("cio_reasoning", "") +
                    " [SPSK hard-enforcement applied: LLM omitted SPSK in RISK_OFF regime.]"
                )
                logger.info(
                    "SPSK enforcement: portfolio rewritten → %s",
                    [(p["ticker"], p["weight"]) for p in picks],
                )

    except Exception as exc:
        logger.error("Agent 3 (CIO) failed: %s", exc)
        result = {
            "status":       "FAILED",
            "error":        str(exc),
            "duration_s":   round(time.time() - t0, 1),
            "selections":   [],
            "weights_sum":  0.0,
            "spsk_override": spsk_override,
            "cio_reasoning": "CIO agent call failed.",
        }
        # Fallback: take top-3 from quant agent by composite score
        top10 = quant_report.get("top10", [])
        if spsk_override:
            spsk_w   = round((_SPSK_WEIGHT_MIN + _SPSK_WEIGHT_MAX) / 2, 2)
            rest_w   = round((1.0 - spsk_w) / 2, 2)
            fallbacks = [
                {
                    "rank": 1, "ticker": _SUKUK_TICKER, "action": "BUY",
                    "weight": spsk_w, "sector": "Fixed Income",
                    "vam_score": 0.0,
                    "thesis_bullets": [
                        "Momentum: Sukuk ETF provides capital preservation in RISK_OFF.",
                        "Fundamentals: Investment-grade sukuk — Sharia-compliant bond proxy.",
                        "Macro alignment: RISK_OFF regime mandates defensive allocation.",
                    ],
                },
            ]
            for i, p in enumerate(top10[:2]):
                fallbacks.append({
                    "rank": i + 2, "ticker": p["ticker"], "action": "BUY",
                    "weight": rest_w, "sector": p.get("sector", "Unknown"),
                    "vam_score": p.get("vam_score", 0.0),
                    "thesis_bullets": [
                        f"Momentum: VAM score {p.get('vam_score', 0):.2f}.",
                        f"Fundamentals: Debt ratio {p.get('debt_ratio', 'N/A')}.",
                        "Macro alignment: Fallback pick — low beta equity.",
                    ],
                })
            result["selections"]  = fallbacks
            result["weights_sum"] = round(sum(f["weight"] for f in fallbacks), 2)
        else:
            for i, p in enumerate(top10[:_FINAL_PICKS]):
                w = [0.45, 0.35, 0.20][i] if i < 3 else 0.0
                result["selections"].append({
                    "rank": i + 1, "ticker": p["ticker"], "action": "BUY",
                    "weight": w, "sector": p.get("sector", "Unknown"),
                    "vam_score": p.get("vam_score", 0.0),
                    "thesis_bullets": [
                        f"Momentum: VAM score {p.get('vam_score', 0):.2f}.",
                        f"Fundamentals: Debt ratio {p.get('debt_ratio', 'N/A')}.",
                        "Macro alignment: Fallback pick by quant rank.",
                    ],
                })
            result["weights_sum"] = round(sum(s["weight"] for s in result["selections"]), 2)
        result["cio_reasoning"] = "CIO agent failed — fallback picks by quant rank."

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_decision_cache() -> dict | None:
    """Return today's cached decision or None."""
    try:
        if DECISION_FILE.exists():
            d = json.loads(DECISION_FILE.read_text())
            if d.get("decision_date") == date.today().isoformat():
                return d
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Telegram heartbeat
# ─────────────────────────────────────────────────────────────────────────────

def _send_equity_heartbeat(decision: dict) -> None:
    try:
        from scripts.telegram_notifier import send_alert
        status  = decision.get("pipeline_status", "UNKNOWN")
        regime  = decision.get("market_regime", "UNKNOWN")
        picks   = decision.get("selections", [])
        vix     = decision.get("macro_context", {}).get("vix")
        spsk_or = decision.get("spsk_override", False)

        vix_str = f"{vix:.1f}" if vix else "N/A"
        icon    = "✅" if status == "SUCCESS" else ("⚠️" if status == "PARTIAL" else "🚨")
        lines   = [
            f"{icon} <b>Halal Equity Decision — {decision.get('decision_date', '')}</b>",
            f"<b>Regime:</b> <code>{regime}</code>   <b>VIX:</b> <code>{vix_str}</code>",
            f"<b>SPSK override:</b> <code>{'YES' if spsk_or else 'no'}</code>",
            "",
            "<b>Selections:</b>",
        ]
        for p in picks:
            lines.append(
                f"  #{p['rank']} <code>{p['ticker']}</code>  "
                f"{int(p.get('weight', 0)*100)}%  VAM {p.get('vam_score', 0):.2f}"
            )
        lines.append(f"\n<i>{_now_utc()}</i>")
        send_alert("\n".join(lines), level="OK" if status == "SUCCESS" else "WARNING")
    except Exception as exc:
        logger.warning("Telegram heartbeat failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(force: bool = False, dry_run: bool = False) -> dict:
    """
    Full pipeline:
      1. Load halal_discovery_candidates.json
      2. Fetch VIX + HY spread (for macro agent)
      3. Run Agent 1 (Quant) + Agent 2 (Macro) in parallel
      4. Determine SPSK override (deterministic from macro data)
      5. Run Agent 3 (CIO) sequentially
      6. Write equity_decision.json (atomic)
      7. Send Telegram heartbeat

    Returns the final decision dict.
    """
    today = date.today().isoformat()

    # ── Cache check ───────────────────────────────────────────────────────────
    if not force:
        cached = _load_decision_cache()
        if cached:
            logger.info("Today's decision already exists — skipping (use --force to override).")
            return cached

    # ── Load screener output ──────────────────────────────────────────────────
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"{CANDIDATES_FILE} not found. "
            "Run: python3 scripts/equity_screener.py"
        )
    screener = json.loads(CANDIDATES_FILE.read_text())
    candidates: list[dict] = screener.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in halal_discovery_candidates.json")

    screen_date = screener.get("run_date", "")
    if screen_date and screen_date != today:
        logger.warning(
            "Screener output is from %s (today is %s). "
            "Consider re-running equity_screener.py first.",
            screen_date, today,
        )

    logger.info("Loaded %d candidates from screener (run_date=%s)", len(candidates), screen_date)

    # ── Fetch market data ─────────────────────────────────────────────────────
    logger.info("Fetching VIX and HY credit spread…")
    vix       = _fetch_vix()
    hy_spread = _fetch_hy_spread()
    regime    = _classify_regime(vix, hy_spread)
    spsk_override = (regime == "RISK_OFF")

    logger.info(
        "Market data: VIX=%.1f  HY=%s bps  Regime=%s  SPSK_override=%s",
        vix or 0.0,
        f"{hy_spread:.0f}" if hy_spread else "N/A",
        regime,
        spsk_override,
    )

    # ── Stages tracking ───────────────────────────────────────────────────────
    stages: dict[str, dict] = {}
    pipeline_status = "SUCCESS"

    # ── Agents 1 + 2 in parallel ──────────────────────────────────────────────
    logger.info("Launching Agent 1 (Quant) and Agent 2 (Macro) in parallel…")
    quant_report: dict = {}
    macro_report: dict = {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_quant = pool.submit(_run_quant_agent, candidates)
        fut_macro = pool.submit(_run_macro_agent, vix, hy_spread)

        for fut in as_completed([fut_quant, fut_macro]):
            try:
                result = fut.result()
                if fut is fut_quant:
                    quant_report = result
                    stages["quant"] = {
                        "status":     result.get("status", "UNKNOWN"),
                        "duration_s": result.get("duration_s", 0.0),
                    }
                    if result.get("status") != "OK":
                        pipeline_status = "PARTIAL"
                else:
                    macro_report = result
                    stages["macro"] = {
                        "status":     result.get("status", "UNKNOWN"),
                        "duration_s": result.get("duration_s", 0.0),
                    }
                    if result.get("status") != "OK":
                        pipeline_status = "PARTIAL"
            except Exception as exc:
                logger.error("Parallel agent raised: %s", exc)
                pipeline_status = "PARTIAL"

    # ── Agent 3: CIO ──────────────────────────────────────────────────────────
    logger.info("Launching Agent 3 (CIO)…")
    cio_report = _run_cio_agent(quant_report, macro_report, spsk_override)
    stages["cio"] = {
        "status":     cio_report.get("status", "UNKNOWN"),
        "duration_s": cio_report.get("duration_s", 0.0),
    }
    if cio_report.get("status") != "OK":
        pipeline_status = "PARTIAL" if pipeline_status == "SUCCESS" else pipeline_status

    # ── Assemble final decision ───────────────────────────────────────────────
    total_s = sum(s.get("duration_s", 0.0) for s in stages.values())

    decision: dict = {
        "decision_date":    today,
        "run_timestamp":    _now_utc(),
        "pipeline_status":  pipeline_status,
        "market_regime":    regime,
        "spsk_override":    spsk_override,
        "selections":       cio_report.get("selections", []),
        "weights_sum":      cio_report.get("weights_sum", 0.0),
        "cio_reasoning":    cio_report.get("cio_reasoning", ""),
        "macro_context": {
            "vix":          vix,
            "hy_spread_bps": hy_spread,
            "regime":       regime,
            "sector_bias":  macro_report.get("sector_bias", {}),
            "key_risk":     macro_report.get("key_risk", ""),
            "narrative":    macro_report.get("narrative", ""),
        },
        "quant_top10":  quant_report.get("top10", []),
        "excluded_tickers": quant_report.get("excluded_tickers", []),
        "screener_run_date": screen_date,
        "stages": stages,
        "total_duration_s": round(total_s, 1),
    }

    # ── Write output ──────────────────────────────────────────────────────────
    if not dry_run:
        _atomic_write(DECISION_FILE, decision)
        _send_equity_heartbeat(decision)

    # ── Log summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Equity Decision — %s  [%s]", today, pipeline_status)
    logger.info("Regime: %s  |  SPSK override: %s", regime, spsk_override)
    for sel in decision["selections"]:
        logger.info(
            "  #%d %-6s  %.0f%%  VAM %.2f",
            sel["rank"], sel["ticker"],
            sel.get("weight", 0) * 100,
            sel.get("vam_score", 0),
        )
    logger.info("Total runtime: %.0fs", total_s)
    logger.info("=" * 60)

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Halal Equity Investment Committee (3-agent DeepSeek)"
    )
    parser.add_argument(
        "--force",   action="store_true",
        help="Ignore today's cached decision and re-run all agents",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run agents but do not write equity_decision.json or send Telegram",
    )
    args = parser.parse_args()

    if not _DS_KEY:
        logger.error("DEEPSEEK_API_KEY not set in .env — cannot run agents.")
        sys.exit(1)

    try:
        decision = run(force=args.force, dry_run=args.dry_run)
        sys.exit(0 if decision.get("pipeline_status") in ("SUCCESS", "PARTIAL") else 1)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
