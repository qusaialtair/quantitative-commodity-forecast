#!/usr/bin/env python3
"""
agents/equity_swarm/sentiment_agent.py
======================================
Sentiment Agent — queries Perplexity Sonar Pro for each top-50 ticker
to surface recent news sentiment, catalysts, and analyst commentary.

Mirrors the cached-JSON pattern used in agents/perplexity_oracle.py
(6h TTL, structured response schema, robust error handling).

Output per ticker:
    score:      float ∈ [-1, +1]   (-1 bearish, +1 bullish)
    signal:     BUY | HOLD | SELL
    summary:    ≤60-word synthesis
    headlines:  list[str] top recent catalysts
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

logger = logging.getLogger("SENT")

CACHE_DIR = ROOT / "data" / "perplexity_cache" / "equity_sentiment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 6 * 3600
API_URL   = "https://api.perplexity.ai/chat/completions"
MODEL     = "sonar-pro"

API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
SENT_WORKERS = 5   # Perplexity soft rate limit

_SYSTEM = (
    "You are a senior Wall Street equity research analyst. "
    "Return ONLY valid JSON — no markdown, no prose, no code fences. "
    "Schema: {\"score\": float in [-1,1], \"signal\": \"BUY\"|\"HOLD\"|\"SELL\", "
    "\"summary\": str (≤60 words), \"headlines\": [str, ...] (≤5 items)}. "
    "Score anchors: -1=materially bearish catalyst in last 7d, 0=balanced, "
    "+1=materially bullish catalyst."
)


def _user_prompt(ticker: str, name: str, sector: str) -> str:
    return (
        f"Summarise the most important news, earnings commentary, analyst "
        f"revisions, and catalysts for {ticker} ({name}, {sector}) from the "
        f"past 7 days. Return only the JSON object per the schema."
    )


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _load_cache(ticker: str) -> dict | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        if time.time() - obj.get("_ts", 0) < CACHE_TTL:
            return obj
    except Exception:
        pass
    return None


def _save_cache(ticker: str, obj: dict) -> None:
    obj["_ts"] = time.time()
    _cache_path(ticker).write_text(json.dumps(obj, indent=2))


def _call_perplexity(ticker: str, name: str, sector: str) -> dict | None:
    if not API_KEY:
        return None
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _user_prompt(ticker, name, sector)},
        ],
        "temperature": 0.1, "max_tokens": 350,
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lower().startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as exc:
        logger.warning("Perplexity sentiment %s failed: %s", ticker, exc)
        return None


def fetch_one(ticker: str, name: str, sector: str) -> dict[str, Any]:
    cached = _load_cache(ticker)
    if cached:
        return {k: cached.get(k) for k in ("score", "signal", "summary", "headlines")} | {"cache": True}
    res = _call_perplexity(ticker, name, sector) or {
        "score": 0.0, "signal": "HOLD",
        "summary": "Perplexity call failed or disabled.", "headlines": [],
    }
    _save_cache(ticker, res)
    return res | {"cache": False}


def fetch_sentiment(tickers: list[dict]) -> dict[str, dict]:
    """tickers = [{ticker, name, sector}, ...] so we give Perplexity context."""
    if not API_KEY:
        logger.warning("PERPLEXITY_API_KEY not set — sentiment agent disabled (zero-filled).")
        return {t["ticker"]: {"score": 0.0, "signal": "HOLD",
                              "summary": "sentiment disabled", "headlines": []}
                for t in tickers}

    logger.info("Perplexity sentiment scan for %d tickers…", len(tickers))
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=SENT_WORKERS) as pool:
        futs = {
            pool.submit(fetch_one, t["ticker"], t.get("name", t["ticker"]),
                        t.get("sector", "Unknown")): t["ticker"]
            for t in tickers
        }
        for i, fut in enumerate(as_completed(futs), 1):
            tk = futs[fut]
            out[tk] = fut.result()
            if i % 10 == 0:
                logger.info("  sentiment %d/%d", i, len(tickers))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    demo = [{"ticker": "AAPL", "name": "Apple Inc", "sector": "Technology"}]
    print(json.dumps(fetch_sentiment(demo), indent=2))
