#!/usr/bin/env python3
"""
agents/equity_swarm/insider_agent.py
====================================
Insider Agent — reads recent SEC Form 4 filings per ticker and
aggregates insider buy/sell signal over the last 90 days.

Uses only free SEC EDGAR endpoints (no key, just a User-Agent header
per SEC fair-access policy):
   https://www.sec.gov/files/company_tickers.json       ticker → CIK
   https://data.sec.gov/submissions/CIK{cik}.json       recent filings
   Form-4 primary document XML (direct fetch & parse)

Output per ticker:
   n_form4_90d         count of Form 4 filings in last 90 days
   net_dollars_90d     net (buy − sell) traded dollars
   buy_dollars_90d     gross buy dollars
   sell_dollars_90d    gross sell dollars
   n_insider_buys      distinct insiders buying
   n_insider_sells     distinct insiders selling
   signal              z-scaled float ∈ [-1, +1] for CIO conviction input
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("INSIDER")

ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data" / "sec_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "GoldTradingAI insider-agent contact@goldtradingai.local",
    "Accept": "application/json, text/xml, */*",
    "Host":   None,
}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUB_URL     = "https://data.sec.gov/submissions/CIK{cik10}.json"

LOOKBACK_DAYS   = 90
INSIDER_WORKERS = 4   # SEC rate limit: 10 req/s global; keep well under


@dataclass
class InsiderStats:
    ticker:           str
    n_form4_90d:      int   = 0
    buy_dollars_90d:  float = 0.0
    sell_dollars_90d: float = 0.0
    net_dollars_90d:  float = 0.0
    n_insider_buys:   int   = 0
    n_insider_sells:  int   = 0
    signal:           float = 0.0
    error:            str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Ticker → CIK map (cached 7 days)
# ══════════════════════════════════════════════════════════════════════════════

def _load_ticker_cik_map() -> dict[str, str]:
    cache = CACHE_DIR / "ticker_to_cik.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
        return json.loads(cache.read_text())

    logger.info("Fetching SEC ticker→CIK registry…")
    h = {k: v for k, v in HEADERS.items() if v is not None}
    r = requests.get(TICKERS_URL, headers=h, timeout=20)
    r.raise_for_status()
    data = r.json()
    # JSON is keyed by row index; values have ticker & cik_str
    m = {row["ticker"].upper(): f"{int(row['cik_str']):010d}"
         for row in data.values()}
    cache.write_text(json.dumps(m))
    logger.info("Ticker→CIK registry: %d entries cached", len(m))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Per-ticker Form-4 aggregation
# ══════════════════════════════════════════════════════════════════════════════

_CODE_BUY  = {"P", "A"}   # Purchase / Award (both add to insider position)
_CODE_SELL = {"S", "D"}   # Sale / Disposition


def _parse_form4_xml(xml_bytes: bytes) -> tuple[float, float, str]:
    """
    Returns (buy_dollars, sell_dollars, insider_name).
    Parses non-derivative transactions only (ownership changes in common stock).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return 0.0, 0.0, ""

    buy = sell = 0.0
    name = ""
    for rpt in root.findall(".//reportingOwner/reportingOwnerId/rptOwnerName"):
        if rpt.text:
            name = rpt.text.strip()
            break

    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el   = txn.find(".//transactionCoding/transactionCode")
        shares_el = txn.find(".//transactionAmounts/transactionShares/value")
        price_el  = txn.find(".//transactionAmounts/transactionPricePerShare/value")
        if code_el is None or shares_el is None:
            continue
        code = (code_el.text or "").strip().upper()
        try:
            shares = float(shares_el.text or 0)
            price  = float(price_el.text or 0) if price_el is not None else 0.0
        except ValueError:
            continue
        dollars = shares * price
        if code in _CODE_BUY:
            buy += dollars
        elif code in _CODE_SELL:
            sell += dollars
    return buy, sell, name


def _fetch_bytes(url: str, retries: int = 2) -> bytes | None:
    h = {k: v for k, v in HEADERS.items() if v is not None}
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=h, timeout=20)
            if r.status_code == 200:
                return r.content
            if r.status_code == 429:
                time.sleep(1.5)
                continue
            return None
        except Exception:
            time.sleep(0.5)
    return None


def _form4_url(cik: str, accession: str) -> tuple[str, str]:
    """Build URL patterns for Form 4 primary XML. Accession 'nnnnnnnnnn-nn-nnnnnn'."""
    acc_nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}"
    index_url = f"{base}/{accession}-index.htm"
    return base, index_url


def _discover_primary_xml(base: str) -> str | None:
    """Form-4 primary XML is typically named *.xml at the filing root."""
    # Fetch the filing index page and regex for the .xml link.
    raw = _fetch_bytes(f"{base}/")
    if not raw:
        return None
    text = raw.decode("utf-8", errors="ignore")
    m = re.search(r'href="([^"]+\.xml)"', text)
    if not m:
        return None
    href = m.group(1)
    return href if href.startswith("http") else f"https://www.sec.gov{href}" if href.startswith("/") else f"{base}/{href.split('/')[-1]}"


def fetch_ticker_insider(ticker: str, cik: str) -> InsiderStats:
    stats = InsiderStats(ticker=ticker)
    sub = _fetch_bytes(SUB_URL.format(cik10=cik))
    if not sub:
        stats.error = "submissions fetch failed"
        return stats

    try:
        data = json.loads(sub)
        recent = data["filings"]["recent"]
        forms = recent["form"]
        accs  = recent["accessionNumber"]
        dates = recent["filingDate"]
    except Exception as exc:
        stats.error = f"parse submissions: {exc}"
        return stats

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)
    form4_rows = [
        (accs[i], dates[i])
        for i in range(len(forms))
        if forms[i] == "4"
        and datetime.strptime(dates[i], "%Y-%m-%d").date() >= cutoff
    ]

    buyers: set[str] = set()
    sellers: set[str] = set()
    for acc, _dt in form4_rows[:30]:  # hard cap per ticker to stay under rate limits
        base, _ = _form4_url(cik, acc)
        xml_url = _discover_primary_xml(base)
        if not xml_url:
            continue
        xml_bytes = _fetch_bytes(xml_url)
        if not xml_bytes:
            continue
        buy, sell, name = _parse_form4_xml(xml_bytes)
        stats.buy_dollars_90d  += buy
        stats.sell_dollars_90d += sell
        if name:
            if buy > 0:  buyers.add(name)
            if sell > 0: sellers.add(name)
        time.sleep(0.12)  # SEC fair-use pacing

    stats.n_form4_90d     = len(form4_rows)
    stats.n_insider_buys  = len(buyers)
    stats.n_insider_sells = len(sellers)
    stats.net_dollars_90d = stats.buy_dollars_90d - stats.sell_dollars_90d

    # Normalise to a [-1, +1] conviction signal. log-scale so a $5M net buy
    # isn't swamped by a $500M CEO sale at Apple.
    import math
    net = stats.net_dollars_90d
    if abs(net) < 1e-6:
        stats.signal = 0.0
    else:
        stats.signal = max(-1.0, min(1.0,
            math.copysign(math.log10(1 + abs(net) / 1e5) / 3.0, net)))
    return stats


def fetch_insiders(tickers: list[str]) -> dict[str, dict]:
    cik_map = _load_ticker_cik_map()
    jobs = [(t, cik_map[t]) for t in tickers if t in cik_map]
    missing = [t for t in tickers if t not in cik_map]
    if missing:
        logger.warning("CIK not found for %d tickers (e.g. %s)", len(missing), missing[:5])

    logger.info("Scanning SEC Form-4 for %d tickers…", len(jobs))
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=INSIDER_WORKERS) as pool:
        futs = {pool.submit(fetch_ticker_insider, t, c): t for t, c in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            out[s.ticker] = s.as_dict()
            if i % 10 == 0:
                logger.info("  insider %d/%d", i, len(jobs))

    for t in missing:
        out[t] = InsiderStats(ticker=t, error="no CIK match").as_dict()
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tks = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    res = fetch_insiders(tks)
    print(json.dumps(res, indent=2, default=str))
