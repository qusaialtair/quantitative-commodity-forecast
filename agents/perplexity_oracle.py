"""
Perplexity AI Oracle — Multi-Metal Real-Time Macro Intelligence Layer

Queries Perplexity Sonar Pro with web-grounding, metal-specific prompts:
  Gold     — Fed sentiment, geopolitical safe-haven, central bank demand
  Silver   — Industrial + safe-haven dual demand, solar/EV angle
  Copper   — China construction, green energy infrastructure
  Lithium  — EV adoption, battery gigafactories, sodium-ion competition
  Platinum — Auto catalysts, hydrogen fuel cell economy, SA supply
  Iron     — China steel output, global infrastructure, mining majors

Results are cached 6 hours per metal to control API spend.
Returns structured float scores [-1, +1] feeding directly into ensemble.

Usage:
    oracle = PerplexityOracle(metal_name="lithium")
    scores = oracle.get_scores()
"""

import os
import json
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_BASE  = Path(__file__).parent.parent / "data" / "perplexity_cache"
CACHE_TTL   = 6 * 3600
API_URL     = "https://api.perplexity.ai/chat/completions"
MODEL       = "sonar-pro"

# ── Score key mapping (same keys regardless of metal for model compat) ─────────
SCORE_KEYS = {
    "fed_sentiment":    "pplx_fed",
    "geopolitical_risk":"pplx_geo_risk",
    "physical_demand":  "pplx_phys_demand",
    "macro_narrative":  "pplx_macro",
}

# ══════════════════════════════════════════════════════════════════════════════
# Metal-specific query templates
# Each metal gets 4 queries with tailored prompts but the same JSON schema.
# ══════════════════════════════════════════════════════════════════════════════

def _json_system(role: str) -> str:
    return (
        f"You are a {role}. Return ONLY valid JSON with no markdown, no explanation. "
        "Schema: {\"score\": float, \"summary\": str (max 40 words)} "
    )

def _narrative_system(asset: str) -> str:
    return (
        f"You are a senior {asset} market strategist. Return ONLY valid JSON with no markdown. "
        "Schema: {\"score\": float between -1.0 and 1.0, \"signal\": \"BUY\"|\"HOLD\"|\"SELL\", "
        "\"summary\": str (max 60 words)} "
        "Score -1.0 = strongly bearish. Score +1.0 = strongly bullish."
    )


METAL_QUERY_TEMPLATES: dict[str, dict] = {

    # ── GOLD ──────────────────────────────────────────────────────────────────
    "gold": {
        "fed_sentiment": {
            "system": (
                _json_system("quantitative macro analyst") +
                "Score -1.0 = maximally hawkish (rate hikes, strong dollar, bad for gold). "
                "Score +1.0 = maximally dovish (rate cuts, weak dollar, very good for gold). "
                "Score 0.0 = neutral."
            ),
            "user": (
                "What is the current Federal Reserve monetary policy stance? "
                "Include latest Fed statements, dot plot, market-implied rate probabilities. "
                "How does this affect gold prices? Return only the JSON object."
            ),
        },
        "geopolitical_risk": {
            "system": (
                _json_system("geopolitical risk analyst for a precious metals fund") +
                "Score 0.0 = geopolitically calm. Score 1.0 = extreme global crisis, max safe-haven demand."
            ),
            "user": (
                "Assess current geopolitical risks driving gold safe-haven demand. "
                "Consider armed conflicts, US-China tensions, Middle East, energy disruptions. "
                "Return only the JSON object."
            ),
        },
        "physical_demand": {
            "system": (
                _json_system("precious metals physical demand analyst") +
                "Score -1.0 = very weak demand. Score +1.0 = very strong physical demand."
            ),
            "user": (
                "Current state of physical gold demand globally? "
                "Include: central bank purchases (China, India, Turkey, UAE), "
                "gold ETF flows (GLD, IAU), Shanghai Gold Exchange premiums, "
                "CFTC managed money positioning, retail demand UAE/India/China. "
                "Return only the JSON object."
            ),
        },
        "macro_narrative": {
            "system": _narrative_system("gold"),
            "user": (
                "Current macro assessment for gold (GC=F / XAU/USD). "
                "Consider: USD trajectory, inflation expectations, real yields, "
                "equity direction, any major macro events last 48 hours. "
                "Return only the JSON object."
            ),
        },
    },

    # ── SILVER ────────────────────────────────────────────────────────────────
    "silver": {
        "fed_sentiment": {
            "system": (
                _json_system("precious and industrial metals macro analyst") +
                "Score -1.0 = hawkish Fed + weak industrial demand (very bad for silver). "
                "Score +1.0 = dovish Fed + strong industrial/EV demand (very bullish silver). "
                "Score 0.0 = neutral."
            ),
            "user": (
                "How does the current Federal Reserve stance and industrial activity outlook "
                "affect silver? Silver has dual demand: safe-haven AND industrial/solar. "
                "Include solar panel demand, electronics, Fed rate expectations. "
                "Return only the JSON object."
            ),
        },
        "geopolitical_risk": {
            "system": (
                _json_system("silver market risk analyst") +
                "Score 0.0 = calm. Score 1.0 = extreme risk boosting silver safe-haven + industrial demand."
            ),
            "user": (
                "What geopolitical events are currently affecting silver demand? "
                "Consider: safe-haven flows, supply chain disruptions affecting solar/EV manufacturing, "
                "US-China solar tariffs, energy transition policy changes. "
                "Return only the JSON object."
            ),
        },
        "physical_demand": {
            "system": (
                _json_system("silver physical demand specialist") +
                "Score -1.0 = weak industrial + investment demand. Score +1.0 = very strong demand."
            ),
            "user": (
                "Current silver demand outlook? Include: "
                "solar photovoltaic silver usage (GW installations), "
                "electronics and semiconductor silver demand, "
                "silver ETF flows (SLV, PSLV), "
                "gold/silver ratio vs 5-year average, "
                "mint coin demand (US Mint, Royal Canadian Mint). "
                "Return only the JSON object."
            ),
        },
        "macro_narrative": {
            "system": _narrative_system("silver"),
            "user": (
                "Current macro assessment for silver (SI=F / XAG/USD). "
                "Silver is both a safe-haven and industrial metal. "
                "Consider: gold/silver ratio, green energy transition pace, "
                "manufacturing PMI globally, USD strength, inflation expectations. "
                "Return only the JSON object."
            ),
        },
    },

    # ── INDUSTRIAL METALS DISABLED ────────────────────────────────────────────
    # Copper, Lithium, Platinum, and Iron Perplexity templates are removed.
    # The system only runs AI inference for Gold and Silver.
    # alt_data_harvester.py still fetches HG=F for the Copper/Gold ratio
    # signal — that is a feature input, not a tradable instrument.
}


class PerplexityOracle:
    def __init__(self, metal_name: str = "gold", api_key: str = None, cache_ttl: int = CACHE_TTL):
        self.metal_name = metal_name.lower()
        self.api_key    = api_key or os.getenv("PERPLEXITY_API_KEY", "")
        self.cache_ttl  = cache_ttl
        self.enabled    = bool(self.api_key)

        # Per-metal cache directory
        self.cache_dir = CACHE_BASE / self.metal_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use gold queries as fallback if metal not defined
        self.queries = METAL_QUERY_TEMPLATES.get(self.metal_name,
                                                  METAL_QUERY_TEMPLATES["gold"])

        if not self.enabled:
            logger.warning(
                "PerplexityOracle: PERPLEXITY_API_KEY not set — macro intelligence disabled."
            )
        else:
            logger.info(f"PerplexityOracle: active [{self.metal_name}] (Sonar Pro, 6h cache)")

    # ── Cache helpers ──────────────────────────────────────────────────────────
    def _cache_path(self, query_name: str) -> Path:
        return self.cache_dir / f"{query_name}.json"

    def _load_cache(self, query_name: str) -> dict | None:
        path = self._cache_path(query_name)
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                cached = json.load(fh)
            if time.time() - cached.get("_ts", 0) < self.cache_ttl:
                return cached
        except Exception:
            pass
        return None

    def _save_cache(self, query_name: str, data: dict):
        path = self._cache_path(query_name)
        data["_ts"] = time.time()
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    # ── API call ───────────────────────────────────────────────────────────────
    def _query(self, name: str, spec: dict) -> dict | None:
        cached = self._load_cache(name)
        if cached:
            return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": spec["system"]},
                {"role": "user",   "content": spec["user"]},
            ],
            "temperature": 0.1,
            "max_tokens":  300,
        }
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            self._save_cache(name, result)
            logger.info(
                f"Perplexity [{self.metal_name}/{name}]: "
                f"score={result.get('score', '?')} | {result.get('summary', '')[:60]}"
            )
            return result
        except requests.exceptions.HTTPError as exc:
            # Intercept 429 and 5xx before the broad except swallows them silently
            status = exc.response.status_code if exc.response is not None else 0
            if status == 429 or status >= 500:
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).parent.parent))
                    from scripts.telegram_notifier import send_urgent
                    send_urgent(
                        "Perplexity",
                        exc,
                        context=f"{self.metal_name}/{name}  HTTP {status}",
                    )
                except Exception:
                    pass
            logger.warning(f"Perplexity query '{self.metal_name}/{name}' failed: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"Perplexity query '{self.metal_name}/{name}' failed: {exc}")
            return None

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_scores(self) -> dict:
        """Returns standardised float scores for the ensemble model."""
        if not self.enabled:
            return {
                "pplx_fed":         0.0,
                "pplx_geo_risk":    0.0,
                "pplx_phys_demand": 0.0,
                "pplx_macro":       0.0,
                "pplx_available":   0.0,
            }

        scores = {"pplx_available": 1.0}
        for name, spec in self.queries.items():
            result = self._query(name, spec)
            key = SCORE_KEYS.get(name, f"pplx_{name}")
            scores[key] = float(result.get("score", 0.0)) if result else 0.0
            time.sleep(0.5)

        return scores

    def get_narrative(self) -> str:
        """Human-readable macro summary for the dashboard."""
        if not self.enabled:
            return "Perplexity oracle offline. Set PERPLEXITY_API_KEY to enable."
        result = self._query("macro_narrative", self.queries["macro_narrative"])
        if result:
            signal  = result.get("signal", "HOLD")
            summary = result.get("summary", "No summary available.")
            score   = result.get("score", 0.0)
            return f"[{signal}] Score: {score:+.2f} | {summary}"
        return "Macro narrative unavailable."

    def get_all_summaries(self) -> dict:
        """Returns {query_name: summary_str} for display in the dashboard."""
        summaries = {}
        for name, spec in self.queries.items():
            result = self._query(name, spec)
            summaries[name] = result.get("summary", "") if result else ""
        return summaries


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    metal = sys.argv[1] if len(sys.argv) > 1 else "gold"
    oracle = PerplexityOracle(metal_name=metal)
    print(f"\n=== {metal.upper()} ORACLE ===")
    print("\nMacro Scores:")
    print(json.dumps(oracle.get_scores(), indent=2))
    print("\nNarrative:")
    print(oracle.get_narrative())
