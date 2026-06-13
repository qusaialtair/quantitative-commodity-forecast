#!/usr/bin/env python3
"""
scripts/daily_trainer.py
========================
Post-market LSTM fine-tuner.  Runs automatically after COMEX Gold close.

Pipeline
--------
1. Fetch today's final closing price            (yfinance)
2. Pull fresh Perplexity oracle intel            (oracle_scout)
3. Load today's CIO decision + oracle scores     (local data files)
4. Ask DeepSeek Reasoner to synthesize a        (deepseek-reasoner)
   calibrated macro feature vector from
   all of the above
5. Fine-tune GoldLSTM-v1 with DeepSeek-          (lstm_predictor.daily_update)
   calibrated Perplexity features
6. Log every run to data/training_log.json

Schedule
--------
Install launchd (22:30 UTC = just after COMEX 17:00 ET close):
    python3 scripts/daily_trainer.py --install-launchd

Manual run (skips market-close guard):
    python3 scripts/daily_trainer.py --force

Why DeepSeek calibration?
--------------------------
Raw Perplexity scores are computed independently for each call.
DeepSeek Reasoner sees TODAY's complete picture — price action, Perplexity
bullets, the CIO's actual decision, oracle history — and produces one
calibrated vector that reflects the *totality* of today's signal before
the weights are baked into the model.
"""

import os, sys, json, argparse, time, subprocess, re
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import yfinance as yf
from openai import OpenAI

DS_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
_client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com") if DS_KEY else None

TRAINING_LOG = ROOT / "data" / "training_log.json"

# ── DeepSeek system prompt ─────────────────────────────────────────────────────

_CALIBRATOR_SYSTEM = """You are the LSTM Feature Calibrator for a precious metals AI trading system.
Given today's complete data feed — price action, Perplexity macro bullets, oracle scores,
the CIO's actual decision, AND today's alternative data readings — produce a calibrated
macro feature vector for fine-tuning the LSTM.

These 4 values are the Perplexity-derived features you must calibrate:
  pplx_fed:          -1.0 (hawkish) to +1.0 (dovish)
  pplx_geo_risk:      0.0 (calm)    to +1.0 (crisis)
  pplx_phys_demand:  -1.0 (weak)    to +1.0 (strong)
  pplx_macro:        -1.0 (bearish for gold) to +1.0 (bullish)

The following 3 alternative data values are provided as READ-ONLY CONTEXT.
They are pre-computed from authoritative sources (FRED, yfinance, CFTC) and
will be passed to the LSTM directly — do NOT output them, only USE them to
improve the calibration of the 4 Perplexity features above:
  real_yield_10y          — FRED DFII10, raw %  (higher = more bearish for gold)
  copper_gold_ratio_zscore — rolling z-score (+2σ = industrial demand surging)
  cot_gold_mm_net_zscore  — rolling z-score of spec net longs (+2σ = crowded long)

Cross-reference rule: If real yields are rising (>2.5%) AND copper/gold z-score
is negative, reduce pplx_macro and pplx_phys_demand by 0.1–0.3. If COT z-score
is above +1.5σ, the market is crowded long — lower pplx_macro by 0.1 (crowded
longs increase mean-reversion risk, not bullish conviction).

Also output:
  training_confidence: 0.0 (skip — noisy day) to 1.0 (clean signal, proceed)
  reasoning: one sentence citing the key cross-asset signal

Rules:
- Be precise to 2 decimal places
- training_confidence < 0.30 means skip fine-tuning today
- If CIO issued VETO or CUT LOSSES, lower training_confidence by 0.2
- Return ONLY a valid JSON object, no extra text"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_gold_close() -> float:
    """Pull today's / most recent COMEX Gold closing price."""
    try:
        df = yf.download("GC=F", period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cl = df["Close"].dropna()
        return float(cl.iloc[-1]) if len(cl) > 0 else 0.0
    except Exception:
        return 0.0


def _load_oracle_bullets() -> tuple[float, list[str]]:
    """Return (latest_score, [bullet1, bullet2, bullet3]) from raw_memory_log."""
    path = ROOT / "data" / "raw_memory_log.json"
    if not path.exists():
        return 0.5, []
    try:
        with open(path) as f:
            log = json.load(f)
        gold_entries = [e for e in log if e.get("ticker") == "GC=F"]
        if not gold_entries:
            return 0.5, []
        latest = sorted(gold_entries, key=lambda e: e.get("date", ""))[-1]
        return float(latest.get("score", 0.5)), latest.get("bullets", [])
    except Exception:
        return 0.5, []


def _load_oracle_score_from_history() -> float:
    """Fallback: latest oracle score from oracle_history.csv."""
    path = ROOT / "data" / "oracle_history.csv"
    if not path.exists():
        return 0.5
    try:
        import pandas as pd
        df = pd.read_csv(path, parse_dates=["date"])
        gold = df[df["ticker"] == "GC=F"].sort_values("date")
        if not gold.empty:
            return float(gold.iloc[-1]["score"])
    except Exception:
        pass
    return 0.5


def _load_latest_cio_decision() -> dict:
    """Return the most recent CIO decision for GC=F from decision_log.json."""
    path = ROOT / "data" / "decision_log.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            log = json.load(f)
        gold = [d for d in log if d.get("ticker") == "GC=F"]
        return gold[-1] if gold else {}
    except Exception:
        return {}


def _load_alt_data_latest() -> dict:
    """
    Return the most recent row of data/alt_data.csv as a dict.
    Keys: real_yield_10y, copper_gold_ratio_raw, copper_gold_ratio_zscore,
          cot_gold_mm_net_raw, cot_gold_mm_net_zscore,
          cot_silver_mm_net_raw, cot_silver_mm_net_zscore, date
    Returns {} if the file doesn't exist or has no rows.
    """
    try:
        from scripts.alt_data_harvester import load_latest_row
        return load_latest_row()
    except Exception as exc:
        print(f"  [alt_data] Could not load latest row: {exc}")
        return {}


def _load_perplexity_raw_scores() -> dict:
    """Get most recent pplx_* scores stored in raw_memory_log."""
    path = ROOT / "data" / "raw_memory_log.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            log = json.load(f)
        gold_entries = [e for e in log if e.get("ticker") == "GC=F"]
        if not gold_entries:
            return {}
        latest = sorted(gold_entries, key=lambda e: e.get("date", ""))[-1]
        return {k: v for k, v in latest.items() if k.startswith("pplx_")}
    except Exception:
        return {}


def _run_oracle_scout() -> dict:
    """
    Call oracle_scout to get fresh Perplexity data.
    Returns the latest entry written, or {} on failure.
    """
    try:
        from scripts.oracle_scout import run_scout
        run_scout(["GC=F"])
    except Exception:
        pass
    return _load_perplexity_raw_scores()


# ── DeepSeek calibration ───────────────────────────────────────────────────────

def calibrate_with_deepseek(
    gold_close: float,
    oracle_score: float,
    oracle_bullets: list,
    perp_raw: dict,
    cio_decision: dict,
    alt_data: dict = None,
) -> dict:
    """
    Ask DeepSeek Reasoner to produce a calibrated macro feature vector
    from all today's data.

    Returns dict with keys:
        pplx_fed, pplx_geo_risk, pplx_phys_demand, pplx_macro,
        training_confidence, reasoning
    """
    fallback = {
        "pplx_fed":          perp_raw.get("pplx_fed",          0.0),
        "pplx_geo_risk":     perp_raw.get("pplx_geo_risk",     0.5),
        "pplx_phys_demand":  perp_raw.get("pplx_phys_demand",  0.0),
        "pplx_macro":        perp_raw.get("pplx_macro",        0.0),
        "training_confidence": 0.6,
        "reasoning":         "DeepSeek unavailable — using raw Perplexity scores.",
    }

    if _client is None:
        print("  [calibrator] DeepSeek key not set — using raw Perplexity scores.")
        return fallback

    bullets_str = "\n".join(f"  - {b}" for b in oracle_bullets) if oracle_bullets else "  (none)"
    raw_str = json.dumps(perp_raw, indent=2) if perp_raw else "  (none)"
    cio_str = (
        f"  Action: {cio_decision.get('action_taken','unknown')}\n"
        f"  Price at time: ${cio_decision.get('price_at_time', 0):,.2f}\n"
        f"  Reasoning: {cio_decision.get('original_reasoning','')}"
    ) if cio_decision else "  (no decision today)"

    # ── Format alternative data block ─────────────────────────────────────────
    if alt_data and any(k in alt_data for k in
                        ["real_yield_10y", "copper_gold_ratio_zscore",
                         "cot_gold_mm_net_zscore"]):
        _ry  = alt_data.get("real_yield_10y",           "N/A")
        _caz = alt_data.get("copper_gold_ratio_zscore",  "N/A")
        _car = alt_data.get("copper_gold_ratio_raw",     "N/A")
        _cgz = alt_data.get("cot_gold_mm_net_zscore",    "N/A")
        _cgr = alt_data.get("cot_gold_mm_net_raw",       "N/A")
        _csz = alt_data.get("cot_silver_mm_net_zscore",  "N/A")
        _csr = alt_data.get("cot_silver_mm_net_raw",     "N/A")
        _adt = alt_data.get("date",                      "N/A")

        def _fmt(v, fmt=".3f"):
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        alt_str = (
            f"  Data date                    : {_adt}\n"
            f"  10Y Real Yield (DFII10)      : {_fmt(_ry)}%  "
            f"(>2.5% = real-rate headwind for gold)\n"
            f"  Copper/Gold Ratio (raw)      : {_fmt(_car, '.6f')}\n"
            f"  Copper/Gold Ratio (252d z)   : {_fmt(_caz, '+.2f')}σ  "
            f"(+2σ = industrial demand surging; -1σ = risk-off)\n"
            f"  COT Gold MM Net (raw)        : {_fmt(_cgr, ',.0f')} contracts\n"
            f"  COT Gold MM Net (252d z)     : {_fmt(_cgz, '+.2f')}σ  "
            f"(>+1.5σ = crowded long, reversal risk)\n"
            f"  COT Silver MM Net (raw)      : {_fmt(_csr, ',.0f')} contracts\n"
            f"  COT Silver MM Net (252d z)   : {_fmt(_csz, '+.2f')}σ"
        )
    else:
        alt_str = "  (alt_data_harvester has not run yet — no data available)"

    user_msg = f"""## Today's Complete Data Feed — Gold (GC=F)

### Price
  Today's closing price: ${gold_close:,.2f}

### Perplexity Oracle Score
  Composite macro score: {oracle_score:.3f}  (0.0=bearish → 1.0=bullish)

### Perplexity Intel Bullets (today's raw search intel):
{bullets_str}

### Raw Perplexity Feature Scores (before calibration):
{raw_str}

### Alternative Data (FRED + yfinance + CFTC — authoritative, no interpretation needed):
{alt_str}

### CIO Decision Today:
{cio_str}

---
Using ALL of the above — especially cross-referencing the alternative data signals
with the Perplexity scores — produce a calibrated macro feature vector for
fine-tuning the LSTM. Return only the JSON object."""

    try:
        resp = _client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": _CALIBRATOR_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=4000,
        )
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()

        # Fallback: extract from reasoning_content if content is empty
        if not raw:
            rc = getattr(msg, "reasoning_content", "") or ""
            m  = re.search(r'\{[^{}]*"pplx_fed"[^{}]*\}', rc, re.DOTALL)
            if m:
                raw = m.group(0)

        if not raw:
            print("  [calibrator] Empty response from DeepSeek — using raw scores.")
            return fallback

        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        result = json.loads(raw)
        # Clamp all feature values to valid ranges
        result["pplx_fed"]          = max(-1.0, min(1.0,  float(result.get("pplx_fed",         0.0))))
        result["pplx_geo_risk"]     = max( 0.0, min(1.0,  float(result.get("pplx_geo_risk",    0.5))))
        result["pplx_phys_demand"]  = max(-1.0, min(1.0,  float(result.get("pplx_phys_demand", 0.0))))
        result["pplx_macro"]        = max(-1.0, min(1.0,  float(result.get("pplx_macro",       0.0))))
        result["training_confidence"] = max(0.0, min(1.0, float(result.get("training_confidence", 0.6))))
        return result

    except Exception as exc:
        print(f"  [calibrator] DeepSeek error: {exc} — using raw scores.")
        return fallback


# ── Training log ───────────────────────────────────────────────────────────────

def _append_training_log(entry: dict) -> None:
    log = []
    if TRAINING_LOG.exists():
        try:
            with open(TRAINING_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    # Keep last 90 days
    if len(log) > 90:
        log = log[-90:]
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_daily_training(force: bool = False) -> bool:
    """
    Execute the full post-market fine-tuning pipeline.
    Returns True if fine-tuning ran, False if skipped.
    """
    print("\n" + "━" * 62)
    print("  WORLDWIDE AI METALS — Daily LSTM Fine-Tuner")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("━" * 62)

    t0 = time.time()

    # ── Step 0a: Refit HMM Regime Models ──────────────────────────────────────
    print("\n[0a/5] Refitting HMM regime models (GC=F, SI=F) …")
    try:
        from scripts.regime_detector import fit as regime_fit
        for _hmm_ticker in ("GC=F", "SI=F"):
            print(f"       Fitting {_hmm_ticker} …", end=" ", flush=True)
            regime_fit(_hmm_ticker)
            print("done")
    except Exception as _hmm_exc:
        print(f"       WARNING: Regime refit failed — {_hmm_exc}")
        print("       Continuing with stale regime model (non-fatal).")

    # ── Step 0b: Update alternative data ──────────────────────────────────────
    print("\n[0b/5] Updating alt data (FRED + Copper/Gold + COT) …")
    try:
        from scripts.alt_data_harvester import run as _alt_run
        _alt_run("update")
        print("       alt_data.csv updated.")
    except Exception as _alt_exc:
        print(f"       WARNING: Alt data update failed — {_alt_exc}")
        print("       Continuing with cached alt data (non-fatal).")

    alt_data = _load_alt_data_latest()
    if alt_data:
        print(f"       Latest alt row: {alt_data.get('date','?')}  "
              f"yield={alt_data.get('real_yield_10y','?')}%  "
              f"cu/au_z={alt_data.get('copper_gold_ratio_zscore','?')}  "
              f"cot_z={alt_data.get('cot_gold_mm_net_zscore','?')}")
    else:
        print("       No alt data available yet.")

    # ── Step 1: Today's closing price ─────────────────────────────────────────
    print("\n[1/5] Fetching Gold closing price …")
    gold_close = _fetch_gold_close()
    print(f"      GC=F close: ${gold_close:,.2f}")

    # ── Step 2: Fresh Perplexity oracle intel ─────────────────────────────────
    print("\n[2/5] Running Perplexity oracle scout …")
    perp_raw = _run_oracle_scout()
    oracle_score, oracle_bullets = _load_oracle_bullets()
    if not oracle_score:
        oracle_score = _load_oracle_score_from_history()
    print(f"      Oracle score: {oracle_score:.3f}   "
          f"pplx_macro={perp_raw.get('pplx_macro', 'N/A')}")

    # ── Step 3: Latest CIO decision ───────────────────────────────────────────
    print("\n[3/5] Loading CIO decision …")
    cio_decision = _load_latest_cio_decision()
    if cio_decision:
        print(f"      CIO: {cio_decision.get('action_taken','?')} "
              f"@ ${cio_decision.get('price_at_time', 0):,.2f} "
              f"({cio_decision.get('date', '?')})")
    else:
        print("      No CIO decision found.")

    # ── Step 4: DeepSeek feature calibration ──────────────────────────────────
    print("\n[4/5] Consulting DeepSeek Reasoner for feature calibration …")
    calibrated = calibrate_with_deepseek(
        gold_close=gold_close,
        oracle_score=oracle_score,
        oracle_bullets=oracle_bullets,
        perp_raw=perp_raw,
        cio_decision=cio_decision,
        alt_data=alt_data,
    )
    conf = calibrated.get("training_confidence", 0.6)
    print(f"      pplx_fed={calibrated['pplx_fed']:+.2f}  "
          f"geo_risk={calibrated['pplx_geo_risk']:.2f}  "
          f"demand={calibrated['pplx_phys_demand']:+.2f}  "
          f"macro={calibrated['pplx_macro']:+.2f}")
    print(f"      Training confidence: {conf:.2f}")
    print(f"      Reasoning: {calibrated.get('reasoning', '')}")

    # Skip fine-tuning if signal is too noisy
    if not force and conf < 0.30:
        print(f"\n  Training skipped — confidence {conf:.2f} < 0.30 threshold.")
        _append_training_log({
            "date":               date.today().isoformat(),
            "timestamp":          datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "gold_close":         round(gold_close, 2),
            "oracle_score":       round(oracle_score, 4),
            "deepseek_calibrated": calibrated,
            "fine_tune_ran":      False,
            "skip_reason":        "low_confidence",
            "duration_s":         round(time.time() - t0, 1),
        })
        print("━" * 62 + "\n")
        return False

    # ── Step 5: Fine-tune LSTM ─────────────────────────────────────────────────
    print("\n[5/5] Fine-tuning GoldLSTM-v1 …")
    try:
        from models.lstm_predictor import daily_update
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lstm_predictor", ROOT / "models" / "lstm_predictor.py")
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        daily_update = mod.daily_update

    perp_for_model = {
        "pplx_fed":         calibrated["pplx_fed"],
        "pplx_geo_risk":    calibrated["pplx_geo_risk"],
        "pplx_phys_demand": calibrated["pplx_phys_demand"],
        "pplx_macro":       calibrated["pplx_macro"],
    }
    # Pass today's alt data as an explicit override so the fine-tune window
    # reflects the current reading rather than relying solely on CSV history
    alt_for_model = {
        k: alt_data[k] for k in
        ["real_yield_10y", "copper_gold_ratio_zscore", "cot_gold_mm_net_zscore"]
        if k in alt_data
    } if alt_data else {}

    result = daily_update(perp_scores=perp_for_model, alt_scores=alt_for_model)
    duration = round(time.time() - t0, 1)

    _append_training_log({
        "date":               date.today().isoformat(),
        "timestamp":          datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "gold_close":         round(gold_close, 2),
        "oracle_score":       round(oracle_score, 4),
        "deepseek_calibrated": calibrated,
        "fine_tune_ran":      True,
        "fine_tune_result":   result,
        "duration_s":         duration,
    })

    print(f"\n  Fine-tune complete in {duration}s")
    if result and result.get("success"):
        print(f"  Epochs: {result.get('epochs_run', 5)}  "
              f"Final loss: {result.get('final_loss', 'N/A')}")
    print("━" * 62 + "\n")
    return True


# ── launchd installer ──────────────────────────────────────────────────────────

def install_launchd():
    """
    Write a launchd plist that runs this script daily at 22:30 UTC
    (5 min after the standard 22:00 UTC = 17:00 ET COMEX Gold close).
    """
    import subprocess as sp
    python  = sys.executable
    script  = str(Path(__file__).resolve())
    workdir = str(ROOT)
    label   = "com.metals.daily_trainer"
    plist   = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logfile = "/tmp/metals_trainer_daily.log"

    # 22:30 UTC — after COMEX close year-round (covers both EST 17:30 and EDT 18:30)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>             <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
    </array>
    <key>WorkingDirectory</key>  <string>{workdir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>          <integer>22</integer>
        <key>Minute</key>        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>   <string>{logfile}</string>
    <key>StandardErrorPath</key> <string>{logfile}</string>
    <key>RunAtLoad</key>         <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(xml)
    print(f"  plist written: {plist}")

    sp.run(["launchctl", "unload", str(plist)], capture_output=True)
    result = sp.run(["launchctl", "load", str(plist)],
                    capture_output=True, text=True)
    if result.returncode == 0:
        print("  launchd job loaded — daily fine-tuning at 22:30 UTC.")
        print(f"  Logs: {logfile}")
        print(f"  Stop: launchctl unload {plist}")
    else:
        print(f"  launchctl load failed: {result.stderr}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-market LSTM fine-tuner with DeepSeek calibration")
    parser.add_argument("--install-launchd", action="store_true",
                        help="Install launchd job to run at 22:30 UTC daily")
    parser.add_argument("--force", action="store_true",
                        help="Run even if training_confidence < 0.30")
    args = parser.parse_args()

    if args.install_launchd:
        install_launchd()
    else:
        run_daily_training(force=args.force)
