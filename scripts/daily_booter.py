#!/usr/bin/env python3
"""
scripts/daily_booter.py
=======================
Morning metals briefing — standalone, no Streamlit dependency.

What it does each run
---------------------
1. Loads yesterday's predictions from data/predictions_log.json
2. Fetches today's actual open prices → grades yesterday's predictions
3. Runs full signal engine + LSTM forecast for all 6 metals
4. Saves today's predictions to predictions_log.json
5. Fires a formatted Telegram message with the grade report + today's signals

Auto-start (launchd — recommended for macOS, survives reboots)
--------------------------------------------------------------
Run this ONCE in your terminal to install:

    python3 scripts/daily_booter.py --install-launchd

It will write ~/Library/LaunchAgents/com.metals.daily_booter.plist
and load it immediately. After that it runs automatically at 08:00 every day.

Manual test:
    python3 scripts/daily_booter.py
"""

import os, sys, json, logging, argparse, subprocess
from pathlib import Path
from datetime import datetime, date, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
PERPLEXITY_KEY   = os.getenv("PERPLEXITY_API_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PREDICTIONS_FILE = ROOT / "data" / "predictions_log.json"

METALS = {
    "Gold":     {"ticker": "GC=F", "sym": "Au", "emoji": "🥇", "key": "gold"},
    "Silver":   {"ticker": "SI=F", "sym": "Ag", "emoji": "🥈", "key": "silver"},
    "Platinum": {"ticker": "PL=F", "sym": "Pt", "emoji": "⬜", "key": "platinum"},
    "Copper":   {"ticker": "HG=F", "sym": "Cu", "emoji": "🟫", "key": "copper"},
    "Lithium":  {"ticker": "ALB",  "sym": "Li", "emoji": "🔵", "key": "lithium"},
    "Iron":     {"ticker": "VALE", "sym": "Fe", "emoji": "🔩", "key": "iron"},
}

# Only precious metals get Perplexity macro calls — saves API credits
_PRECIOUS_KEYS = {"gold", "silver"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_booter")


# ── Telegram ───────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=20)
        r.raise_for_status()
        log.info("Telegram ✓ delivered")
        return True
    except Exception as exc:
        log.error(f"Telegram ✗  {exc}")
        return False


# ── Prediction log ─────────────────────────────────────────────────────────────

def load_predictions() -> dict:
    if PREDICTIONS_FILE.exists():
        try:
            return json.loads(PREDICTIONS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_predictions(log_data: dict):
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep only last 30 days to avoid unbounded growth
    if len(log_data) > 30:
        keys = sorted(log_data.keys())
        for old in keys[:-30]:
            del log_data[old]
    PREDICTIONS_FILE.write_text(json.dumps(log_data, indent=2))


def grade_yesterday(pred_log: dict, today_prices: dict) -> list[dict]:
    """
    Compare yesterday's predicted signals against today's actual price move.
    A BUY is CORRECT if today's price >= yesterday's close.
    A SELL is CORRECT if today's price <= yesterday's close.
    A HOLD is CORRECT if abs move < 0.5%.
    """
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    yesterday = pred_log.get(yesterday_str, {})
    if not yesterday:
        return []

    grades = []
    for name, pred in yesterday.items():
        if name == "_lstm":
            continue
        actual_close = today_prices.get(name, 0)
        prev_close   = pred.get("close", 0)
        if prev_close <= 0 or actual_close <= 0:
            continue

        move_pct = (actual_close - prev_close) / prev_close * 100
        signal   = pred.get("signal", "HOLD")

        if signal == "BUY":
            correct = move_pct >= 0
        elif signal == "SELL":
            correct = move_pct <= 0
        else:  # HOLD
            correct = abs(move_pct) < 0.5

        grades.append({
            "name":     name,
            "signal":   signal,
            "move_pct": round(move_pct, 2),
            "correct":  correct,
        })

    # Also grade LSTM Gold forecast if present
    lstm_pred = yesterday.get("_lstm", {})
    if lstm_pred and "target" in lstm_pred:
        gold_actual  = today_prices.get("Gold", 0)
        gold_prev    = lstm_pred.get("current_close", 0)
        lstm_target  = lstm_pred.get("target", 0)
        if gold_prev > 0 and gold_actual > 0:
            actual_dir  = "UP" if gold_actual >= gold_prev else "DOWN"
            pred_dir    = lstm_pred.get("direction", "UP")
            correct     = actual_dir == pred_dir
            actual_move = (gold_actual - gold_prev) / gold_prev * 100
            grades.append({
                "name":     "LSTM Gold",
                "signal":   f"→ {pred_dir}  (target ${lstm_target:,.0f})",
                "move_pct": round(actual_move, 2),
                "correct":  correct,
            })

    return grades


# ── Data helpers ───────────────────────────────────────────────────────────────

def fetch_ohlc(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception as exc:
        log.warning(f"{ticker}: fetch failed — {exc}")
        return pd.DataFrame()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d  = series.diff()
    up = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + up / dn.replace(0, float("nan")))


def compute_signal(df: pd.DataFrame, perp: dict) -> dict:
    """Exact mirror of signal_engine() in app.py."""
    empty = dict(signal="HOLD", conf=50, cur=0.0, ret1=0.0, ret5=0.0,
                 rsi_val=50.0, ma50=0.0, ma200=0.0, vol21=0.0)
    if df.empty or len(df) < 50:
        return empty

    cl    = df["Close"]
    cur   = float(cl.iloc[-1])
    ma50  = float(cl.rolling(50).mean().iloc[-1])
    ma200 = float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else ma50
    rsi_v = float(rsi(cl).iloc[-1])
    macd  = float((cl.ewm(span=12).mean() - cl.ewm(span=26).mean()).iloc[-1])
    ret1  = float(cl.pct_change(1).iloc[-1]  * 100)
    ret5  = float(cl.pct_change(5).iloc[-1]  * 100)
    ret21 = float(cl.pct_change(21).iloc[-1] * 100)
    ret63 = float(cl.pct_change(63).iloc[-1] * 100) if len(cl) >= 63 else 0.0
    vol21 = float(cl.pct_change().rolling(21).std().iloc[-1] * 100)

    tscore = sum([
        1 if cur > ma50   else -1,
        1 if cur > ma200  else -1,
        1 if ma50 > ma200 else -1,
        1 if ret5  > 0    else -1,
        1 if ret21 > 0    else -1,
        1 if ret63 > 0    else -1,
        1 if rsi_v < 70   else -1,
        1 if macd  > 0    else -1,
    ]) / 8

    macro  = perp.get("pplx_macro",       0.0)
    demand = perp.get("pplx_phys_demand",  0.0)
    geo    = perp.get("pplx_geo_risk",     0.0)
    comb   = tscore * 0.42 + macro * 0.33 + demand * 0.15 + (geo - 0.3) * 0.10

    veto  = macro < -0.3 or rsi_v > 75 or cur < ma200 * 0.97
    force = macro < -0.6 or (cur < ma50 and cur < ma200 and ret5 < -3)

    if force or (comb < -0.15 and veto):
        sig, conf = "SELL", min(94, int(abs(comb) * 100 + 20))
    elif not veto and comb > 0.2:
        sig, conf = "BUY",  min(94, int(comb * 100 + 30))
    else:
        sig, conf = "HOLD", min(94, int(50 + abs(comb) * 30))

    return dict(signal=sig, conf=conf, cur=cur, ret1=ret1, ret5=ret5,
                rsi_val=rsi_v, ma50=ma50, ma200=ma200, vol21=vol21)


def fetch_perplexity(metal_key: str) -> dict:
    try:
        from agents.perplexity_oracle import PerplexityOracle
        oracle = PerplexityOracle(metal_name=metal_key, api_key=PERPLEXITY_KEY)
        return oracle.get_scores()
    except Exception as exc:
        log.warning(f"Perplexity [{metal_key}]: {exc}")
        return {}


def get_lstm_gold() -> dict:
    try:
        from models.lstm_predictor import predict_next
        return predict_next()
    except Exception as exc:
        log.warning(f"LSTM skipped: {exc}")
        return {}


# ── Message builder ────────────────────────────────────────────────────────────

SIG_ICON = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}

def build_message(results: list, lstm: dict, grades: list) -> str:
    now = datetime.utcnow().strftime("%A, %d %b %Y — %H:%M UTC")

    lines = [
        "🌍  <b>Worldwide AI Metals Briefing</b>",
        f"📅  {now}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # ── Yesterday's scorecard ──
    if grades:
        correct = sum(1 for g in grades if g["correct"])
        total   = len(grades)
        pct     = round(correct / total * 100)
        medal   = "🏆" if pct >= 80 else ("👍" if pct >= 60 else "📉")
        lines.append(f"\n{medal}  <b>Yesterday's Accuracy: {correct}/{total}  ({pct}%)</b>")
        for g in grades:
            tick  = "✅" if g["correct"] else "❌"
            arrow = "▲" if g["move_pct"] >= 0 else "▼"
            lines.append(
                f"  {tick}  {g['name']:10s}  {g['signal']:6s}  "
                f"actual {arrow} {abs(g['move_pct']):.2f}%"
            )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Today's signals ──
    lines.append("\n📊  <b>Today's Signals</b>")
    for r in results:
        icon  = SIG_ICON.get(r["signal"], "⚪")
        arrow = "▲" if r["ret1"] >= 0 else "▼"
        price = r["cur"]
        lines.append(
            f"\n{r['emoji']}  <b>{r['name']}</b>  <code>({r['sym']})</code>\n"
            f"   Price   <code>${price:>10,.2f}</code>\n"
            f"   Signal  {icon} <b>{r['signal']}</b>  ·  {r['conf']}% conf\n"
            f"   1d/5d   {arrow} {r['ret1']:+.2f}%  /  {r['ret5']:+.2f}%  ·  RSI {r['rsi']:.0f}"
        )

    # ── LSTM Gold forecast ──
    if lstm and "error" not in lstm and "adjusted_price" in lstm:
        d_arrow = "▲" if lstm.get("direction") == "UP" else "▼"
        chg     = lstm.get("predicted_change_pct", 0)
        lines.append(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖  <b>GoldLSTM‑v1 Forecast</b>  <i>(next session)</i>\n"
            f"   Current  <code>${lstm.get('current_close', 0):>10,.2f}</code>\n"
            f"   Target   <code>${lstm.get('adjusted_price', 0):>10,.2f}</code>"
            f"  {d_arrow} {chg:+.2f}%"
        )

    lines += [
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️  <i>Personal use only</i>",
    ]
    return "\n".join(lines)


# ── launchd installer ──────────────────────────────────────────────────────────

def install_launchd():
    """Write a launchd plist and load it — runs at 08:00 every day automatically."""
    python  = sys.executable
    script  = str(Path(__file__).resolve())
    workdir = str(ROOT)
    label   = "com.metals.daily_booter"
    plist   = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logfile = "/tmp/metals_booter.log"

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
        <key>Hour</key>          <integer>8</integer>
        <key>Minute</key>        <integer>0</integer>
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
    print(f"✅  plist written → {plist}")

    # Unload if already loaded, then load fresh
    subprocess.run(["launchctl", "unload", str(plist)],
                   capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("✅  launchd job loaded — will run daily at 08:00 automatically.")
        print(f"   Logs → {logfile}")
        print(f"   Stop : launchctl unload {plist}")
    else:
        print(f"❌  launchctl load failed: {result.stderr}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("━━━  Daily Booter starting  ━━━")

    pred_log     = load_predictions()
    today_str    = date.today().isoformat()
    today_prices = {}   # filled as we fetch each metal

    # ── 1. Run signal engine + fetch prices for ALL metals ──
    results = []
    today_entry = {}

    for name, md in METALS.items():
        log.info(f"  Processing {name} ({md['ticker']}) …")
        df   = fetch_ohlc(md["ticker"])
        perp = fetch_perplexity(md["key"]) if md["key"] in _PRECIOUS_KEYS else {}
        sig  = compute_signal(df, perp)

        today_prices[name] = sig["cur"]

        results.append({
            "name":   name,
            "sym":    md["sym"],
            "emoji":  md["emoji"],
            "cur":    sig["cur"],
            "signal": sig["signal"],
            "conf":   sig["conf"],
            "ret1":   sig["ret1"],
            "ret5":   sig["ret5"],
            "rsi":    sig["rsi_val"],
            "vol":    sig["vol21"],
        })

        # Store today's prediction for tomorrow's grading
        today_entry[name] = {
            "signal": sig["signal"],
            "close":  sig["cur"],
            "conf":   sig["conf"],
        }

    # ── 2. LSTM Gold forecast ──
    log.info("  Running LSTM Gold forecast …")
    lstm = get_lstm_gold()
    if lstm and "error" not in lstm and "adjusted_price" in lstm:
        today_entry["_lstm"] = {
            "target":        lstm.get("adjusted_price"),
            "current_close": lstm.get("current_close"),
            "direction":     lstm.get("direction"),
        }

    # ── 3. Grade yesterday's predictions ──
    log.info("  Grading yesterday's predictions …")
    grades = grade_yesterday(pred_log, today_prices)
    if grades:
        correct = sum(1 for g in grades if g["correct"])
        log.info(f"  Score: {correct}/{len(grades)}")
    else:
        log.info("  No prior predictions to grade (first run?)")

    # ── 4. Save today's predictions ──
    pred_log[today_str] = today_entry
    save_predictions(pred_log)
    log.info(f"  Predictions saved → {PREDICTIONS_FILE}")

    # ── 5. Build + send Telegram message ──
    message = build_message(results, lstm, grades)
    log.info("  Sending Telegram …")
    send_telegram(message)

    log.info("━━━  Done  ━━━")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-launchd", action="store_true",
                        help="Install launchd job for automatic 08:00 daily runs")
    args = parser.parse_args()

    if args.install_launchd:
        install_launchd()
    else:
        main()
