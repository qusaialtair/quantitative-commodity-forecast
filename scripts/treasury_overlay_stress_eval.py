#!/usr/bin/env python3
"""
Treasury Overlay Stress Evaluator — Phase XXV-b
================================================
Wraps the multi_asset_stress_backtester output by applying the
treasury_hedge_overlay's rule matrix to each of the 8 historical crisis
windows.  Tests the hypothesis: does adding the recommended TLT/IEF sleeve
to the combined book actually rescue 2022 inflation and lift average
Sharpe materially?

Method
------
For each window:
  1. Classify it into (regime_quadrant, crisis_tier) by domain knowledge.
  2. Look up the rule-matrix recommendation (instrument, allocation %).
  3. Download actual TLT/IEF daily prices over the window via yfinance.
  4. Compute treasury annualised return + vol from real data.
  5. Blend with the existing combined-book stats (annualised, Sharpe, vol)
     using the Markowitz portfolio formula assuming zero correlation
     (conservative — in deflation regimes the actual correlation is
     negative and would amplify the benefit).
  6. Compare verdict before vs after.

The result is honest: if 2022 IEF still loses 15% the rule matrix gets
flagged for revision.  No cherry-picking, no hindsight.

Output
------
data/treasury_overlay_stress_eval.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "multi_asset_stress_backtest.json"
OUTPUT_FILE = DATA_DIR / "treasury_overlay_stress_eval.json"

SEP = "━" * 78

# Window → (regime_quadrant, crisis_tier).
# Mapped by hand from the named regime descriptions in
# multi_asset_stress_backtest.json. These are defensible labels, not
# back-fits — see source descriptions for justification.
WINDOW_CLASSIFICATION: dict[str, tuple[str, str]] = {
    "2008 GFC":              ("DEFLATION",   "CRISIS"),   # Sept-Mar deflationary crash
    "2011 Euro crisis":      ("DEFLATION",   "STRESS"),   # safe-haven gold spike
    "2013 taper tantrum":    ("REFLATION",   "STRESS"),   # bond rates rising, gold sell-off
    "2015 China rout":       ("DEFLATION",   "STRESS"),   # PBoC + commodity collapse
    "2018 vol-mageddon":     ("GOLDILOCKS",  "STRESS"),   # Feb VIX spike, Q4 risk-off
    "2020 COVID":            ("DEFLATION",   "CRISIS"),   # pandemic crash + ZIRP
    "2022 inflation rout":   ("STAGFLATION", "STRESS"),   # the headline 60/40 disaster
    "2024-26 (tuning window)": ("GOLDILOCKS","NORMAL"),   # control — recent benign regime
}


def _fetch_window_returns(ticker: str, start: str, end: str) -> tuple[float, float, float, int]:
    """Return (cum_return_pct, ann_return_pct, ann_vol_pct, n_days) for ticker over window."""
    import yfinance as yf
    import numpy as np
    df = yf.download(
        ticker, start=start, end=end,
        progress=False, auto_adjust=True, threads=False,
    )
    if df is None or df.empty:
        return (0.0, 0.0, 0.0, 0)
    close_col = "Close" if "Close" in df.columns else df.columns[0]
    close = df[close_col].dropna().to_numpy().flatten()
    if len(close) < 2:
        return (0.0, 0.0, 0.0, 0)
    rets = (close[1:] / close[:-1]) - 1.0
    n = len(rets)
    cum = (close[-1] / close[0]) - 1.0
    ann = (1 + cum) ** (252 / n) - 1.0 if n > 0 else 0.0
    vol = float(np.std(rets, ddof=1)) * math.sqrt(252) if n > 1 else 0.0
    return (cum * 100.0, ann * 100.0, vol * 100.0, n)


def _blend(
    base_ann_pct: float, base_sharpe: float,
    treas_ann_pct: float, treas_vol_pct: float,
    hedge_pct: float,
    corr: float = 0.0,
) -> tuple[float, float]:
    """Markowitz two-asset blend. Returns (new_ann_pct, new_sharpe).

    Conservative assumption: zero correlation between book and treasuries.
    In deflation regimes the actual correlation is negative — so reality is
    typically better than this estimate.
    """
    h = hedge_pct / 100.0
    new_ann = (1 - h) * base_ann_pct + h * treas_ann_pct

    # Derive base vol from base Sharpe (assuming rf~=0)
    if abs(base_sharpe) < 1e-6:
        return (new_ann, 0.0)
    base_vol = abs(base_ann_pct / base_sharpe)

    new_var = (
        (1 - h) ** 2 * base_vol ** 2
        + h ** 2 * treas_vol_pct ** 2
        + 2 * h * (1 - h) * corr * base_vol * treas_vol_pct
    )
    new_vol = math.sqrt(max(new_var, 1e-9))
    new_sharpe = new_ann / new_vol if new_vol > 0 else 0.0
    return (new_ann, new_sharpe)


def _verdict(sharpe: float, ann_pct: float) -> str:
    if sharpe >= 1.0 and ann_pct >= 5:
        return "STRONG"
    if sharpe >= 0.4 and ann_pct >= 0:
        return "PASS"
    if sharpe >= 0 or ann_pct > -3:
        return "DEGRADED"
    return "FAIL"


def run() -> dict:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found — run multi_asset_stress_backtester.py first"
        )
    src = json.loads(INPUT_FILE.read_text())

    from scripts.treasury_hedge_overlay import HEDGE_RULES

    rows: list[dict] = []
    sharpe_before_sum = 0.0
    sharpe_after_sum  = 0.0
    n_valid = 0
    n_rescued = 0   # transitioned FAIL/DEGRADED → PASS/STRONG
    n_regressed = 0 # transitioned the other way

    for w in src.get("windows", []):
        label = w.get("label", "?")
        combined = w.get("combined") or {}
        c_sharpe = combined.get("sharpe")
        c_ann = combined.get("annualised_pct")
        if c_sharpe is None or c_ann is None:
            continue

        cls = WINDOW_CLASSIFICATION.get(label, ("UNKNOWN", "NORMAL"))
        quadrant, tier = cls
        rule_key = (quadrant, tier)
        if rule_key not in HEDGE_RULES:
            instrument, pct, reason = (None, 0.0, "no rule")
        else:
            instrument, pct, reason = HEDGE_RULES[rule_key]

        # Fetch actual treasury returns over the window
        treas_cum = treas_ann = treas_vol = 0.0
        n_days = 0
        if instrument and pct > 0:
            treas_cum, treas_ann, treas_vol, n_days = _fetch_window_returns(
                instrument, w["start"], w["end"]
            )

        new_ann, new_sharpe = _blend(c_ann, c_sharpe, treas_ann, treas_vol, pct)
        v_before = _verdict(c_sharpe, c_ann)
        v_after  = _verdict(new_sharpe, new_ann)

        rank = {"FAIL": 0, "DEGRADED": 1, "PASS": 2, "STRONG": 3}
        if rank[v_after] > rank[v_before]:
            n_rescued += 1
        elif rank[v_after] < rank[v_before]:
            n_regressed += 1

        sharpe_before_sum += c_sharpe
        sharpe_after_sum  += new_sharpe
        n_valid += 1

        rows.append({
            "label":            label,
            "start":            w["start"],
            "end":              w["end"],
            "regime_quadrant":  quadrant,
            "crisis_tier":      tier,
            "instrument":       instrument,
            "allocation_pct":   pct,
            "rule_reason":      reason,
            "treas_n_days":     n_days,
            "treas_cum_pct":    round(treas_cum, 2),
            "treas_ann_pct":    round(treas_ann, 2),
            "treas_ann_vol_pct":round(treas_vol, 2),
            "before": {
                "sharpe":   round(c_sharpe, 3),
                "ann_pct":  round(c_ann, 2),
                "verdict":  v_before,
            },
            "after": {
                "sharpe":   round(new_sharpe, 3),
                "ann_pct":  round(new_ann, 2),
                "verdict":  v_after,
            },
            "delta_sharpe": round(new_sharpe - c_sharpe, 3),
            "delta_ann_pp": round(new_ann - c_ann, 2),
        })

    avg_before = sharpe_before_sum / n_valid if n_valid else 0.0
    avg_after  = sharpe_after_sum  / n_valid if n_valid else 0.0

    if n_rescued >= 2 and avg_after >= avg_before + 0.1:
        agg_verdict = "OVERLAY_BENEFICIAL"
        agg_note = (
            f"{n_rescued} window(s) rescued, avg Sharpe lifted "
            f"{avg_before:.2f}→{avg_after:.2f}. Worth activating after Sharia review."
        )
    elif n_regressed > n_rescued or avg_after < avg_before - 0.05:
        agg_verdict = "OVERLAY_NEGATIVE"
        agg_note = (
            f"{n_regressed} regressed vs {n_rescued} rescued, "
            f"avg Sharpe {avg_before:.2f}→{avg_after:.2f}. "
            "Rule matrix needs revision before activation."
        )
    else:
        agg_verdict = "OVERLAY_MIXED"
        agg_note = (
            f"{n_rescued} rescued, {n_regressed} regressed, "
            f"avg Sharpe {avg_before:.2f}→{avg_after:.2f}. "
            "Marginal benefit — keep SIGNAL_ONLY until specific regimes are isolated."
        )

    result = {
        "schema_version": "1.0",
        "engine":          "treasury_overlay_stress_eval",
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":          "data/multi_asset_stress_backtest.json",
        "n_windows":       n_valid,
        "avg_sharpe_before": round(avg_before, 3),
        "avg_sharpe_after":  round(avg_after,  3),
        "delta_avg_sharpe":  round(avg_after - avg_before, 3),
        "n_rescued":       n_rescued,
        "n_regressed":     n_regressed,
        "verdict":         agg_verdict,
        "note":            agg_note,
        "rows":            rows,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  TREASURY OVERLAY STRESS EVALUATION  (Phase XXV-b)\n{SEP}")
    fmt = "  {label:<24s} {regime:<11s} {tier:<8s} {ins:>4s} {pct:>5}%  " \
          "{bSh:>+6.2f}→{aSh:>+6.2f}  {bAn:>+6.1f}%→{aAn:>+6.1f}%  {v}"
    print(fmt.format(
        label="Window", regime="Regime", tier="Tier", ins="Ins",
        pct="Pct", bSh=0, aSh=0, bAn=0, aAn=0, v="Verdict Δ").replace(
        "+0.00", "before", 1).replace("+0.00", "after", 1).replace(
        "+0.0", "bef",1).replace("+0.0","aft",1))
    print(f"  {'-'*24} {'-'*11} {'-'*8} {'-'*4} {'-'*6}  {'-'*15}  {'-'*15}  {'-'*20}")
    for row in r["rows"]:
        ins = row["instrument"] or "—"
        v_str = f"{row['before']['verdict']} → {row['after']['verdict']}"
        print(fmt.format(
            label=row["label"],
            regime=row["regime_quadrant"],
            tier=row["crisis_tier"],
            ins=ins,
            pct=int(row["allocation_pct"]),
            bSh=row["before"]["sharpe"],
            aSh=row["after"]["sharpe"],
            bAn=row["before"]["ann_pct"],
            aAn=row["after"]["ann_pct"],
            v=v_str,
        ))
    print(SEP)
    print(f"  AGG Sharpe before:  {r['avg_sharpe_before']:+.3f}")
    print(f"  AGG Sharpe after:   {r['avg_sharpe_after']:+.3f}  (Δ {r['delta_avg_sharpe']:+.3f})")
    print(f"  Rescued / Regressed: {r['n_rescued']} / {r['n_regressed']}")
    print(f"  Verdict:            {r['verdict']}")
    print(f"  Note:               {r['note']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    res = run()
    _print_report(res)
