#!/usr/bin/env python3
"""
Portfolio Stress Testing Engine
=================================
Simulates portfolio performance through historical crisis scenarios
to quantify tail risk exposure and recovery characteristics.

Scenarios:
  1. 2008 GFC (Sep-Nov 2008): Gold initially fell with everything
  2. 2011 Gold Crash (Sep 2011-Jun 2012): Gold lost 20% after record high
  3. 2013 Taper Tantrum (Apr-Jun 2013): Gold crashed 25% on Fed taper signal
  4. 2020 COVID Crash (Feb-Apr 2020): Liquidity crisis hit gold briefly
  5. 2022 Rate Shock (Mar-Oct 2022): Aggressive Fed hikes crushed gold
  6. Custom tail scenario: Parametric worst-case from return distribution

Usage:
    python3 scripts/stress_tester.py
    python3 scripts/stress_tester.py --portfolio 100000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "stress_test_results.json"

LINE_W = 72
SEP = "\u2501" * LINE_W

CRISIS_SCENARIOS = {
    "2008_GFC": {
        "name": "2008 Global Financial Crisis",
        "start": "2008-09-01",
        "end": "2008-11-30",
        "description": "Lehman collapse, global deleveraging",
    },
    "2011_GOLD_CRASH": {
        "name": "2011 Gold Crash",
        "start": "2011-09-01",
        "end": "2012-06-30",
        "description": "Gold fell 20% after hitting $1,920 record",
    },
    "2013_TAPER_TANTRUM": {
        "name": "2013 Taper Tantrum",
        "start": "2013-04-01",
        "end": "2013-06-30",
        "description": "Fed signals taper, gold crashes 25%",
    },
    "2020_COVID": {
        "name": "2020 COVID Liquidity Crisis",
        "start": "2020-02-20",
        "end": "2020-04-15",
        "description": "Margin calls forced gold liquidation",
    },
    "2022_RATE_SHOCK": {
        "name": "2022 Rate Shock",
        "start": "2022-03-01",
        "end": "2022-10-31",
        "description": "Aggressive Fed hikes, DXY surge to 114",
    },
}


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    period: str
    description: str
    gold_return_pct: float
    portfolio_return_pct: float
    max_drawdown_pct: float
    drawdown_days: int
    recovery_days: int
    worst_daily_loss_pct: float
    best_daily_gain_pct: float
    vol_annualized_pct: float
    sharpe_in_crisis: float
    correlation_breakdown: dict


def _fetch_crisis_data(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch gold and multi-asset data for a crisis period."""
    if yf is None:
        raise ImportError("yfinance required")

    tickers = {
        "GC=F": "Gold",
        "SPY": "S&P 500",
        "TLT": "Treasuries",
        "DX-Y.NYB": "Dollar",
        "^VIX": "VIX",
    }

    frames = {}
    for ticker, name in tickers.items():
        try:
            # Add buffer before and after for recovery calculation
            buf_start = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            buf_end = (pd.Timestamp(end) + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
            raw = yf.download(ticker, start=buf_start, end=buf_end,
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if not raw.empty:
                frames[ticker] = raw
        except Exception:
            pass
    return frames


def _compute_scenario(
    scenario_id: str,
    config: dict,
    portfolio_value: float,
    gold_weight: float,
) -> ScenarioResult:
    """Run a single historical scenario."""
    start = config["start"]
    end = config["end"]

    frames = _fetch_crisis_data(start, end)

    if "GC=F" not in frames or frames["GC=F"].empty:
        return ScenarioResult(
            scenario_id=scenario_id, name=config["name"],
            period=f"{start} to {end}", description=config["description"],
            gold_return_pct=0, portfolio_return_pct=0, max_drawdown_pct=0,
            drawdown_days=0, recovery_days=0, worst_daily_loss_pct=0,
            best_daily_gain_pct=0, vol_annualized_pct=0, sharpe_in_crisis=0,
            correlation_breakdown={},
        )

    gold_df = frames["GC=F"]
    # Filter to crisis period
    crisis_mask = (gold_df.index >= start) & (gold_df.index <= end)
    crisis_gold = gold_df.loc[crisis_mask, "Close"].dropna()

    if len(crisis_gold) < 2:
        return ScenarioResult(
            scenario_id=scenario_id, name=config["name"],
            period=f"{start} to {end}", description=config["description"],
            gold_return_pct=0, portfolio_return_pct=0, max_drawdown_pct=0,
            drawdown_days=0, recovery_days=0, worst_daily_loss_pct=0,
            best_daily_gain_pct=0, vol_annualized_pct=0, sharpe_in_crisis=0,
            correlation_breakdown={},
        )

    gold_returns = crisis_gold.pct_change().dropna()
    gold_total_ret = (crisis_gold.iloc[-1] / crisis_gold.iloc[0] - 1) * 100

    # Portfolio return (gold_weight in gold, rest in cash)
    portfolio_ret = gold_total_ret * gold_weight

    # Drawdown calculation
    cummax = crisis_gold.cummax()
    drawdown = (crisis_gold - cummax) / cummax * 100
    max_dd = float(drawdown.min())

    # Days in drawdown
    dd_days = int((drawdown < -1.0).sum())

    # Recovery: how many days after crisis end to recover peak
    post_crisis = gold_df.loc[gold_df.index > end, "Close"].dropna()
    pre_crisis_peak = float(crisis_gold.max())
    recovery_days = 0
    if not post_crisis.empty:
        recovered = post_crisis[post_crisis >= pre_crisis_peak]
        if not recovered.empty:
            recovery_date = recovered.index[0]
            crisis_end_date = pd.Timestamp(end)
            recovery_days = (recovery_date - crisis_end_date).days

    # Daily extremes
    worst_day = float(gold_returns.min()) * 100 if len(gold_returns) > 0 else 0
    best_day = float(gold_returns.max()) * 100 if len(gold_returns) > 0 else 0

    # Volatility
    vol_ann = float(gold_returns.std() * np.sqrt(252) * 100) if len(gold_returns) > 0 else 0

    # Sharpe during crisis
    mean_daily = float(gold_returns.mean()) if len(gold_returns) > 0 else 0
    std_daily = float(gold_returns.std()) if len(gold_returns) > 0 else 1
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    # Correlation breakdown: gold vs other assets during crisis
    corr_breakdown = {}
    for ticker, name in [("SPY", "S&P500"), ("TLT", "Treasuries"), ("DX-Y.NYB", "Dollar")]:
        if ticker in frames:
            other = frames[ticker].loc[crisis_mask, "Close"].dropna()
            if len(other) > 5:
                # Align dates
                aligned = pd.DataFrame({"gold": crisis_gold, "other": other}).dropna()
                if len(aligned) > 5:
                    g_ret = aligned["gold"].pct_change().dropna()
                    o_ret = aligned["other"].pct_change().dropna()
                    if len(g_ret) > 3 and len(o_ret) > 3:
                        corr = float(g_ret.corr(o_ret))
                        corr_breakdown[name] = round(corr, 3)

    return ScenarioResult(
        scenario_id=scenario_id,
        name=config["name"],
        period=f"{start} to {end}",
        description=config["description"],
        gold_return_pct=round(gold_total_ret, 2),
        portfolio_return_pct=round(portfolio_ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        drawdown_days=dd_days,
        recovery_days=recovery_days,
        worst_daily_loss_pct=round(worst_day, 2),
        best_daily_gain_pct=round(best_day, 2),
        vol_annualized_pct=round(vol_ann, 2),
        sharpe_in_crisis=round(sharpe, 3),
        correlation_breakdown=corr_breakdown,
    )


def _parametric_tail_scenario(portfolio_value: float) -> dict:
    """
    Generate a parametric tail scenario from the actual return distribution.
    Uses the 1st percentile of historical daily returns, sustained for 10 days.
    """
    try:
        raw = yf.download("GC=F", period="10y", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        returns = raw["Close"].pct_change().dropna()

        p1 = float(np.percentile(returns, 1))
        p5 = float(np.percentile(returns, 5))

        # 10-day sustained tail event
        tail_10d = (1 + p1) ** 10 - 1
        # 5-day severe crash
        tail_5d_severe = (1 + p1 * 1.5) ** 5 - 1

        return {
            "name": "Parametric Tail (1st percentile, 10 days)",
            "daily_1pct_return": round(p1 * 100, 3),
            "daily_5pct_return": round(p5 * 100, 3),
            "10d_sustained_loss_pct": round(tail_10d * 100, 2),
            "5d_severe_crash_pct": round(tail_5d_severe * 100, 2),
            "portfolio_10d_loss": round(portfolio_value * abs(tail_10d), 2),
            "portfolio_5d_severe_loss": round(portfolio_value * abs(tail_5d_severe), 2),
        }
    except Exception:
        return {"error": "Could not compute parametric tail"}


def run_stress_test(portfolio_value: float = 100_000.0, gold_weight: float = 1.0) -> dict:
    """Run all stress scenarios and save results."""
    results = []
    for sid, config in CRISIS_SCENARIOS.items():
        print(f"  Running scenario: {config['name']}...")
        sr = _compute_scenario(sid, config, portfolio_value, gold_weight)
        results.append(asdict(sr))

    # Parametric tail
    print(f"  Computing parametric tail scenario...")
    parametric = _parametric_tail_scenario(portfolio_value)

    # Aggregate risk metrics
    gold_returns = [r["gold_return_pct"] for r in results]
    max_dds = [r["max_drawdown_pct"] for r in results]

    aggregate = {
        "avg_crisis_return_pct": round(np.mean(gold_returns), 2),
        "worst_crisis_return_pct": round(min(gold_returns), 2),
        "worst_crisis_scenario": results[np.argmin(gold_returns)]["name"],
        "avg_max_drawdown_pct": round(np.mean(max_dds), 2),
        "worst_max_drawdown_pct": round(min(max_dds), 2),
        "avg_recovery_days": round(np.mean([r["recovery_days"] for r in results if r["recovery_days"] > 0])),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio_value": portfolio_value,
        "gold_weight": gold_weight,
        "scenarios": results,
        "parametric_tail": parametric,
        "aggregate": aggregate,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    _print_report(output)
    return output


def _print_report(output: dict) -> None:
    print(f"\n{SEP}")
    print(f"  PORTFOLIO STRESS TEST")
    print(f"  Portfolio: ${output['portfolio_value']:,.0f}  |  Gold Weight: {output['gold_weight']:.0%}")
    print(SEP)

    # Header
    print(f"  {'Scenario':<28s} {'Return':>8s} {'Max DD':>8s} {'Worst Day':>10s} {'Recov':>7s} {'Vol':>7s}")
    print(f"  {'─' * 68}")

    for s in output["scenarios"]:
        ret_color = "\033[32m" if s["gold_return_pct"] > 0 else "\033[31m"
        print(
            f"  {s['name'][:27]:<28s} "
            f"{ret_color}{s['gold_return_pct']:+7.2f}%\033[0m "
            f"{s['max_drawdown_pct']:+7.2f}% "
            f"{s['worst_daily_loss_pct']:+9.2f}% "
            f"{s['recovery_days']:>5d}d "
            f"{s['vol_annualized_pct']:>6.1f}%"
        )

    print()
    agg = output["aggregate"]
    print(f"  AGGREGATE")
    print(f"  {'─' * 50}")
    print(f"  Avg Crisis Return:     {agg['avg_crisis_return_pct']:+.2f}%")
    print(f"  Worst Crisis:          {agg['worst_crisis_return_pct']:+.2f}%  ({agg['worst_crisis_scenario']})")
    print(f"  Avg Max Drawdown:      {agg['avg_max_drawdown_pct']:.2f}%")
    print(f"  Avg Recovery:          {agg['avg_recovery_days']:.0f} days")

    pt = output.get("parametric_tail", {})
    if "error" not in pt:
        print()
        print(f"  PARAMETRIC TAIL RISK")
        print(f"  {'─' * 50}")
        print(f"  Daily 1st percentile:  {pt.get('daily_1pct_return', 0):+.3f}%")
        print(f"  10-day sustained loss: {pt.get('10d_sustained_loss_pct', 0):+.2f}%"
              f"  (${pt.get('portfolio_10d_loss', 0):,.0f})")
        print(f"  5-day severe crash:    {pt.get('5d_severe_crash_pct', 0):+.2f}%"
              f"  (${pt.get('portfolio_5d_severe_loss', 0):,.0f})")

    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Stress Testing")
    parser.add_argument("--portfolio", type=float, default=100_000.0)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    args = parser.parse_args()
    run_stress_test(portfolio_value=args.portfolio, gold_weight=args.gold_weight)
