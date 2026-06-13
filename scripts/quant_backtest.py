#!/usr/bin/env python3
"""
scripts/quant_backtest.py
=========================
Institutional-grade deterministic backtest for GC=F (Gold Futures).
Reflects the OUNCE ACCUMULATION mandate: default state is LONG (holding metal);
the strategy adds value ONLY by identifying regime-level corrections and
re-entering at a lower cost-per-oz.

Portfolio model:
  Starting capital : $100,000 — deployed into gold on first valid bar
  Default state    : LONG (100% metal)
  STRATEGIC_EXIT   : sell all → 100% fiat  (rare, high-conviction only)
  RE_ENTER         : deploy all fiat → 100% metal
  Fees             : 15 bps per transaction (physical gold bid/ask spread)
  Slippage         : 5 bps per side

Signal logic:
  STRATEGIC_EXIT fires when BOTH:
    [1] HMM regime == BEARISH  (structural deterioration)
    [2] real_yield_10y > 2.5%  OR  cot_gold_z > +2.0  (macro confirmation)

  RE_ENTER fires when ALL:
    [1] Currently in FIAT (after a STRATEGIC_EXIT)
    [2] HMM regime != BEARISH  (regime stabilised)
    [3] SMA20 > SMA50  (momentum confirmed)
    [4] cot_gold_z < +1.0  (not re-entering into a crowded long)

  Over-trading is naturally penalised by the 15 bps fee — each unnecessary
  cycle costs oz, not dollars.

Primary metric — Ounce Efficiency:
  oz_final_strategy / oz_final_bh
  > 1.0  → the strategy accumulated MORE ounces than buy-and-hold
  < 1.0  → the strategy destroyed oz through bad timing or costs

Usage:
  python3 scripts/quant_backtest.py                        # 8-year backtest
  python3 scripts/quant_backtest.py --start 2020-01-01
  python3 scripts/quant_backtest.py --no-tearsheet         # terminal metrics only
  python3 scripts/quant_backtest.py --oos                  # 2023-present only
"""

from __future__ import annotations
import argparse, sys, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────────

TICKER           = "GC=F"
INIT_CASH        = 100_000.0
FEES_BPS         = 15.0        # physical gold bid/ask spread
SLIPPAGE_BPS     = 5.0         # per side
DEFAULT_YEARS    = 8
OOS_START        = "2023-01-01"

REAL_YIELD_MAX   = 2.5         # FRED DFII10 threshold for exit confirmation
COT_CROWD_EXIT   = 2.0         # COT z-score — crowded long, exit confirmation
COT_CROWD_ENTER  = 1.0         # COT z-score ceiling for re-entry

OUTPUT_DIR       = ROOT / "data"
REPORT_PATH      = OUTPUT_DIR / "backtest_report.html"
TRADES_PATH      = OUTPUT_DIR / "backtest_trades.csv"
EQUITY_PATH      = OUTPUT_DIR / "backtest_equity.csv"
ALT_DATA_CSV     = OUTPUT_DIR / "alt_data.csv"


# ── Data loading ─────────────────────────────────────────────────────────────────

def _load_prices(start: str, end: str) -> pd.Series:
    import yfinance as yf
    print(f"  Downloading {TICKER} ({start} → {end}) …")
    raw = yf.download(TICKER, start=start, end=end,
                      interval="1d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cl = raw["Close"].dropna()
    cl.index = pd.to_datetime(cl.index).tz_localize(None).normalize()
    print(f"  {len(cl)} trading days loaded.")
    return cl


def _load_alt_data(start: str, end: str) -> pd.DataFrame:
    if not ALT_DATA_CSV.exists():
        print("  WARNING: alt_data.csv not found — real yield + COT signals disabled.")
        return pd.DataFrame()
    alt = pd.read_csv(ALT_DATA_CSV, index_col=0, parse_dates=True)
    alt.index = pd.to_datetime(alt.index).tz_localize(None).normalize()
    return alt.loc[start:end]


# ── HMM regime labels ─────────────────────────────────────────────────────────────

def _compute_regime_labels(close: pd.Series) -> pd.Series:
    """
    Fit a 3-state Gaussian HMM on the full price history.
    Features are zero-mean / unit-variance standardised for stable EM convergence.
    Returns a daily Series: 'BULLISH' | 'RANGING' | 'BEARISH'.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  WARNING: hmmlearn not installed — regime signal disabled (pip install hmmlearn).")
        return pd.Series("RANGING", index=close.index)

    log_ret  = np.log(close / close.shift(1)).dropna()
    roll_vol = log_ret.rolling(21).std().reindex(log_ret.index).bfill().ffill()
    roll_ret = log_ret.rolling(21).sum().reindex(log_ret.index).bfill().ffill()

    raw = np.column_stack([log_ret.values, roll_vol.values, roll_ret.values])
    means_ = raw.mean(axis=0)
    stds_  = raw.std(axis=0)
    stds_[stds_ == 0] = 1.0
    X = (raw - means_) / stds_

    model = GaussianHMM(
        n_components=3, covariance_type="full",
        n_iter=1000, random_state=42, tol=1e-3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    states = model.predict(X)

    means      = [X[states == i, 0].mean() for i in range(3)]
    sorted_idx = np.argsort(means)
    label_map  = {
        sorted_idx[0]: "BEARISH",
        sorted_idx[1]: "RANGING",
        sorted_idx[2]: "BULLISH",
    }
    labels = pd.Series([label_map[s] for s in states], index=log_ret.index)
    return labels.reindex(close.index).fillna("RANGING")


# ── Signal construction ───────────────────────────────────────────────────────────

def _build_signals(
    close:  pd.Series,
    alt:    pd.DataFrame,
    regime: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (initial_entry, strategic_exit, re_entry) boolean Series.

    initial_entry : True on the first valid bar — start long immediately.
    strategic_exit: True when HMM=BEARISH AND macro confirms (yield or COT).
    re_entry      : True when regime stabilises AND momentum AND COT clear.
    """
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    # Alt data (default to permissive if missing)
    if not alt.empty and "real_yield_10y" in alt.columns:
        ry = alt["real_yield_10y"].reindex(close.index).ffill().bfill()
    else:
        ry = pd.Series(0.0, index=close.index)

    if not alt.empty and "cot_gold_mm_net_zscore" in alt.columns:
        cot_z = alt["cot_gold_mm_net_zscore"].reindex(close.index).ffill().bfill()
    else:
        cot_z = pd.Series(0.0, index=close.index)

    # HMM booleans
    is_bearish  = (regime == "BEARISH")
    not_bearish = ~is_bearish

    # STRATEGIC_EXIT: structural + macro confirmation (both conditions required)
    macro_confirm = (ry > REAL_YIELD_MAX) | (cot_z > COT_CROWD_EXIT)
    strategic_exit = (is_bearish & macro_confirm).fillna(False)

    # RE_ENTER: regime stable + momentum + positioning clear
    re_entry = (
        not_bearish &
        (sma20 > sma50) &
        (cot_z < COT_CROWD_ENTER)
    ).fillna(False)

    # Mask warmup (need 50 bars for SMA50)
    warmup = 50
    strategic_exit.iloc[:warmup] = False
    re_entry.iloc[:warmup]       = False

    # Initial entry: first bar after warmup where regime != BEARISH
    initial_entry = pd.Series(False, index=close.index)
    valid_start = not_bearish.iloc[warmup:].idxmax()
    if valid_start:
        initial_entry.loc[valid_start] = True

    return initial_entry, strategic_exit, re_entry


# ── Vectorbt simulation ───────────────────────────────────────────────────────────

def _run_backtest(
    close:         pd.Series,
    initial_entry: pd.Series,
    exits:         pd.Series,
    re_entries:    pd.Series,
) -> object:
    try:
        import vectorbt as vbt
    except ImportError:
        print("ERROR: vectorbt not installed.  pip install vectorbt")
        sys.exit(1)

    # Combine initial entry + re-entry into a single entries series
    entries = (initial_entry | re_entries)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        direction="longonly",
        init_cash=INIT_CASH,
        fees=FEES_BPS / 10_000,
        slippage=SLIPPAGE_BPS / 10_000,
        freq="1D",
        size=np.inf,       # invest all available cash per signal
    )
    return pf


# ── Strategic cycle analysis ──────────────────────────────────────────────────────

def _analyse_cycles(pf, close: pd.Series) -> list[dict]:
    """
    Extract each STRATEGIC_EXIT → RE_ENTER cycle from vectorbt trade records.
    Returns list of dicts with exit_price, reenter_price, oz_delta, etc.
    """
    cycles = []
    try:
        trades = pf.trades.records_readable
        if trades.empty:
            return cycles
        for _, t in trades.iterrows():
            exit_p    = float(t.get("Avg Exit Price", 0) or 0)
            enter_p   = float(t.get("Avg Entry Price", 0) or 0)
            oz_before = INIT_CASH / enter_p if enter_p > 0 else 0
            oz_after  = float(t.get("PnL", 0) or 0) / exit_p + oz_before if exit_p > 0 else oz_before
            discount  = (exit_p - enter_p) / enter_p * 100 if enter_p > 0 else 0
            cycles.append({
                "entry_date":  str(t.get("Entry Index", "")),
                "exit_date":   str(t.get("Exit Index", "")),
                "entry_price": round(enter_p, 2),
                "exit_price":  round(exit_p, 2),
                "discount_pct": round(discount, 2),
                "pnl_usd":     round(float(t.get("PnL", 0) or 0), 2),
            })
    except Exception:
        pass
    return cycles


# ── Metrics helpers ───────────────────────────────────────────────────────────────

def _safe(fn, default=float("nan")):
    try:
        v = fn()
        return float(v) if v is not None else default
    except Exception:
        return default


def _compute_oz_efficiency(pf, close: pd.Series) -> tuple[float, float, float]:
    """
    Returns (strategy_final_oz, bh_final_oz, oz_efficiency_ratio).
    Both are measured as final_portfolio_value / final_spot_price.
    """
    final_spot = float(close.iloc[-1])
    first_spot = float(close.iloc[50]) if len(close) > 50 else float(close.iloc[0])

    strategy_value = _safe(lambda: pf.value().iloc[-1])
    strategy_oz    = strategy_value / final_spot if final_spot > 0 else 0.0

    bh_oz          = INIT_CASH / first_spot if first_spot > 0 else 0.0
    bh_final_value = bh_oz * final_spot
    bh_oz_end      = bh_final_value / final_spot if final_spot > 0 else 0.0

    efficiency = strategy_oz / bh_oz_end if bh_oz_end > 0 else float("nan")
    return strategy_oz, bh_oz_end, efficiency


def _print_metrics(
    pf,
    close:      pd.Series,
    start:      str,
    end:        str,
    exits:      pd.Series,
    re_entries: pd.Series,
    cycles:     list[dict],
) -> None:
    s_ret     = _safe(lambda: pf.total_return() * 100)
    s_cagr    = _safe(lambda: pf.annualized_return() * 100)
    s_sharpe  = _safe(lambda: pf.sharpe_ratio())
    s_sort    = _safe(lambda: pf.sortino_ratio())
    s_mdd     = _safe(lambda: pf.max_drawdown() * 100)
    s_calmar  = _safe(lambda: pf.calmar_ratio())
    s_ntrades = _safe(lambda: pf.trades.count())

    try:
        in_mkt = float(pf.stats()["Long Exposure [%]"])
    except Exception:
        in_mkt = float("nan")

    n_exits    = int(exits.sum())
    n_reentries = int(re_entries.sum())

    # Buy-and-hold baseline
    n_years  = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    bh_ret   = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    bh_cagr  = float(((close.iloc[-1] / close.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100)
    bh_rets  = close.pct_change().dropna()
    bh_mdd   = float((close / close.cummax() - 1).min() * 100)
    bh_sharpe = (float(bh_rets.mean() / bh_rets.std() * 252 ** 0.5)
                 if bh_rets.std() > 0 else float("nan"))

    strat_oz, bh_oz, oz_eff = _compute_oz_efficiency(pf, close)

    # Cycle quality: % of cycles where exit > re-entry (strategy worked)
    good_cycles = [c for c in cycles if c["discount_pct"] < 0]  # sold high, bought low
    cycle_quality = len(good_cycles) / len(cycles) * 100 if cycles else float("nan")

    W = 66
    print("\n" + "━" * W)
    print("  GOLD TRADING AI — Ounce Accumulation Backtest")
    print(f"  {TICKER}  |  {start} → {end}  |  ${INIT_CASH:,.0f} initial capital")
    print(f"  Fees: {FEES_BPS}bps (physical)  |  Slippage: {SLIPPAGE_BPS}bps  |  LONG-only")
    print("━" * W)

    def _row(lbl, sv, bv, fmt="+.1f", sfx="%"):
        s = f"{sv:{fmt}}{sfx}" if not np.isnan(sv) else "N/A"
        b = f"{bv:{fmt}}{sfx}" if not np.isnan(bv) else "N/A"
        print(f"  {lbl:<32}  {s:>9}  {b:>9}")

    def _rowp(lbl, sv, bv, fmt=".2f", sfx=""):
        s = f"{sv:{fmt}}{sfx}" if not np.isnan(sv) else "N/A"
        b = f"{bv:{fmt}}{sfx}" if not np.isnan(bv) else "N/A"
        print(f"  {lbl:<32}  {s:>9}  {b:>9}")

    print(f"  {'Metric':<32}  {'Strategy':>9}  {'Buy&Hold':>9}")
    print("  " + "─" * (W - 2))
    _row("Total Return",          s_ret,    bh_ret)
    _row("CAGR",                  s_cagr,   bh_cagr)
    _rowp("Sharpe Ratio",         s_sharpe, bh_sharpe)
    _rowp("Sortino Ratio",        s_sort,   float("nan"))
    _row("Max Drawdown",          s_mdd,    bh_mdd)
    _rowp("Calmar Ratio",         s_calmar, float("nan"))
    _row("Time in Market",        in_mkt,   100.0, fmt=".1f")

    print("\n  ── Ounce Accumulation ─────────────────────────────────────")
    _rowp("Final oz (strategy)",  strat_oz, float("nan"), fmt=".4f", sfx=" oz")
    _rowp("Final oz (buy&hold)",  bh_oz,    float("nan"), fmt=".4f", sfx=" oz")
    eff_str = f"{oz_eff:.4f}x" if not np.isnan(oz_eff) else "N/A"
    print(f"  {'Oz efficiency ratio':<32}  {eff_str:>9}  {'1.0000x':>9}")

    print("\n  ── Strategic Cycle Analysis ────────────────────────────────")
    print(f"  Strategic exits triggered    {n_exits:>9}")
    print(f"  Re-entries triggered         {n_reentries:>9}")
    print(f"  Completed cycles (trades)    {int(s_ntrades):>9}" if not np.isnan(s_ntrades) else "  Completed cycles            N/A")
    print(f"  Exit-quality (sold>rebought) {cycle_quality:>8.1f}%" if not np.isnan(cycle_quality) else "  Exit quality                N/A")

    if cycles:
        print("\n  ── Individual Cycles ───────────────────────────────────────")
        print(f"  {'Entry':>12}  {'Exit':>12}  {'EntryP':>8}  {'ExitP':>8}  {'Δ%':>7}  {'PnL':>10}")
        for c in cycles[-10:]:   # last 10
            sign = "+" if c["pnl_usd"] >= 0 else ""
            disc = f"{c['discount_pct']:+.1f}%"
            print(f"  {c['entry_date']:>12}  {c['exit_date']:>12}  "
                  f"${c['entry_price']:>7,.0f}  ${c['exit_price']:>7,.0f}  "
                  f"{disc:>7}  {sign}${c['pnl_usd']:>8,.0f}")

    print("━" * W + "\n")


# ── Output writers ────────────────────────────────────────────────────────────────

def _save_outputs(pf) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        pf.trades.records_readable.to_csv(TRADES_PATH)
        print(f"  Trades   → {TRADES_PATH.name}")
    except Exception as exc:
        print(f"  WARNING: trades CSV failed: {exc}")
    try:
        pf.value().to_csv(EQUITY_PATH, header=["portfolio_value"])
        print(f"  Equity   → {EQUITY_PATH.name}")
    except Exception as exc:
        print(f"  WARNING: equity CSV failed: {exc}")


def _generate_tearsheet(pf, close: pd.Series) -> None:
    try:
        import quantstats as qs
    except ImportError:
        print("  WARNING: quantstats not installed — skipping  (pip install quantstats)")
        return
    strat = pf.returns().rename("Gold Accumulation Strategy")
    bench = close.pct_change().dropna().rename("GC=F Buy & Hold")
    idx   = strat.index.intersection(bench.index)
    try:
        qs.reports.html(
            strat.loc[idx],
            benchmark=bench.loc[idx],
            output=str(REPORT_PATH),
            title="Gold Trading AI — Ounce Accumulation Backtest",
            download_filename=str(REPORT_PATH),
        )
        print(f"  Tearsheet → {REPORT_PATH.name}")
    except Exception as exc:
        print(f"  WARNING: quantstats error: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str, tearsheet: bool = True) -> None:
    print("\n" + "━" * 66)
    print("  GOLD TRADING AI — Ounce Accumulation Backtest Engine")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("━" * 66 + "\n")

    close = _load_prices(start, end)
    alt   = _load_alt_data(start, end)

    print("  Fitting HMM regime model …")
    regime    = _compute_regime_labels(close)
    bull_pct  = (regime == "BULLISH").mean() * 100
    rang_pct  = (regime == "RANGING").mean() * 100
    bear_pct  = (regime == "BEARISH").mean() * 100
    print(f"  Regime → BULLISH {bull_pct:.1f}%  RANGING {rang_pct:.1f}%  BEARISH {bear_pct:.1f}%")

    print("  Building signal arrays …")
    initial_entry, exits, re_entries = _build_signals(close, alt, regime)
    print(f"  STRATEGIC_EXIT signals: {int(exits.sum())}  |  RE_ENTER signals: {int(re_entries.sum())}")

    print("  Running vectorbt simulation …")
    pf = _run_backtest(close, initial_entry, exits, re_entries)

    cycles = _analyse_cycles(pf, close)
    _print_metrics(pf, close, start, end, exits, re_entries, cycles)

    _save_outputs(pf)
    if tearsheet:
        print("  Generating quantstats tearsheet …")
        _generate_tearsheet(pf, close)

    print("  Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Gold Accumulation Backtest — STRATEGIC_EXIT + RE_ENTER signals")
    parser.add_argument("--start", default=None,
        help="Start date YYYY-MM-DD (default: 8 years ago)")
    parser.add_argument("--end", default=None,
        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-tearsheet", action="store_true",
        help="Skip quantstats HTML tearsheet")
    parser.add_argument("--oos", action="store_true",
        help=f"Out-of-sample window only (start={OOS_START})")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.oos:
        start = OOS_START
    elif args.start:
        start = args.start
    else:
        start = (datetime.now(timezone.utc) - timedelta(days=DEFAULT_YEARS * 365)).strftime("%Y-%m-%d")
    end = args.end or today

    run_backtest(start=start, end=end, tearsheet=not args.no_tearsheet)


if __name__ == "__main__":
    main()
