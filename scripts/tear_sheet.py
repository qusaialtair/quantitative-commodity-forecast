#!/usr/bin/env python3
"""
Form-PF Lite + Investor Tear Sheet  (Phase X Stages 54-55)
============================================================
Consolidates the full institutional stack into a single markdown report
suitable for monthly investor distribution and a stripped-down "Form PF
lite" regulatory summary.

Sections produced:
  1. Executive snapshot     portfolio, regime, action
  2. Performance            Sharpe, return, vol, drawdown, Brinson α
  3. Treasury hedge sleeve  Phase XXV — hedge mode, instrument, notional,
                            by-strategy P&L attribution (hedge drag vs alpha)
  4. Risk profile           CVaR, EVT tail, DCC stress, drawdown tier
  5. Factor decomposition   Fama-French α / β / R²
  6. Signal diagnostics     alpha attribution, IC/IR, BMA, decision quality
  7. Macro & regime         HMM, macro quadrant, vol surface, term structure
  8. Capacity & costs       TCA, capacity ceiling, SOR
  9. Governance             MRM champion, audit chain, DR backup

Writes:
  data/tear_sheet.md            monthly investor report
  data/form_pf_lite.json        machine-readable regulatory snapshot
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
TEAR_SHEET = DATA_DIR / "tear_sheet.md"
FORM_PF_FILE = DATA_DIR / "form_pf_lite.json"

LINE_W = 62
SEP = "━" * LINE_W


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _fmt_pct(v, places: int = 2, sign: bool = True) -> str:
    if v is None:
        return "n/a"
    fmt = f"{{:+.{places}f}}%" if sign else f"{{:.{places}f}}%"
    try:
        return fmt.format(float(v))
    except Exception:
        return "n/a"


def _fmt_usd(v) -> str:
    if v is None:
        return "n/a"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _section_executive() -> tuple[str, dict]:
    ps = _load("pipeline_state.json")
    pf = ps.get("portfolio", {})
    rg = ps.get("regime", {})
    cm = ps.get("committee", {})
    out = []
    out.append("## 1. Executive Snapshot")
    out.append("")
    out.append(f"- **Date**: {ps.get('run_date', '—')}")
    out.append(f"- **Pipeline status**: {ps.get('pipeline_status', '—')}")
    out.append(f"- **Portfolio value**: {_fmt_usd(pf.get('portfolio_value'))}")
    out.append(f"- **Cash**: {_fmt_usd(pf.get('cash_usd'))}    Gold (oz): {pf.get('gold_oz', 0)}")
    out.append(f"- **Unrealised P&L**: {_fmt_usd(pf.get('unrealised_pnl'))}  ({_fmt_pct(pf.get('unrealised_pct'))})")
    out.append(f"- **HMM regime**: {rg.get('hmm_state', '—')}  (p_bull={rg.get('p_bullish', 0):.2f})")
    out.append(f"- **Today's CIO action**: {cm.get('action_taken', '—')}  (veto={cm.get('veto_active', False)})")
    out.append("")
    snapshot = {
        "run_date":         ps.get("run_date"),
        "pipeline_status":  ps.get("pipeline_status"),
        "portfolio_value":  pf.get("portfolio_value"),
        "cash_usd":         pf.get("cash_usd"),
        "gold_oz":          pf.get("gold_oz"),
        "unrealised_pnl":   pf.get("unrealised_pnl"),
        "regime":           rg.get("hmm_state"),
        "action":           cm.get("action_taken"),
    }
    return "\n".join(out), snapshot


def _section_performance() -> tuple[str, dict]:
    tre = _load("tail_risk_engine.json")
    bri = _load("brinson_attribution.json")
    bt = _load("backtest_results.json")
    perf = tre.get("performance", {})
    out = []
    out.append("## 2. Performance")
    out.append("")
    out.append(f"- **Sharpe (tail-risk engine)**: {perf.get('sharpe_ratio', '—')}")
    out.append(f"- **Sortino**: {perf.get('sortino_ratio', '—')}")
    out.append(f"- **Excess kurtosis**: {perf.get('excess_kurtosis', '—')}")
    out.append(f"- **Brinson excess**: {_fmt_pct(bri.get('excess_return_pct'))}")
    out.append(f"- **Allocation effect**: {_fmt_pct(bri.get('allocation_effect_pct'))}")
    out.append(f"- **Selection effect**: {_fmt_pct(bri.get('selection_effect_pct'))}")
    if bt:
        strat = bt.get("strategy", {})
        out.append(f"- **Backtest Sharpe**: {strat.get('sharpe_ratio', '—')}  win rate: {_fmt_pct(strat.get('win_rate_pct'), sign=False)}  profit factor: {strat.get('profit_factor', '—')}x")
    out.append("")
    snapshot = {
        "sharpe":              perf.get("sharpe_ratio"),
        "sortino":             perf.get("sortino_ratio"),
        "brinson_excess_pct":  bri.get("excess_return_pct"),
        "allocation_effect_pct":bri.get("allocation_effect_pct"),
    }
    return "\n".join(out), snapshot


def _section_hedge_sleeve() -> tuple[str, dict]:
    """§3 — Phase XXV Treasury Hedge Sleeve attribution."""
    mst = _load("multi_strategy_trader.json")
    th  = _load("treasury_hedge.json")
    hs  = mst.get("hedge_state") or {}
    by_s = mst.get("by_strategy") or {}
    hedge_notional = mst.get("hedge_notional_usd", 0.0) or 0.0
    hedge_frac_pct = (mst.get("hedge_fraction") or 0.0) * 100.0
    book_equity    = mst.get("book_equity_usd", 0.0) or 0.0

    mode       = hs.get("mode") or th.get("mode") or "SIGNAL_ONLY"
    instrument = hs.get("instrument")
    sub_tag    = hs.get("sub_tag")
    gate       = hs.get("gate_action") or th.get("gate_action") or "—"

    out = []
    out.append("## 3. Treasury Hedge Sleeve (Phase XXV)")
    out.append("")
    out.append(f"- **Mode**: `{mode}`  |  **Gate**: `{gate}`")

    if instrument:
        instr_label = (
            f"{instrument}  *(sharia fallback — physical-gold proxy)*"
            if sub_tag == "sharia_fallback_gld"
            else f"{instrument}  *(sovereign duration)*"
        )
        out.append(f"- **Active instrument**: {instr_label}")
        out.append(
            f"- **Notional**: {_fmt_usd(hedge_notional)}"
            f"  ({_fmt_pct(hedge_frac_pct, sign=False)} of book equity {_fmt_usd(book_equity)})"
        )
    else:
        out.append("- **Sleeve**: off — no hedge warranted under current regime")

    # Overlay recommendation from treasury_hedge.json
    if th:
        sharia_cleared = th.get("sharia_cleared")
        out.append(
            f"- **Regime**: {th.get('regime_quadrant', '—')} / {th.get('crisis_tier', '—')}"
            f"  (conf {th.get('regime_confidence', '—')})"
        )
        out.append(
            f"- **Rule-matrix rec.**: {th.get('instrument') or '—'}"
            f"  {_fmt_pct(th.get('allocation_pct', 0), sign=False)}"
        )
        out.append(
            f"- **Effective (post-gate)**: {th.get('effective_instrument') or '—'}"
            f"  {_fmt_pct(th.get('effective_allocation_pct', 0), sign=False)}"
            f"  |  Sharia cleared: {'yes' if sharia_cleared else '**NO — GLD fallback active**'}"
        )
    out.append("")

    # By-strategy P&L attribution table
    out.append("### Strategy Attribution")
    out.append("")
    out.append("| Strategy | Realized P&L | Open Notional | Open | Closed |")
    out.append("|---|---:|---:|---:|---:|")

    # Treasury hedge row first so it stands out
    hd = by_s.get("TREASURY_HEDGE") or {}
    out.append(
        f"| **TREASURY_HEDGE**"
        + (f" `{sub_tag}`" if sub_tag else "")
        + f" | {_fmt_usd(hd.get('realized_pl_usd', 0))}"
        f" | {_fmt_usd(hd.get('open_notional_usd', 0))}"
        f" | {hd.get('n_open', 0)} | {hd.get('n_closed', 0)} |"
    )

    # Alpha strategies in alphabetical order
    for key in sorted(k for k in by_s if k != "TREASURY_HEDGE"):
        d = by_s[key]
        out.append(
            f"| {key}"
            f" | {_fmt_usd(d.get('realized_pl_usd', 0))}"
            f" | {_fmt_usd(d.get('open_notional_usd', 0))}"
            f" | {d.get('n_open', 0)} | {d.get('n_closed', 0)} |"
        )

    # Totals row
    total_realized = sum((v.get("realized_pl_usd") or 0.0) for v in by_s.values())
    total_notional = sum((v.get("open_notional_usd") or 0.0) for v in by_s.values())
    alpha_realized = sum(
        (v.get("realized_pl_usd") or 0.0)
        for k, v in by_s.items() if k != "TREASURY_HEDGE"
    )
    hedge_realized = (by_s.get("TREASURY_HEDGE") or {}).get("realized_pl_usd") or 0.0

    out.append(
        f"| **TOTAL**"
        f" | {_fmt_usd(total_realized)}"
        f" | {_fmt_usd(total_notional)}"
        f" | — | — |"
    )
    out.append("")

    if by_s:
        out.append(
            f"> Hedge drag vs alpha: "
            f"TREASURY_HEDGE {_fmt_usd(hedge_realized)} / "
            f"alpha strategies {_fmt_usd(alpha_realized)}"
        )
    out.append("")

    snapshot = {
        "hedge_mode":             mode,
        "hedge_instrument":       instrument,
        "hedge_sub_tag":          sub_tag,
        "hedge_gate_action":      gate,
        "hedge_notional_usd":     round(hedge_notional, 2),
        "hedge_fraction_pct":     round(hedge_frac_pct, 2),
        "hedge_realized_pl_usd":  round(hedge_realized, 2),
        "alpha_realized_pl_usd":  round(alpha_realized, 2),
        "total_realized_pl_usd":  round(total_realized, 2),
        "sharia_cleared":         th.get("sharia_cleared"),
        "by_strategy":            by_s,
    }
    return "\n".join(out), snapshot


def _section_risk() -> tuple[str, dict]:
    tre = _load("tail_risk_engine.json")
    mc = _load("monte_carlo_simulation.json")
    dd = _load("drawdown_controller.json")
    dcc = _load("dcc_garch.json")
    st = _load("stress_test_results.json")
    evt = tre.get("tail_risk", {}).get("methods", {}).get("evt_pot", {})

    out = []
    out.append("## 4. Risk Profile")
    out.append("")
    out.append(f"- **EVT CVaR-99**: {evt.get('cvar_990', '—')}%")
    out.append(f"- **MC CVaR-95 (daily)**: {mc.get('risk', {}).get('cvar_95_pct', '—')}%")
    out.append(f"- **Tail-fatness premium vs Gaussian**: {tre.get('tail_risk', {}).get('tail_fatness_premium_pct', '—')}%")
    out.append(f"- **Drawdown tier**: {dd.get('tier_name', '—')}  sizing × {dd.get('sizing_multiplier', '—')}")
    out.append(f"- **DCC stressed pairs**: {dcc.get('n_stressed', 0)}")
    out.append(f"- **Worst historical stress test**: {_fmt_pct(st.get('aggregate', {}).get('worst_crisis_return_pct'))}  ({st.get('aggregate', {}).get('worst_crisis_scenario', '—')})")
    out.append("")
    snapshot = {
        "evt_cvar_99_pct":  evt.get("cvar_990"),
        "mc_cvar_95_daily": mc.get("risk", {}).get("cvar_95_pct"),
        "drawdown_tier":    dd.get("tier_name"),
        "dcc_stressed":     dcc.get("n_stressed", 0),
    }
    return "\n".join(out), snapshot


def _section_factors() -> tuple[str, dict]:
    ff = _load("fama_french.json")
    out = []
    out.append("## 5. Factor Decomposition (Fama-French-Carhart)")
    out.append("")
    out.append(f"- **α annualised**: {_fmt_pct(ff.get('alpha_annualised_pct'))}  t = {ff.get('alpha_t_stat', '—')}  (significant: {ff.get('alpha_significant', False)})")
    out.append(f"- **R²**: {ff.get('r_squared', '—')}    Information Ratio: {ff.get('information_ratio', '—')}")
    out.append(f"- **Dominant factor**: {ff.get('dominant_factor', '—')}")
    out.append(f"- **Significant factors**: {ff.get('n_significant_factors', 0)} of {len(ff.get('factor_summary', []))}")
    out.append("")
    out.append("| Factor | β | t | Significant |")
    out.append("|---|---|---|---|")
    for f in ff.get("factor_summary", []):
        out.append(f"| {f['factor']} | {f['beta']:+.4f} | {f['t_stat']:+.2f} | {'yes' if f['significant'] else 'no'} |")
    out.append("")
    snapshot = {
        "alpha_annualised_pct": ff.get("alpha_annualised_pct"),
        "alpha_t_stat":         ff.get("alpha_t_stat"),
        "r_squared":            ff.get("r_squared"),
        "dominant_factor":      ff.get("dominant_factor"),
    }
    return "\n".join(out), snapshot


def _section_signals() -> tuple[str, dict]:
    aa = _load("alpha_attribution.json")
    ii = _load("ic_ir_tracker.json")
    bma = _load("bma_weights.json")
    dq = _load("decision_quality.json")
    out = []
    out.append("## 6. Signal Diagnostics")
    out.append("")
    out.append(f"- **Top by Sharpe**: {aa.get('ranked_by_sharpe', ['—'])[0]}")
    out.append(f"- **Top by IR**: {ii.get('ranked_by_ir', ['—'])[0]}")
    out.append(f"- **BMA top**: {bma.get('top_source', '—')}")
    out.append(f"- **Decision-quality best**: {dq.get('best_signal', '—')}  (Brier {dq.get('best_brier', '—')})")
    out.append(f"- **Deployable signals (IR > 0.5)**: {', '.join(ii.get('deployable_signals', [])) or 'none'}")
    out.append(f"- **Diversification ratio**: {aa.get('combined', {}).get('diversification_ratio', '—')}x")
    out.append("")
    snapshot = {
        "top_sharpe":      aa.get("ranked_by_sharpe", [None])[0],
        "top_ir":          ii.get("ranked_by_ir", [None])[0],
        "bma_top":         bma.get("top_source"),
        "deployable":      ii.get("deployable_signals", []),
    }
    return "\n".join(out), snapshot


def _section_macro() -> tuple[str, dict]:
    mr = _load("macro_regime.json")
    vs = _load("vol_surface.json")
    ts = _load("term_structure.json")
    nc = _load("macro_nowcast.json")
    out = []
    out.append("## 7. Macro & Regime")
    out.append("")
    out.append(f"- **Macro quadrant**: {mr.get('quadrant', '—')}  (conf {mr.get('confidence', 0):.1%})")
    out.append(f"- **Vol regime**: {vs.get('vol_regime', '—')}  phase: {vs.get('phase', '—')}")
    out.append(f"- **Curve shape**: {ts.get('curve_shape', '—')}  slope: {_fmt_pct(ts.get('overall_slope_pct'))}")
    out.append(f"- **Macro nowcast**: {nc.get('regime', '—')}  score: {nc.get('composite_score', '—')}")
    out.append("")
    snapshot = {
        "quadrant":     mr.get("quadrant"),
        "vol_regime":   vs.get("vol_regime"),
        "curve_shape":  ts.get("curve_shape"),
        "nowcast":      nc.get("regime"),
    }
    return "\n".join(out), snapshot


def _section_capacity() -> tuple[str, dict]:
    cap = _load("capacity_analyzer.json")
    tca = _load("transaction_cost_model.json")
    sor = _load("smart_order_router.json")
    out = []
    out.append("## 8. Capacity & Execution Costs")
    out.append("")
    out.append(f"- **Expected alpha (top source)**: {_fmt_pct(cap.get('expected_alpha_pct'))}")
    out.append(f"- **Physical execution cap (25% α decay)**: {_fmt_usd(cap.get('thresholds_physical', {}).get('decay_25pct', {}).get('aum_cap_usd'))}")
    out.append(f"- **Paper execution cap (25% α decay)**: {_fmt_usd(cap.get('thresholds_paper', {}).get('decay_25pct', {}).get('aum_cap_usd'))}")
    out.append(f"- **TCA avg one-way**: {tca.get('aggregate', {}).get('avg_oneway_cost_bps', '—')} bps")
    out.append(f"- **SOR recommended algo**: {sor.get('recommended_algo', '—')}  cost: {sor.get('recommended_cost', {}).get('total_oneway_bps', '—')} bps")
    out.append("")
    snapshot = {
        "expected_alpha_pct": cap.get("expected_alpha_pct"),
        "physical_cap_usd":   cap.get("thresholds_physical", {}).get("decay_25pct", {}).get("aum_cap_usd"),
        "paper_cap_usd":      cap.get("thresholds_paper", {}).get("decay_25pct", {}).get("aum_cap_usd"),
        "tca_avg_bps":        tca.get("aggregate", {}).get("avg_oneway_cost_bps"),
    }
    return "\n".join(out), snapshot


def _section_governance() -> tuple[str, dict]:
    mrm = _load("mrm_champion.json")
    at = _load("audit_trail_status.json")
    dr = _load("dr_backup.json")
    out = []
    out.append("## 9. Governance")
    out.append("")
    out.append(f"- **Current champion**: {mrm.get('current_champion', '—')}")
    out.append(f"- **MRM decision**: {mrm.get('decision', '—')}  Δ-score: {mrm.get('score_delta', '—')}")
    out.append(f"- **Audit chain valid**: {at.get('chain_valid', '—')}  total rows: {at.get('n_total', 0)}")
    out.append(f"- **Last DR snapshot**: {dr.get('snapshot', {}).get('size_mb', '—')} MB  encrypted: {dr.get('snapshot', {}).get('encrypted', False)}")
    out.append("")
    snapshot = {
        "champion":         mrm.get("current_champion"),
        "mrm_decision":     mrm.get("decision"),
        "audit_chain_valid":at.get("chain_valid"),
        "audit_rows":       at.get("n_total"),
        "dr_size_mb":       dr.get("snapshot", {}).get("size_mb"),
        "dr_encrypted":     dr.get("snapshot", {}).get("encrypted"),
    }
    return "\n".join(out), snapshot


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_tear_sheet() -> dict:
    parts = []
    parts.append(f"# Monthly Tear Sheet")
    parts.append("")
    parts.append(f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}*")
    parts.append("")

    sections = [
        _section_executive(),
        _section_performance(),
        _section_hedge_sleeve(),   # §3 — Phase XXV hedge attribution
        _section_risk(),
        _section_factors(),
        _section_signals(),
        _section_macro(),
        _section_capacity(),
        _section_governance(),
    ]
    pf_snapshot = {}
    for md, snap in sections:
        parts.append(md)
        pf_snapshot.update(snap)

    body = "\n".join(parts)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEAR_SHEET.write_text(body)

    # Form PF lite — flatten everything
    pf_payload = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type":             "FORM_PF_LITE",
        "fund_id":          "GOLD_AI_OPERATOR",
        **pf_snapshot,
    }
    FORM_PF_FILE.write_text(json.dumps(pf_payload, indent=2, default=str))

    print(f"\n{SEP}\n  TEAR SHEET + FORM PF LITE\n{SEP}")
    print(f"  Markdown:   {TEAR_SHEET}")
    print(f"  Form PF:    {FORM_PF_FILE}")
    print(f"  Sections:   {len(sections)}")
    print(SEP)

    return {
        "tear_sheet_path": str(TEAR_SHEET),
        "form_pf_path":    str(FORM_PF_FILE),
        "n_sections":      len(sections),
        "snapshot":        pf_snapshot,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tear Sheet + Form PF lite")
    args = parser.parse_args()
    run_tear_sheet()
