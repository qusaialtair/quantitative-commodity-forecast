#!/usr/bin/env python3
"""
scripts/risk_manager.py
=======================
Deterministic risk layer. Intercepts CIO output BEFORE shadow_trader execution.
No LLM calls, no DB writes. Pure function: inputs -> RiskDecision.

Risk checks (applied in order):
  1. Volatility-Adjusted Sizing (target-vol formula)
       base_weight  = TARGET_VOL_ANNUAL / max(sigma_21d, garch_forecast)
       scaled by conviction [0.50, 1.00] and HMM multiplier [0.75, 1.00]
       hard ceiling at MAX_POSITION_PCT (95%)

  2. GARCH(1,1) Volatility Forecast
       sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1)
       Calibrated for gold: omega=0.000002, alpha=0.06, beta=0.93
       Uses max(realized_vol, garch_vol) for conservative sizing.

  3. Historical VaR + CVaR Safety Net (rolling 252-day, 95% confidence)
       VaR_95   = 5th-percentile daily log-return
       CVaR_95  = mean of returns below VaR_95 (expected shortfall)
       position_VaR_95 > 2.5% portfolio -> analytical downsize

  4. Maximum Drawdown Brake
       Rolling 252-day max drawdown > 15% -> position size halved

  5. Correlation Risk (Gold-DXY)
       21-day correlation > +0.30 (structural break) -> size reduced 25%

STRATEGIC_EXIT is never downsized -- exits always liquidate fully.
HOLD_METAL passes through immediately with zero sizing.

Usage (standalone audit):
  python3 scripts/risk_manager.py --action ACCUMULATE --quant 7 --macro 5
  python3 scripts/risk_manager.py --action ACCUMULATE --quant 3 --macro 2 --hmm RANGING
  python3 scripts/risk_manager.py --action STRATEGIC_EXIT --oz 21.5
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# -- Configuration ------------------------------------------------------------

TARGET_VOL_ANNUAL   = 0.15    # 15% -- commodity trend-following standard (AQR / Winton)
MAX_POSITION_PCT    = 0.95    # never deploy more than 95% of cash (5% friction buffer)
VAR_CONFIDENCE      = 0.95    # 95% one-tailed -- 5th percentile of daily returns
MAX_DAILY_LOSS_PCT  = 0.025   # 2.5% of total portfolio value hard stop
VAR_LOOKBACK_DAYS   = 252     # rolling 1-year window
VOL_WINDOW_DAYS     = 21      # realised vol lookback (one trading month)
CONVICTION_FLOOR    = 0.50    # minimum conviction scalar -- even weak conviction deploys 50%
MIN_VIABLE_DEPLOY   = 100.0   # minimum $100 notional; below this -> HOLD_METAL

# GARCH(1,1) calibrated parameters for gold futures
GARCH_OMEGA = 0.000002
GARCH_ALPHA = 0.06
GARCH_BETA  = 0.93

# Drawdown brake
MAX_DRAWDOWN_THRESHOLD = 0.15   # 15% rolling max drawdown triggers 50% size reduction
DRAWDOWN_PENALTY       = 0.50   # multiply position by this factor when drawdown threshold hit

# Correlation risk (Gold vs DXY)
DXY_TICKER                = "DX-Y.NYB"
CORR_WINDOW               = 21     # trading days for correlation window
CORR_POSITIVE_THRESHOLD   = 0.30   # gold-DXY correlation above this -> structural break
CORR_PENALTY              = 0.75   # multiply position by this factor on positive correlation

# HMM regime -> sizing multiplier
# BEARISH should never reach here (veto fires upstream), but hard-zeroed as safety
_HMM_MULT = {"BULLISH": 1.00, "RANGING": 0.75, "BEARISH": 0.00}

DEFAULT_TICKER = "GC=F"


# -- Output dataclass ---------------------------------------------------------

@dataclass
class RiskDecision:
    # Final approved action (may be downgraded to HOLD_METAL)
    approved_action:        str
    ticker:                 str

    # Sizing
    target_weight:          float   # fraction of cash approved for deployment (0 - 0.95)
    deploy_usd:             float   # dollar amount approved
    oz_to_transact:         float   # positive = buy, negative = sell (STRATEGIC_EXIT)

    # Sizing audit
    realized_vol_21d_annual: float
    base_weight:             float   # TARGET_VOL / sigma (pre-cap)
    conviction_scalar:       float   # [FLOOR, 1.0]
    hmm_multiplier:          float   # from _HMM_MULT
    raw_weight:              float   # before MAX_POSITION_PCT cap

    # VaR audit
    var_95_daily:            float   # 5th-pct daily return (negative)
    var_99_daily:            float   # 1st-pct daily return (negative)
    position_var_usd:        float   # expected 1-day loss at 95% confidence
    max_daily_loss_usd:      float   # hard limit applied
    var_check_passed:        bool
    var_override:            bool    # True if VaR downsized the position

    # Human-readable audit trail
    risk_notes:              str
    computed_at:             str

    # --- New fields (v2) --- defaults ensure backward compatibility -----------
    cvar_95_daily:           float  = 0.0    # Expected Shortfall (mean of losses beyond VaR_95)
    garch_vol_forecast:      float  = 0.0    # GARCH(1,1) annualized vol forecast for next day
    drawdown_pct:            float  = 0.0    # Rolling 252-day max drawdown (0.0 to 1.0)
    dxy_correlation:         float  = 0.0    # 21-day Gold-DXY correlation (-1 to +1)


# -- Internal helpers ----------------------------------------------------------

def _fetch_log_returns(ticker: str, lookback: int = VAR_LOOKBACK_DAYS) -> np.ndarray:
    """
    Download ~lookback+60 calendar days of price history and return the last
    `lookback` daily log-return observations as a 1-D numpy array.
    Falls back to an empty array on any network error.
    """
    try:
        import yfinance as yf
        # Request extra days to guarantee enough business-day observations
        period_days = int(lookback / 252 * 365) + 90
        raw = yf.download(
            ticker, period=f"{period_days}d",
            interval="1d", progress=False, auto_adjust=True,
        )
        if raw.empty:
            return np.array([])
        import pandas as pd
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cl = raw["Close"].dropna().values.astype(float)
        log_ret = np.diff(np.log(cl))
        return log_ret[-lookback:]
    except Exception:
        return np.array([])


def _fetch_dxy_returns(lookback: int = VAR_LOOKBACK_DAYS) -> np.ndarray:
    """
    Fetch DXY log returns for correlation analysis.
    Falls back to empty array on any error.
    """
    try:
        import yfinance as yf
        import pandas as pd
        period_days = int(lookback / 252 * 365) + 90
        raw = yf.download(
            DXY_TICKER, period=f"{period_days}d",
            interval="1d", progress=False, auto_adjust=True,
        )
        if raw.empty:
            return np.array([])
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cl = raw["Close"].dropna().values.astype(float)
        log_ret = np.diff(np.log(cl))
        return log_ret[-lookback:]
    except Exception:
        return np.array([])


def _realized_vol_annual(returns: np.ndarray, window: int = VOL_WINDOW_DAYS) -> float:
    """Annualised realised vol from the last `window` observations. Returns TARGET_VOL on failure."""
    if len(returns) < window:
        return TARGET_VOL_ANNUAL
    return float(np.std(returns[-window:], ddof=1) * np.sqrt(252))


def _historical_var(returns: np.ndarray) -> tuple[float, float]:
    """
    Historical simulation VaR (no distributional assumptions).
    Returns (VaR_95, VaR_99) -- both negative floats representing daily loss floor.
    Falls back to parametric estimate (z * sigma) on insufficient data.
    """
    if len(returns) < 30:
        sigma = np.std(returns) if len(returns) > 1 else TARGET_VOL_ANNUAL / np.sqrt(252)
        return float(-1.645 * sigma), float(-2.326 * sigma)

    sorted_ret = np.sort(returns)          # ascending: worst day first
    n          = len(sorted_ret)
    idx_95     = max(int(np.floor(0.05 * n)), 0)
    idx_99     = max(int(np.floor(0.01 * n)), 0)
    return float(sorted_ret[idx_95]), float(sorted_ret[idx_99])


def _conditional_var(returns: np.ndarray, var_95: float) -> float:
    """
    Conditional VaR / Expected Shortfall at 95% confidence.
    Average of all returns that fall at or below the VaR_95 threshold.
    This captures tail risk better than VaR alone -- it answers:
    "Given that we are in the worst 5% of days, what is the average loss?"

    Falls back to 1.4x VaR_95 (typical ES/VaR ratio for fat-tailed assets)
    on insufficient data.
    """
    if len(returns) < 30:
        return float(var_95 * 1.4)

    tail = returns[returns <= var_95]
    if len(tail) == 0:
        return float(var_95 * 1.4)
    return float(np.mean(tail))


def _garch_forecast_vol(returns: np.ndarray) -> float:
    """
    Simple recursive GARCH(1,1) volatility forecast for the next trading day.

    Model: sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1)

    Calibrated for gold futures:
        omega = 0.000002  (long-run daily variance floor)
        alpha = 0.06      (innovation/shock sensitivity)
        beta  = 0.93      (persistence)
        alpha + beta = 0.99 (high persistence, standard for commodities)

    Uses the last 252 observations to warm up the recursive filter, then
    forecasts one step ahead. Returns annualised volatility.

    Falls back to TARGET_VOL_ANNUAL if insufficient data.
    """
    if len(returns) < 30:
        return TARGET_VOL_ANNUAL

    omega = GARCH_OMEGA
    alpha = GARCH_ALPHA
    beta  = GARCH_BETA

    # Initialise sigma^2 with sample variance of the data
    sigma2 = float(np.var(returns))

    # Run the recursive filter through all observations
    for r in returns:
        sigma2 = omega + alpha * (r * r) + beta * sigma2

    # sigma2 is now the one-step-ahead forecast (daily variance)
    daily_vol = np.sqrt(max(sigma2, 1e-12))
    annual_vol = daily_vol * np.sqrt(252)

    return float(annual_vol)


def _max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from a series of log returns (last 252 days).
    Returns a positive float between 0.0 and 1.0 representing peak-to-trough decline.

    Falls back to 0.0 on insufficient data (conservative -- no drawdown brake applied).
    """
    if len(returns) < 10:
        return 0.0

    # Reconstruct cumulative return series (price index from 1.0)
    cum_returns = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdowns as fraction of peak
    drawdowns = (running_max - cum_returns) / running_max

    return float(np.max(drawdowns))


def _gold_dxy_correlation(gold_returns: np.ndarray, dxy_returns: np.ndarray,
                          window: int = CORR_WINDOW) -> float:
    """
    Compute rolling Pearson correlation between gold and DXY over the last `window` days.

    Gold and DXY are normally negatively correlated (~-0.40 to -0.60).
    A strongly positive correlation signals a structural break (e.g., both
    falling during a liquidity crisis, or unusual macro regime).

    Returns 0.0 (neutral) if data is insufficient for reliable correlation.
    """
    if len(gold_returns) < window or len(dxy_returns) < window:
        return 0.0

    # Align to same length (take last `window` of each)
    g = gold_returns[-window:]
    d = dxy_returns[-window:]

    # Guard against constant series (zero std -> undefined correlation)
    g_std = np.std(g)
    d_std = np.std(d)
    if g_std < 1e-12 or d_std < 1e-12:
        return 0.0

    corr = float(np.corrcoef(g, d)[0, 1])

    # Guard against NaN from degenerate data
    if np.isnan(corr):
        return 0.0

    return corr


def _conviction_scalar(quant: int, macro: int) -> float:
    """
    Map average conviction -> [CONVICTION_FLOOR, 1.0].
    Negative conviction on a BUY action is floored (CIO should not emit BUY at <0).
    """
    avg     = (int(quant) + int(macro)) / 2.0
    clamped = max(0.0, min(10.0, avg))
    return CONVICTION_FLOOR + clamped / 10.0 * (1.0 - CONVICTION_FLOOR)


def _build_passthrough_decision(
    action: str, ticker: str, now: str, risk_notes: str = "",
    gold_oz: float = 0.0, deploy_usd: float = 0.0,
    oz_to_transact: float = 0.0, max_daily_loss_usd: float = 0.0,
    var_check_passed: bool = True,
    # Optional sizing audit for downgraded decisions
    vol_21d: float = 0.0, base_weight: float = 0.0,
    c_scalar: float = 0.0, h_mult: float = 0.0,
    raw_weight: float = 0.0, var_95: float = 0.0, var_99: float = 0.0,
    position_var_usd: float = 0.0, var_override: bool = False,
) -> RiskDecision:
    """
    Build a RiskDecision for pass-through cases (HOLD_METAL, STRATEGIC_EXIT,
    no-cash, below-minimum). Centralises construction to avoid repeating all
    the new default fields at every call site.
    """
    return RiskDecision(
        approved_action=action, ticker=ticker,
        target_weight=0.0, deploy_usd=deploy_usd, oz_to_transact=oz_to_transact,
        realized_vol_21d_annual=vol_21d,
        base_weight=base_weight,
        conviction_scalar=c_scalar, hmm_multiplier=h_mult,
        raw_weight=raw_weight,
        var_95_daily=var_95, var_99_daily=var_99,
        position_var_usd=position_var_usd,
        max_daily_loss_usd=max_daily_loss_usd,
        var_check_passed=var_check_passed,
        var_override=var_override,
        risk_notes=risk_notes,
        computed_at=now,
        # New v2 fields default to safe values
        cvar_95_daily=0.0,
        garch_vol_forecast=0.0,
        drawdown_pct=0.0,
        dxy_correlation=0.0,
    )


# -- Public API ----------------------------------------------------------------

def evaluate(
    action:           str,
    quant_conviction: int,
    macro_conviction: int,
    hmm_state:        str,
    hmm_veto:         bool,
    portfolio:        dict,
    ticker:           str   = DEFAULT_TICKER,
    spot_price:       float = 0.0,
    commission:       float = 2.50,
    slippage_bps:     float = 5.0,
) -> RiskDecision:
    """
    Evaluate a CIO action and return a RiskDecision with approved sizing.

    Parameters
    ----------
    action           : CIO action string
    quant_conviction : Quant Agent score  (-10 to +10)
    macro_conviction : Macro Agent score  (-10 to +10)
    hmm_state        : 'BULLISH' | 'RANGING' | 'BEARISH'
    hmm_veto         : True if HMM regime veto is active
    portfolio        : {'cash_usd': float, 'gold_oz': float, 'portfolio_value': float}
    ticker           : yfinance symbol for price-history fetch
    spot_price       : current fill reference price (fetched if 0.0)
    commission       : flat fee per transaction in USD
    slippage_bps     : slippage per side in basis points
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cash_usd        = float(portfolio.get("cash_usd",        0.0))
    gold_oz         = float(portfolio.get("gold_oz",         0.0))
    portfolio_value = float(portfolio.get("portfolio_value", max(cash_usd, 1.0)))

    # -- Resolve live spot if not provided -------------------------------------
    if spot_price <= 0.0:
        try:
            import yfinance as yf
            import pandas as pd
            raw = yf.download(ticker, period="2d", interval="1d",
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            spot_price = float(raw["Close"].dropna().iloc[-1])
        except Exception:
            spot_price = 0.0

    # -- HOLD_METAL pass-through -----------------------------------------------
    if action == "HOLD_METAL":
        return _build_passthrough_decision(
            action="HOLD_METAL", ticker=ticker, now=now,
            risk_notes="HOLD_METAL -- no sizing required.",
        )

    # -- STRATEGIC_EXIT -- always liquidate 100%, no VaR gate on exits ---------
    if action == "STRATEGIC_EXIT":
        if gold_oz < 1e-6:
            return _build_passthrough_decision(
                action="HOLD_METAL", ticker=ticker, now=now,
                risk_notes="STRATEGIC_EXIT -- already in FIAT (0 oz held).",
            )
        fill_exit = spot_price * (1.0 - slippage_bps / 10_000)
        proceeds  = gold_oz * fill_exit - commission
        return _build_passthrough_decision(
            action="STRATEGIC_EXIT", ticker=ticker, now=now,
            oz_to_transact=-gold_oz,
            max_daily_loss_usd=portfolio_value * MAX_DAILY_LOSS_PCT,
            risk_notes=(
                f"STRATEGIC_EXIT -- full liquidation of {gold_oz:.4f} oz "
                f"@ ${fill_exit:,.2f} fill.  Proceeds ~${proceeds:,.0f}.  "
                f"VaR not applicable on exits."
            ),
        )

    # -- ACCUMULATE / RE_ENTER sizing path -------------------------------------
    if cash_usd < 1.0:
        return _build_passthrough_decision(
            action="HOLD_METAL", ticker=ticker, now=now,
            risk_notes=f"{action} -- no cash available to deploy (already fully in metal).",
        )

    notes: list[str] = []

    # -- Step 1: Fetch historical returns (gold + DXY in parallel concept) -----
    returns     = _fetch_log_returns(ticker, VAR_LOOKBACK_DAYS)
    dxy_returns = _fetch_dxy_returns(VAR_LOOKBACK_DAYS)

    data_degraded = len(returns) < 50
    if data_degraded:
        notes.append(
            f"WARNING: only {len(returns)} return observations (need 252) -- "
            "using conservative fallback vol."
        )

    # -- Step 2: Realised vol (21-day) -----------------------------------------
    vol_21d = _realized_vol_annual(returns, VOL_WINDOW_DAYS)

    # -- Step 3: GARCH(1,1) vol forecast ---------------------------------------
    garch_vol = _garch_forecast_vol(returns)
    notes.append(f"GARCH(1,1): forecast_vol={garch_vol:.1%} vs realized_vol={vol_21d:.1%}")

    # Use the more conservative (higher) of realised and GARCH-forecast vol
    sizing_vol = max(vol_21d, garch_vol)
    if garch_vol > vol_21d:
        notes.append(
            f"GARCH vol ({garch_vol:.1%}) > realized vol ({vol_21d:.1%}) -- "
            "using GARCH forecast for conservative sizing."
        )

    # -- Step 4: Base weight (target-vol formula) ------------------------------
    # Protect against near-zero vol (illiquid / holiday data)
    vol_floored = max(sizing_vol, 0.02)
    base_weight = TARGET_VOL_ANNUAL / vol_floored

    # -- Step 5: Conviction scalar [FLOOR, 1.0] --------------------------------
    c_scalar = _conviction_scalar(quant_conviction, macro_conviction)

    # -- Step 6: HMM multiplier ------------------------------------------------
    h_mult = _HMM_MULT.get(hmm_state, 0.75)

    # -- Step 7: Raw weight and cap --------------------------------------------
    raw_weight    = base_weight * c_scalar * h_mult
    target_weight = min(raw_weight, MAX_POSITION_PCT)
    deploy_usd    = cash_usd * target_weight

    notes.append(
        f"Sizing: sigma={sizing_vol:.1%}  base_wt={base_weight:.3f}  "
        f"conviction_scalar={c_scalar:.2f}  hmm_mult={h_mult:.2f}  "
        f"raw_wt={raw_weight:.3f}  -> target_wt={target_weight:.3f}  "
        f"deploy=${deploy_usd:,.0f}"
    )

    # -- Step 8: Maximum Drawdown brake ----------------------------------------
    drawdown_pct = _max_drawdown(returns)
    drawdown_applied = False
    if drawdown_pct > MAX_DRAWDOWN_THRESHOLD:
        drawdown_applied = True
        pre_dd_deploy = deploy_usd
        deploy_usd   *= DRAWDOWN_PENALTY
        target_weight = deploy_usd / cash_usd
        notes.append(
            f"DRAWDOWN BRAKE: {drawdown_pct:.1%} drawdown > {MAX_DRAWDOWN_THRESHOLD:.0%} "
            f"threshold -- size cut {DRAWDOWN_PENALTY:.0%}: "
            f"${pre_dd_deploy:,.0f} -> ${deploy_usd:,.0f}"
        )
    else:
        notes.append(f"Drawdown: {drawdown_pct:.1%} (threshold {MAX_DRAWDOWN_THRESHOLD:.0%}) -- OK")

    # -- Step 9: DXY Correlation risk ------------------------------------------
    dxy_corr = _gold_dxy_correlation(returns, dxy_returns, CORR_WINDOW)
    corr_applied = False
    if dxy_corr > CORR_POSITIVE_THRESHOLD:
        corr_applied = True
        pre_corr_deploy = deploy_usd
        deploy_usd   *= CORR_PENALTY
        target_weight = deploy_usd / cash_usd
        notes.append(
            f"CORRELATION RISK: Gold-DXY corr={dxy_corr:+.2f} > +{CORR_POSITIVE_THRESHOLD:.2f} "
            f"(structural break) -- size cut {(1-CORR_PENALTY):.0%}: "
            f"${pre_corr_deploy:,.0f} -> ${deploy_usd:,.0f}"
        )
    else:
        notes.append(f"DXY correlation: {dxy_corr:+.2f} (threshold +{CORR_POSITIVE_THRESHOLD:.2f}) -- OK")

    # -- Step 10: Historical VaR + CVaR (95%, rolling 252-day) -----------------
    var_95, var_99         = _historical_var(returns)
    cvar_95                = _conditional_var(returns, var_95)
    max_daily_loss_usd     = portfolio_value * MAX_DAILY_LOSS_PCT
    position_var_usd       = deploy_usd * abs(var_95)
    var_check_passed       = True
    var_override           = False

    notes.append(
        f"VaR: 5th-pctl={var_95:.3%}  CVaR={cvar_95:.3%}  1st-pctl={var_99:.3%}  "
        f"position_VaR=${position_var_usd:,.0f}  limit=${max_daily_loss_usd:,.0f}"
    )

    # -- Step 11: Hard stop -- downsize analytically if VaR breaches limit -----
    if position_var_usd > max_daily_loss_usd:
        var_override     = True
        var_check_passed = False

        # Analytical solution: maximum deploy s.t. deploy * |VaR_95| <= limit
        max_deploy_by_var = max_daily_loss_usd / abs(var_95)
        notes.append(
            f"VaR OVERRIDE: ${deploy_usd:,.0f} -> ${max_deploy_by_var:,.0f}  "
            f"(VaR would be ${position_var_usd:,.0f}/day vs limit ${max_daily_loss_usd:,.0f}/day)"
        )

        deploy_usd       = max_deploy_by_var
        target_weight    = deploy_usd / cash_usd
        position_var_usd = deploy_usd * abs(var_95)
        var_check_passed = True   # now within limits

    # -- Step 12: Minimum viable check -----------------------------------------
    if deploy_usd < MIN_VIABLE_DEPLOY:
        notes.append(
            f"Deploy ${deploy_usd:.0f} < min viable ${MIN_VIABLE_DEPLOY:.0f} "
            f"-- downgrading to HOLD_METAL."
        )
        return RiskDecision(
            approved_action="HOLD_METAL", ticker=ticker,
            target_weight=0.0, deploy_usd=0.0, oz_to_transact=0.0,
            realized_vol_21d_annual=vol_21d,
            base_weight=min(base_weight, MAX_POSITION_PCT),
            conviction_scalar=c_scalar, hmm_multiplier=h_mult,
            raw_weight=min(raw_weight, MAX_POSITION_PCT),
            var_95_daily=var_95, var_99_daily=var_99,
            position_var_usd=position_var_usd,
            max_daily_loss_usd=max_daily_loss_usd,
            var_check_passed=var_check_passed,
            var_override=var_override,
            risk_notes="  |  ".join(notes),
            computed_at=now,
            cvar_95_daily=cvar_95,
            garch_vol_forecast=garch_vol,
            drawdown_pct=drawdown_pct,
            dxy_correlation=dxy_corr,
        )

    # -- Step 13: Compute oz ---------------------------------------------------
    fill_price     = spot_price * (1.0 + slippage_bps / 10_000)
    oz_to_transact = (deploy_usd - commission) / fill_price if fill_price > 0 else 0.0

    return RiskDecision(
        approved_action=action, ticker=ticker,
        target_weight=round(target_weight, 6),
        deploy_usd=round(deploy_usd, 2),
        oz_to_transact=round(oz_to_transact, 6),
        realized_vol_21d_annual=round(vol_21d, 6),
        base_weight=round(min(base_weight, 10.0), 6),  # cap display at 10x
        conviction_scalar=round(c_scalar, 4),
        hmm_multiplier=h_mult,
        raw_weight=round(raw_weight, 6),
        var_95_daily=round(var_95, 6),
        var_99_daily=round(var_99, 6),
        position_var_usd=round(position_var_usd, 2),
        max_daily_loss_usd=round(max_daily_loss_usd, 2),
        var_check_passed=var_check_passed,
        var_override=var_override,
        risk_notes="  |  ".join(notes),
        computed_at=now,
        cvar_95_daily=round(cvar_95, 6),
        garch_vol_forecast=round(garch_vol, 6),
        drawdown_pct=round(drawdown_pct, 6),
        dxy_correlation=round(dxy_corr, 4),
    )


# -- CLI audit tool ------------------------------------------------------------

def _print_decision(rd: RiskDecision) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  RISK MANAGER v2 -- Decision Audit")
    print(f"  {rd.computed_at}  |  {rd.ticker}")
    print("=" * W)
    print(f"  Approved action     : {rd.approved_action}")
    print(f"  Target weight       : {rd.target_weight:.2%}")
    print(f"  Deploy USD          : ${rd.deploy_usd:>10,.2f}")
    print(f"  Oz to transact      : {rd.oz_to_transact:>+.4f}")
    print()
    print("  -- Sizing inputs -------------------------------------------------")
    print(f"  Realised vol (21d)  : {rd.realized_vol_21d_annual:.2%}  annualised")
    print(f"  GARCH(1,1) forecast : {rd.garch_vol_forecast:.2%}  annualised")
    print(f"  Base weight         : {rd.base_weight:.3f}  (15% / max(sigma, garch))")
    print(f"  Conviction scalar   : {rd.conviction_scalar:.3f}")
    print(f"  HMM multiplier      : {rd.hmm_multiplier:.2f}")
    print(f"  Raw weight          : {rd.raw_weight:.3f}")
    print()
    print("  -- VaR / CVaR (historical 252-day, 95% confidence) ---------------")
    print(f"  VaR_95 daily        : {rd.var_95_daily:+.3%}")
    print(f"  CVaR_95 (Exp.Short.): {rd.cvar_95_daily:+.3%}")
    print(f"  VaR_99 daily        : {rd.var_99_daily:+.3%}")
    print(f"  Position VaR        : ${rd.position_var_usd:>10,.2f}  / day")
    print(f"  Max daily loss limit: ${rd.max_daily_loss_usd:>10,.2f}")
    print(f"  VaR check passed    : {rd.var_check_passed}")
    print(f"  VaR override fired  : {rd.var_override}")
    print()
    print("  -- Drawdown & Correlation brakes ----------------------------------")
    print(f"  Max drawdown (252d) : {rd.drawdown_pct:.2%}  "
          f"(threshold {MAX_DRAWDOWN_THRESHOLD:.0%})"
          f"{'  ** BRAKE ACTIVE **' if rd.drawdown_pct > MAX_DRAWDOWN_THRESHOLD else ''}")
    print(f"  Gold-DXY corr (21d) : {rd.dxy_correlation:+.3f}  "
          f"(threshold +{CORR_POSITIVE_THRESHOLD:.2f})"
          f"{'  ** STRUCTURAL BREAK **' if rd.dxy_correlation > CORR_POSITIVE_THRESHOLD else ''}")
    print()
    if rd.risk_notes:
        for note in rd.risk_notes.split("  |  "):
            print(f"  NOTE: {note}")
    print("=" * W + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Risk Manager v2 -- standalone audit tool")
    parser.add_argument("--action",  default="ACCUMULATE",
        choices=["ACCUMULATE", "RE_ENTER", "STRATEGIC_EXIT", "HOLD_METAL"])
    parser.add_argument("--quant",   type=int, default=6,  help="Quant conviction -10..+10")
    parser.add_argument("--macro",   type=int, default=5,  help="Macro conviction -10..+10")
    parser.add_argument("--hmm",     default="BULLISH",
        choices=["BULLISH", "RANGING", "BEARISH"])
    parser.add_argument("--veto",    action="store_true")
    parser.add_argument("--cash",    type=float, default=100_000.0)
    parser.add_argument("--oz",      type=float, default=0.0,   help="Gold oz currently held")
    parser.add_argument("--value",   type=float, default=0.0,   help="Total portfolio value")
    parser.add_argument("--ticker",  default=DEFAULT_TICKER)
    parser.add_argument("--spot",    type=float, default=0.0,   help="Spot price (fetched if 0)")
    args = parser.parse_args()

    portfolio = {
        "cash_usd":        args.cash,
        "gold_oz":         args.oz,
        "portfolio_value": args.value or args.cash + args.oz * max(args.spot, 3000.0),
    }

    rd = evaluate(
        action=args.action,
        quant_conviction=args.quant,
        macro_conviction=args.macro,
        hmm_state=args.hmm,
        hmm_veto=args.veto,
        portfolio=portfolio,
        ticker=args.ticker,
        spot_price=args.spot,
    )
    _print_decision(rd)


if __name__ == "__main__":
    main()
