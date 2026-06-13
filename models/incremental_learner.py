#!/usr/bin/env python3
"""
Phase 4 — Heterogeneous Ensemble Backtesting Engine
Simulates a Sharia-compliant physical gold investor (UAE Gold Souq).

Architecture:
  Base Learner 1 — RandomForestRegressor   (macro features,    5-day target)
  Base Learner 2 — LightGBM / GradientBoosting (technical features, 5-day target)
  Base Learner 3 — SGDRegressor            (all features, online 5-day updates)
  Meta-Learner   — Ridge                   (combines base predictions)
  Oracle         — Perplexity Sonar Pro    (live macro scores, cached 6h)

Signal Quality Filters (ALL must pass for BUY):
  ✓ Ensemble predicts > 2.5% gain over next 5 days
  ✓ At least 2 of 3 base learners agree (bullish direction)
  ✓ Gold above 200-day moving average (long-term uptrend)
  ✓ RSI(14) < 75 (not severely overbought)
  ✓ DXY not in strong uptrend (no major dollar headwind)
  ✓ Perplexity macro score ≥ 0 (net macro tailwind) — when available

SELL triggers (ANY):
  ✗ Ensemble predicts < -1% over next 5 days
  ✗ Stop-loss: position down > 6% from entry
  ✗ Gold breaks below 50-day MA (trend breakdown)

Fee: 60 AED flat per BUY. No SELL fee.
Starting capital: 10,000 AED. Currency: USD/AED at 3.6725.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.linear_model    import SGDRegressor, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error

# Optional LightGBM — fall back to GradientBoostingRegressor
try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

# Add project root to path so sibling packages resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipelines.multi_asset_feed import build_features
from agents.perplexity_oracle        import PerplexityOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
USD_TO_AED           = 3.6725
BUY_FEE_AED          = 60.0
STARTING_AED         = 10_000.0
DATA_PERIOD          = "10y"

WARMUP_DAYS          = 500    # ~2 years before trading starts
RETRAIN_EVERY        = 20     # retrain RF + LGB every N live days
TARGET_HORIZON       = 5      # predict 5-trading-day forward return

# Signal thresholds
BUY_THRESHOLD        = 0.025  # ensemble must predict > 2.5% 5-day gain
SELL_THRESHOLD       = -0.01  # sell when ensemble predicts < -1%
STOP_LOSS_PCT        = -0.06  # hard stop at -6% from entry
CONSENSUS_REQUIRED   = 2      # at least 2 of 3 base learners must agree

# Feature subsets for each base learner
MACRO_FEATURE_KEYWORDS    = ["dxy", "tnx", "vix", "spx", "oil", "gs_", "real_", "yield",
                              "inverted", "risk_off", "pplx", "gold_oil", "month", "q4"]
TECHNICAL_FEATURE_KEYWORDS = ["g_lag", "g_ma", "g_pma", "g_ret", "g_rsi", "g_macd",
                               "g_bb", "g_vol", "g_atr"]


def split_features(feature_cols: list) -> tuple:
    """Split feature columns into macro and technical subsets."""
    macro_cols  = [c for c in feature_cols if any(k in c for k in MACRO_FEATURE_KEYWORDS)]
    tech_cols   = [c for c in feature_cols if any(k in c for k in TECHNICAL_FEATURE_KEYWORDS)]
    # Ensure no empty sets — fall back to all features
    if not macro_cols:
        macro_cols = feature_cols
    if not tech_cols:
        tech_cols = feature_cols
    return macro_cols, tech_cols


class HeterogeneousEnsemble:
    """
    Three base learners + Ridge meta-learner.
    Base 1 (RF)  : macro-focused features, 5-day target
    Base 2 (LGB) : technical features, 5-day target
    Base 3 (SGD) : all features, online partial_fit, 5-day target
    Meta (Ridge) : combines [RF_pred, LGB_pred, SGD_pred] → final prediction
    """

    def __init__(self, macro_cols: list, tech_cols: list, all_cols: list):
        self.macro_cols = macro_cols
        self.tech_cols  = tech_cols
        self.all_cols   = all_cols

        self.rf  = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features=0.5, random_state=42, n_jobs=-1
        )
        if HAS_LGB:
            self.lgb = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=6,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbose=-1
            )
        else:
            self.lgb = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=5,
                min_samples_leaf=10, subsample=0.8, random_state=42
            )

        self.sgd    = SGDRegressor(loss="squared_error", learning_rate="invscaling",
                                   eta0=0.01, power_t=0.25, random_state=42, max_iter=1, tol=None)
        self.meta   = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

        self.rf_fitted   = False
        self.lgb_fitted  = False
        self.sgd_fitted  = False
        self.meta_fitted = False

        # Rolling MAE trackers for dynamic weighting
        self._rf_errors  = []
        self._lgb_errors = []
        self._sgd_errors = []

    def train_batch(self, X_df: pd.DataFrame, y: np.ndarray):
        """Retrain RF and LGB on expanding window. Also fits scaler."""
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            return

        X_all = X_df[self.all_cols].values[valid]
        X_mac = X_df[self.macro_cols].values[valid]
        X_tec = X_df[self.tech_cols].values[valid]
        y_v   = y[valid]

        self.scaler.fit(X_all)

        logger.info(f"  [Ensemble] Batch training on {valid.sum()} samples...")
        self.rf.fit(X_mac, y_v)
        self.lgb.fit(X_tec, y_v)
        self.rf_fitted  = True
        self.lgb_fitted = True

        # Fit meta-learner on in-sample base predictions
        rf_p  = self.rf.predict(X_mac)
        lgb_p = self.lgb.predict(X_tec)

        if self.sgd_fitted:
            X_sc  = self.scaler.transform(X_all)
            sgd_p = self.sgd.predict(X_sc)
        else:
            sgd_p = np.zeros(len(y_v))

        meta_X = np.column_stack([rf_p, lgb_p, sgd_p])
        self.meta.fit(meta_X, y_v)
        self.meta_fitted = True
        logger.info(f"  [Ensemble] Batch training complete. RF+LGB+Meta fitted.")

    def update_online(self, x_row: np.ndarray, y_val: float):
        """Online update for SGD using partial_fit with 5-day target."""
        if np.isnan(y_val):
            return
        x_2d = x_row.reshape(1, -1)
        if not hasattr(self.scaler, 'n_samples_seen_') or self.scaler.n_samples_seen_ == 0:
            return  # scaler not fitted yet
        try:
            x_sc = self.scaler.transform(x_2d)
            self.sgd.partial_fit(x_sc, [y_val])
            self.sgd_fitted = True
        except Exception:
            pass

    def predict(self, x_row: np.ndarray, x_mac: np.ndarray, x_tec: np.ndarray) -> tuple:
        """
        Returns (ensemble_prediction, [rf_pred, lgb_pred, sgd_pred], bullish_count).
        All predictions are 5-day forward return estimates.
        """
        preds = []

        rf_p  = self.rf.predict(x_mac.reshape(1, -1))[0]  if self.rf_fitted  else 0.0
        lgb_p = self.lgb.predict(x_tec.reshape(1, -1))[0] if self.lgb_fitted else 0.0

        if self.sgd_fitted and hasattr(self.scaler, 'n_samples_seen_') and self.scaler.n_samples_seen_ > 0:
            try:
                x_sc  = self.scaler.transform(x_row.reshape(1, -1))
                sgd_p = self.sgd.predict(x_sc)[0]
            except Exception:
                sgd_p = 0.0
        else:
            sgd_p = 0.0

        base_preds = [rf_p, lgb_p, sgd_p]
        bullish    = sum(1 for p in base_preds if p > 0)

        if self.meta_fitted:
            ensemble = float(self.meta.predict(np.array([[rf_p, lgb_p, sgd_p]]))[0])
        else:
            fitted = [p for p, fitted in zip(base_preds, [self.rf_fitted, self.lgb_fitted, self.sgd_fitted]) if fitted]
            ensemble = float(np.mean(fitted)) if fitted else 0.0

        return ensemble, base_preds, bullish

    def update_errors(self, actual: float, preds: list):
        """Track rolling MAE per base learner for diagnostics."""
        for store, pred in zip([self._rf_errors, self._lgb_errors, self._sgd_errors], preds):
            store.append(abs(actual - pred))
            if len(store) > 60:
                store.pop(0)

    def mae_report(self) -> dict:
        return {
            "rf_mae":  round(np.mean(self._rf_errors),  5) if self._rf_errors  else None,
            "lgb_mae": round(np.mean(self._lgb_errors), 5) if self._lgb_errors else None,
            "sgd_mae": round(np.mean(self._sgd_errors), 5) if self._sgd_errors else None,
        }


def _quality_filters(row: pd.Series, ensemble_pred: float, bullish_count: int,
                     perplexity_scores: dict) -> tuple:
    """
    Returns (pass: bool, reason: str).
    ALL conditions must hold for a BUY signal.
    """
    # Sanity-clamp: if prediction is absurdly large, the model is extrapolating
    # outside its training distribution (e.g. COVID crash regime). Reject.
    if abs(ensemble_pred) > 0.30:
        return False, f"Prediction ({ensemble_pred*100:.1f}%) out-of-distribution — skipping"

    if ensemble_pred <= BUY_THRESHOLD:
        return False, f"Ensemble {ensemble_pred*100:.2f}% < {BUY_THRESHOLD*100}% threshold"

    if bullish_count < CONSENSUS_REQUIRED:
        return False, f"Consensus {bullish_count}/3 < {CONSENSUS_REQUIRED} required"

    pma200 = row.get("g_pma200", np.nan)
    if not np.isnan(pma200) and pma200 < 1.0:
        return False, f"Price below 200-day MA (g_pma200={pma200:.3f})"

    rsi14 = row.get("g_rsi14", np.nan)
    if not np.isnan(rsi14) and rsi14 > 75:
        return False, f"RSI overbought ({rsi14:.1f} > 75)"

    dxy_strong = row.get("dxy_strong", 0.0)
    if dxy_strong == 1.0:
        return False, "DXY in strong uptrend — dollar headwind for gold"

    # Volatility regime filter: don't buy into extreme fear / black-swan events
    vix = row.get("vix_level", np.nan)
    if not np.isnan(vix) and vix > 30:
        return False, f"VIX too high ({vix:.1f} > 30) — extreme uncertainty, HOLD"

    # Perplexity macro veto (only when oracle is live)
    if perplexity_scores.get("pplx_available", 0.0) == 1.0:
        macro_score = perplexity_scores.get("pplx_macro", 0.0)
        if macro_score < -0.3:
            return False, f"Perplexity macro bearish ({macro_score:+.2f})"

    return True, "All quality filters passed"


def _sell_triggered(gold_price_usd: float, entry_price_usd: float,
                    row: pd.Series, ensemble_pred: float,
                    perplexity_scores: dict) -> tuple:
    """Returns (sell: bool, reason: str)."""
    current_return = (gold_price_usd - entry_price_usd) / entry_price_usd

    if current_return < STOP_LOSS_PCT:
        return True, f"Stop-loss hit ({current_return*100:.2f}%)"

    pma50 = row.get("g_pma50", np.nan)
    if not np.isnan(pma50) and pma50 < 0.97:
        return True, f"Gold broke below 50-day MA (pma50={pma50:.3f})"

    if ensemble_pred < SELL_THRESHOLD:
        return True, f"Ensemble bearish ({ensemble_pred*100:.2f}% < {SELL_THRESHOLD*100}%)"

    if perplexity_scores.get("pplx_available", 0.0) == 1.0:
        macro_score = perplexity_scores.get("pplx_macro", 0.0)
        if macro_score < -0.5:
            return True, f"Perplexity macro strongly bearish ({macro_score:+.2f})"

    return False, ""


# ── Main Backtest ──────────────────────────────────────────────────────────────

def run_backtest(use_perplexity_live: bool = False) -> dict:
    """
    Walk-forward backtest over full data period.
    use_perplexity_live: if True, calls Perplexity API for live scores on current day only.
                         Backtest uses market-data macro proxies (no API calls during history).
    """
    logger.info("=" * 70)
    logger.info("  Phase 4 — Heterogeneous Ensemble Gold Trading Engine")
    logger.info(f"  BUY threshold: {BUY_THRESHOLD*100}% | Stop-loss: {STOP_LOSS_PCT*100}% | Fee: {BUY_FEE_AED} AED")
    logger.info("=" * 70)

    # ── 1. Download and build features ───────────────────────────────────────
    feat_df, feature_cols, gold_close = build_features(period=DATA_PERIOD)
    macro_cols, tech_cols = split_features(feature_cols)
    logger.info(f"Macro features: {len(macro_cols)} | Technical features: {len(tech_cols)} | Total: {len(feature_cols)}")

    # ── 2. Precompute 5-day forward targets (training labels) ─────────────────
    # target[i] = return from buying on day i and selling on day i+5
    # This is NOT a feature — it's the label. Available only after 5 days have passed.
    target_5d = gold_close.pct_change(5).shift(-5)   # shape: same as feat_df

    # ── 3. Perplexity oracle (live queries for current day, if enabled) ───────
    oracle = PerplexityOracle()
    live_pplx_scores = {}   # will be populated only for the current/last day if enabled

    # ── 4. Build ensemble ────────────────────────────────────────────────────
    ensemble = HeterogeneousEnsemble(macro_cols, tech_cols, feature_cols)

    # ── 5. Portfolio state ────────────────────────────────────────────────────
    balance_aed     = STARTING_AED
    position_oz     = 0.0
    entry_price_usd = 0.0
    trades          = []
    snapshots       = []
    live_days       = 0
    last_retrain    = 0

    n = len(feat_df)
    dates  = feat_df.index
    logger.info(f"Data: {n} days [{dates[0].date()} → {dates[-1].date()}]")
    logger.info(f"Warm-up: {WARMUP_DAYS} days | Live trading from day {WARMUP_DAYS}")

    for i, (date, row) in enumerate(feat_df.iterrows()):

        gold_price = float(gold_close.loc[date])
        X_all  = row[feature_cols].values.astype(float)
        X_mac  = row[macro_cols].values.astype(float)
        X_tec  = row[tech_cols].values.astype(float)

        # Online SGD update: we know the 5-day target for the day 5 days ago
        if i >= TARGET_HORIZON:
            past_date = dates[i - TARGET_HORIZON]
            past_row  = feat_df.loc[past_date]
            past_X    = past_row[feature_cols].values.astype(float)
            past_y    = float(target_5d.loc[past_date]) if not np.isnan(target_5d.loc[past_date]) else np.nan
            ensemble.update_online(past_X, past_y)

        # ── Warm-up phase ─────────────────────────────────────────────────────
        if i < WARMUP_DAYS:
            # During warm-up: fit scaler progressively
            if i % 50 == 0 and i > 50:
                X_hist = feat_df.iloc[:i][feature_cols].values.astype(float)
                finite  = np.all(np.isfinite(X_hist), axis=1)
                if finite.sum() > 10:
                    ensemble.scaler.fit(X_hist[finite])
            continue

        # ── First live day: full batch train ─────────────────────────────────
        if live_days == 0:
            train_df  = feat_df.iloc[:i - TARGET_HORIZON]
            train_y   = target_5d.iloc[:i - TARGET_HORIZON].values
            ensemble.train_batch(train_df, train_y)
            last_retrain = i
            logger.info(f"[{date.date()}] Warm-up complete. Ensemble trained. Live trading begins.")

        # ── Periodic batch retraining ─────────────────────────────────────────
        elif (i - last_retrain) >= RETRAIN_EVERY:
            train_df = feat_df.iloc[:i - TARGET_HORIZON]
            train_y  = target_5d.iloc[:i - TARGET_HORIZON].values
            ensemble.train_batch(train_df, train_y)
            last_retrain = i

        live_days += 1

        # ── Get Perplexity scores ─────────────────────────────────────────────
        # During backtest we only call Perplexity for the very last (current) day
        is_last_day = (i == n - 1)
        if use_perplexity_live and is_last_day:
            logger.info("Fetching live Perplexity macro scores for current day...")
            live_pplx_scores = oracle.get_scores()
        perplexity_scores = live_pplx_scores if is_last_day else {}

        # ── Generate ensemble prediction ──────────────────────────────────────
        try:
            ensemble_pred, base_preds, bullish_count = ensemble.predict(X_all, X_mac, X_tec)
        except Exception as exc:
            logger.debug(f"Prediction error on {date.date()}: {exc}")
            ensemble_pred, base_preds, bullish_count = 0.0, [0.0, 0.0, 0.0], 0

        # ── SELL logic (check open position first) ────────────────────────────
        if position_oz > 0.0:
            sell, reason = _sell_triggered(gold_price, entry_price_usd,
                                           row, ensemble_pred, perplexity_scores)
            if sell:
                proceeds_aed  = position_oz * gold_price * USD_TO_AED
                cost_basis_aed = position_oz * entry_price_usd * USD_TO_AED
                pnl_aed        = proceeds_aed - cost_basis_aed
                trades.append({
                    "date":           str(date.date()),
                    "action":         "SELL",
                    "price_usd":      round(gold_price, 2),
                    "oz_sold":        round(position_oz, 6),
                    "proceeds_aed":   round(proceeds_aed, 2),
                    "pnl_aed":        round(pnl_aed, 2),
                    "ensemble_pred":  round(ensemble_pred * 100, 3),
                    "reason":         reason,
                })
                balance_aed     = proceeds_aed
                position_oz     = 0.0
                entry_price_usd = 0.0
                logger.info(
                    f"  [SELL] {date.date()} | ${gold_price:,.2f} | "
                    f"PnL: {pnl_aed:+,.2f} AED | {reason}"
                )

        # ── BUY logic (only when no position open) ────────────────────────────
        elif position_oz == 0.0:
            passes, reason = _quality_filters(row, ensemble_pred, bullish_count, perplexity_scores)
            if passes:
                capital_after_fee = balance_aed - BUY_FEE_AED
                if capital_after_fee > 0:
                    price_aed   = gold_price * USD_TO_AED
                    position_oz = capital_after_fee / price_aed
                    entry_price_usd = gold_price
                    trades.append({
                        "date":            str(date.date()),
                        "action":          "BUY",
                        "price_usd":       round(gold_price, 2),
                        "price_aed":       round(price_aed, 2),
                        "fee_aed":         BUY_FEE_AED,
                        "oz_bought":       round(position_oz, 6),
                        "ensemble_pred":   round(ensemble_pred * 100, 3),
                        "base_preds":      [round(p * 100, 3) for p in base_preds],
                        "bullish_count":   bullish_count,
                        "balance_before":  round(balance_aed, 2),
                    })
                    balance_aed = 0.0
                    logger.info(
                        f"  [BUY ] {date.date()} | ${gold_price:,.2f} | "
                        f"Ensemble: +{ensemble_pred*100:.2f}% | Consensus: {bullish_count}/3"
                    )

        # ── Update error trackers ─────────────────────────────────────────────
        actual_5d = float(target_5d.loc[date]) if not np.isnan(target_5d.loc[date]) else 0.0
        ensemble.update_errors(actual_5d, base_preds)

        # ── Daily portfolio snapshot (every 5 days for log size) ──────────────
        if live_days % 5 == 0:
            port_aed = balance_aed + (position_oz * gold_price * USD_TO_AED)
            snapshots.append({
                "date":          str(date.date()),
                "portfolio_aed": round(port_aed, 2),
                "holding_oz":    round(position_oz, 6),
                "gold_usd":      round(gold_price, 2),
                "ensemble_pred": round(ensemble_pred * 100, 3),
            })

    # ── Liquidate open position at end of backtest ────────────────────────────
    if position_oz > 0.0:
        last_price = float(gold_close.iloc[-1])
        proceeds   = position_oz * last_price * USD_TO_AED
        cost_basis = position_oz * entry_price_usd * USD_TO_AED
        pnl_aed    = proceeds - cost_basis
        trades.append({
            "date":         str(dates[-1].date()),
            "action":       "LIQUIDATE",
            "price_usd":    round(last_price, 2),
            "oz":           round(position_oz, 6),
            "proceeds_aed": round(proceeds, 2),
            "pnl_aed":      round(pnl_aed, 2),
        })
        balance_aed = proceeds
        logger.info(f"  [LIQ ] {dates[-1].date()} | ${last_price:,.2f} | {proceeds:,.2f} AED")

    buys  = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] in ("SELL", "LIQUIDATE")]
    roi   = (balance_aed - STARTING_AED) / STARTING_AED * 100

    return {
        "summary": {
            "data_period":        DATA_PERIOD,
            "total_days":         n,
            "warmup_days":        WARMUP_DAYS,
            "live_days":          live_days,
            "feature_count":      len(feature_cols),
            "macro_features":     len(macro_cols),
            "technical_features": len(tech_cols),
            "starting_aed":       STARTING_AED,
            "final_balance_aed":  round(balance_aed, 2),
            "pnl_aed":            round(balance_aed - STARTING_AED, 2),
            "roi_percent":        round(roi, 2),
            "buy_count":          len(buys),
            "sell_count":         len(sells),
            "total_trades":       len(trades),
            "model_mae":          ensemble.mae_report(),
            "perplexity_active":  oracle.enabled,
            "buy_fee_aed":        BUY_FEE_AED,
            "buy_threshold_pct":  BUY_THRESHOLD * 100,
            "stop_loss_pct":      STOP_LOSS_PCT * 100,
        },
        "trade_log":           trades,
        "snapshots_tail":      snapshots[-20:],
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    use_pplx = bool(os.getenv("PERPLEXITY_API_KEY", ""))
    result   = run_backtest(use_perplexity_live=use_pplx)
    s        = result["summary"]
    sep      = "=" * 70

    print(f"\n{sep}")
    print("  HETEROGENEOUS ENSEMBLE BACKTEST — PHYSICAL GOLD (UAE)")
    print(sep)
    print(f"  Data period       : {s['data_period']}  ({s['warmup_days']}d warm-up | {s['live_days']}d live)")
    print(f"  Features          : {s['feature_count']} total  ({s['macro_features']} macro + {s['technical_features']} technical)")
    print(f"  Perplexity oracle : {'ACTIVE' if s['perplexity_active'] else 'OFFLINE (set PERPLEXITY_API_KEY)'}")
    print(f"  Starting balance  : {s['starting_aed']:>12,.2f} AED")
    print(f"  Final balance     : {s['final_balance_aed']:>12,.2f} AED")
    print(f"  PnL               : {s['pnl_aed']:>+12,.2f} AED")
    print(f"  ROI               : {s['roi_percent']:>+12.2f}%")
    print(f"  BUY signals       : {s['buy_count']}")
    print(f"  SELL signals      : {s['sell_count']}")
    print(f"  Total trades      : {s['total_trades']}")
    print(f"  Model MAE         : RF={s['model_mae']['rf_mae']}  LGB={s['model_mae']['lgb_mae']}  SGD={s['model_mae']['sgd_mae']}")
    print(sep)

    print("\n── Trade Log ──")
    for t in result["trade_log"]:
        if t["action"] == "BUY":
            print(f"  BUY  {t['date']} | ${t['price_usd']:,.2f} | Ensemble +{t['ensemble_pred']}% | Consensus {t['bullish_count']}/3")
        elif t["action"] == "SELL":
            print(f"  SELL {t['date']} | ${t['price_usd']:,.2f} | PnL {t['pnl_aed']:+,.2f} AED | {t.get('reason','')}")
        elif t["action"] == "LIQUIDATE":
            print(f"  LIQ  {t['date']} | ${t['price_usd']:,.2f} | PnL {t['pnl_aed']:+,.2f} AED")

    print("\nFull JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
