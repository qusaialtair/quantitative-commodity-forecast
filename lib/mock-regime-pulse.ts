import type { RegimePulsePayload } from "@/lib/api/regime-pulse";

/** Mirrors the live 2026-06 STRESS readings so the sandbox demo is honest. */
export const MOCK_REGIME_PULSE: RegimePulsePayload = {
  score: 0.6122,
  tier: "STRESS",
  fast: {
    vol_spike_ratio: 1.309,
    ewma_vol_ann_pct: 32.0,
    vol_21d_ann_pct: 24.5,
    rsi_14: 36.4,
    macd_hist_pct: -0.67,
    drift_10d_pct: -5.93,
  },
  volBreaker: {
    active: true,
    sizeMultiplier: 0.517,
    rvBlendAnnPct: 42.6,
    volTargetAnnPct: 22.0,
  },
  generatedAt: "2026-06-12T06:00:00+00:00",
};
