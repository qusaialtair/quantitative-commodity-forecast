import { SNAPSHOT_API_URL } from "@/lib/config";

export type CrisisTier = "NORMAL" | "ELEVATED" | "STRESS" | "CRISIS";

/** Phase XXVII fast dials emitted by scripts/crisis_detector.py. */
export interface RegimeFastMetrics {
  vol_spike_ratio?: number;
  ewma_vol_ann_pct?: number;
  vol_21d_ann_pct?: number;
  rsi_14?: number | null;
  macd_hist_pct?: number | null;
  drift_10d_pct?: number;
}

export interface RegimePulsePayload {
  score: number;
  tier: CrisisTier;
  fast: RegimeFastMetrics;
  volBreaker: {
    active: boolean;
    sizeMultiplier: number;
    rvBlendAnnPct: number;
    volTargetAnnPct: number;
  } | null;
  generatedAt: string | null;
}

interface CrisisResponse {
  score?: number;
  tier?: string;
  fast_metrics?: RegimeFastMetrics;
  generated_at?: string;
}

interface TraderSummaryResponse {
  vol_breaker?: {
    active?: boolean;
    size_multiplier?: number;
    rv_blend_ann_pct?: number;
    vol_target_ann_pct?: number;
  };
}

function normalizeTier(raw: string | undefined): CrisisTier {
  const tier = (raw ?? "NORMAL").toUpperCase();
  if (tier === "ELEVATED" || tier === "STRESS" || tier === "CRISIS") {
    return tier;
  }
  return "NORMAL";
}

export async function fetchRegimePulse(
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<RegimePulsePayload> {
  const [crisisRes, traderRes] = await Promise.all([
    fetch(`${baseUrl}/api/crisis`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal,
    }),
    // Vol breaker block written by multi_strategy_trader; optional.
    fetch(`${baseUrl}/api/engines/multi_strategy_trader`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal,
    }).catch(() => null),
  ]);

  if (!crisisRes.ok) {
    throw new Error(`Regime pulse request failed (${crisisRes.status})`);
  }

  const crisis = (await crisisRes.json()) as CrisisResponse;
  let breaker: TraderSummaryResponse["vol_breaker"];
  if (traderRes?.ok) {
    breaker = ((await traderRes.json()) as TraderSummaryResponse).vol_breaker;
  }

  return {
    score: typeof crisis.score === "number" ? crisis.score : 0,
    tier: normalizeTier(crisis.tier),
    fast: crisis.fast_metrics ?? {},
    volBreaker: breaker
      ? {
          active: Boolean(breaker.active),
          sizeMultiplier: breaker.size_multiplier ?? 1,
          rvBlendAnnPct: breaker.rv_blend_ann_pct ?? 0,
          volTargetAnnPct: breaker.vol_target_ann_pct ?? 0,
        }
      : null,
    generatedAt: crisis.generated_at ?? null,
  };
}
