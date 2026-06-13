import type { HoldingRow, SessionPerformanceHero } from "@/lib/types";

/** Active sleeve positions — aligned with Phase XXV GLD fallback mock book ($100K). */
export const MOCK_HOLDINGS: HoldingRow[] = [
  {
    ticker: "GLD",
    assetName: "SPDR Gold Shares",
    positionType: "LONG",
    allocationPct: 20.0,
    notionalUsd: 20_000,
    livePnlUsd: 32,
    livePnlPct: 0.16,
    strategyId: "defensive-hedge",
  },
  {
    ticker: "GC=F",
    assetName: "Gold Futures (COMEX)",
    positionType: "LONG",
    allocationPct: 38.0,
    notionalUsd: 38_000,
    livePnlUsd: 612,
    livePnlPct: 1.61,
    strategyId: "alpha-core",
  },
  {
    ticker: "SI=F",
    assetName: "Silver Futures (COMEX)",
    positionType: "LONG",
    allocationPct: 28.0,
    notionalUsd: 28_000,
    livePnlUsd: 448,
    livePnlPct: 1.6,
    strategyId: "alpha-core",
  },
  {
    ticker: "IAU",
    assetName: "iShares Gold Trust",
    positionType: "LONG",
    allocationPct: 14.0,
    notionalUsd: 14_000,
    livePnlUsd: 148,
    livePnlPct: 1.06,
    strategyId: "alpha-core",
  },
];

export const MOCK_PERFORMANCE_HERO: SessionPerformanceHero = {
  strategyId: "alpha-core",
  strategyName: "Alpha Core Strategies",
  winRatePct: 68.4,
  pnlContributionUsd: 1_208,
  pnlContributionPct: 1.51,
  sessionLabel: "SESSION LEADER",
  accentColor: "#a1a1aa",
};
