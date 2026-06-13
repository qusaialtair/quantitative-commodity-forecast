import { buildShariaGateState } from "@/lib/compliance";
import {
  MOCK_HOLDINGS,
  MOCK_PERFORMANCE_HERO,
} from "@/lib/mock-holdings";
import { buildStrategies } from "@/lib/strategies";
import type { DashboardState, StrategyAllocation } from "@/lib/types";

const TOTAL_EQUITY = 100_000;

/** Phase XXV Sharia GLD fallback — default demo posture for public deployments. */
const compliance = buildShariaGateState(false, TOTAL_EQUITY);

const strategies: StrategyAllocation[] = buildStrategies(
  TOTAL_EQUITY,
  compliance
).map((strategy) => {
  if (strategy.id === "alpha-core") {
    return {
      ...strategy,
      pnlContributionUsd: 1_208,
      pnlContributionPct: 1.51,
    };
  }
  if (strategy.id === "defensive-hedge") {
    return {
      ...strategy,
      pnlContributionUsd: 32,
      pnlContributionPct: 0.16,
      color: "#d4af37",
    };
  }
  return strategy;
});

export const MOCK_DASHBOARD: DashboardState = {
  book: {
    totalEquity: TOTAL_EQUITY,
    grossExposurePct: 96.0,
    dailyPnlPct: 1.24,
    dailyPnlDollar: 1_240,
  },
  compliance,
  strategies,
  holdings: MOCK_HOLDINGS,
  performanceHero: MOCK_PERFORMANCE_HERO,
};
