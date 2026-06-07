import { buildShariaGateState } from "@/lib/compliance";
import { buildStrategies } from "@/lib/strategies";
import type { DashboardState } from "@/lib/types";

const TOTAL_EQUITY = 100_000;
const compliance = buildShariaGateState(true, TOTAL_EQUITY);

export const MOCK_DASHBOARD: DashboardState = {
  book: {
    totalEquity: TOTAL_EQUITY,
    grossExposurePct: 96.0,
    dailyPnlPct: 1.24,
    dailyPnlDollar: 1_240,
  },
  compliance,
  strategies: buildStrategies(TOTAL_EQUITY, compliance),
};
