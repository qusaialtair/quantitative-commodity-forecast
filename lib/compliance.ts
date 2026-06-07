import type { ShariaGateState } from "@/lib/types";

const DEFENSIVE_BUDGET_PCT = 20.0;

export function buildShariaGateState(
  treasuryShariaCleared: boolean,
  totalEquity: number
): ShariaGateState {
  const defensiveBudgetDollar = totalEquity * (DEFENSIVE_BUDGET_PCT / 100);

  if (treasuryShariaCleared) {
    return {
      treasuryShariaCleared: true,
      sovereignStatus: "CLEARED_SOVEREIGN",
      defensiveBudgetPct: DEFENSIVE_BUDGET_PCT,
      defensiveBudgetDollar,
      instrument: "TLT/IEF",
      zeroChurn: true,
      lastCheckedTimestamp: new Date().toISOString(),
      checksum: "0x7f3a9c2e",
    };
  }

  return {
    treasuryShariaCleared: false,
    sovereignStatus: "SHARIA_FALLBACK_GLD",
    defensiveBudgetPct: DEFENSIVE_BUDGET_PCT,
    defensiveBudgetDollar,
    instrument: "GLD",
    zeroChurn: true,
    warningMessage:
      "Treasury route blocked — defensive budget auto-rerouted to physical gold",
    lastCheckedTimestamp: new Date().toISOString(),
    checksum: "0x7f3a9c2e",
  };
}
