import type { ShariaGateState, StrategyAllocation } from "@/lib/types";

const ALPHA_CORE: Omit<StrategyAllocation, "notionalUsd"> = {
  id: "alpha-core",
  name: "Alpha Core Strategies",
  shortCode: "AC",
  allocationPct: 80.0,
  pnlContributionUsd: 1_208,
  pnlContributionPct: 1.51,
  instruments: ["GC=F", "SI=F", "GLD", "IAU"],
  color: "#a1a1aa",
};

export function buildStrategies(
  totalEquity: number,
  compliance: ShariaGateState
): StrategyAllocation[] {
  const defensivePct = compliance.defensiveBudgetPct;
  const alphaPct = 100 - defensivePct;
  const defensiveInstrument = compliance.instrument;

  const defensive: StrategyAllocation = {
    id: "defensive-hedge",
    name: "Defensive Treasury/GLD Hedge",
    shortCode: defensiveInstrument === "GLD" ? "GLD" : "DTH",
    allocationPct: defensivePct,
    notionalUsd: totalEquity * (defensivePct / 100),
    pnlContributionUsd: 32,
    pnlContributionPct: 0.16,
    instruments:
      defensiveInstrument === "GLD" ? ["GLD"] : ["TLT", "IEF"],
    color: defensiveInstrument === "GLD" ? "#d4af37" : "#71717a",
  };

  return [
    {
      ...ALPHA_CORE,
      allocationPct: alphaPct,
      notionalUsd: totalEquity * (alphaPct / 100),
    },
    defensive,
  ];
}
