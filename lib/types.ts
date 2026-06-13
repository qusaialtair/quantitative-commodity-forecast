export interface BookMetrics {
  totalEquity: number;
  grossExposurePct: number;
  dailyPnlPct: number;
  dailyPnlDollar: number;
}

export type ComplianceStatus = "CLEARED_SOVEREIGN" | "SHARIA_FALLBACK_GLD";
export type DefensiveInstrument = "TLT/IEF" | "GLD";

export interface ShariaGateState {
  treasuryShariaCleared: boolean;
  sovereignStatus: ComplianceStatus;
  defensiveBudgetPct: number;
  defensiveBudgetDollar: number;
  instrument: DefensiveInstrument;
  zeroChurn: boolean;
  warningMessage?: string;
  lastCheckedTimestamp: string;
  checksum: string;
}

export interface StrategyAllocation {
  id: string;
  name: string;
  shortCode: string;
  allocationPct: number;
  notionalUsd: number;
  pnlContributionUsd: number;
  pnlContributionPct: number;
  instruments: string[];
  color: string;
}

export type ApiConnectionStatus =
  | "CONNECTED"
  | "DISCONNECTED"
  | "LOCAL_SIMULATION";

export type PositionType = "LONG" | "SHORT" | "FLAT";

export interface HoldingRow {
  ticker: string;
  assetName: string;
  positionType: PositionType;
  allocationPct: number;
  notionalUsd: number;
  livePnlUsd: number;
  livePnlPct: number;
  strategyId?: string;
}

export interface SessionPerformanceHero {
  strategyId: string;
  strategyName: string;
  winRatePct: number;
  pnlContributionUsd: number;
  pnlContributionPct: number;
  sessionLabel?: string;
  accentColor?: string;
}

export interface DashboardState {
  book: BookMetrics;
  compliance: ShariaGateState;
  strategies: StrategyAllocation[];
  holdings: HoldingRow[];
  performanceHero: SessionPerformanceHero;
}
