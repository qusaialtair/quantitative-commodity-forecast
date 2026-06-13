import type {
  DashboardState,
  HoldingRow,
  SessionPerformanceHero,
} from "@/lib/types";

/** Raw holding row from GET /api/snapshot or /api/holdings (snake_case). */
export interface ApiHoldingRow {
  ticker: string;
  asset_name: string;
  position_type: "LONG" | "SHORT" | "FLAT";
  allocation_pct: number;
  notional_usd: number;
  live_pnl_usd: number;
  live_pnl_pct: number;
  strategy_id?: string;
}

/** Best-performing strategy block from the live API packet. */
export interface ApiSessionPerformanceHero {
  strategy_id: string;
  strategy_name: string;
  win_rate_pct: number;
  pnl_contribution_usd: number;
  pnl_contribution_pct: number;
  session_label?: string;
  accent_color?: string;
}

export interface ApiHoldingsSection {
  holdings: ApiHoldingRow[];
  performance_hero: ApiSessionPerformanceHero;
}

export function mapApiHoldingRow(row: ApiHoldingRow): HoldingRow {
  return {
    ticker: row.ticker,
    assetName: row.asset_name,
    positionType: row.position_type,
    allocationPct: row.allocation_pct,
    notionalUsd: row.notional_usd,
    livePnlUsd: row.live_pnl_usd,
    livePnlPct: row.live_pnl_pct,
    strategyId: row.strategy_id,
  };
}

export function mapApiPerformanceHero(
  hero: ApiSessionPerformanceHero
): SessionPerformanceHero {
  return {
    strategyId: hero.strategy_id,
    strategyName: hero.strategy_name,
    winRatePct: hero.win_rate_pct,
    pnlContributionUsd: hero.pnl_contribution_usd,
    pnlContributionPct: hero.pnl_contribution_pct,
    sessionLabel: hero.session_label,
    accentColor: hero.accent_color,
  };
}

export function mapApiHoldingsSection(
  section: ApiHoldingsSection
): Pick<DashboardState, "holdings" | "performanceHero"> {
  return {
    holdings: section.holdings.map(mapApiHoldingRow),
    performanceHero: mapApiPerformanceHero(section.performance_hero),
  };
}
