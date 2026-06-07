import { buildShariaGateState } from "@/lib/compliance";
import { buildStrategies } from "@/lib/strategies";
import type {
  BookMetrics,
  DashboardState,
  ShariaGateState,
  StrategyAllocation,
} from "@/lib/types";

/** Raw payload from GET /api/snapshot (snake_case backend keys). */
export interface StrategySnapshotRow {
  name: string;
  allocation_pct: number;
  notional_usd: number;
  pnl_contribution_usd: number;
  pnl_contribution_pct: number;
  instruments: string[];
  color?: string;
}

export interface ApiSnapshot {
  total_equity: number;
  gross_exposure: number;
  daily_pnl: number;
  daily_pnl_usd?: number;
  treasury_sharia_cleared: boolean;
  by_strategy: {
    alpha_core: StrategySnapshotRow;
    defensive_hedge: StrategySnapshotRow;
  };
  generated_at: string;
  checksum?: string;
}

export class SnapshotFetchError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "SnapshotFetchError";
  }
}

function mapStrategyRow(
  id: string,
  row: StrategySnapshotRow,
  fallbackColor: string
): StrategyAllocation {
  return {
    id,
    name: row.name,
    shortCode: id === "alpha_core" ? "AC" : row.instruments[0] === "GLD" ? "GLD" : "DTH",
    allocationPct: row.allocation_pct,
    notionalUsd: row.notional_usd,
    pnlContributionUsd: row.pnl_contribution_usd,
    pnlContributionPct: row.pnl_contribution_pct,
    instruments: row.instruments,
    color: row.color ?? fallbackColor,
  };
}

export function mapSnapshotToDashboard(snapshot: ApiSnapshot): DashboardState {
  const book: BookMetrics = {
    totalEquity: snapshot.total_equity,
    grossExposurePct: snapshot.gross_exposure,
    dailyPnlPct: snapshot.daily_pnl,
    dailyPnlDollar:
      snapshot.daily_pnl_usd ??
      snapshot.total_equity * (snapshot.daily_pnl / 100),
  };

  const compliance: ShariaGateState = {
    ...buildShariaGateState(
      snapshot.treasury_sharia_cleared,
      snapshot.total_equity
    ),
    lastCheckedTimestamp: snapshot.generated_at,
    checksum: snapshot.checksum ?? "0x7f3a9c2e",
  };

  const strategies: StrategyAllocation[] = [
    mapStrategyRow(
      "alpha-core",
      snapshot.by_strategy.alpha_core,
      "#a1a1aa"
    ),
    mapStrategyRow(
      "defensive-hedge",
      snapshot.by_strategy.defensive_hedge,
      snapshot.treasury_sharia_cleared ? "#71717a" : "#d4af37"
    ),
  ];

  return { book, compliance, strategies };
}

export async function fetchSnapshot(
  baseUrl: string,
  signal?: AbortSignal
): Promise<ApiSnapshot> {
  const response = await fetch(`${baseUrl}/api/snapshot`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new SnapshotFetchError(
      `Snapshot request failed (${response.status})`,
      response.status
    );
  }

  return response.json() as Promise<ApiSnapshot>;
}
