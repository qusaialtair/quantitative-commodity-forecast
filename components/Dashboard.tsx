"use client";

import { useCallback, useState } from "react";
import { buildShariaGateState } from "@/lib/compliance";
import { isSandboxMode } from "@/lib/config";
import { buildStrategies } from "@/lib/strategies";
import { useDashboardSnapshot } from "@/hooks/useDashboardSnapshot";
import type { DashboardState, ShariaGateState } from "@/lib/types";
import DashboardShell from "@/components/dashboard/DashboardShell";
import MetricsBar from "@/components/MetricsBar";
import CompliancePanel from "@/components/CompliancePanel";
import StrategyAttribution from "@/components/StrategyAttribution";

interface DashboardProps {
  initial: DashboardState;
}

export default function Dashboard({ initial }: DashboardProps) {
  const sandbox = isSandboxMode();
  const { data, apiStatus } = useDashboardSnapshot(initial);
  const [sandboxCompliance, setSandboxCompliance] =
    useState<ShariaGateState | null>(null);

  const compliance = sandboxCompliance ?? data.compliance;
  const strategies =
    sandboxCompliance !== null
      ? buildStrategies(data.book.totalEquity, compliance)
      : data.strategies;

  const handleShariaToggle = useCallback(
    (treasuryShariaCleared: boolean) => {
      if (!sandbox) return;
      setSandboxCompliance(
        buildShariaGateState(treasuryShariaCleared, data.book.totalEquity)
      );
    },
    [sandbox, data.book.totalEquity]
  );

  return (
    <DashboardShell
      apiStatus={apiStatus}
      metricsBar={<MetricsBar metrics={data.book} />}
      compliancePanel={
        <CompliancePanel
          state={compliance}
          onToggle={handleShariaToggle}
          readOnly={!sandbox}
        />
      }
      strategyAttribution={
        <StrategyAttribution
          strategies={strategies}
          totalEquity={data.book.totalEquity}
        />
      }
    />
  );
}
