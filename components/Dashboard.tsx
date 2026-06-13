"use client";

import { useCallback, useState } from "react";
import { buildShariaGateState } from "@/lib/compliance";
import { buildStrategies } from "@/lib/strategies";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { useDashboardSnapshot } from "@/hooks/useDashboardSnapshot";
import type { DashboardState, ShariaGateState } from "@/lib/types";
import DashboardShell from "@/components/dashboard/DashboardShell";
import MetricsBar from "@/components/MetricsBar";
import CompliancePanel from "@/components/CompliancePanel";
import StrategyAttribution from "@/components/StrategyAttribution";
import ExecutiveBriefing from "@/components/ExecutiveBriefing";

interface DashboardProps {
  initial: DashboardState;
}

export default function Dashboard({ initial }: DashboardProps) {
  const { isSandbox } = useDeployment();
  const { data, apiStatus } = useDashboardSnapshot(initial);
  const [sandboxCompliance, setSandboxCompliance] =
    useState<ShariaGateState | null>(null);

  const book = isSandbox ? initial.book : data.book;
  const compliance =
    sandboxCompliance ?? (isSandbox ? initial.compliance : data.compliance);
  const strategies = buildStrategies(book.totalEquity, compliance);
  const holdings = isSandbox ? initial.holdings : data.holdings;
  const performanceHero = isSandbox
    ? initial.performanceHero
    : data.performanceHero;

  const handleShariaToggle = useCallback(
    (treasuryShariaCleared: boolean) => {
      if (!isSandbox) return;
      setSandboxCompliance(
        buildShariaGateState(treasuryShariaCleared, book.totalEquity)
      );
    },
    [isSandbox, book.totalEquity]
  );

  return (
    <DashboardShell
      apiStatus={apiStatus}
      executiveBriefing={<ExecutiveBriefing />}
      metricsBar={<MetricsBar metrics={book} />}
      compliancePanel={
        <CompliancePanel
          state={compliance}
          onToggle={handleShariaToggle}
          readOnly={!isSandbox}
        />
      }
      strategyAttribution={
        <StrategyAttribution strategies={strategies} totalEquity={book.totalEquity} />
      }
      holdings={holdings}
      performanceHero={performanceHero}
    />
  );
}
