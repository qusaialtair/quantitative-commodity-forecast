"use client";

import { useState } from "react";
import Header from "@/components/shell/Header";
import type { ApiConnectionStatus } from "@/lib/types";
import NavTabs, { type NavSection } from "@/components/shell/NavTabs";
import HomeSection from "@/components/dashboard/sections/HomeSection";
import MetalsSection from "@/components/dashboard/sections/MetalsSection";
import EquitiesSection from "@/components/dashboard/sections/EquitiesSection";
import AgentSection from "@/components/dashboard/sections/AgentSection";
import PerformanceSection from "@/components/dashboard/sections/PerformanceSection";

const SECTIONS: Record<NavSection, React.ComponentType> = {
  home: HomeSection,
  metals: MetalsSection,
  equities: EquitiesSection,
  agent: AgentSection,
  performance: PerformanceSection,
};

interface DashboardShellProps {
  apiStatus: ApiConnectionStatus;
  executiveBriefing: React.ReactNode;
  metricsBar: React.ReactNode;
  compliancePanel: React.ReactNode;
  strategyAttribution: React.ReactNode;
}

export default function DashboardShell({
  apiStatus,
  executiveBriefing,
  metricsBar,
  compliancePanel,
  strategyAttribution,
}: DashboardShellProps) {
  const [active, setActive] = useState<NavSection>("home");
  const ActiveSection = SECTIONS[active];

  return (
    <div className="flex min-h-screen flex-col bg-ebony">
      <Header apiStatus={apiStatus} />
      <NavTabs active={active} onChange={setActive} />
      {active === "home" && (
        <div className="border-b border-border">{executiveBriefing}</div>
      )}
      <div className="border-b border-border">{metricsBar}</div>
      <main className="flex-1 overflow-auto p-px">
        {active === "home" ? (
          <HomeSection
            compliancePanel={compliancePanel}
            strategyAttribution={strategyAttribution}
          />
        ) : (
          <ActiveSection />
        )}
      </main>
    </div>
  );
}
