"use client";

import { useState } from "react";
import Header from "@/components/shell/Header";
import type {
  ApiConnectionStatus,
  HoldingRow,
  SessionPerformanceHero,
} from "@/lib/types";
import NavTabs, { type NavSection } from "@/components/shell/NavTabs";
import HomeSection from "@/components/dashboard/sections/HomeSection";
import MetalsSection from "@/components/dashboard/sections/MetalsSection";
import EquitiesSection from "@/components/dashboard/sections/EquitiesSection";
import AgentSection from "@/components/dashboard/sections/AgentSection";
import PerformanceSection from "@/components/dashboard/sections/PerformanceSection";

const SECTIONS: Record<Exclude<NavSection, "home">, React.ComponentType> = {
  metals: MetalsSection,
  equities: EquitiesSection,
  agent: AgentSection,
  performance: PerformanceSection,
};

function SectionRouter({ section }: { section: Exclude<NavSection, "home"> }) {
  const Section = SECTIONS[section];
  return <Section />;
}

interface DashboardShellProps {
  apiStatus: ApiConnectionStatus;
  executiveBriefing: React.ReactNode;
  metricsBar: React.ReactNode;
  compliancePanel: React.ReactNode;
  strategyAttribution: React.ReactNode;
  holdings: HoldingRow[];
  performanceHero: SessionPerformanceHero;
}

export default function DashboardShell({
  apiStatus,
  executiveBriefing,
  metricsBar,
  compliancePanel,
  strategyAttribution,
  holdings,
  performanceHero,
}: DashboardShellProps) {
  const [active, setActive] = useState<NavSection>("home");

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
            holdings={holdings}
            performanceHero={performanceHero}
          />
        ) : (
          <SectionRouter section={active} />
        )}
      </main>
    </div>
  );
}
