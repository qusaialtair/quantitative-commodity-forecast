import ModuleCard from "@/components/shell/ModuleCard";
import HoldingsTable from "@/components/HoldingsTable";
import PerformanceHero from "@/components/PerformanceHero";
import EquityRunnerUps from "@/components/EquityRunnerUps";
import MetalsIntelligenceTeaser from "@/components/MetalsIntelligenceTeaser";
import RegimePulse from "@/components/RegimePulse";
import { MOCK_SECTIONS } from "@/lib/mock-sections";
import type { HoldingRow, SessionPerformanceHero } from "@/lib/types";

interface HomeSectionProps {
  compliancePanel?: React.ReactNode;
  strategyAttribution?: React.ReactNode;
  holdings: HoldingRow[];
  performanceHero: SessionPerformanceHero;
}

export default function HomeSection({
  compliancePanel,
  strategyAttribution,
  holdings,
  performanceHero,
}: HomeSectionProps) {
  return (
    <div className="flex flex-col gap-px bg-border">
      <RegimePulse />
      {compliancePanel}
      {strategyAttribution}

      <ModuleCard title="PERFORMANCE HERO" subtitle="Phase XIV target">
        <PerformanceHero hero={performanceHero} />
      </ModuleCard>

      <ModuleCard title="HOLDINGS" subtitle="Live prices / P&L">
        <HoldingsTable holdings={holdings} />
      </ModuleCard>

      <div className="grid gap-px bg-border lg:grid-cols-2">
        <ModuleCard title="EQUITY RUNNER-UPS" subtitle="Screener radar">
          <EquityRunnerUps rows={MOCK_SECTIONS.homeTeasers.equityRunnerUps} />
        </ModuleCard>
        <ModuleCard title="METALS INTELLIGENCE" subtitle="HMM + Oracle">
          <MetalsIntelligenceTeaser teaser={MOCK_SECTIONS.homeTeasers.metalsTeaser} />
        </ModuleCard>
      </div>
    </div>
  );
}
