import ModuleCard from "@/components/shell/ModuleCard";

interface HomeSectionProps {
  compliancePanel?: React.ReactNode;
  strategyAttribution?: React.ReactNode;
}

export default function HomeSection({
  compliancePanel,
  strategyAttribution,
}: HomeSectionProps) {
  return (
    <div className="flex flex-col gap-px bg-border">
      {compliancePanel}
      {strategyAttribution}

      <div className="grid gap-px bg-border lg:grid-cols-2">
        <ModuleCard title="EXECUTIVE BRIEFING" subtitle="Chief of Staff" className="min-h-[180px]" />
        <ModuleCard title="PERFORMANCE HERO" subtitle="Phase XIV target" className="min-h-[180px]" />
      </div>

      <ModuleCard title="HOLDINGS" subtitle="Live prices / P&L" className="min-h-[280px]" />

      <div className="grid gap-px bg-border lg:grid-cols-2">
        <ModuleCard title="EQUITY RUNNER-UPS" subtitle="Screener radar" className="min-h-[220px]" />
        <ModuleCard title="METALS INTELLIGENCE" subtitle="HMM + Oracle" className="min-h-[220px]" />
      </div>
    </div>
  );
}
