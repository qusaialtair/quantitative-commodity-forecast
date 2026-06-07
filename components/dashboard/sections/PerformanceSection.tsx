import MetricTile from "@/components/shell/MetricTile";
import ModuleCard from "@/components/shell/ModuleCard";

export default function PerformanceSection() {
  return (
    <div className="flex flex-col gap-px bg-border">
      <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-4">
        <MetricTile label="VERDICT" sub="Phase XV" />
        <MetricTile label="TOTAL RETURN" accent="positive" />
        <MetricTile label="SHARPE" />
        <MetricTile label="MAX DRAWDOWN" accent="negative" />
      </div>

      <ModuleCard title="NAV HISTORY" subtitle="phase14_nav.csv" className="min-h-[280px]" />

      <div className="grid gap-px bg-border lg:grid-cols-2">
        <ModuleCard title="STRATEGY ATTRIBUTION" subtitle="Alpha Core · Treasury Hedge" className="min-h-[240px]" />
        <ModuleCard title="TREASURY HEDGE SLEEVE" subtitle="Phase XXV · Sharia gate" className="min-h-[240px]" />
      </div>

      <ModuleCard title="ML CONVICTION GATE" subtitle="Phase XXVI walk-forward" className="min-h-[200px]" />
    </div>
  );
}
