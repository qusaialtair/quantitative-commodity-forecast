"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn, formatCurrency, formatPct } from "@/lib/utils";
import { MOCK_SECTIONS } from "@/lib/mock-sections";
import MetricTile from "@/components/shell/MetricTile";
import ModuleCard from "@/components/shell/ModuleCard";

export default function PerformanceSection() {
  const { performance } = MOCK_SECTIONS;

  return (
    <div className="flex flex-col gap-px bg-border">
      <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-4">
        <MetricTile label="VERDICT" value={performance.verdict} sub="Phase XV" />
        <MetricTile label="TOTAL RETURN" value={performance.totalReturn} accent="positive" />
        <MetricTile label="SHARPE" value={performance.sharpe} />
        <MetricTile label="MAX DRAWDOWN" value={performance.maxDrawdown} accent="negative" />
      </div>

      <ModuleCard title="NAV HISTORY" subtitle="phase14_nav.csv">
        <div className="h-[240px] w-full min-w-0 p-4">
          <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
            <LineChart data={performance.navHistory} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="2 4" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#71717a", fontSize: 9, fontFamily: "var(--font-mono)" }}
                axisLine={{ stroke: "#27272a" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#71717a", fontSize: 9, fontFamily: "var(--font-mono)" }}
                axisLine={false}
                tickLine={false}
                width={56}
                tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
              />
              <Tooltip
                contentStyle={{
                  background: "#09090b",
                  border: "1px solid #3f3f46",
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                }}
                formatter={(value) => [formatCurrency(Number(value)), "NAV"]}
              />
              <Line
                type="monotone"
                dataKey="nav"
                stroke="#22c55e"
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ModuleCard>

      <div className="grid gap-px bg-border lg:grid-cols-2">
        <ModuleCard title="STRATEGY ATTRIBUTION" subtitle="Alpha Core · Treasury Hedge">
          <div className="space-y-3 p-4">
            {performance.strategyAttribution.map((row) => (
              <div key={row.name}>
                <div className="mb-1 flex justify-between font-mono text-[10px]">
                  <span className="text-text-secondary">{row.name}</span>
                  <span className="text-positive tabular-nums">
                    {formatPct(row.contributionPct, true)}
                  </span>
                </div>
                <div className="h-px bg-border">
                  <div
                    className="h-px"
                    style={{
                      width: `${Math.min(row.contributionPct * 40, 100)}%`,
                      backgroundColor: row.color,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </ModuleCard>
        <ModuleCard title="TREASURY HEDGE SLEEVE" subtitle="Phase XXV · Sharia gate">
          <div className="grid grid-cols-2 gap-px bg-border p-4">
            <Stat label="Instrument" value={performance.treasuryHedge.instrument} />
            <Stat
              label="Allocation"
              value={formatPct(performance.treasuryHedge.allocationPct)}
            />
            <Stat label="Status" value={performance.treasuryHedge.status} className="text-warning" />
            <Stat
              label="Session P&L"
              value={formatCurrency(performance.treasuryHedge.pnlUsd, { signed: true })}
              className="text-positive"
            />
          </div>
        </ModuleCard>
      </div>

      <ModuleCard title="ML CONVICTION GATE" subtitle="Phase XXVI walk-forward">
        <div className="grid grid-cols-2 gap-px bg-border p-4 md:grid-cols-4">
          <Stat label="Gate" value={performance.mlConviction.gate} className="text-positive" />
          <Stat label="WF Sharpe" value={performance.mlConviction.walkForwardSharpe.toFixed(2)} />
          <Stat
            label="OOS Win Rate"
            value={formatPct(performance.mlConviction.oosWinRate)}
          />
          <Stat label="Last Retrain" value={performance.mlConviction.lastRetrain} />
        </div>
      </ModuleCard>
    </div>
  );
}

function Stat({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="border border-border bg-charcoal-dark px-3 py-2.5">
      <p className="text-label mb-1">{label}</p>
      <p className={cn("font-mono text-sm font-semibold tabular-nums text-text-primary", className)}>
        {value}
      </p>
    </div>
  );
}
