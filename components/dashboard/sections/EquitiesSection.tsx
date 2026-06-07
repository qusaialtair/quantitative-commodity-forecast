"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import MetricTile from "@/components/shell/MetricTile";
import ModuleCard from "@/components/shell/ModuleCard";

const SUB_TABS = ["Master God-View", "Global Screener", "Ticker Terminal"] as const;

export default function EquitiesSection() {
  const [subTab, setSubTab] = useState<(typeof SUB_TABS)[number]>("Master God-View");

  return (
    <div className="flex flex-col gap-px bg-border">
      {/* Sub-tabs from 2_Equity.py */}
      <div className="flex border-b border-border bg-charcoal-dark">
        {SUB_TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setSubTab(tab)}
            className={cn(
              "px-3 py-2 font-mono text-[9px] tracking-[0.1em]",
              subTab === tab
                ? "border-b border-text-primary text-text-primary"
                : "text-text-muted hover:text-text-secondary"
            )}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Vault KPI row — 5 columns */}
      <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-3 xl:grid-cols-5">
        <MetricTile label="TOTAL EQUITY" />
        <MetricTile label="INVESTED" />
        <MetricTile label="CASH / SUKUK" accent="warning" />
        <MetricTile label="LIFETIME P&L" accent="positive" />
        <MetricTile label="POSITIONS" />
      </div>

      {subTab === "Master God-View" && (
        <>
          <div className="grid gap-px bg-border lg:grid-cols-[1.1fr_1.1fr_1.4fr]">
            <ModuleCard title="REGIME GAUGE" subtitle="VIX" className="min-h-[220px]" />
            <ModuleCard title="EXPOSURE DONUT" subtitle="Sector weights" className="min-h-[220px]" />
            <ModuleCard title="MACRO SNAPSHOT" subtitle="Decision context" className="min-h-[220px]" />
          </div>
          <div className="grid gap-px bg-border lg:grid-cols-[1.3fr_1.7fr]">
            <ModuleCard title="SECTOR HEATMAP" className="min-h-[260px]" />
            <ModuleCard title="TOP MOVERS" subtitle="Conviction ranked" className="min-h-[260px]" />
          </div>
        </>
      )}

      {subTab === "Global Screener" && (
        <ModuleCard
          title="GLOBAL SCREENER"
          subtitle="Sharia gate · sector filter · AI score"
          className="min-h-[400px]"
        />
      )}

      {subTab === "Ticker Terminal" && (
        <div className="grid gap-px bg-border lg:grid-cols-[1fr_4fr]">
          <ModuleCard title="TICKER SELECT" className="min-h-[400px]" />
          <div className="flex flex-col gap-px bg-border">
            <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-4">
              <MetricTile label="PRICE" />
              <MetricTile label="CHANGE" />
              <MetricTile label="AI SCORE" />
              <MetricTile label="SHARIA" accent="positive" />
            </div>
            <div className="grid gap-px bg-border lg:grid-cols-[1.9fr_1.1fr]">
              <ModuleCard title="PRICE CHART" className="min-h-[300px]" />
              <ModuleCard title="FUNDAMENTAL RADAR" className="min-h-[300px]" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
