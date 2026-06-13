"use client";

import { useState } from "react";
import { cn, formatPct } from "@/lib/utils";
import { MOCK_SECTIONS } from "@/lib/mock-sections";
import type { EquityMover } from "@/lib/types/sections";
import MetricTile from "@/components/shell/MetricTile";
import ModuleCard from "@/components/shell/ModuleCard";

const SUB_TABS = ["Master God-View", "Global Screener", "Ticker Terminal"] as const;

export default function EquitiesSection() {
  const [subTab, setSubTab] = useState<(typeof SUB_TABS)[number]>("Master God-View");
  const { equities } = MOCK_SECTIONS;

  return (
    <div className="flex flex-col gap-px bg-border">
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

      <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-3 xl:grid-cols-5">
        <MetricTile label="TOTAL EQUITY" value={equities.vault.totalEquity} />
        <MetricTile label="INVESTED" value={equities.vault.invested} />
        <MetricTile label="CASH / SUKUK" value={equities.vault.cashSukuk} accent="warning" />
        <MetricTile label="LIFETIME P&L" value={equities.vault.lifetimePnl} accent="positive" />
        <MetricTile label="POSITIONS" value={equities.vault.positions} />
      </div>

      {subTab === "Master God-View" && (
        <>
          <div className="grid gap-px bg-border lg:grid-cols-[1.1fr_1.1fr_1.4fr]">
            <ModuleCard title="REGIME GAUGE" subtitle="VIX">
              <div className="p-4">
                <p className="font-mono text-2xl font-bold text-positive tabular-nums">
                  {equities.regimeVix}
                </p>
                <p className="mt-2 font-sans text-[11px] text-text-muted">
                  Complacency regime — equity adds permitted under conviction gate.
                </p>
              </div>
            </ModuleCard>
            <ModuleCard title="EXPOSURE DONUT" subtitle="Sector weights">
              <div className="space-y-2 p-4">
                {equities.exposureTop.map((row) => (
                  <div key={row.sector}>
                    <div className="mb-1 flex justify-between font-mono text-[9px]">
                      <span className="text-text-muted">{row.sector}</span>
                      <span className="text-text-secondary">{row.weightPct}%</span>
                    </div>
                    <div className="h-px bg-border">
                      <div
                        className="h-px bg-text-secondary"
                        style={{ width: `${row.weightPct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </ModuleCard>
            <ModuleCard title="MACRO SNAPSHOT" subtitle="Decision context">
              <p className="p-4 font-sans text-[12px] leading-relaxed text-text-secondary">
                {equities.macroSnapshot}
              </p>
            </ModuleCard>
          </div>
          <div className="grid gap-px bg-border lg:grid-cols-[1.3fr_1.7fr]">
            <ModuleCard title="SECTOR HEATMAP">
              <div className="grid grid-cols-2 gap-px bg-border p-px sm:grid-cols-3">
                {equities.sectorHeatmap.map((row) => (
                  <div
                    key={row.sector}
                    className="border border-border bg-charcoal px-3 py-2.5"
                  >
                    <p className="text-label">{row.sector}</p>
                    <p
                      className={cn(
                        "mt-1 font-mono text-sm font-semibold tabular-nums",
                        row.returnPct >= 0 ? "text-positive" : "text-negative"
                      )}
                    >
                      {formatPct(row.returnPct, true)}
                    </p>
                  </div>
                ))}
              </div>
            </ModuleCard>
            <ModuleCard title="TOP MOVERS" subtitle="Conviction ranked">
              <MoverTable rows={equities.topMovers} />
            </ModuleCard>
          </div>
        </>
      )}

      {subTab === "Global Screener" && (
        <ModuleCard title="GLOBAL SCREENER" subtitle="Sharia gate · sector filter · AI score">
          <MoverTable rows={equities.screenerRows} />
        </ModuleCard>
      )}

      {subTab === "Ticker Terminal" && (
        <div className="grid gap-px bg-border lg:grid-cols-[1fr_4fr]">
          <ModuleCard title="TICKER SELECT">
            <div className="p-2">
              {equities.topMovers.map((row) => (
                <div
                  key={row.ticker}
                  className="border-b border-border px-2 py-2 font-mono text-[10px] text-text-secondary last:border-0"
                >
                  {row.ticker}
                  <span className="ml-2 text-text-muted">{row.name}</span>
                </div>
              ))}
            </div>
          </ModuleCard>
          <div className="flex flex-col gap-px bg-border">
            <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-4">
              <MetricTile label="PRICE" value="$128.42" />
              <MetricTile label="CHANGE" value="+2.4%" accent="positive" />
              <MetricTile label="AI SCORE" value="92" />
              <MetricTile label="SHARIA" value="CLEARED" accent="positive" />
            </div>
            <div className="grid gap-px bg-border lg:grid-cols-[1.9fr_1.1fr]">
              <ModuleCard title="PRICE CHART">
                <p className="p-4 font-mono text-[11px] text-text-muted">
                  NVDA · 30D trend +1.8σ · volume 1.2× 20D avg
                </p>
              </ModuleCard>
              <ModuleCard title="FUNDAMENTAL RADAR">
                <div className="space-y-2 p-4">
                  {[
                    ["Growth", 88],
                    ["Quality", 76],
                    ["Momentum", 91],
                    ["Value", 42],
                  ].map(([label, score]) => (
                    <div key={label as string}>
                      <div className="mb-1 flex justify-between font-mono text-[9px] text-text-muted">
                        <span>{label}</span>
                        <span>{score}</span>
                      </div>
                      <div className="h-px bg-border">
                        <div className="h-px bg-positive/60" style={{ width: `${score}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </ModuleCard>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MoverTable({ rows }: { rows: EquityMover[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse">
        <thead>
          <tr className="border-b border-border bg-charcoal-dark">
            {["Ticker", "Sector", "Change", "AI Score", "Sharia"].map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-mono text-[9px] font-medium tracking-[0.12em] text-text-muted uppercase"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.ticker} className="border-b border-border hover:bg-surface/40">
              <td className="px-3 py-2.5">
                <p className="font-mono text-xs font-semibold text-text-primary">{row.ticker}</p>
                <p className="font-mono text-[9px] text-text-muted">{row.name}</p>
              </td>
              <td className="px-3 py-2.5 font-mono text-[10px] text-text-secondary">
                {row.sector}
              </td>
              <td
                className={cn(
                  "px-3 py-2.5 font-mono text-sm font-semibold tabular-nums",
                  row.changePct >= 0 ? "text-positive" : "text-negative"
                )}
              >
                {formatPct(row.changePct, true)}
              </td>
              <td className="px-3 py-2.5 font-mono text-sm text-text-primary tabular-nums">
                {row.aiScore}
              </td>
              <td className="px-3 py-2.5 font-mono text-[9px] text-positive uppercase">
                {row.sharia ? "Cleared" : "Blocked"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
