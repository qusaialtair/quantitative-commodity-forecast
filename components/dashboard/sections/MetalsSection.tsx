"use client";

import { useState } from "react";
import { cn, formatPct } from "@/lib/utils";
import { MOCK_SECTIONS } from "@/lib/mock-sections";
import type { MetalQuote } from "@/lib/types/sections";
import ModuleCard from "@/components/shell/ModuleCard";
import SignalBadge from "@/components/sections/SignalBadge";
import MetalPriceChart from "@/components/metals/MetalPriceChart";

function formatSpot(price: number, unit: string): string {
  const decimals = unit === "oz" ? 2 : unit === "lb" ? 2 : 2;
  return `$${price.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}/${unit}`;
}

function MetricBlock({
  label,
  value,
  tone = "text-text-primary",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="border border-border bg-charcoal-dark px-3 py-2.5">
      <p className="text-label mb-1">{label}</p>
      <p className={cn("font-mono text-sm font-semibold tabular-nums", tone)}>{value}</p>
    </div>
  );
}

export default function MetalsSection() {
  const { metals } = MOCK_SECTIONS;
  const [active, setActive] = useState<MetalQuote>(metals.instruments[0]);

  return (
    <div className="flex flex-col gap-px bg-border">
      <div className="grid grid-cols-3 gap-px bg-border sm:grid-cols-6">
        {metals.instruments.map((metal) => (
          <button
            key={metal.id}
            type="button"
            onClick={() => setActive(metal)}
            className={cn(
              "border border-border px-2 py-2 font-mono text-[9px] tracking-wide transition-colors",
              active.id === metal.id
                ? "bg-surface-raised text-text-primary"
                : "bg-charcoal text-text-muted hover:border-border-strong hover:text-text-secondary"
            )}
          >
            {metal.label.toUpperCase()}
          </button>
        ))}
      </div>

      <div>
        <div className="border-b border-border bg-charcoal-dark px-3 py-1.5 font-mono text-[9px] tracking-[0.14em] text-text-muted">
          MARKET OVERVIEW
        </div>
        <div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-3 xl:grid-cols-6">
          {metals.instruments.map((metal) => {
            const up = metal.changePct >= 0;
            return (
              <button
                key={`ov-${metal.id}`}
                type="button"
                onClick={() => setActive(metal)}
                className={cn(
                  "border border-border bg-charcoal px-3 py-3 text-left transition-colors hover:bg-surface/30",
                  active.id === metal.id && "ring-1 ring-inset ring-border-strong"
                )}
              >
                <div className="font-mono text-[9px] text-text-muted">{metal.label}</div>
                <div className="mt-1 font-mono text-sm font-semibold text-text-primary tabular-nums">
                  {formatSpot(metal.spotPrice, metal.unit)}
                </div>
                <div
                  className={cn(
                    "mt-1 font-mono text-[10px] tabular-nums",
                    up ? "text-positive" : "text-negative"
                  )}
                >
                  {formatPct(metal.changePct, true)} · {metal.ticker}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <ModuleCard title="MARKET SUMMARY" subtitle="AI-generated">
        <p className="px-4 py-3 font-sans text-[12px] leading-relaxed text-text-secondary">
          {metals.marketSummary}
        </p>
      </ModuleCard>

      <div className="grid gap-px bg-border lg:grid-cols-3">
        <ModuleCard title="SIGNAL" subtitle="BUY / HOLD / SELL">
          <div className="space-y-3 p-4">
            <div className="flex items-center justify-between">
              <span className="text-label">{active.ticker}</span>
              <SignalBadge signal={active.signal} />
            </div>
            <p className="font-sans text-[12px] leading-relaxed text-text-secondary">
              {active.summary}
            </p>
            <div className="grid grid-cols-2 gap-px bg-border">
              <MetricBlock label="HMM Regime" value={active.hmmRegime} />
              <MetricBlock
                label="Macro Score"
                value={`${active.macroScore}/100`}
                tone="text-warning"
              />
            </div>
          </div>
        </ModuleCard>

        <ModuleCard title="MACRO INTELLIGENCE" subtitle="Perplexity scores">
          <div className="space-y-3 p-4">
            <p className="font-sans text-[12px] leading-relaxed text-text-secondary">
              Oracle composite favours precious metals on easing real yields and USD softness.
              Industrial complex neutral — selective lithium recovery only.
            </p>
            <div className="space-y-2">
              {[
                { label: "Rates / Real Yields", score: 72 },
                { label: "USD Liquidity", score: 68 },
                { label: "China Demand", score: 54 },
                { label: "Geopolitical Risk", score: 81 },
              ].map((row) => (
                <div key={row.label}>
                  <div className="mb-1 flex justify-between font-mono text-[9px] text-text-muted">
                    <span>{row.label}</span>
                    <span>{row.score}</span>
                  </div>
                  <div className="h-px bg-border">
                    <div
                      className="h-px bg-text-secondary"
                      style={{ width: `${row.score}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </ModuleCard>

        <ModuleCard title="TECHNICALS" subtitle="RSI · MA-50 · MA-200">
          <div className="grid grid-cols-1 gap-px bg-border p-4 sm:grid-cols-3">
            <MetricBlock label="RSI (14)" value={active.rsi.toFixed(1)} />
            <MetricBlock label="MA-50" value={formatSpot(active.ma50, active.unit)} />
            <MetricBlock label="MA-200" value={formatSpot(active.ma200, active.unit)} />
          </div>
        </ModuleCard>
      </div>

      <ModuleCard title="PRICE CHART" subtitle={`${active.ticker} · YTD proxy`}>
        <div className="p-4">
          <MetalPriceChart
            data={active.priceHistory}
            color={active.color}
            unit={active.unit}
          />
        </div>
      </ModuleCard>

      <ModuleCard title="ANALYST CHAT" subtitle="Gemini metals analyst">
        <div className="space-y-3 p-4">
          <p className="font-sans text-[12px] leading-relaxed text-text-primary">
            {metals.analystNote}
          </p>
          <p className="border-l border-border-strong pl-3 font-mono text-[10px] leading-relaxed text-text-muted">
            Active focus: {active.label} ({active.ticker}) — {active.signal} · regime{" "}
            {active.hmmRegime}
          </p>
        </div>
      </ModuleCard>
    </div>
  );
}
