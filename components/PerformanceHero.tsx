import { TrendingUp } from "lucide-react";
import { cn, formatCurrency, formatPct } from "@/lib/utils";
import type { SessionPerformanceHero } from "@/lib/types";

export interface PerformanceHeroProps {
  hero: SessionPerformanceHero;
}

export default function PerformanceHero({ hero }: PerformanceHeroProps) {
  const pnlPositive = hero.pnlContributionUsd >= 0;
  const accent = hero.accentColor ?? "#a1a1aa";

  return (
    <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-stretch">
      <div
        className="hidden w-0.5 shrink-0 sm:block"
        style={{ backgroundColor: accent }}
      />

      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div
              className="flex h-10 w-10 shrink-0 items-center justify-center border"
              style={{
                borderColor: `${accent}40`,
                backgroundColor: `${accent}14`,
              }}
            >
              <TrendingUp
                className="h-4 w-4"
                style={{ color: accent }}
                strokeWidth={1.5}
              />
            </div>
            <div>
              <p className="text-label">Best Performing Strategy of the Session</p>
              <h3 className="mt-1 font-mono text-sm font-bold tracking-[0.1em] text-text-primary uppercase">
                {hero.strategyName}
              </h3>
              <p className="mt-0.5 font-mono text-[9px] tracking-widest text-text-muted uppercase">
                {hero.strategyId.replace(/-/g, " ")}
              </p>
            </div>
          </div>
          {hero.sessionLabel && (
            <span className="border border-border-strong bg-ebony px-2 py-1 font-mono text-[9px] font-semibold tracking-[0.14em] text-text-secondary uppercase">
              {hero.sessionLabel}
            </span>
          )}
        </div>

        <div className="grid gap-px bg-border sm:grid-cols-3">
          {[
            {
              label: "Win Rate",
              value: formatPct(hero.winRatePct),
              tone: "text-text-primary",
            },
            {
              label: "Isolated P&L",
              value: formatCurrency(hero.pnlContributionUsd, { signed: true }),
              tone: pnlPositive ? "text-positive" : "text-negative",
            },
            {
              label: "Book Contribution",
              value: formatPct(hero.pnlContributionPct, true),
              tone: pnlPositive ? "text-positive" : "text-negative",
            },
          ].map((metric) => (
            <div
              key={metric.label}
              className="border border-border bg-charcoal-dark px-3 py-3"
            >
              <p className="text-label mb-1.5">{metric.label}</p>
              <p
                className={cn(
                  "font-mono text-lg font-bold tabular-nums tracking-tight",
                  metric.tone
                )}
              >
                {metric.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
