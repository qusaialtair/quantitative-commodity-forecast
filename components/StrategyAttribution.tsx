import dynamic from "next/dynamic";
import { cn, formatCurrency, formatPct } from "@/lib/utils";
import type { StrategyAllocation } from "@/lib/types";

const AttributionChart = dynamic(() => import("@/components/AttributionChart"), {
  ssr: false,
  loading: () => (
    <div className="min-h-[200px] animate-pulse border border-border bg-charcoal" />
  ),
});

export interface StrategyAttributionProps {
  strategies: StrategyAllocation[];
  totalEquity: number;
}

function StrategyTable({ strategies }: { strategies: StrategyAllocation[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse">
        <thead>
          <tr className="border-b border-border bg-charcoal-dark">
            {[
              "Strategy Name",
              "Allocation %",
              "Current Notional (USD)",
              "Strategy P&L Contribution",
            ].map((col) => (
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
          {strategies.map((strategy) => {
            const pnlPositive = strategy.pnlContributionUsd >= 0;

            return (
              <tr
                key={strategy.id}
                className="border-b border-border transition-colors hover:bg-surface/40"
              >
                <td className="px-3 py-3">
                  <div className="flex items-center gap-2.5">
                    <span
                      className="h-8 w-0.5 shrink-0"
                      style={{ backgroundColor: strategy.color }}
                    />
                    <div>
                      <p className="font-mono text-xs font-semibold tracking-wide text-text-primary">
                        {strategy.name}
                      </p>
                      <p className="mt-0.5 font-mono text-[9px] tracking-wide text-text-muted">
                        {strategy.instruments.join(" · ")}
                      </p>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-3 font-mono text-sm font-semibold text-text-primary tabular-nums">
                  {formatPct(strategy.allocationPct)}
                </td>
                <td className="px-3 py-3 font-mono text-sm text-text-secondary tabular-nums">
                  {formatCurrency(strategy.notionalUsd)}
                </td>
                <td className="px-3 py-3">
                  <p
                    className={cn(
                      "font-mono text-sm font-semibold tabular-nums",
                      pnlPositive ? "text-positive" : "text-negative"
                    )}
                  >
                    {formatCurrency(strategy.pnlContributionUsd, { signed: true })}
                  </p>
                  <p
                    className={cn(
                      "mt-0.5 font-mono text-[10px] tabular-nums",
                      pnlPositive ? "text-positive/80" : "text-negative/80"
                    )}
                  >
                    {formatPct(strategy.pnlContributionPct, true)} session
                  </p>
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t border-border-strong bg-ebony">
            <td className="px-3 py-2.5 font-mono text-[9px] tracking-widest text-text-muted uppercase">
              Total Book
            </td>
            <td className="px-3 py-2.5 font-mono text-xs font-semibold text-text-primary tabular-nums">
              {formatPct(
                strategies.reduce((sum, s) => sum + s.allocationPct, 0)
              )}
            </td>
            <td className="px-3 py-2.5 font-mono text-xs font-semibold text-text-primary tabular-nums">
              {formatCurrency(
                strategies.reduce((sum, s) => sum + s.notionalUsd, 0)
              )}
            </td>
            <td className="px-3 py-2.5 font-mono text-xs font-semibold text-positive tabular-nums">
              {formatCurrency(
                strategies.reduce((sum, s) => sum + s.pnlContributionUsd, 0),
                { signed: true }
              )}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function AllocationBars({ strategies }: { strategies: StrategyAllocation[] }) {
  return (
    <div className="space-y-3 border-t border-border px-3 py-3">
      {strategies.map((strategy) => (
        <div key={strategy.id}>
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="font-mono text-[9px] tracking-wide text-text-muted uppercase">
              {strategy.shortCode}
            </span>
            <span className="font-mono text-[9px] text-text-secondary tabular-nums">
              {formatPct(strategy.allocationPct)}
            </span>
          </div>
          <div className="h-px w-full bg-border">
            <div
              className="h-px transition-all duration-500"
              style={{
                width: `${strategy.allocationPct}%`,
                backgroundColor: strategy.color,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function StrategyAttribution({
  strategies,
  totalEquity,
}: StrategyAttributionProps) {
  return (
    <section className="border border-border bg-charcoal">
      <header className="flex items-center justify-between gap-4 border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-6 w-px bg-border-strong" />
          <div>
            <h2 className="font-mono text-xs font-bold tracking-[0.14em] text-text-primary uppercase">
              Strategy Attribution
            </h2>
            <p className="mt-0.5 font-mono text-[10px] tracking-widest text-text-muted uppercase">
              Sub-allocation breakdown · {strategies.length} modules
            </p>
          </div>
        </div>
        <span className="border border-border px-2 py-1 font-mono text-[9px] tracking-widest text-text-muted uppercase">
          Phase XXV
        </span>
      </header>

      <div className="grid gap-px bg-border lg:grid-cols-[1.6fr_1fr]">
        <div className="bg-charcoal">
          <StrategyTable strategies={strategies} />
          <AllocationBars strategies={strategies} />
        </div>
        <div className="border-l border-border bg-charcoal p-4">
          <p className="text-label mb-3">Allocation Split</p>
          <AttributionChart strategies={strategies} totalEquity={totalEquity} />
        </div>
      </div>
    </section>
  );
}
