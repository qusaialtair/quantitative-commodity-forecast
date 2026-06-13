import { cn, formatCurrency, formatPct } from "@/lib/utils";
import type { HoldingRow, PositionType } from "@/lib/types";

export interface HoldingsTableProps {
  holdings: HoldingRow[];
}

const POSITION_STYLES: Record<
  PositionType,
  { label: string; className: string }
> = {
  LONG: {
    label: "Long",
    className: "border-positive/25 bg-positive/10 text-positive",
  },
  SHORT: {
    label: "Short",
    className: "border-negative/25 bg-negative/10 text-negative",
  },
  FLAT: {
    label: "Flat",
    className: "border-border-strong bg-charcoal-dark text-text-muted",
  },
};

function PositionBadge({ type }: { type: PositionType }) {
  const style = POSITION_STYLES[type];
  return (
    <span
      className={cn(
        "inline-block border px-1.5 py-0.5 font-mono text-[9px] font-semibold tracking-[0.12em] uppercase",
        style.className
      )}
    >
      {style.label}
    </span>
  );
}

export default function HoldingsTable({ holdings }: HoldingsTableProps) {
  const totalNotional = holdings.reduce((sum, row) => sum + row.notionalUsd, 0);
  const totalPnl = holdings.reduce((sum, row) => sum + row.livePnlUsd, 0);
  const totalAllocation = holdings.reduce(
    (sum, row) => sum + row.allocationPct,
    0
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse">
        <thead>
          <tr className="border-b border-border bg-charcoal-dark">
            {[
              "Ticker / Asset",
              "Position Type",
              "Allocation %",
              "Notional Value",
              "Live P&L",
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
          {holdings.map((row) => {
            const pnlPositive = row.livePnlUsd >= 0;

            return (
              <tr
                key={row.ticker}
                className="border-b border-border transition-colors hover:bg-surface/40"
              >
                <td className="px-3 py-3">
                  <p className="font-mono text-xs font-semibold tracking-wide text-text-primary">
                    {row.ticker}
                  </p>
                  <p className="mt-0.5 font-mono text-[9px] tracking-wide text-text-muted">
                    {row.assetName}
                  </p>
                </td>
                <td className="px-3 py-3">
                  <PositionBadge type={row.positionType} />
                </td>
                <td className="px-3 py-3 font-mono text-sm font-semibold text-text-primary tabular-nums">
                  {formatPct(row.allocationPct)}
                </td>
                <td className="px-3 py-3 font-mono text-sm text-text-secondary tabular-nums">
                  {formatCurrency(row.notionalUsd)}
                </td>
                <td className="px-3 py-3">
                  <p
                    className={cn(
                      "font-mono text-sm font-semibold tabular-nums",
                      pnlPositive ? "text-positive" : "text-negative"
                    )}
                  >
                    {formatCurrency(row.livePnlUsd, { signed: true })}
                  </p>
                  <p
                    className={cn(
                      "mt-0.5 font-mono text-[10px] tabular-nums",
                      pnlPositive ? "text-positive/80" : "text-negative/80"
                    )}
                  >
                    {formatPct(row.livePnlPct, true)} session
                  </p>
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t border-border-strong bg-ebony">
            <td className="px-3 py-2.5 font-mono text-[9px] tracking-widest text-text-muted uppercase">
              Active Sleeve
            </td>
            <td className="px-3 py-2.5" />
            <td className="px-3 py-2.5 font-mono text-xs font-semibold text-text-primary tabular-nums">
              {formatPct(totalAllocation)}
            </td>
            <td className="px-3 py-2.5 font-mono text-xs font-semibold text-text-primary tabular-nums">
              {formatCurrency(totalNotional)}
            </td>
            <td
              className={cn(
                "px-3 py-2.5 font-mono text-xs font-semibold tabular-nums",
                totalPnl >= 0 ? "text-positive" : "text-negative"
              )}
            >
              {formatCurrency(totalPnl, { signed: true })}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
