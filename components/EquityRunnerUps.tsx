import { cn, formatPct } from "@/lib/utils";
import type { EquityMover } from "@/lib/types/sections";

export default function EquityRunnerUps({ rows }: { rows: EquityMover[] }) {
  return (
    <div className="w-full">
      {rows.map((row) => (
        <div
          key={row.ticker}
          className="flex items-center justify-between border-b border-border px-4 py-2.5 last:border-0"
        >
          <div>
            <p className="font-mono text-xs font-semibold text-text-primary">{row.ticker}</p>
            <p className="font-mono text-[9px] text-text-muted">{row.sector}</p>
          </div>
          <div className="text-right">
            <p
              className={cn(
                "font-mono text-sm font-semibold tabular-nums",
                row.changePct >= 0 ? "text-positive" : "text-negative"
              )}
            >
              {formatPct(row.changePct, true)}
            </p>
            <p className="font-mono text-[9px] text-text-muted">AI {row.aiScore}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
