import { cn } from "@/lib/utils";
import type { TradeSignal } from "@/lib/types/sections";

const STYLES: Record<TradeSignal, string> = {
  BUY: "border-positive/30 bg-positive/10 text-positive",
  HOLD: "border-warning/30 bg-warning/10 text-warning",
  SELL: "border-negative/30 bg-negative/10 text-negative",
};

export default function SignalBadge({ signal }: { signal: TradeSignal }) {
  return (
    <span
      className={cn(
        "inline-block border px-2 py-0.5 font-mono text-[9px] font-bold tracking-[0.14em] uppercase",
        STYLES[signal]
      )}
    >
      {signal}
    </span>
  );
}
