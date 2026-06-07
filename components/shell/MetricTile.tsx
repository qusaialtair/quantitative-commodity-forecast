import { cn } from "@/lib/utils";

interface MetricTileProps {
  label: string;
  value?: string;
  sub?: string;
  accent?: "default" | "positive" | "negative" | "warning";
}

export default function MetricTile({
  label,
  value = "—",
  sub,
  accent = "default",
}: MetricTileProps) {
  return (
    <div className="border border-border bg-charcoal px-3 py-3">
      <div className="font-mono text-[9px] tracking-[0.14em] text-text-muted">
        {label}
      </div>
      <div
        className={cn(
          "mt-1 font-mono text-lg font-semibold tracking-tight",
          accent === "positive" && "text-positive",
          accent === "negative" && "text-negative",
          accent === "warning" && "text-warning",
          accent === "default" && "text-text-primary"
        )}
      >
        {value}
      </div>
      {sub && (
        <div className="mt-0.5 font-mono text-[9px] text-text-muted">{sub}</div>
      )}
    </div>
  );
}
