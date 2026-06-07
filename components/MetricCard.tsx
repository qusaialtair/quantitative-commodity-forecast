import { TrendingDown, TrendingUp, type LucideIcon } from "lucide-react";
import { cn, formatPct } from "@/lib/utils";

export type MetricVariant = "default" | "positive" | "negative" | "warning" | "danger";

export interface MetricCardProps {
  label: string;
  sublabel?: string;
  value: string;
  subvalue?: string;
  delta?: string;
  variant?: MetricVariant;
  icon?: LucideIcon;
  badge?: string;
  progressPct?: number;
  progressWarningThreshold?: number;
  progressDangerThreshold?: number;
  footnote?: string;
}

const VARIANT_STYLES: Record<
  MetricVariant,
  { border: string; valueColor: string }
> = {
  default: { border: "border-border", valueColor: "text-text-primary" },
  positive: { border: "border-positive/25", valueColor: "text-positive" },
  negative: { border: "border-negative/25", valueColor: "text-negative" },
  warning: { border: "border-warning/30", valueColor: "text-warning" },
  danger: { border: "border-negative/40", valueColor: "text-negative" },
};

function ExposureBar({
  pct,
  warningAt,
  dangerAt,
}: {
  pct: number;
  warningAt: number;
  dangerAt: number;
}) {
  const clamped = Math.min(100, Math.max(0, pct));
  const overLimit = pct > 100;
  const barColor = overLimit
    ? "bg-negative"
    : clamped >= dangerAt
      ? "bg-negative"
      : clamped >= warningAt
        ? "bg-warning"
        : "bg-border-strong";

  return (
    <div className="mt-3 h-px w-full bg-border">
      <div
        className={cn("h-px transition-all duration-500", barColor)}
        style={{ width: `${Math.min(clamped, 100)}%` }}
      />
    </div>
  );
}

export default function MetricCard({
  label,
  sublabel,
  value,
  subvalue,
  delta,
  variant = "default",
  icon: Icon,
  badge,
  progressPct,
  progressWarningThreshold = 90,
  progressDangerThreshold = 95,
  footnote,
}: MetricCardProps) {
  const styles = VARIANT_STYLES[variant];

  return (
    <div
      className={cn(
        "flex flex-col border bg-charcoal px-4 py-3",
        styles.border
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className="text-label">{label}</span>
          {sublabel && (
            <p className="mt-0.5 font-mono text-[9px] tracking-wide text-text-muted">
              {sublabel}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {badge && (
            <span
              className={cn(
                "border px-1.5 py-0.5 font-mono text-[9px] tracking-widest uppercase",
                variant === "danger" || variant === "warning"
                  ? "border-warning/30 text-warning"
                  : "border-border text-text-muted"
              )}
            >
              {badge}
            </span>
          )}
          {Icon && (
            <Icon className={cn("h-3.5 w-3.5", styles.valueColor)} strokeWidth={1.5} />
          )}
        </div>
      </div>

      <div className="mt-2 flex items-end gap-2">
        <span
          className={cn(
            "font-mono text-2xl font-semibold tabular-nums tracking-tight",
            styles.valueColor
          )}
        >
          {value}
        </span>
        {delta && (
          <span
            className={cn(
              "mb-0.5 inline-flex items-center gap-0.5 font-mono text-[10px] font-medium",
              variant === "positive" && "text-positive",
              variant === "negative" && "text-negative",
              variant === "default" && "text-text-muted"
            )}
          >
            {variant === "positive" && (
              <TrendingUp className="h-2.5 w-2.5" strokeWidth={2.5} />
            )}
            {variant === "negative" && (
              <TrendingDown className="h-2.5 w-2.5" strokeWidth={2.5} />
            )}
            {delta}
          </span>
        )}
      </div>

      {subvalue && (
        <span className="mt-0.5 font-mono text-xs tabular-nums text-text-secondary">
          {subvalue}
        </span>
      )}

      {progressPct !== undefined && (
        <ExposureBar
          pct={progressPct}
          warningAt={progressWarningThreshold}
          dangerAt={progressDangerThreshold}
        />
      )}

      {footnote && (
        <p className="mt-2 border-t border-border pt-2 font-mono text-[9px] leading-relaxed text-text-muted">
          {footnote}
        </p>
      )}
    </div>
  );
}

export function exposureVariant(pct: number): MetricVariant {
  if (pct > 100) return "danger";
  if (pct >= 95) return "warning";
  return "default";
}

export function exposureBadge(pct: number): string | undefined {
  if (pct > 100) return "OVER LIMIT";
  if (pct >= 95) return "NEAR LIMIT";
  return undefined;
}

export function exposureFootnote(pct: number): string | undefined {
  if (pct > 100) {
    return `Gross exposure at ${formatPct(pct)} — exceeds 100% hard limit.`;
  }
  if (pct >= 95) {
    return `Approaching maximum gross exposure limit of 100%.`;
  }
  return undefined;
}
