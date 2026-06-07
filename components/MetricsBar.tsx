import { Activity, BarChart3, Wallet } from "lucide-react";
import MetricCard, {
  exposureBadge,
  exposureFootnote,
  exposureVariant,
} from "@/components/MetricCard";
import { formatCurrency, formatPct } from "@/lib/utils";
import type { BookMetrics } from "@/lib/types";

interface MetricsBarProps {
  metrics: BookMetrics;
}

export default function MetricsBar({ metrics }: MetricsBarProps) {
  const exposureVar = exposureVariant(metrics.grossExposurePct);
  const isPositivePnl = metrics.dailyPnlPct >= 0;

  return (
    <div className="grid grid-cols-1 gap-px bg-border md:grid-cols-3">
      <MetricCard
        label="Total Book Equity"
        value={formatCurrency(metrics.totalEquity)}
        subvalue="Mark-to-market NAV"
        icon={Wallet}
        variant="default"
      />
      <MetricCard
        label="Gross Exposure"
        value={formatPct(metrics.grossExposurePct)}
        variant={exposureVar}
        icon={Activity}
        badge={exposureBadge(metrics.grossExposurePct)}
        progressPct={metrics.grossExposurePct}
        progressWarningThreshold={90}
        progressDangerThreshold={95}
        footnote={exposureFootnote(metrics.grossExposurePct)}
      />
      <MetricCard
        label="Daily P&L"
        value={formatPct(metrics.dailyPnlPct, true)}
        subvalue={formatCurrency(metrics.dailyPnlDollar)}
        delta={formatPct(metrics.dailyPnlPct, true)}
        variant={isPositivePnl ? "positive" : "negative"}
        icon={BarChart3}
      />
    </div>
  );
}
