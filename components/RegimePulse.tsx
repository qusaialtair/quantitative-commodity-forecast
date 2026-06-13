"use client";

import { cn } from "@/lib/utils";
import { useRegimePulse } from "@/hooks/useRegimePulse";
import type { CrisisTier, RegimePulsePayload } from "@/lib/api/regime-pulse";

const TIER_COLOR: Record<CrisisTier, string> = {
  NORMAL: "#22c55e",
  ELEVATED: "#eab308",
  STRESS: "#f97316",
  CRISIS: "#ef4444",
};

// Score thresholds from scripts/crisis_detector.py TIER_THRESHOLDS.
const TIER_TICKS = [0.3, 0.5, 0.7];

function volSpikeStatus(ratio: number | undefined): {
  label: string;
  color: string;
} {
  if (ratio === undefined) return { label: "—", color: "#71717a" };
  if (ratio >= 1.35) return { label: "Igniting", color: "#ef4444" };
  if (ratio >= 1.15) return { label: "Warming", color: "#eab308" };
  return { label: "Calm", color: "#22c55e" };
}

function rsiStatus(rsi: number | null | undefined): {
  label: string;
  color: string;
} {
  if (rsi === null || rsi === undefined) return { label: "—", color: "#71717a" };
  if (rsi < 30) return { label: "Oversold", color: "#ef4444" };
  if (rsi < 45) return { label: "Weak", color: "#f97316" };
  if (rsi <= 55) return { label: "Neutral", color: "#a1a1aa" };
  if (rsi <= 70) return { label: "Firm", color: "#22c55e" };
  return { label: "Overbought", color: "#eab308" };
}

function Cell({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-1.5 bg-charcoal px-4 py-3.5", className)}>
      <p className="text-label">{label}</p>
      {children}
    </div>
  );
}

function CrisisGauge({ score, tier }: { score: number; tier: CrisisTier }) {
  const color = TIER_COLOR[tier];
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2.5">
        <span className="font-mono text-xl leading-none font-semibold text-text-primary">
          {score.toFixed(3)}
        </span>
        <span
          className={cn(
            "border px-1.5 py-0.5 font-mono text-[8px] font-bold tracking-[0.14em] uppercase",
            tier === "CRISIS" && "animate-pulse"
          )}
          style={{ color, borderColor: `${color}66`, backgroundColor: `${color}1a` }}
        >
          {tier}
        </span>
      </div>
      <div className="relative h-1.5 w-full bg-border-subtle">
        <div
          className="absolute inset-y-0 left-0 transition-[width] duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
        {TIER_TICKS.map((t) => (
          <div
            key={t}
            className="absolute inset-y-0 w-px bg-border-focus"
            style={{ left: `${t * 100}%` }}
          />
        ))}
      </div>
      <p className="font-mono text-[9px] tracking-wide text-text-muted">
        ticks · elevated 0.30 / stress 0.50 / crisis 0.70
      </p>
    </div>
  );
}

function RsiBar({ rsi }: { rsi: number | null | undefined }) {
  const status = rsiStatus(rsi);
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-base leading-none font-semibold text-text-primary">
          {rsi === null || rsi === undefined ? "—" : rsi.toFixed(0)}
        </span>
        <span
          className="font-mono text-[9px] tracking-[0.12em] uppercase"
          style={{ color: status.color }}
        >
          {status.label}
        </span>
      </div>
      <div className="relative h-1 w-full bg-border-subtle">
        {rsi !== null && rsi !== undefined && (
          <div
            className="absolute top-1/2 h-2.5 w-[3px] -translate-y-1/2"
            style={{ left: `${Math.max(0, Math.min(100, rsi))}%`, backgroundColor: status.color }}
          />
        )}
        <div className="absolute inset-y-0 w-px bg-border-focus" style={{ left: "30%" }} />
        <div className="absolute inset-y-0 w-px bg-border-focus" style={{ left: "70%" }} />
      </div>
    </div>
  );
}

function Reading({
  value,
  status,
  statusColor,
  valueColor,
}: {
  value: string;
  status: string;
  statusColor: string;
  valueColor?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span
        className="font-mono text-base leading-none font-semibold"
        style={{ color: valueColor ?? "#fafafa" }}
      >
        {value}
      </span>
      <span
        className="font-mono text-[9px] tracking-[0.12em] uppercase"
        style={{ color: statusColor }}
      >
        {status}
      </span>
    </div>
  );
}

function PulseGrid({ payload }: { payload: RegimePulsePayload }) {
  const { fast, volBreaker } = payload;
  const spike = volSpikeStatus(fast.vol_spike_ratio);
  const macd = fast.macd_hist_pct;
  const drift = fast.drift_10d_pct;
  const sizingPct =
    volBreaker === null ? null : Math.round(volBreaker.sizeMultiplier * 100);

  return (
    <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-3 xl:grid-cols-7">
      <Cell label="Crisis Score" className="col-span-2 md:col-span-1 xl:col-span-2">
        <CrisisGauge score={payload.score} tier={payload.tier} />
      </Cell>

      <Cell label="Fast Vol · EWMA/21D">
        <Reading
          value={fast.vol_spike_ratio !== undefined ? `${fast.vol_spike_ratio.toFixed(2)}×` : "—"}
          status={spike.label}
          statusColor={spike.color}
        />
        {fast.ewma_vol_ann_pct !== undefined && fast.vol_21d_ann_pct !== undefined && (
          <p className="font-mono text-[9px] tracking-wide text-text-muted">
            {fast.ewma_vol_ann_pct.toFixed(0)}% fast vs {fast.vol_21d_ann_pct.toFixed(0)}% base
          </p>
        )}
      </Cell>

      <Cell label="RSI-14">
        <RsiBar rsi={fast.rsi_14} />
      </Cell>

      <Cell label="MACD Histogram">
        <Reading
          value={macd === null || macd === undefined ? "—" : `${macd > 0 ? "+" : ""}${macd.toFixed(2)}%`}
          status={
            macd === null || macd === undefined
              ? "—"
              : macd < -0.05
                ? "Bearish"
                : macd > 0.05
                  ? "Bullish"
                  : "Flat"
          }
          statusColor={
            macd === null || macd === undefined
              ? "#71717a"
              : macd < -0.05
                ? "#ef4444"
                : macd > 0.05
                  ? "#22c55e"
                  : "#a1a1aa"
          }
          valueColor={
            macd === null || macd === undefined ? "#71717a" : macd < 0 ? "#ef4444" : "#22c55e"
          }
        />
      </Cell>

      <Cell label="10D Drift">
        <Reading
          value={drift !== undefined ? `${drift > 0 ? "+" : ""}${drift.toFixed(2)}%` : "—"}
          status={drift === undefined ? "—" : drift < -3 ? "Breaking down" : drift > 3 ? "Running" : "Sideways"}
          statusColor={drift === undefined ? "#71717a" : drift < -3 ? "#ef4444" : drift > 3 ? "#22c55e" : "#a1a1aa"}
          valueColor={drift === undefined ? "#71717a" : drift < 0 ? "#ef4444" : "#22c55e"}
        />
      </Cell>

      <Cell label="Vol Breaker · Sizing">
        <Reading
          value={sizingPct === null ? "—" : `${sizingPct}%`}
          status={
            volBreaker === null ? "No data" : volBreaker.active ? "Breaker engaged" : "Full size"
          }
          statusColor={volBreaker?.active ? "#f97316" : "#22c55e"}
          valueColor={volBreaker?.active ? "#f97316" : "#fafafa"}
        />
        {volBreaker && (
          <p className="font-mono text-[9px] tracking-wide text-text-muted">
            rv {volBreaker.rvBlendAnnPct.toFixed(0)}% vs target{" "}
            {volBreaker.volTargetAnnPct.toFixed(0)}%
          </p>
        )}
      </Cell>
    </div>
  );
}

function PulseSkeleton() {
  return (
    <div
      className="grid grid-cols-2 gap-px bg-border md:grid-cols-3 xl:grid-cols-7"
      aria-hidden
    >
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className={cn(
            "space-y-2.5 bg-charcoal px-4 py-3.5",
            i === 0 && "col-span-2 md:col-span-1 xl:col-span-2"
          )}
        >
          <div className="h-2 w-20 animate-pulse bg-border-strong/40" />
          <div className="h-4 w-16 animate-pulse bg-border-strong/40" />
          <div className="h-1.5 w-full animate-pulse bg-border-strong/30" />
        </div>
      ))}
    </div>
  );
}

/**
 * Phase XXVII Regime Pulse — fast crisis/momentum dials feeding the
 * falling-knife veto and the volatility circuit breaker. Reads
 * /api/crisis (fast_metrics) + the trader summary's vol_breaker block.
 */
export default function RegimePulse() {
  const { payload, isLoading, error } = useRegimePulse();

  return (
    <section className="border border-border bg-charcoal">
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <h2 className="font-mono text-[10px] font-medium tracking-[0.14em] text-text-secondary uppercase">
          Regime Pulse
        </h2>
        <span className="font-mono text-[9px] tracking-wide text-text-muted uppercase">
          Fast vol + momentum dials · Phase XXVII
        </span>
      </header>
      {isLoading ? (
        <PulseSkeleton />
      ) : error ? (
        <p className="px-4 py-3 font-mono text-[10px] text-negative">{error}</p>
      ) : payload ? (
        <PulseGrid payload={payload} />
      ) : (
        <p className="px-4 py-3 font-mono text-[10px] text-text-muted">
          Regime data unavailable — run scripts/crisis_detector.py.
        </p>
      )}
    </section>
  );
}
