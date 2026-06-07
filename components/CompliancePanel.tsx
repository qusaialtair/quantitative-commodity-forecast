"use client";

import {
  AlertTriangle,
  ArrowRight,
  Hash,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  ToggleLeft,
  ToggleRight,
} from "lucide-react";
import { cn, formatCurrency, formatPct } from "@/lib/utils";
import type { ShariaGateState } from "@/lib/types";

export interface CompliancePanelProps {
  state: ShariaGateState;
  onToggle: (treasuryShariaCleared: boolean) => void;
  readOnly?: boolean;
}

function ClearedState({
  budgetPct,
  budgetDollar,
}: {
  budgetPct: number;
  budgetDollar: number;
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center border border-positive/25 bg-positive/10">
          <ShieldCheck className="h-5 w-5 text-positive" strokeWidth={1.5} />
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm font-bold tracking-[0.12em] text-positive uppercase">
              CLEARED_SOVEREIGN
            </span>
            <span className="border border-positive/25 bg-positive/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold tracking-widest text-positive uppercase">
              ACTIVE
            </span>
          </div>
          <p className="mt-0.5 font-mono text-[11px] text-text-muted">
            Sharia compliance verified — sovereign bond routing authorized
          </p>
        </div>
      </div>

      <div className="border border-positive/20 bg-positive/5 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-label">Defensive Budget Routing</span>
          <span className="font-mono text-[10px] tracking-widest text-text-muted uppercase">
            Cleared Route
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1 text-center">
            <p className="text-label mb-1">Allocation</p>
            <p className="font-mono text-xl font-bold text-text-primary tabular-nums">
              {formatPct(budgetPct)}
            </p>
            <p className="mt-0.5 font-mono text-xs text-text-muted tabular-nums">
              {formatCurrency(budgetDollar)}
            </p>
          </div>
          <ArrowRight className="h-4 w-4 text-positive/40" strokeWidth={1.5} />
          <div className="flex-1 text-center">
            <p className="text-label mb-1">Instrument</p>
            <p className="font-mono text-xl font-bold tracking-widest text-positive">
              TLT / IEF
            </p>
            <p className="mt-0.5 font-mono text-[10px] tracking-wide text-text-muted uppercase">
              US Treasury ETF
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function FallbackState({
  budgetPct,
  budgetDollar,
}: {
  budgetPct: number;
  budgetDollar: number;
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center border border-warning/30 bg-warning/10">
          <ShieldAlert className="h-5 w-5 text-warning" strokeWidth={1.5} />
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm font-bold tracking-[0.12em] text-warning uppercase">
              SHARIA_FALLBACK_GLD
            </span>
            <span className="border border-warning/30 bg-warning/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold tracking-widest text-warning uppercase">
              OVERRIDE ACTIVE
            </span>
          </div>
          <p className="mt-0.5 font-mono text-[11px] text-text-muted">
            Sovereign route blocked — budget rerouted to physical gold
          </p>
        </div>
      </div>

      <div className="flex items-start gap-2.5 border border-warning/25 bg-warning/5 px-3.5 py-3">
        <AlertTriangle
          className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning"
          strokeWidth={2}
        />
        <div>
          <p className="font-mono text-[10px] font-semibold tracking-wide text-warning uppercase">
            Sharia Compliance Gate — Treasury Route Blocked
          </p>
          <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-text-muted">
            Flag{" "}
            <code className="text-text-primary">TREASURY_SHARIA_CLEARED = False</code>
            . Defensive 20% budget auto-rerouted to GLD.
          </p>
          <p className="mt-2 inline-block border border-positive/25 bg-positive/5 px-2 py-1 font-mono text-[9px] font-semibold tracking-wide text-positive uppercase">
            Zero Asset Churn Invariant Maintained
          </p>
        </div>
      </div>

      <div className="border border-warning/20 bg-warning/5 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-label">Fallback Routing</span>
          <span className="font-mono text-[10px] tracking-widest text-warning uppercase">
            Auto-Override
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1 text-center">
            <p className="text-label mb-1">Allocation</p>
            <p className="font-mono text-xl font-bold text-text-primary tabular-nums">
              {formatPct(budgetPct)}
            </p>
            <p className="mt-0.5 font-mono text-xs text-text-muted tabular-nums">
              {formatCurrency(budgetDollar)}
            </p>
          </div>
          <ArrowRight className="h-4 w-4 text-warning/40" strokeWidth={1.5} />
          <div className="flex-1 text-center">
            <p className="text-label mb-1">Instrument</p>
            <p className="font-mono text-xl font-bold tracking-widest text-[#d4af37]">
              GLD
            </p>
            <p className="mt-0.5 font-mono text-[10px] tracking-wide text-text-muted uppercase">
              Physical Gold ETF
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ShariaToggle({
  value,
  onChange,
  disabled,
}: {
  value: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onChange(!value)}
      disabled={disabled}
      aria-label="Simulate TREASURY_SHARIA_CLEARED"
      className={cn(
        "group flex items-center gap-2.5 border px-3.5 py-2 transition-colors select-none",
        disabled
          ? "cursor-default opacity-70"
          : "cursor-pointer",
        value
          ? "border-positive/25 bg-positive/10 hover:bg-positive/15"
          : "border-warning/25 bg-warning/10 hover:bg-warning/15"
      )}
    >
      {value ? (
        <ToggleRight
          className="h-4 w-4 text-positive transition-transform group-hover:scale-110"
          strokeWidth={1.5}
        />
      ) : (
        <ToggleLeft
          className="h-4 w-4 text-warning transition-transform group-hover:scale-110"
          strokeWidth={1.5}
        />
      )}
      <div className="text-left">
        <p className="font-mono text-[9px] tracking-widest text-text-muted uppercase">
          Simulate TREASURY_SHARIA_CLEARED
        </p>
        <p
          className={cn(
            "font-mono text-[10px] font-semibold tracking-wide",
            value ? "text-positive" : "text-warning"
          )}
        >
          = {value ? "True" : "False"}
        </p>
      </div>
    </button>
  );
}

export default function CompliancePanel({
  state,
  onToggle,
  readOnly = false,
}: CompliancePanelProps) {
  const isCleared = state.treasuryShariaCleared;

  return (
    <div
      className={cn(
        "border bg-charcoal",
        isCleared ? "border-positive/20" : "border-warning/25"
      )}
    >
      <div className="flex items-center justify-between gap-4 border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div
            className={cn("h-6 w-px", isCleared ? "bg-positive" : "bg-warning")}
          />
          <div>
            <h2 className="font-mono text-xs font-bold tracking-[0.14em] text-text-primary uppercase">
              Sovereign Risk &amp; Sharia Gate
            </h2>
            <p className="mt-0.5 font-mono text-[10px] tracking-widest text-text-muted uppercase">
              Phase XXV Compliance Overlay
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ShariaToggle
            value={isCleared}
            onChange={onToggle}
            disabled={readOnly}
          />
          <div className="flex items-center gap-1.5 border border-border bg-ebony px-2.5 py-2">
            <RefreshCw className="h-2.5 w-2.5 text-text-muted" strokeWidth={2} />
            <span className="font-mono text-[9px] tracking-widest text-text-muted uppercase">
              Live
            </span>
          </div>
        </div>
      </div>

      <div className="p-4">
        {isCleared ? (
          <ClearedState
            budgetPct={state.defensiveBudgetPct}
            budgetDollar={state.defensiveBudgetDollar}
          />
        ) : (
          <FallbackState
            budgetPct={state.defensiveBudgetPct}
            budgetDollar={state.defensiveBudgetDollar}
          />
        )}
      </div>

      <div className="flex items-center justify-between gap-4 border-t border-border bg-ebony px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Hash className="h-3 w-3 text-text-muted" strokeWidth={1.5} />
          <span className="font-mono text-[9px] tracking-widest text-text-muted">
            CHECKSUM {state.checksum}
          </span>
        </div>
        <span className="font-mono text-[9px] text-text-muted">
          ZERO-CHURN:{" "}
          <span
            className={cn(
              "font-semibold",
              state.zeroChurn ? "text-positive" : "text-negative"
            )}
          >
            {state.zeroChurn ? "ENABLED" : "DISABLED"}
          </span>
        </span>
      </div>
    </div>
  );
}
