"use client";

import { cn } from "@/lib/utils";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import type { DashboardMode } from "@/lib/config";

const MODES: DashboardMode[] = ["RECRUITER SANDBOX", "PRODUCTION AUTOMATED"];

export default function DeploymentModeControl() {
  const {
    mode,
    isHostedProduction,
    isDeploymentToggleEnabled,
    setDeploymentMode,
  } = useDeployment();

  if (isHostedProduction) {
    return (
      <div className="border border-border-strong bg-charcoal px-2.5 py-1">
        <span className="font-mono text-[9px] font-medium tracking-[0.12em] text-text-secondary uppercase">
          Demo · Recruiter Sandbox
        </span>
      </div>
    );
  }

  if (!isDeploymentToggleEnabled) {
    return null;
  }

  return (
    <div
      className="flex border border-border bg-charcoal"
      role="group"
      aria-label="Deployment mode"
    >
      {MODES.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => setDeploymentMode(option)}
          className={cn(
            "px-2 py-1 font-mono text-[8px] tracking-[0.1em] uppercase transition-colors",
            mode === option
              ? "bg-surface-raised text-text-primary"
              : "text-text-muted hover:text-text-secondary"
          )}
        >
          {option === "RECRUITER SANDBOX" ? "Sandbox" : "Live API"}
        </button>
      ))}
    </div>
  );
}
