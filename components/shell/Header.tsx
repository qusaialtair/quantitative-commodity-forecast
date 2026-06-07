"use client";

import { Circle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ApiConnectionStatus } from "@/lib/types";
import { useClock } from "@/hooks/useClock";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import DeploymentModeControl from "@/components/shell/DeploymentModeControl";

interface HeaderProps {
  apiStatus: ApiConnectionStatus;
}

export default function Header({ apiStatus }: HeaderProps) {
  const time = useClock();
  const { isHostedProduction } = useDeployment();
  const connected = apiStatus === "CONNECTED";

  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-border px-4 lg:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <span className="font-mono text-[11px] font-semibold tracking-[0.2em] text-text-primary">
          ALTAIR MK1
        </span>
        <span className="hidden text-border-strong sm:inline">//</span>
        <span className="hidden font-mono text-[10px] tracking-[0.14em] text-text-muted sm:inline">
          SYSTEM OPERATIONAL
        </span>
      </div>

      <div className="flex items-center gap-3">
        <DeploymentModeControl />

        <span
          className="hidden font-mono text-[10px] tracking-wide text-text-muted md:inline"
          suppressHydrationWarning
        >
          {time || "—"}
        </span>

        {isHostedProduction ? (
          <div className="flex items-center gap-2 border border-border-strong bg-charcoal px-2.5 py-1">
            <Circle className="h-2 w-2 fill-text-muted text-text-muted" />
            <span className="font-mono text-[9px] font-medium tracking-[0.12em] text-text-secondary uppercase">
              Static Demo Data
            </span>
          </div>
        ) : (
          <div
            className={cn(
              "flex items-center gap-2 border px-2.5 py-1",
              connected
                ? "border-positive/30 bg-positive/5"
                : "border-negative/30 bg-negative/5"
            )}
          >
            <Circle
              className={cn(
                "h-2 w-2",
                connected
                  ? "fill-positive text-positive"
                  : "fill-negative text-negative"
              )}
            />
            <span
              className={cn(
                "font-mono text-[9px] font-medium tracking-[0.12em]",
                connected ? "text-positive" : "text-negative"
              )}
            >
              BACKEND API: {connected ? "CONNECTED" : "DISCONNECTED"}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
