"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useExecutiveSummary } from "@/hooks/useExecutiveSummary";
import {
  postOverride,
  type OverrideAction,
} from "@/lib/api/override";

type ButtonPhase = "idle" | "loading" | "confirmed";

const CONFIRMED_MS = 3_000;
const SANDBOX_TOAST_MS = 3_000;

function BriefingSkeleton() {
  return (
    <div className="space-y-2.5" aria-hidden>
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-2 animate-pulse bg-border-strong/40"
          style={{ width: `${88 - i * 8}%` }}
        />
      ))}
    </div>
  );
}

function formatBriefingText(text: string): string {
  return text.replace(/\*\*/g, "").trim();
}

function buttonLabel(
  action: OverrideAction,
  phase: ButtonPhase
): string {
  if (phase === "loading") return "Processing…";
  if (phase === "confirmed") return "Confirmed";
  return action === "AUTHORIZE"
    ? "Authorize Pipeline Execution"
    : "Emergency Halt // Liquidate to Cash";
}

export default function ExecutiveBriefing() {
  const { summary, generatedAt, isLoading, error, isSandbox } =
    useExecutiveSummary();
  const [authorizePhase, setAuthorizePhase] = useState<ButtonPhase>("idle");
  const [haltPhase, setHaltPhase] = useState<ButtonPhase>("idle");
  const [sandboxToast, setSandboxToast] = useState(false);
  const [overrideError, setOverrideError] = useState<string | null>(null);
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleReset = useCallback((action: OverrideAction) => {
    if (resetTimerRef.current) clearTimeout(resetTimerRef.current);
    resetTimerRef.current = setTimeout(() => {
      if (action === "AUTHORIZE") setAuthorizePhase("idle");
      else setHaltPhase("idle");
    }, CONFIRMED_MS);
  }, []);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current) clearTimeout(resetTimerRef.current);
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    };
  }, []);

  const runOverride = useCallback(
    async (action: OverrideAction) => {
      setOverrideError(null);
      const setPhase =
        action === "AUTHORIZE" ? setAuthorizePhase : setHaltPhase;

      if (isSandbox) {
        setSandboxToast(true);
        if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
        toastTimerRef.current = setTimeout(
          () => setSandboxToast(false),
          SANDBOX_TOAST_MS
        );
        console.log("[ALTAIR MK1] SANDBOX override simulated", { action });
        return;
      }

      setPhase("loading");
      try {
        await postOverride(action);
        setPhase("confirmed");
        scheduleReset(action);
      } catch (err) {
        setPhase("idle");
        setOverrideError(
          err instanceof Error ? err.message : "Override command failed"
        );
      }
    },
    [isSandbox, scheduleReset]
  );

  const anyLoading = authorizePhase === "loading" || haltPhase === "loading";

  return (
    <section className="relative border border-border bg-charcoal">
      {sandboxToast && (
        <div className="absolute top-3 right-3 z-10 border border-border-strong bg-ebony px-3 py-1.5 font-mono text-[9px] font-semibold tracking-[0.12em] text-text-primary uppercase shadow-lg">
          Sandbox: Action simulated successfully.
        </div>
      )}

      <header className="flex items-center justify-between gap-4 border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-6 w-px bg-text-primary" />
          <div>
            <h2 className="font-mono text-xs font-bold tracking-[0.16em] text-text-primary uppercase">
              Executive Briefing // DeepSeek Systems
            </h2>
            <p className="mt-0.5 font-mono text-[10px] tracking-widest text-text-muted uppercase">
              Dumb Mode · Manual Override Surface
            </p>
          </div>
        </div>
        {generatedAt && !isLoading && (
          <span className="font-mono text-[9px] tracking-wide text-text-muted">
            {generatedAt.slice(0, 10)}
          </span>
        )}
      </header>

      <div className="border-b border-border bg-ebony-alt px-4 py-4">
        {isLoading ? (
          <BriefingSkeleton />
        ) : error ? (
          <p className="font-mono text-[11px] leading-relaxed text-negative">
            {error}
          </p>
        ) : summary ? (
          <p className="font-sans text-[13px] leading-[1.75] text-text-primary">
            {formatBriefingText(summary)}
          </p>
        ) : (
          <p className="font-mono text-[11px] text-text-muted">
            No executive summary available. Run executive_briefer.py or wait
            for the next pipeline cycle.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-px bg-border sm:flex-row">
        <button
          type="button"
          disabled={anyLoading}
          onClick={() => void runOverride("AUTHORIZE")}
          className={cn(
            "flex flex-1 items-center justify-center gap-2 border px-4 py-3 font-mono text-[10px] font-semibold tracking-[0.12em] uppercase transition-colors disabled:cursor-not-allowed disabled:opacity-60",
            authorizePhase === "confirmed"
              ? "border-positive/40 bg-positive/20 text-positive"
              : "border-positive/30 bg-positive/10 text-positive hover:bg-positive/15"
          )}
        >
          {authorizePhase === "loading" && (
            <Loader2 className="h-3 w-3 animate-spin" strokeWidth={2} />
          )}
          {buttonLabel("AUTHORIZE", authorizePhase)}
        </button>
        <button
          type="button"
          disabled={anyLoading}
          onClick={() => void runOverride("HALT")}
          className={cn(
            "flex flex-1 items-center justify-center gap-2 border px-4 py-3 font-mono text-[10px] font-semibold tracking-[0.12em] uppercase transition-colors disabled:cursor-not-allowed disabled:opacity-60",
            haltPhase === "confirmed"
              ? "border-negative/40 bg-negative/20 text-negative"
              : "border-negative/30 bg-negative/10 text-negative hover:bg-negative/15"
          )}
        >
          {haltPhase === "loading" && (
            <Loader2 className="h-3 w-3 animate-spin" strokeWidth={2} />
          )}
          {buttonLabel("HALT", haltPhase)}
        </button>
      </div>

      {overrideError && (
        <div className="border-t border-negative/25 bg-negative/5 px-4 py-2 font-mono text-[9px] font-semibold tracking-[0.12em] text-negative uppercase">
          {overrideError}
        </div>
      )}
    </section>
  );
}
