"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { useExecutiveSummary } from "@/hooks/useExecutiveSummary";
import {
  postOverride,
  type OverrideAction,
} from "@/lib/api/override";
import { fetchPipelineStatus } from "@/lib/api/pipeline-status";
import type { PipelineStatusPayload } from "@/lib/api/pipeline-status";
import {
  appendOperatorNote,
  liveOverrideResult,
  sandboxOverrideResult,
  type OperatorControlState,
} from "@/lib/operator-control";
import type { ExecutiveSummaryPayload } from "@/lib/api/executive-summary";

type ButtonPhase = "idle" | "loading" | "confirmed";

const CONFIRMED_MS = 3_000;
const PIPELINE_POLL_MS = 4_000;
const PIPELINE_POLL_MAX = 90;

function BriefingSkeleton() {
  return (
    <div className="space-y-6 py-1" aria-hidden>
      <div className="h-5 w-3/4 animate-pulse bg-border-strong/40" />
      <div className="space-y-2.5">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={`lede-${i}`}
            className="h-2 animate-pulse bg-border-strong/40"
            style={{ width: `${94 - i * 9}%` }}
          />
        ))}
      </div>
      {Array.from({ length: 3 }).map((_, block) => (
        <div key={`block-${block}`} className="space-y-2.5">
          <div className="h-2 w-24 animate-pulse bg-border-strong/30" />
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="h-2 animate-pulse bg-border-strong/40"
              style={{ width: `${90 - i * 12}%` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

function formatBriefingText(text: string): string {
  return text.replace(/\*\*/g, "").trim();
}

function buttonLabel(action: OverrideAction, phase: ButtonPhase): string {
  if (phase === "loading") return "Processing…";
  if (phase === "confirmed") return "Confirmed";
  return action === "AUTHORIZE"
    ? "Authorize Pipeline Execution"
    : "Emergency Halt // Liquidate to Cash";
}

/**
 * Newsletter layout for "The QCTF Daily".
 *
 * THE READ renders as an unboxed lede under the headline; the remaining
 * sections get labelled blocks with generous leading so the long-form
 * advisor text breathes. THE CALL is visually elevated — it carries the
 * actionable recommendation.
 */
function BriefingSections({ payload }: { payload: ExecutiveSummaryPayload }) {
  const headline = payload.headline?.trim();
  const sections = [
    { key: "holdings", label: "Positioning" },
    { key: "watchlist", label: "Watchlist" },
  ] as const;

  const hasStructure = Boolean(
    payload.market || payload.holdings || payload.watchlist || payload.action
  );

  if (!hasStructure) {
    return (
      <p className="max-w-[72ch] font-sans text-[13.5px] leading-[1.9] whitespace-pre-line text-text-primary">
        {formatBriefingText(payload.summary)}
      </p>
    );
  }

  return (
    <article className="max-w-[76ch] space-y-7">
      {headline && (
        <h3 className="font-sans text-[19px] leading-snug font-semibold tracking-tight text-text-primary">
          {headline}
        </h3>
      )}

      {payload.market && (
        <div className="space-y-2">
          <p className="text-label">The Read</p>
          <p className="font-sans text-[13.5px] leading-[1.9] text-text-primary">
            {formatBriefingText(payload.market)}
          </p>
        </div>
      )}

      {sections.map((row) => {
        const value = payload[row.key];
        if (!value) return null;
        return (
          <div
            key={row.key}
            className="space-y-2 border-l border-border-strong pl-4"
          >
            <p className="text-label">{row.label}</p>
            <p className="font-sans text-[13px] leading-[1.85] text-text-secondary">
              {formatBriefingText(value)}
            </p>
          </div>
        );
      })}

      {payload.action && (
        <div className="space-y-2 border border-border-strong/60 bg-charcoal-light/50 px-5 py-4">
          <p className="text-label !text-text-secondary">The Call</p>
          <p className="font-sans text-[13.5px] leading-[1.85] font-medium text-text-primary">
            {formatBriefingText(payload.action)}
          </p>
        </div>
      )}
    </article>
  );
}

function OperatorStatusBanner({ state }: { state: OperatorControlState }) {
  const halted = state.halted;

  return (
    <div
      className={cn(
        "border-t px-4 py-3",
        halted
          ? "border-negative/30 bg-negative/10"
          : "border-positive/30 bg-positive/10"
      )}
    >
      <div className="flex items-start gap-3">
        {halted ? (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-negative" strokeWidth={1.5} />
        ) : (
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-positive" strokeWidth={1.5} />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "font-mono text-[9px] font-bold tracking-[0.14em] uppercase",
                halted ? "text-negative" : "text-positive"
              )}
            >
              {halted ? "Trading Halted" : "Pipeline Authorized"}
            </span>
            {state.simulated && (
              <span className="border border-border-strong bg-ebony px-1.5 py-0.5 font-mono text-[8px] tracking-widest text-text-muted uppercase">
                Demo — switch to Live API for real execution
              </span>
            )}
            {state.pipeline && (
              <span className="font-mono text-[8px] tracking-widest text-text-muted uppercase">
                Pipeline · {state.pipeline}
              </span>
            )}
          </div>
          <p
            className={cn(
              "mt-1 font-sans text-[12px] leading-relaxed",
              halted ? "text-negative/90" : "text-positive/90"
            )}
          >
            {state.message}
          </p>
        </div>
        {!halted && (
          <CheckCircle2 className="h-4 w-4 shrink-0 text-positive" strokeWidth={1.5} />
        )}
      </div>
    </div>
  );
}

function PipelineStatusBanner({ status }: { status: PipelineStatusPayload }) {
  const running = !status.finished_at && status.started_at;
  const success = status.success === true;

  return (
    <div className="border-t border-border bg-charcoal-dark px-4 py-2.5">
      <p className="font-mono text-[9px] tracking-[0.12em] text-text-muted uppercase">
        master_controller.py
      </p>
      <p className="mt-1 font-mono text-[10px] text-text-secondary">
        {running && (
          <span className="inline-flex items-center gap-1.5 text-warning">
            <Loader2 className="h-3 w-3 animate-spin" />
            Running… started {status.started_at?.slice(11, 19)} UTC
          </span>
        )}
        {!running && status.finished_at && (
          <span className={success ? "text-positive" : "text-negative"}>
            Finished exit {status.exit_code} · {status.finished_at?.slice(11, 19)} UTC
          </span>
        )}
      </p>
    </div>
  );
}

export default function ExecutiveBriefing() {
  const { isSandbox } = useDeployment();
  const { payload, summary, generatedAt, isLoading, error, refetch } =
    useExecutiveSummary();
  const [authorizePhase, setAuthorizePhase] = useState<ButtonPhase>("idle");
  const [haltPhase, setHaltPhase] = useState<ButtonPhase>("idle");
  const [operatorState, setOperatorState] = useState<OperatorControlState | null>(null);
  const [displayPayload, setDisplayPayload] = useState<ExecutiveSummaryPayload | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusPayload | null>(null);
  const [overrideError, setOverrideError] = useState<string | null>(null);
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollAbortRef = useRef(false);

  // Reset the operator-annotated copy whenever a fresh briefing arrives.
  // Render-phase reset (not an effect) per react.dev guidance on derived state.
  const [lastPayload, setLastPayload] = useState(payload);
  if (lastPayload !== payload) {
    setLastPayload(payload);
    setDisplayPayload(null);
  }

  const scheduleReset = useCallback((action: OverrideAction) => {
    if (resetTimerRef.current) clearTimeout(resetTimerRef.current);
    resetTimerRef.current = setTimeout(() => {
      if (action === "AUTHORIZE") setAuthorizePhase("idle");
      else setHaltPhase("idle");
    }, CONFIRMED_MS);
  }, []);

  useEffect(() => {
    return () => {
      pollAbortRef.current = true;
      if (resetTimerRef.current) clearTimeout(resetTimerRef.current);
    };
  }, []);

  const pollPipelineCompletion = useCallback(async () => {
    pollAbortRef.current = false;
    for (let i = 0; i < PIPELINE_POLL_MAX; i++) {
      if (pollAbortRef.current) return;
      await new Promise((r) => setTimeout(r, PIPELINE_POLL_MS));
      try {
        const status = await fetchPipelineStatus();
        setPipelineStatus(status);
        if (status.finished_at) {
          await refetch();
          return;
        }
      } catch {
        /* keep polling */
      }
    }
  }, [refetch]);

  const applyResult = useCallback(
    (action: OverrideAction, result: OperatorControlState) => {
      const setPhase = action === "AUTHORIZE" ? setAuthorizePhase : setHaltPhase;
      setOperatorState(result);
      const base = payload ?? { summary };
      if (base.summary) {
        setDisplayPayload({
          ...base,
          summary: appendOperatorNote(base.summary, result),
        });
      }
      setPhase("confirmed");
      scheduleReset(action);
    },
    [payload, scheduleReset, summary]
  );

  const runOverride = useCallback(
    async (action: OverrideAction) => {
      setOverrideError(null);
      setPipelineStatus(null);
      const setPhase = action === "AUTHORIZE" ? setAuthorizePhase : setHaltPhase;

      if (isSandbox) {
        applyResult(action, sandboxOverrideResult(action));
        return;
      }

      setPhase("loading");
      try {
        const response = await postOverride(action);
        applyResult(action, liveOverrideResult(action, response));

        if (action === "AUTHORIZE") {
          if (response.pipeline === "STARTED") {
            setPipelineStatus({ started_at: new Date().toISOString(), success: null });
            void pollPipelineCompletion();
          } else if (response.pipeline === "ALREADY_RUNNING") {
            setPipelineStatus({ status: "ALREADY_RUNNING", success: null });
            void pollPipelineCompletion();
          }
        }

        await refetch();
      } catch (err) {
        setPhase("idle");
        setOverrideError(
          err instanceof Error ? err.message : "Override command failed"
        );
      }
    },
    [applyResult, isSandbox, pollPipelineCompletion, refetch]
  );

  const anyLoading = authorizePhase === "loading" || haltPhase === "loading";
  const briefingPayload = displayPayload ?? payload;

  return (
    <section className="relative border border-border bg-charcoal">
      <header className="flex items-center justify-between gap-4 border-b border-border px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="h-7 w-px bg-text-primary" />
          <div>
            <h2 className="font-mono text-xs font-bold tracking-[0.16em] text-text-primary uppercase">
              The QCTF Daily // Executive Briefing
            </h2>
            <p className="mt-0.5 font-mono text-[10px] tracking-widest text-text-muted uppercase">
              DeepSeek Advisor Desk · Manual Override Surface
              {isSandbox ? " · Sandbox" : " · Live API"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {operatorState?.halted && (
            <span className="border border-negative/40 bg-negative/15 px-2 py-1 font-mono text-[8px] font-bold tracking-[0.12em] text-negative uppercase animate-pulse">
              Halt Active
            </span>
          )}
          {generatedAt && !isLoading && (
            <span className="font-mono text-[9px] tracking-[0.14em] text-text-muted uppercase">
              Edition · {generatedAt.slice(0, 10)}
            </span>
          )}
        </div>
      </header>

      <div className="border-b border-border bg-ebony-alt px-6 py-7 sm:px-8">
        {isLoading ? (
          <BriefingSkeleton />
        ) : error ? (
          <p className="font-mono text-[11px] leading-relaxed text-negative">
            {error}
          </p>
        ) : briefingPayload ? (
          <BriefingSections payload={briefingPayload} />
        ) : (
          <p className="font-mono text-[11px] text-text-muted">
            No briefing available yet. Run executive_briefer.py or wait for
            the next pipeline cycle.
          </p>
        )}
      </div>

      {operatorState && <OperatorStatusBanner state={operatorState} />}
      {pipelineStatus && <PipelineStatusBanner status={pipelineStatus} />}

      <div className="flex flex-col gap-px bg-border sm:flex-row">
        <button
          type="button"
          disabled={anyLoading || operatorState?.halted === true}
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
