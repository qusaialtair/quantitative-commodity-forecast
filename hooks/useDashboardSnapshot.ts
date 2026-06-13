"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { SNAPSHOT_API_URL, SNAPSHOT_POLL_MS } from "@/lib/config";
import {
  fetchSnapshot,
  mapSnapshotToDashboard,
  SnapshotFetchError,
} from "@/lib/api/snapshot";
import type { ApiConnectionStatus, DashboardState } from "@/lib/types";

interface UseDashboardSnapshotResult {
  data: DashboardState;
  apiStatus: ApiConnectionStatus;
  lastUpdatedAt: string | null;
  error: string | null;
}

function logPollFailure(err: unknown): void {
  if (process.env.NODE_ENV !== "development") return;

  const message = err instanceof Error ? err.message : String(err);
  const status =
    err instanceof SnapshotFetchError ? err.status : undefined;
  const corsHint =
    message === "Failed to fetch" || message.includes("NetworkError")
      ? " — check FastAPI is running on :8000 and CORS/proxy (/qctf-backend)"
      : "";

  console.warn(
    `[QCTF MODEL] snapshot poll failed${status ? ` (${status})` : ""}: ${message}${corsHint}`
  );
}

export function useDashboardSnapshot(
  initial: DashboardState
): UseDashboardSnapshotResult {
  const { isSandbox } = useDeployment();
  const [data, setData] = useState<DashboardState>(initial);
  const [apiStatus, setApiStatus] = useState<ApiConnectionStatus>(
    isSandbox ? "LOCAL_SIMULATION" : "DISCONNECTED"
  );
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlightRef = useRef(false);

  useEffect(() => {
    if (isSandbox) {
      setData(initial);
      setApiStatus("LOCAL_SIMULATION");
      setLastUpdatedAt(null);
      setError(null);
    }
  }, [isSandbox, initial]);

  const poll = useCallback(async (signal: AbortSignal) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    try {
      const snapshot = await fetchSnapshot(SNAPSHOT_API_URL, signal);
      if (signal.aborted) return;

      setData(mapSnapshotToDashboard(snapshot));
      setApiStatus("CONNECTED");
      setLastUpdatedAt(snapshot.generated_at);
      setError(null);

      if (process.env.NODE_ENV === "development") {
        console.debug(
          `[QCTF MODEL] snapshot OK · ${snapshot.generated_at} · ${SNAPSHOT_API_URL}/api/snapshot`
        );
      }
    } catch (err) {
      if (signal.aborted) return;
      setApiStatus("DISCONNECTED");
      setError(err instanceof Error ? err.message : "Snapshot fetch failed");
      logPollFailure(err);
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (isSandbox) {
      return;
    }

    setApiStatus("DISCONNECTED");

    const controller = new AbortController();
    void poll(controller.signal);

    const intervalId = window.setInterval(() => {
      void poll(controller.signal);
    }, SNAPSHOT_POLL_MS);

    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [isSandbox, poll]);

  return { data, apiStatus, lastUpdatedAt, error };
}
