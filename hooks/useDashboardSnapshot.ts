"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  isSandboxMode,
  SNAPSHOT_API_URL,
  SNAPSHOT_POLL_MS,
} from "@/lib/config";
import { fetchSnapshot, mapSnapshotToDashboard } from "@/lib/api/snapshot";
import type { ApiConnectionStatus, DashboardState } from "@/lib/types";

interface UseDashboardSnapshotResult {
  data: DashboardState;
  apiStatus: ApiConnectionStatus;
  lastUpdatedAt: string | null;
  error: string | null;
}

export function useDashboardSnapshot(
  initial: DashboardState
): UseDashboardSnapshotResult {
  const sandbox = isSandboxMode();
  const [data, setData] = useState<DashboardState>(initial);
  const [apiStatus, setApiStatus] = useState<ApiConnectionStatus>(
    sandbox ? "DISCONNECTED" : "DISCONNECTED"
  );
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlightRef = useRef(false);

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
    } catch (err) {
      if (signal.aborted) return;
      setApiStatus("DISCONNECTED");
      setError(err instanceof Error ? err.message : "Snapshot fetch failed");
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (sandbox) {
      return;
    }

    const controller = new AbortController();
    void poll(controller.signal);

    const intervalId = window.setInterval(() => {
      void poll(controller.signal);
    }, SNAPSHOT_POLL_MS);

    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [sandbox, poll]);

  return { data, apiStatus, lastUpdatedAt, error };
}
