"use client";

import { useCallback, useEffect, useState } from "react";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { SNAPSHOT_API_URL } from "@/lib/config";
import {
  fetchRegimePulse,
  type RegimePulsePayload,
} from "@/lib/api/regime-pulse";
import { MOCK_REGIME_PULSE } from "@/lib/mock-regime-pulse";

const REGIME_POLL_MS = 60_000;

interface UseRegimePulseResult {
  payload: RegimePulsePayload | null;
  isLoading: boolean;
  error: string | null;
}

export function useRegimePulse(): UseRegimePulseResult {
  const { isSandbox } = useDeployment();
  const [payload, setPayload] = useState<RegimePulsePayload | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await fetchRegimePulse(SNAPSHOT_API_URL, signal);
      if (signal?.aborted) return;
      setPayload(data);
      setError(null);
    } catch (err) {
      if (signal?.aborted) return;
      setError(err instanceof Error ? err.message : "Regime pulse unavailable");
    } finally {
      if (!signal?.aborted) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isSandbox) return;
    const controller = new AbortController();
    // Defer the first fetch to a macrotask so no setState lands synchronously
    // inside the effect body (react-hooks/set-state-in-effect).
    const kickoff = setTimeout(() => void load(controller.signal), 0);
    const timer = setInterval(
      () => void load(controller.signal),
      REGIME_POLL_MS
    );
    return () => {
      controller.abort();
      clearTimeout(kickoff);
      clearInterval(timer);
    };
  }, [load, isSandbox]);

  // Sandbox mode renders the canned snapshot without touching effect state,
  // so toggling deployment modes stays instant and lint-clean.
  if (isSandbox) {
    return { payload: MOCK_REGIME_PULSE, isLoading: false, error: null };
  }
  return { payload, isLoading, error };
}
