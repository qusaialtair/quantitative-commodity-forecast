"use client";

import { useCallback, useEffect, useState } from "react";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { SNAPSHOT_API_URL } from "@/lib/config";
import {
  fetchExecutiveSummary,
  type ExecutiveSummaryPayload,
} from "@/lib/api/executive-summary";
import { MOCK_EXECUTIVE_SUMMARY } from "@/lib/mock-executive-summary";

interface UseExecutiveSummaryResult {
  payload: ExecutiveSummaryPayload | null;
  summary: string;
  generatedAt: string | null;
  isLoading: boolean;
  error: string | null;
  isSandbox: boolean;
  refetch: () => Promise<void>;
}

export function useExecutiveSummary(): UseExecutiveSummaryResult {
  const { isSandbox } = useDeployment();
  const [payload, setPayload] = useState<ExecutiveSummaryPayload | null>(
    isSandbox ? MOCK_EXECUTIVE_SUMMARY : null
  );
  const [isLoading, setIsLoading] = useState(!isSandbox);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    if (isSandbox) {
      setPayload(MOCK_EXECUTIVE_SUMMARY);
      setIsLoading(false);
      setError(null);
      return;
    }

    setIsLoading(true);
    try {
      const data = await fetchExecutiveSummary(SNAPSHOT_API_URL, signal);
      if (signal?.aborted) return;
      setPayload(data);
      setError(null);
    } catch (err) {
      if (signal?.aborted) return;
      setError(
        err instanceof Error ? err.message : "Executive summary unavailable"
      );
    } finally {
      if (!signal?.aborted) {
        setIsLoading(false);
      }
    }
  }, [isSandbox]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const refetch = useCallback(async () => {
    await load();
  }, [load]);

  return {
    payload,
    summary: payload?.summary ?? "",
    generatedAt: payload?.generated_at ?? null,
    isLoading,
    error,
    isSandbox,
    refetch,
  };
}
