"use client";

import { useEffect, useState } from "react";
import { useDeployment } from "@/components/providers/DeploymentProvider";
import { SNAPSHOT_API_URL } from "@/lib/config";
import {
  fetchExecutiveSummary,
  type ExecutiveSummaryPayload,
} from "@/lib/api/executive-summary";
import { MOCK_EXECUTIVE_SUMMARY } from "@/lib/mock-executive-summary";

interface UseExecutiveSummaryResult {
  summary: string;
  generatedAt: string | null;
  isLoading: boolean;
  error: string | null;
  isSandbox: boolean;
}

export function useExecutiveSummary(): UseExecutiveSummaryResult {
  const { isSandbox } = useDeployment();
  const [payload, setPayload] = useState<ExecutiveSummaryPayload | null>(
    isSandbox ? MOCK_EXECUTIVE_SUMMARY : null
  );
  const [isLoading, setIsLoading] = useState(!isSandbox);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isSandbox) {
      setPayload(MOCK_EXECUTIVE_SUMMARY);
      setIsLoading(false);
      setError(null);
      return;
    }

    const controller = new AbortController();

    const load = async () => {
      setIsLoading(true);
      try {
        const data = await fetchExecutiveSummary(
          SNAPSHOT_API_URL,
          controller.signal
        );
        if (controller.signal.aborted) return;
        setPayload(data);
        setError(null);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(
          err instanceof Error ? err.message : "Executive summary unavailable"
        );
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    };

    void load();
    return () => controller.abort();
  }, [isSandbox]);

  return {
    summary: payload?.summary ?? "",
    generatedAt: payload?.generated_at ?? null,
    isLoading,
    error,
    isSandbox,
  };
}
