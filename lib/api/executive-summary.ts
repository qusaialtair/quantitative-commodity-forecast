import { SNAPSHOT_API_URL } from "@/lib/config";

export interface ExecutiveSummaryPayload {
  summary: string;
  generated_at?: string;
  run_date?: string;
}

export class ExecutiveSummaryFetchError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "ExecutiveSummaryFetchError";
  }
}

export async function fetchExecutiveSummary(
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<ExecutiveSummaryPayload> {
  const response = await fetch(`${baseUrl}/api/executive-summary`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new ExecutiveSummaryFetchError(
      `Executive summary request failed (${response.status})`,
      response.status
    );
  }

  return response.json() as Promise<ExecutiveSummaryPayload>;
}
