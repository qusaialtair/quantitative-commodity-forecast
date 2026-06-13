import { SNAPSHOT_API_URL } from "@/lib/config";

export interface PipelineStatusPayload {
  status?: string;
  source?: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number | null;
  success?: boolean | null;
  stdout_tail?: string;
  stderr_tail?: string;
}

export async function fetchPipelineStatus(
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<PipelineStatusPayload> {
  const response = await fetch(`${baseUrl}/api/pipeline-status`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(`Pipeline status request failed (${response.status})`);
  }

  return response.json() as Promise<PipelineStatusPayload>;
}
