import { SNAPSHOT_API_URL } from "@/lib/config";

export type OverrideAction = "AUTHORIZE" | "HALT";

export interface OverridePayload {
  action: OverrideAction;
}

export interface OverrideResponse {
  status: string;
  action: OverrideAction;
  halted?: boolean;
  pipeline?: string;
  message?: string;
}

export class OverrideFetchError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "OverrideFetchError";
  }
}

export async function postOverride(
  action: OverrideAction,
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<OverrideResponse> {
  const response = await fetch(`${baseUrl}/api/override`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ action } satisfies OverridePayload),
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new OverrideFetchError(
      `Override request failed (${response.status})`,
      response.status
    );
  }

  return response.json() as Promise<OverrideResponse>;
}
