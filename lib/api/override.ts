import { SNAPSHOT_API_URL } from "@/lib/config";
import { AdminFetchError, postAdminJson } from "@/lib/api/admin-fetch";

export type OverrideAction = "AUTHORIZE" | "HALT";

export interface OverridePayload {
  action: OverrideAction;
}

export interface OverrideResponse {
  status: string;
  action: OverrideAction;
  halted?: boolean;
  trading_halted?: boolean;
  pipeline?: string | null;
  message?: string;
  cleared_for_date?: string | null;
}

export { AdminFetchError as OverrideFetchError };

export async function postOverride(
  action: OverrideAction,
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<OverrideResponse> {
  return postAdminJson<OverrideResponse>(
    "/api/override",
    { action } satisfies OverridePayload,
    baseUrl,
    signal
  );
}
