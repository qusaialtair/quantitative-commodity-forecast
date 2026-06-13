import { SNAPSHOT_API_URL } from "@/lib/config";
import { postAdminJson } from "@/lib/api/admin-fetch";

export interface HaltTradingPayload {
  halted: boolean;
}

export interface HaltTradingResponse {
  status: string;
  halted: boolean;
}

export async function postHaltTrading(
  halted: boolean,
  baseUrl: string = SNAPSHOT_API_URL,
  signal?: AbortSignal
): Promise<HaltTradingResponse> {
  return postAdminJson<HaltTradingResponse>(
    "/api/halt-trading",
    { halted } satisfies HaltTradingPayload,
    baseUrl,
    signal
  );
}
