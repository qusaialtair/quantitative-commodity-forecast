import { qctfAdminHeaders } from "@/lib/config";

export class AdminFetchError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "AdminFetchError";
  }
}

/** POST helper for FastAPI control-plane endpoints that require X-QCTF-Admin-Token. */
export async function postAdminJson<TResponse>(
  path: string,
  body: unknown,
  baseUrl: string,
  signal?: AbortSignal
): Promise<TResponse> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...qctfAdminHeaders(),
    },
    body: JSON.stringify(body),
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new AdminFetchError(
      `Admin request failed (${response.status})`,
      response.status
    );
  }

  return response.json() as Promise<TResponse>;
}
