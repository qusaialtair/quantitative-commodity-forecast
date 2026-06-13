export type DashboardMode = "RECRUITER SANDBOX" | "PRODUCTION AUTOMATED";

export const DEPLOYMENT_MODE_STORAGE_KEY = "qctf-deployment-mode";

/** True on Vercel / static production builds — API polling is hard-disabled. */
export const IS_HOSTED_PRODUCTION = process.env.NODE_ENV === "production";

const MODE_ENV = process.env.NEXT_PUBLIC_DASHBOARD_MODE ?? "RECRUITER SANDBOX";

export function resolveDashboardMode(
  clientOverride?: DashboardMode | null
): DashboardMode {
  if (IS_HOSTED_PRODUCTION) {
    return "RECRUITER SANDBOX";
  }
  if (clientOverride === "PRODUCTION AUTOMATED" || clientOverride === "RECRUITER SANDBOX") {
    return clientOverride;
  }
  if (MODE_ENV === "PRODUCTION AUTOMATED") {
    return "PRODUCTION AUTOMATED";
  }
  return "RECRUITER SANDBOX";
}

export function isSandboxModeFor(mode: DashboardMode): boolean {
  return mode !== "PRODUCTION AUTOMATED";
}

/** @deprecated Prefer `useDeployment()` in client components. */
export function getDashboardMode(): DashboardMode {
  return resolveDashboardMode();
}

/** @deprecated Prefer `useDeployment()` in client components. */
export function isProductionMode(): boolean {
  return !isSandboxModeFor(getDashboardMode());
}

/** @deprecated Prefer `useDeployment()` in client components. */
export function isSandboxMode(): boolean {
  return isSandboxModeFor(getDashboardMode());
}

export function isDeploymentToggleEnabled(): boolean {
  return !IS_HOSTED_PRODUCTION;
}

export const SNAPSHOT_POLL_MS = 3_000;

/**
 * API base URL for browser fetches.
 * Dev default uses the Next.js rewrite at /qctf-backend to avoid CORS (3000 → 8000).
 * Set NEXT_PUBLIC_API_URL=http://localhost:8000 to hit the backend directly (requires CORS).
 */
export const SNAPSHOT_API_URL = resolveSnapshotApiUrl();

function resolveSnapshotApiUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (explicit) {
    return explicit.replace(/\/$/, "");
  }
  if (process.env.NODE_ENV === "development") {
    return "/qctf-backend";
  }
  return "";
}

const DEV_ADMIN_TOKEN_FALLBACK = "";

/**
 * Admin token for mutating FastAPI control endpoints.
 * Returns null on hosted production (Vercel) so no secret is bundled or sent.
 */
export function resolveQctfAdminToken(): string | null {
  if (IS_HOSTED_PRODUCTION) {
    return null;
  }

  const fromEnv = process.env.NEXT_PUBLIC_QCTF_ADMIN_TOKEN?.trim();
  if (fromEnv) {
    return fromEnv;
  }

  if (process.env.NODE_ENV === "development") {
    return DEV_ADMIN_TOKEN_FALLBACK;
  }

  return null;
}

/** Headers for POST /api/override, /api/halt-trading, and other admin routes. */
export function qctfAdminHeaders(): Record<string, string> {
  const token = resolveQctfAdminToken();
  if (!token) {
    return {};
  }
  return { "X-QCTF-Admin-Token": token };
}
