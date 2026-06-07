export type DashboardMode = "RECRUITER SANDBOX" | "PRODUCTION AUTOMATED";

export const DEPLOYMENT_MODE_STORAGE_KEY = "altair-deployment-mode";

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

export const SNAPSHOT_POLL_MS = 5_000;

export const SNAPSHOT_API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
