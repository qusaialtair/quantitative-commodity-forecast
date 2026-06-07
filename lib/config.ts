export type DashboardMode = "RECRUITER SANDBOX" | "PRODUCTION AUTOMATED";

const MODE_ENV = process.env.NEXT_PUBLIC_DASHBOARD_MODE ?? "RECRUITER SANDBOX";

export function getDashboardMode(): DashboardMode {
  if (MODE_ENV === "PRODUCTION AUTOMATED") {
    return "PRODUCTION AUTOMATED";
  }
  return "RECRUITER SANDBOX";
}

export function isProductionMode(): boolean {
  return getDashboardMode() === "PRODUCTION AUTOMATED";
}

export function isSandboxMode(): boolean {
  return !isProductionMode();
}

export const SNAPSHOT_POLL_MS = 5_000;

export const SNAPSHOT_API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
