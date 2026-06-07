"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  DEPLOYMENT_MODE_STORAGE_KEY,
  IS_HOSTED_PRODUCTION,
  resolveDashboardMode,
  type DashboardMode,
} from "@/lib/config";

interface DeploymentContextValue {
  mode: DashboardMode;
  isSandbox: boolean;
  isHostedProduction: boolean;
  isDeploymentToggleEnabled: boolean;
  setDeploymentMode: (mode: DashboardMode) => void;
}

const DeploymentContext = createContext<DeploymentContextValue | null>(null);

function readStoredMode(): DashboardMode | null {
  if (typeof window === "undefined") return null;
  const stored = window.localStorage.getItem(DEPLOYMENT_MODE_STORAGE_KEY);
  if (stored === "PRODUCTION AUTOMATED" || stored === "RECRUITER SANDBOX") {
    return stored;
  }
  return null;
}

export function DeploymentProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<DashboardMode>(() =>
    resolveDashboardMode(null)
  );

  useEffect(() => {
    if (IS_HOSTED_PRODUCTION) {
      window.localStorage.removeItem(DEPLOYMENT_MODE_STORAGE_KEY);
      setMode("RECRUITER SANDBOX");
      return;
    }
    setMode(resolveDashboardMode(readStoredMode()));
  }, []);

  const setDeploymentMode = useCallback((next: DashboardMode) => {
    if (IS_HOSTED_PRODUCTION) return;
    const resolved = resolveDashboardMode(next);
    setMode(resolved);
    window.localStorage.setItem(DEPLOYMENT_MODE_STORAGE_KEY, resolved);
  }, []);

  const value = useMemo<DeploymentContextValue>(
    () => ({
      mode,
      isSandbox: mode !== "PRODUCTION AUTOMATED",
      isHostedProduction: IS_HOSTED_PRODUCTION,
      isDeploymentToggleEnabled: !IS_HOSTED_PRODUCTION,
      setDeploymentMode,
    }),
    [mode, setDeploymentMode]
  );

  return (
    <DeploymentContext.Provider value={value}>
      {children}
    </DeploymentContext.Provider>
  );
}

export function useDeployment(): DeploymentContextValue {
  const ctx = useContext(DeploymentContext);
  if (!ctx) {
    throw new Error("useDeployment must be used within DeploymentProvider");
  }
  return ctx;
}
