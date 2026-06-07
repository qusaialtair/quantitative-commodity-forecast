import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(
  value: number,
  options?: { signed?: boolean; compact?: boolean }
): string {
  const { signed = false, compact = false } = options ?? {};
  const prefix = signed && value > 0 ? "+" : signed && value < 0 ? "" : "";

  if (compact) {
    const abs = Math.abs(value);
    if (abs >= 1_000_000) {
      return `${prefix}$${(value / 1_000_000).toFixed(1)}M`;
    }
    if (abs >= 1_000) {
      return `${prefix}$${(value / 1_000).toFixed(1)}K`;
    }
  }

  const formatted = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(Math.abs(value));

  if (signed && value < 0) {
    return `-${formatted}`;
  }
  return `${prefix}${formatted}`;
}

export function formatPct(value: number, signed = false): string {
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}%`;
}
