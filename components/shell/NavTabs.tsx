"use client";

import { cn } from "@/lib/utils";

export type NavSection =
  | "home"
  | "metals"
  | "equities"
  | "agent"
  | "performance";

const TABS: { id: NavSection; label: string }[] = [
  { id: "home", label: "Command Centre" },
  { id: "metals", label: "Metals" },
  { id: "equities", label: "Equities" },
  { id: "agent", label: "Wealth Agent" },
  { id: "performance", label: "Performance" },
];

interface NavTabsProps {
  active: NavSection;
  onChange: (section: NavSection) => void;
}

export default function NavTabs({ active, onChange }: NavTabsProps) {
  return (
    <nav className="flex shrink-0 items-center gap-0 border-b border-border bg-charcoal-dark px-2 lg:px-4">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={cn(
            "relative px-3 py-2.5 font-mono text-[10px] tracking-[0.1em] transition-colors",
            active === tab.id
              ? "text-text-primary after:absolute after:inset-x-0 after:bottom-0 after:h-px after:bg-text-primary"
              : "text-text-muted hover:text-text-secondary"
          )}
        >
          {tab.label.toUpperCase()}
        </button>
      ))}
    </nav>
  );
}
