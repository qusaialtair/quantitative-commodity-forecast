import SignalBadge from "@/components/sections/SignalBadge";
import type { HomeTeaserData } from "@/lib/types/sections";

export default function MetalsIntelligenceTeaser({
  teaser,
}: {
  teaser: HomeTeaserData["metalsTeaser"];
}) {
  return (
    <div className="grid grid-cols-2 gap-px bg-border p-4 sm:grid-cols-4">
      <div className="border border-border bg-charcoal-dark px-3 py-2.5">
        <p className="text-label mb-1">HMM Regime</p>
        <p className="font-mono text-sm font-bold text-positive">{teaser.regime}</p>
        <p className="mt-0.5 font-mono text-[9px] text-text-muted">
          {teaser.confidencePct}% conf
        </p>
      </div>
      <div className="border border-border bg-charcoal-dark px-3 py-2.5">
        <p className="text-label mb-1">Primary Signal</p>
        <div className="mt-1">
          <SignalBadge signal={teaser.primarySignal} />
        </div>
      </div>
      <div className="border border-border bg-charcoal-dark px-3 py-2.5">
        <p className="text-label mb-1">Spot Gold</p>
        <p className="font-mono text-sm font-bold text-text-primary tabular-nums">
          {teaser.spotGold}
        </p>
      </div>
      <div className="border border-border bg-charcoal-dark px-3 py-2.5 sm:col-span-1">
        <p className="text-label mb-1">Oracle Bias</p>
        <p className="font-sans text-[11px] leading-snug text-text-secondary">
          {teaser.oracleBias}
        </p>
      </div>
    </div>
  );
}
