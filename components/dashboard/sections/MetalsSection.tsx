import ModuleCard from "@/components/shell/ModuleCard";

const METALS = ["Au Gold", "Ag Silver", "Pt Platinum", "Cu Copper", "Li Lithium", "Fe Iron"];

export default function MetalsSection() {
  return (
    <div className="flex flex-col gap-px bg-border">
      {/* Instrument selector strip */}
      <div className="grid grid-cols-3 gap-px bg-border sm:grid-cols-6">
        {METALS.map((metal) => (
          <button
            key={metal}
            type="button"
            className="border border-border bg-charcoal px-2 py-2 font-mono text-[9px] tracking-wide text-text-muted transition-colors hover:border-border-strong hover:text-text-secondary"
          >
            {metal.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Market overview — 6-column strip from app.py */}
      <div>
        <div className="border-b border-border bg-charcoal-dark px-3 py-1.5 font-mono text-[9px] tracking-[0.14em] text-text-muted">
          MARKET OVERVIEW
        </div>
        <div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-3 xl:grid-cols-6">
          {METALS.map((metal) => (
            <div
              key={`ov-${metal}`}
              className="border border-border bg-charcoal px-3 py-3"
            >
              <div className="font-mono text-[9px] text-text-muted">{metal}</div>
              <div className="mt-1 font-mono text-sm text-text-primary">—</div>
              <div className="mt-1 font-mono text-[10px] text-text-muted">—</div>
            </div>
          ))}
        </div>
      </div>

      <ModuleCard title="MARKET SUMMARY" subtitle="AI-generated" className="min-h-[120px]" />

      {/* Signal / Macro / Technicals — 3-column analysis row */}
      <div className="grid gap-px bg-border lg:grid-cols-3">
        <ModuleCard title="SIGNAL" subtitle="BUY / HOLD / SELL" className="min-h-[200px]" />
        <ModuleCard title="MACRO INTELLIGENCE" subtitle="Perplexity scores" className="min-h-[200px]" />
        <ModuleCard title="TECHNICALS" subtitle="RSI · MA-50 · MA-200" className="min-h-[200px]" />
      </div>

      <ModuleCard title="PRICE CHART" subtitle="YTD · 1Y · 5Y · 10Y" className="min-h-[320px]" />

      <ModuleCard title="ANALYST CHAT" subtitle="Gemini metals analyst" className="min-h-[200px]" />
    </div>
  );
}
