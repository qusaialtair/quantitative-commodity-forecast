import { MOCK_SECTIONS } from "@/lib/mock-sections";
import ModuleCard from "@/components/shell/ModuleCard";

export default function AgentSection() {
  const { agent } = MOCK_SECTIONS;

  return (
    <div className="grid gap-px bg-border lg:grid-cols-[3fr_1fr]">
      <ModuleCard title="WEALTH AGENT" subtitle="DeepSeek · tool calling">
        <div className="flex min-h-[480px] flex-col">
          <div className="flex-1 space-y-3 overflow-auto p-4">
            {agent.messages.map((msg, i) => (
              <div
                key={i}
                className={
                  msg.role === "user"
                    ? "border border-border bg-ebony px-3 py-2"
                    : "border border-border-strong bg-charcoal-dark px-3 py-2"
                }
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-mono text-[9px] font-semibold tracking-[0.12em] text-text-muted uppercase">
                    {msg.role === "user" ? "Operator" : "Wealth Agent"}
                  </span>
                  <span className="font-mono text-[8px] text-text-muted">{msg.timestamp}</span>
                </div>
                <p className="font-sans text-[12px] leading-relaxed text-text-primary">
                  {msg.content}
                </p>
              </div>
            ))}
          </div>
          <div className="border-t border-border px-4 py-3">
            <p className="font-mono text-[9px] text-text-muted">
              Ask about your metals portfolio… (sandbox — responses simulated)
            </p>
          </div>
        </div>
      </ModuleCard>
      <div className="flex flex-col gap-px bg-border">
        <ModuleCard title="PORTFOLIO CONTEXT">
          <p className="p-3 font-sans text-[11px] leading-relaxed text-text-secondary">
            {agent.portfolioContext}
          </p>
        </ModuleCard>
        <ModuleCard title="ORACLE SCORES">
          <div className="space-y-2 p-3">
            {agent.oracleScores.map((row) => (
              <div key={row.label}>
                <div className="mb-1 flex justify-between font-mono text-[9px] text-text-muted">
                  <span>{row.label}</span>
                  <span>{row.score}</span>
                </div>
                <div className="h-px bg-border">
                  <div className="h-px bg-warning/70" style={{ width: `${row.score}%` }} />
                </div>
              </div>
            ))}
          </div>
        </ModuleCard>
        <ModuleCard title="LESSONS LEARNED" subtitle="Reflexion engine">
          <ul className="space-y-2 p-3">
            {agent.lessonsLearned.map((lesson) => (
              <li
                key={lesson}
                className="border-l border-border-strong pl-2 font-sans text-[11px] leading-relaxed text-text-secondary"
              >
                {lesson}
              </li>
            ))}
          </ul>
        </ModuleCard>
      </div>
    </div>
  );
}
