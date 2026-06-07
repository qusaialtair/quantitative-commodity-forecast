import ModuleCard from "@/components/shell/ModuleCard";

export default function AgentSection() {
  return (
    <div className="grid gap-px bg-border lg:grid-cols-[3fr_1fr]">
      <ModuleCard
        title="WEALTH AGENT"
        subtitle="DeepSeek · tool calling"
        className="min-h-[520px]"
      />
      <div className="flex flex-col gap-px bg-border">
        <ModuleCard title="PORTFOLIO CONTEXT" className="min-h-[160px]" />
        <ModuleCard title="ORACLE SCORES" className="min-h-[160px]" />
        <ModuleCard title="LESSONS LEARNED" subtitle="Reflexion engine" className="min-h-[160px]" />
      </div>
    </div>
  );
}
