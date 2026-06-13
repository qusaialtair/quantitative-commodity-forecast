import { cn } from "@/lib/utils";

interface ModuleCardProps {
  title: string;
  subtitle?: string;
  className?: string;
  children?: React.ReactNode;
  span?: string;
}

export default function ModuleCard({
  title,
  subtitle,
  className,
  children,
  span,
}: ModuleCardProps) {
  return (
    <section
      className={cn(
        "flex min-h-[140px] flex-col border border-border bg-charcoal",
        span,
        className
      )}
    >
      <div className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <h2 className="font-mono text-[10px] font-medium tracking-[0.14em] text-text-secondary">
          {title}
        </h2>
        {subtitle && (
          <span className="font-mono text-[9px] tracking-wide text-text-muted">
            {subtitle}
          </span>
        )}
      </div>
      <div
        className={cn(
          "flex flex-1 flex-col",
          children ? "min-h-0" : "items-center justify-center p-4"
        )}
      >
        {children ?? (
          <span className="font-mono text-[10px] tracking-wide text-text-muted">
            MODULE PENDING
          </span>
        )}
      </div>
    </section>
  );
}
