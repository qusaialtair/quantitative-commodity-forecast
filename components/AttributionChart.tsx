"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { StrategyAllocation } from "@/lib/types";
import { formatCurrency, formatPct } from "@/lib/utils";

interface AttributionChartProps {
  strategies: StrategyAllocation[];
  totalEquity: number;
}

interface ChartRow {
  name: string;
  shortCode: string;
  allocationPct: number;
  notionalUsd: number;
  color: string;
}

interface TooltipPayload {
  payload: ChartRow;
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;

  return (
    <div className="border border-border-strong bg-ebony px-3 py-2">
      <p className="font-mono text-[9px] tracking-widest text-text-muted uppercase">
        {row.shortCode}
      </p>
      <p className="mt-1 font-mono text-sm font-semibold text-text-primary tabular-nums">
        {formatPct(row.allocationPct)}
      </p>
      <p className="font-mono text-[11px] text-text-secondary tabular-nums">
        {formatCurrency(row.notionalUsd)}
      </p>
    </div>
  );
}

export default function AttributionChart({
  strategies,
  totalEquity,
}: AttributionChartProps) {
  const data: ChartRow[] = strategies.map((s) => ({
    name: s.name,
    shortCode: s.shortCode,
    allocationPct: s.allocationPct,
    notionalUsd: s.notionalUsd,
    color: s.color,
  }));

  return (
    <div className="flex h-full flex-col">
      <div className="relative min-h-[200px] flex-1">
        <ResponsiveContainer width="100%" height="100%" minWidth={1}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 8, left: 4, bottom: 4 }}
            barCategoryGap="28%"
          >
            <CartesianGrid
              horizontal={false}
              stroke="#27272a"
              strokeDasharray="2 4"
            />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fill: "#71717a", fontSize: 9, fontFamily: "monospace" }}
              axisLine={{ stroke: "#3f3f46" }}
              tickLine={{ stroke: "#3f3f46" }}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis
              type="category"
              dataKey="shortCode"
              width={36}
              tick={{ fill: "#a1a1aa", fontSize: 10, fontFamily: "monospace" }}
              axisLine={{ stroke: "#3f3f46" }}
              tickLine={false}
            />
            <Tooltip
              content={<ChartTooltip />}
              cursor={{ fill: "rgba(255,255,255,0.02)" }}
            />
            <Bar dataKey="allocationPct" barSize={14} radius={0}>
              {data.map((entry) => (
                <Cell
                  key={entry.shortCode}
                  fill={entry.color}
                  fillOpacity={0.55}
                  stroke={entry.color}
                  strokeWidth={1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 border-t border-border pt-3">
        <div className="flex items-center justify-between">
          <span className="text-label">Book NAV</span>
          <span className="font-mono text-sm font-semibold text-text-primary tabular-nums">
            {formatCurrency(totalEquity, { compact: true })}
          </span>
        </div>
        <div className="mt-2 space-y-1.5">
          {data.map((row) => (
            <div key={row.shortCode} className="flex items-center gap-2">
              <span
                className="h-2 w-2 shrink-0"
                style={{ backgroundColor: row.color }}
              />
              <span className="flex-1 truncate font-mono text-[10px] text-text-muted">
                {row.name}
              </span>
              <span className="font-mono text-[10px] text-text-primary tabular-nums">
                {formatPct(row.allocationPct)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
