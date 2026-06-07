import type { ExecutiveSummaryPayload } from "@/lib/api/executive-summary";

export const MOCK_EXECUTIVE_SUMMARY: ExecutiveSummaryPayload = {
  summary:
    "Portfolio NAV holds at $100,000 with gross exposure at 96%. The HMM regime reads BULL with 82% confidence; the defensive sleeve routes through cleared sovereign instruments (TLT/IEF) at 20% notional. Alpha Core contributes +1.51% session P&L. No active risk veto. Awaiting operator authorization before the nightly pipeline executes new orders.",
  generated_at: "2026-06-02T12:00:00.000Z",
  run_date: "2026-06-02",
};
