import type { ExecutiveSummaryPayload } from "@/lib/api/executive-summary";

const MOCK_HEADLINE = "Gold's rough month demands patience, not panic";

const MOCK_READ =
  "Gold took a serious hit this month, sliding from the highs near $4,700 down " +
  "to the low $4,200s, and the regime model is about as bearish as it gets " +
  "right now. Our crisis dial is sitting deep in the stress zone with fast " +
  "volatility running well above its monthly baseline — the kind of tape where " +
  "heroes get carried out. Silver is getting dragged along for the ride, and " +
  "until volatility cools off I'd treat every bounce with suspicion.";

const MOCK_POSITIONING =
  "We're deliberately light: the book is mostly cash after the strategic exit, " +
  "which is exactly where I want us. The Treasury hedge is running in its GLD " +
  "fallback (the Sharia gate isn't cleared for TLT/IEF), acting as our airbag " +
  "if this selloff deepens into a full regime break. New entries are " +
  "automatically running at roughly half size while the volatility breaker is " +
  "engaged.";

const MOCK_WATCHLIST =
  "NVDA and XOM stay on the radar for the equity sleeve — I'd want a " +
  "Sharia-cleared, high-conviction signal plus two calm weeks of tape before " +
  "committing. If MSFT holds its range while metals stabilise, that's our " +
  "tell that the risk-off rotation is contained.";

const MOCK_CALL =
  "Hold off on buying gold here — let the knife hit the floor first. Stay " +
  "defensive, keep the hedge on, and let the volatility breaker do its job. " +
  "The one thing that would change my mind: the regime model flipping back to " +
  "bullish with the crisis dial dropping out of the stress zone.";

export const MOCK_EXECUTIVE_SUMMARY: ExecutiveSummaryPayload = {
  summary:
    `HEADLINE: ${MOCK_HEADLINE}\n` +
    `THE READ: ${MOCK_READ}\n` +
    `POSITIONING: ${MOCK_POSITIONING}\n` +
    `WATCHLIST: ${MOCK_WATCHLIST}\n` +
    `THE CALL: ${MOCK_CALL}`,
  headline: MOCK_HEADLINE,
  market: MOCK_READ,
  holdings: MOCK_POSITIONING,
  watchlist: MOCK_WATCHLIST,
  action: MOCK_CALL,
  generated_at: "2026-06-12T06:00:00.000Z",
  run_date: "2026-06-12",
  offline: true,
};
