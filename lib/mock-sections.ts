import type { TerminalSections } from "@/lib/types/sections";

function navHistory(base: number): { date: string; nav: number }[] {
  const dates = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
  return dates.map((date, i) => ({
    date,
    nav: Math.round(base * (0.94 + i * 0.012 + (i % 2) * 0.008)),
  }));
}

function priceHistory(base: number, drift = 0.004): { date: string; price: number }[] {
  const labels = ["W-4", "W-3", "W-2", "W-1", "Now"];
  return labels.map((date, i) => ({
    date,
    price: Math.round(base * (1 - (4 - i) * drift) * 100) / 100,
  }));
}

export const MOCK_SECTIONS: TerminalSections = {
  metals: {
    marketSummary:
      "Precious complex leads risk-on bid: gold holds above $3,380 with HMM BULL at 82% confidence. Silver tracks beta 1.4x. Industrial metals steady — copper supported by grid demand, lithium rebounding on EV restocking. Iron ore flat on China property headwinds.",
    analystNote:
      "GC=F maintains structure above MA-200. Committee quant conviction HIGH — defensive GLD sleeve intact under Phase XXV Sharia gate. Recommend holding core metals exposure; add on SI=F pullbacks only if vol surface compresses.",
    instruments: [
      {
        id: "gold",
        label: "Au Gold",
        symbol: "Au",
        ticker: "GC=F",
        color: "#C9A84C",
        spotPrice: 3382.4,
        unit: "oz",
        changePct: 0.84,
        changeUsd: 28.2,
        signal: "BUY",
        hmmRegime: "BULL",
        rsi: 58.2,
        ma50: 3310.5,
        ma200: 3188.0,
        macroScore: 78,
        summary:
          "Safe-haven bid intact. Real yields easing supports continuation above $3,350 pivot.",
        priceHistory: priceHistory(3382.4),
      },
      {
        id: "silver",
        label: "Ag Silver",
        symbol: "Ag",
        ticker: "SI=F",
        color: "#9E9E9E",
        spotPrice: 36.72,
        unit: "oz",
        changePct: 1.24,
        changeUsd: 0.45,
        signal: "BUY",
        hmmRegime: "BULL",
        rsi: 62.1,
        ma50: 35.8,
        ma200: 32.4,
        macroScore: 74,
        summary: "Outperforming gold beta. Industrial demand tailwind from solar fab cycle.",
        priceHistory: priceHistory(36.72, 0.006),
      },
      {
        id: "platinum",
        label: "Pt Platinum",
        symbol: "Pt",
        ticker: "PL=F",
        color: "#E5E4E2",
        spotPrice: 1048.6,
        unit: "oz",
        changePct: -0.32,
        changeUsd: -3.4,
        signal: "HOLD",
        hmmRegime: "NEUTRAL",
        rsi: 49.5,
        ma50: 1052.0,
        ma200: 1018.3,
        macroScore: 52,
        summary: "Auto catalyst demand mixed. Range-bound until PGM deficit narrative reasserts.",
        priceHistory: priceHistory(1048.6, 0.002),
      },
      {
        id: "copper",
        label: "Cu Copper",
        symbol: "Cu",
        ticker: "HG=F",
        color: "#B87333",
        spotPrice: 4.86,
        unit: "lb",
        changePct: 0.51,
        changeUsd: 0.02,
        signal: "HOLD",
        hmmRegime: "NEUTRAL",
        rsi: 54.8,
        ma50: 4.79,
        ma200: 4.62,
        macroScore: 61,
        summary: "Grid and data-centre wiring demand offsets China property drag.",
        priceHistory: priceHistory(4.86, 0.003),
      },
      {
        id: "lithium",
        label: "Li Lithium",
        symbol: "Li",
        ticker: "ALB",
        color: "#6C8EBF",
        spotPrice: 78.42,
        unit: "sh",
        changePct: 2.14,
        changeUsd: 1.64,
        signal: "BUY",
        hmmRegime: "RECOVERY",
        rsi: 66.3,
        ma50: 74.1,
        ma200: 68.9,
        macroScore: 69,
        summary: "EV restocking cycle. ALB proxy rebounding from oversold Q1 base.",
        priceHistory: priceHistory(78.42, 0.008),
      },
      {
        id: "iron",
        label: "Fe Iron",
        symbol: "Fe",
        ticker: "VALE",
        color: "#A0826D",
        spotPrice: 12.38,
        unit: "sh",
        changePct: 0.91,
        changeUsd: 0.11,
        signal: "HOLD",
        hmmRegime: "NEUTRAL",
        rsi: 51.2,
        ma50: 12.1,
        ma200: 11.8,
        macroScore: 48,
        summary: "VALE equity tracks ore fines stabilisation. China stimulus watch.",
        priceHistory: priceHistory(12.38, 0.004),
      },
    ],
  },
  equities: {
    vault: {
      totalEquity: "$100,000",
      invested: "$96,000",
      cashSukuk: "$4,000",
      lifetimePnl: "+$12,480",
      positions: "8",
    },
    regimeVix: "14.2 · LOW VOL",
    exposureTop: [
      { sector: "Materials", weightPct: 42 },
      { sector: "Energy", weightPct: 18 },
      { sector: "Technology", weightPct: 16 },
      { sector: "Healthcare", weightPct: 12 },
      { sector: "Cash", weightPct: 12 },
    ],
    macroSnapshot:
      "HMM regime BULL · Fed pause priced · USD index softening · Sharia gate active on treasury sleeve",
    sectorHeatmap: [
      { sector: "Materials", returnPct: 1.8 },
      { sector: "Energy", returnPct: 0.9 },
      { sector: "Tech", returnPct: 1.2 },
      { sector: "Healthcare", returnPct: 0.4 },
      { sector: "Financials", returnPct: -0.3 },
      { sector: "Utilities", returnPct: 0.1 },
    ],
    topMovers: [
      { ticker: "NVDA", name: "NVIDIA Corp", sector: "Technology", changePct: 2.4, aiScore: 92, sharia: true },
      { ticker: "XOM", name: "Exxon Mobil", sector: "Energy", changePct: 1.1, aiScore: 78, sharia: true },
      { ticker: "LIN", name: "Linde plc", sector: "Materials", changePct: 0.8, aiScore: 74, sharia: true },
      { ticker: "UNH", name: "UnitedHealth", sector: "Healthcare", changePct: -0.6, aiScore: 61, sharia: true },
    ],
    screenerRows: [
      { ticker: "AAPL", name: "Apple Inc", sector: "Technology", changePct: 0.7, aiScore: 85, sharia: true },
      { ticker: "MSFT", name: "Microsoft", sector: "Technology", changePct: 1.0, aiScore: 88, sharia: true },
      { ticker: "JNJ", name: "Johnson & Johnson", sector: "Healthcare", changePct: 0.3, aiScore: 72, sharia: true },
      { ticker: "PG", name: "Procter & Gamble", sector: "Consumer", changePct: 0.2, aiScore: 69, sharia: true },
    ],
  },
  performance: {
    verdict: "PASS · OOS",
    totalReturn: "+18.4%",
    sharpe: "1.62",
    maxDrawdown: "-6.8%",
    navHistory: navHistory(100_000),
    strategyAttribution: [
      { name: "Alpha Core", contributionPct: 1.51, color: "#a1a1aa" },
      { name: "GLD Hedge", contributionPct: 0.16, color: "#d4af37" },
    ],
    treasuryHedge: {
      instrument: "GLD",
      allocationPct: 20,
      status: "SHARIA_FALLBACK_GLD",
      pnlUsd: 32,
    },
    mlConviction: {
      gate: "OPEN",
      walkForwardSharpe: 1.48,
      oosWinRate: 68.4,
      lastRetrain: "2026-06-04",
    },
  },
  agent: {
    messages: [
      {
        role: "user",
        content: "Summarise metals exposure and Sharia gate status.",
        timestamp: "12:04 UTC",
      },
      {
        role: "agent",
        content:
          "Book holds 80% Alpha Core across GC=F, SI=F, IAU with 20% defensive GLD under Phase XXV Sharia fallback. Treasury routing blocked — zero churn invariant maintained. HMM reads BULL at 82%. No new orders until operator authorises pipeline.",
        timestamp: "12:04 UTC",
      },
    ],
    portfolioContext:
      "NAV $100,000 · Gross 96% · 4 active metals sleeves · Halt flag clear · Last pipeline 2026-06-04 SUCCESS",
    oracleScores: [
      { label: "Macro Oracle", score: 78 },
      { label: "Sentiment", score: 71 },
      { label: "Geopolitical", score: 64 },
      { label: "Rates Path", score: 55 },
    ],
    lessonsLearned: [
      "Reduce SI=F add size when vol surface > 90th percentile.",
      "GLD fallback outperformed TLT proxy in rate-cut pause regimes.",
      "Skip equity adds when Sharia gate blocks treasury clearance.",
    ],
  },
  homeTeasers: {
    equityRunnerUps: [
      { ticker: "NVDA", name: "NVIDIA", sector: "Technology", changePct: 2.4, aiScore: 92, sharia: true },
      { ticker: "MSFT", name: "Microsoft", sector: "Technology", changePct: 1.0, aiScore: 88, sharia: true },
      { ticker: "XOM", name: "Exxon Mobil", sector: "Energy", changePct: 1.1, aiScore: 78, sharia: true },
    ],
    metalsTeaser: {
      regime: "BULL",
      confidencePct: 82,
      primarySignal: "BUY",
      spotGold: "$3,382.40",
      oracleBias: "Risk-on precious metals · GLD sleeve active",
    },
  },
};
