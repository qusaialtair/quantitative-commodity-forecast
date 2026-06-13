export type TradeSignal = "BUY" | "HOLD" | "SELL";

export interface MetalQuote {
  id: string;
  label: string;
  symbol: string;
  ticker: string;
  color: string;
  spotPrice: number;
  unit: string;
  changePct: number;
  changeUsd: number;
  signal: TradeSignal;
  hmmRegime: string;
  rsi: number;
  ma50: number;
  ma200: number;
  macroScore: number;
  summary: string;
  priceHistory: { date: string; price: number }[];
}

export interface MetalsPanelData {
  instruments: MetalQuote[];
  marketSummary: string;
  analystNote: string;
}

export interface EquityVaultKpis {
  totalEquity: string;
  invested: string;
  cashSukuk: string;
  lifetimePnl: string;
  positions: string;
}

export interface EquityMover {
  ticker: string;
  name: string;
  sector: string;
  changePct: number;
  aiScore: number;
  sharia: boolean;
}

export interface EquitiesPanelData {
  vault: EquityVaultKpis;
  regimeVix: string;
  exposureTop: { sector: string; weightPct: number }[];
  macroSnapshot: string;
  sectorHeatmap: { sector: string; returnPct: number }[];
  topMovers: EquityMover[];
  screenerRows: EquityMover[];
}

export interface PerformancePanelData {
  verdict: string;
  totalReturn: string;
  sharpe: string;
  maxDrawdown: string;
  navHistory: { date: string; nav: number }[];
  strategyAttribution: { name: string; contributionPct: number; color: string }[];
  treasuryHedge: {
    instrument: string;
    allocationPct: number;
    status: string;
    pnlUsd: number;
  };
  mlConviction: {
    gate: string;
    walkForwardSharpe: number;
    oosWinRate: number;
    lastRetrain: string;
  };
}

export interface AgentMessage {
  role: "user" | "agent";
  content: string;
  timestamp: string;
}

export interface AgentPanelData {
  messages: AgentMessage[];
  portfolioContext: string;
  oracleScores: { label: string; score: number }[];
  lessonsLearned: string[];
}

export interface HomeTeaserData {
  equityRunnerUps: EquityMover[];
  metalsTeaser: {
    regime: string;
    confidencePct: number;
    primarySignal: TradeSignal;
    spotGold: string;
    oracleBias: string;
  };
}

export interface TerminalSections {
  metals: MetalsPanelData;
  equities: EquitiesPanelData;
  performance: PerformancePanelData;
  agent: AgentPanelData;
  homeTeasers: HomeTeaserData;
}
