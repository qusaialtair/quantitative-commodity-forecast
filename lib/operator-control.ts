import type { OverrideAction, OverrideResponse } from "@/lib/api/override";

export interface OperatorControlState {
  action: OverrideAction;
  halted: boolean;
  message: string;
  pipeline?: string | null;
  at: string;
  simulated: boolean;
}

export function sandboxOverrideResult(action: OverrideAction): OperatorControlState {
  const at = new Date().toISOString();
  if (action === "HALT") {
    return {
      action,
      halted: true,
      at,
      simulated: true,
      message:
        "Emergency halt engaged (demo). All routing suspended — liquidate-to-cash posture armed. No live broker orders sent in sandbox.",
    };
  }
  return {
    action,
    halted: false,
    at,
    simulated: true,
    pipeline: "SIMULATED",
      message:
        "Pipeline authorization accepted (demo). Toggle Live API + Authorize to spawn master_controller.py on the backend.",
  };
}

export function liveOverrideResult(
  action: OverrideAction,
  response: OverrideResponse
): OperatorControlState {
  return {
    action,
    halted: Boolean(response.halted ?? response.trading_halted),
    message: response.message ?? `Override ${action} acknowledged`,
    pipeline: response.pipeline ?? null,
    at: new Date().toISOString(),
    simulated: false,
  };
}

export function appendOperatorNote(
  summary: string,
  state: OperatorControlState
): string {
  const tag = state.halted ? "TRADING HALTED" : "PIPELINE AUTHORIZED";
  const mode = state.simulated ? " · DEMO" : "";
  return `${summary}\n\n[OPERATOR ${tag}${mode}] ${state.message}`;
}
