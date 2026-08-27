import type { FinalDecision, RiskLevel } from "./types"

export function formatINR(value: number, opts?: { compact?: boolean }): string {
  if (opts?.compact && Math.abs(value) >= 100000) {
    return `₹${(value / 100000).toFixed(2)}L`
  }
  return `₹${value.toLocaleString("en-IN", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })}`
}

export const DECISION_META: Record<
  FinalDecision,
  { label: string; short: string; chartColor: string; description: string }
> = {
  NORMAL: {
    label: "Auto-cleared",
    short: "Normal",
    chartColor: "var(--chart-1)",
    description: "Reconciled cleanly within control limits.",
  },
  FINANCIAL_EXCEPTION: {
    label: "Financial exception",
    short: "Exception",
    chartColor: "var(--chart-2)",
    description: "A reconciliation exception the rules caught.",
  },
  ML_REVIEW: {
    label: "ML review",
    short: "ML review",
    chartColor: "var(--chart-3)",
    description: "Statistical anomaly flagged by the model.",
  },
  AI_ESCALATED_REVIEW: {
    label: "AI-escalated",
    short: "AI-escalated",
    chartColor: "var(--chart-4)",
    description: "Rules said normal — the LLM disagreed.",
  },
  CONFIRMED_HIGH_PRIORITY: {
    label: "Confirmed high priority",
    short: "Critical",
    chartColor: "var(--chart-5)",
    description: "Rule risk and ML anomaly both fired.",
  },
}

export const RISK_META: Record<
  RiskLevel,
  { token: string; label: string }
> = {
  LOW: { token: "risk-low", label: "Low" },
  MEDIUM: { token: "risk-medium", label: "Medium" },
  HIGH: { token: "risk-high", label: "High" },
  CRITICAL: { token: "risk-critical", label: "Critical" },
}

// tailwind utility class map (avoids dynamic class purging issues)
export const RISK_TEXT: Record<RiskLevel, string> = {
  LOW: "text-risk-low",
  MEDIUM: "text-risk-medium",
  HIGH: "text-risk-high",
  CRITICAL: "text-risk-critical",
}

export const RISK_BG: Record<RiskLevel, string> = {
  LOW: "bg-risk-low/12 text-risk-low border-risk-low/25",
  MEDIUM: "bg-risk-medium/12 text-risk-medium border-risk-medium/25",
  HIGH: "bg-risk-high/15 text-risk-high border-risk-high/30",
  CRITICAL: "bg-risk-critical/15 text-risk-critical border-risk-critical/35",
}

export const DECISION_BG: Record<FinalDecision, string> = {
  NORMAL: "bg-risk-low/12 text-risk-low border-risk-low/25",
  FINANCIAL_EXCEPTION: "bg-risk-medium/12 text-risk-medium border-risk-medium/25",
  ML_REVIEW: "bg-risk-high/14 text-risk-high border-risk-high/28",
  AI_ESCALATED_REVIEW:
    "bg-chart-4/15 text-chart-4 border-chart-4/30",
  CONFIRMED_HIGH_PRIORITY:
    "bg-risk-critical/15 text-risk-critical border-risk-critical/35",
}

export function scenarioLabel(scenario: string): string {
  return scenario
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}
