import { DashboardHeader } from "@/components/dashboard-header"
import { KpiCards } from "@/components/kpi-cards"
import { DecisionChart, RiskDonut } from "@/components/overview-charts"
import { Workspace } from "@/components/workspace"
import {
  getSummary,
  getTransactions,
  getPriorityQueue,
  getDataSource,
} from "@/lib/pipeline"

export default async function Page() {
  const [summary, transactions, queue, source] = await Promise.all([
    getSummary(),
    getTransactions(),
    getPriorityQueue(),
    getDataSource(),
  ])

  return (
    <div className="min-h-svh">
      <DashboardHeader
        total={summary.total}
        llmEvaluated={summary.llmEvaluated}
        source={source}
      />

      <main className="mx-auto max-w-[1400px] space-y-5 px-5 py-6 md:px-8 md:py-8">
        <section aria-label="Key metrics">
          <KpiCards summary={summary} />
        </section>

        <section className="grid gap-4 lg:grid-cols-2" aria-label="Overview charts">
          <DecisionChart counts={summary.decisionCounts} />
          <RiskDonut counts={summary.riskCounts} />
        </section>

        <section aria-label="Operations workspace">
          <Workspace transactions={transactions} queue={queue} summary={summary} />
        </section>

        <footer className="border-t border-border pt-5 text-xs text-muted-foreground">
          <p className="text-pretty">
            AI Finance Controller — multi-agent settlement reconciliation and
            fraud-control pipeline. Verdicts are computed live from{" "}
            {summary.total} synthetic transactions running the full
            reconciliation → duplicate → anomaly → risk → investigation → LLM →
            decision → action chain.
          </p>
        </footer>
      </main>
    </div>
  )
}
