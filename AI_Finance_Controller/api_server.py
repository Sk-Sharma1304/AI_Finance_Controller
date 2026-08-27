"""
AI Finance Controller — API layer
===================================

This is the piece that merges the Python ML backend with the
Next.js frontend. It runs the exact same 8-agent pipeline as
``main.py`` (same IsolationForest, trained fresh on every request,
same rule-based agents) and serves the result as JSON so the
dashboard can render *real* model output instead of static data.

Run with:

    uvicorn api_server:app --reload --port 8000

Endpoints:
    GET  /api/health   -> service + model status
    GET  /api/results  -> {transactions, summary, generatedAt} (cached)
    POST /api/rerun     -> forces the pipeline to re-run and re-train
"""

import math
import os
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.reconciliation_agent import run_reconciliation_agent
from agents.duplicates_detection_agent import run_duplicate_agent
from agents.anomaly_agent import run_anomaly_agent
from agents.risk_agent import run_risk_agent
from agents.investigation_agent import run_investigation_agent
from agents.llm_investigation_agent import run_llm_investigation_agent
from agents.decision_agent import run_decision_agent
from agents.action_agent import run_action_agent

TRANSACTIONS_FILE = "data/finance_controller_dataset.csv"
GROUND_TRUTH_FILE = "data/ground_truth.csv"
OUTPUT_DIR = "outputs"

app = FastAPI(title="AI Finance Controller API", version="1.0")

_origins = os.environ.get(
    "FRONTEND_ORIGIN", "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, Any] = {}


def _load_data() -> pd.DataFrame:
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    df = transactions.merge(ground_truth, on="payment_id")
    df["amount"] = df["payment_amount"]
    return df


def _run_pipeline() -> pd.DataFrame:
    """Runs every agent in the same order as orchestrator.main(),
    including a fresh IsolationForest fit inside run_anomaly_agent.
    """
    df = _load_data()
    df = run_reconciliation_agent(df)
    df = run_duplicate_agent(df)
    df = run_anomaly_agent(df)
    df = run_risk_agent(df)
    df = run_investigation_agent(df)
    df = run_llm_investigation_agent(df)
    df = run_decision_agent(df)
    df = run_action_agent(df)
    return df


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _row_to_json(row: "pd.Series") -> dict:
    record = {k: _clean(v) for k, v in row.to_dict().items()}
    record["duplicate_flag"] = bool(record.get("duplicate_flag", False))
    record["ml_anomaly"] = int(record.get("ml_anomaly", 1))
    evidence = record.get("evidence") or ""
    record["evidence"] = [
        e.strip() for e in str(evidence).split("|") if e.strip()
    ]
    return record


def _summarize(records: list[dict]) -> dict:
    decision_counts = {
        "NORMAL": 0,
        "FINANCIAL_EXCEPTION": 0,
        "ML_REVIEW": 0,
        "AI_ESCALATED_REVIEW": 0,
        "CONFIRMED_HIGH_PRIORITY": 0,
    }
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    action_counts: dict[str, int] = {}

    total_impact = 0.0
    at_risk_impact = 0.0
    ml_anomalies = 0
    duplicates = 0
    llm_evaluated = 0

    for r in records:
        decision_counts[r["final_decision"]] = (
            decision_counts.get(r["final_decision"], 0) + 1
        )
        risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1
        action_counts[r["recommended_action"]] = (
            action_counts.get(r["recommended_action"], 0) + 1
        )

        impact = r.get("financial_impact") or 0
        total_impact += impact
        if r["final_decision"] != "NORMAL":
            at_risk_impact += impact
        if r["ml_anomaly"] == -1:
            ml_anomalies += 1
        if r["duplicate_flag"]:
            duplicates += 1
        if r.get("llm_risk_opinion") != "NOT_EVALUATED":
            llm_evaluated += 1

    total = len(records)
    auto_cleared = decision_counts["NORMAL"]

    return {
        "total": total,
        "autoCleared": auto_cleared,
        "autoClearedPct": (auto_cleared / total * 100) if total else 0,
        "flagged": total - auto_cleared,
        "aiEscalated": decision_counts["AI_ESCALATED_REVIEW"],
        "confirmedHighPriority": decision_counts["CONFIRMED_HIGH_PRIORITY"],
        "mlAnomalies": ml_anomalies,
        "duplicates": duplicates,
        "totalImpact": total_impact,
        "atRiskImpact": at_risk_impact,
        "decisionCounts": decision_counts,
        "riskCounts": risk_counts,
        "actionCounts": action_counts,
        "llmEvaluated": llm_evaluated,
    }


def _build_payload(persist: bool = True) -> dict:
    df = _run_pipeline()

    if persist:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(os.path.join(OUTPUT_DIR, "action_results.csv"), index=False)

    records = [_row_to_json(row) for _, row in df.iterrows()]
    payload = {
        "transactions": records,
        "summary": _summarize(records),
        "generatedAt": time.time(),
        "llmEnabled": bool(os.environ.get("OPENAI_API_KEY")),
    }
    _cache["data"] = payload
    return payload


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "IsolationForest(contamination=0.25, random_state=42)",
        "llm_enabled": bool(os.environ.get("OPENAI_API_KEY")),
        "cached": "data" in _cache,
    }


@app.get("/api/results")
def results(refresh: bool = False):
    """Returns the pipeline output. Cached after first run so the
    dashboard doesn't retrain on every page load; pass ?refresh=true
    (or hit /api/rerun) to force a fresh run.
    """
    if refresh or "data" not in _cache:
        return _build_payload()
    return _cache["data"]


@app.post("/api/rerun")
def rerun():
    """Forces the full pipeline (including a fresh IsolationForest
    fit) to run again."""
    return _build_payload()


@app.get("/")
def root():
    return {
        "service": "AI Finance Controller API",
        "endpoints": ["/api/health", "/api/results", "/api/rerun"],
    }
