import os
import pandas as pd


class InvestigationAgent:

    def __init__(self):
        self.name = "Investigation Agent"

    def investigate(self, row):

        evidence = []
        explanation = []

        scenario = str(
            row.get("scenario", "")
        ).lower()

        risk_level = str(
            row.get("risk_level", "")
        )

        financial_impact = row.get(
            "financial_impact",
            0
        )

        ml_anomaly = row.get(
            "ml_anomaly",
            1
        )

        duplicate_flag = row.get(
            "duplicate_flag",
            False
        )

        # =========================================================
        # Scenario investigation
        # =========================================================

        if scenario == "missing_settlement":

            evidence.append(
                "Settlement amount is missing or incomplete."
            )

            explanation.append(
                "The transaction was processed but the expected "
                "settlement was not received."
            )

        elif scenario == "duplicate_transaction":

            evidence.append(
                "Duplicate transaction pattern detected."
            )

            explanation.append(
                "Multiple transactions appear to represent "
                "the same financial event."
            )

        elif scenario == "wrong_settlement":

            evidence.append(
                "Settlement amount does not match expected amount."
            )

            explanation.append(
                "The actual settlement differs from the expected "
                "settlement value."
            )

        elif scenario == "unexplained_difference":

            evidence.append(
                "Unexplained financial difference detected."
            )

            explanation.append(
                "A discrepancy exists between expected and observed "
                "financial values."
            )

        # =========================================================
        # ML investigation
        # =========================================================

        if ml_anomaly == -1:

            evidence.append(
                "Machine learning model classified the transaction "
                "as anomalous."
            )

            explanation.append(
                "The transaction has characteristics that differ "
                "significantly from normal transaction patterns."
            )

        # =========================================================
        # Duplicate investigation
        # =========================================================

        if duplicate_flag:

            evidence.append(
                "Duplicate detection agent confirmed a duplicate."
            )

        # =========================================================
        # Financial impact
        # =========================================================

        try:
            financial_impact = float(
                financial_impact
            )
        except:
            financial_impact = 0

        if financial_impact > 0:

            evidence.append(
                f"Financial impact: ₹{financial_impact:.2f}"
            )

        # =========================================================
        # Overall explanation
        # =========================================================

        if not explanation:

            explanation.append(
                "No significant financial anomaly was identified."
            )

        investigation_summary = " ".join(
            explanation
        )

        evidence_summary = " | ".join(
            evidence
        )

        # =========================================================
        # Recommendation
        # =========================================================

        if risk_level == "CRITICAL":

            recommendation = (
                "Immediate manual investigation and financial "
                "control review required."
            )

        elif risk_level == "HIGH":

            recommendation = (
                "Transaction should be investigated before "
                "financial settlement is finalized."
            )

        elif risk_level == "MEDIUM":

            recommendation = (
                "Transaction should be monitored and reviewed "
                "if additional anomalies occur."
            )

        else:

            recommendation = (
                "No immediate investigation required."
            )

        return {
            "investigation_summary": investigation_summary,
            "evidence": evidence_summary,
            "investigation_recommendation": recommendation
        }

    def run(self, df):

        result = df.copy()

        investigations = result.apply(
            self.investigate,
            axis=1
        )

        investigation_df = pd.DataFrame(
            investigations.tolist(),
            index=result.index
        )

        result = pd.concat(
            [
                result,
                investigation_df
            ],
            axis=1
        )

        return result


def run_investigation_agent(df):

    agent = InvestigationAgent()

    return agent.run(df)


if __name__ == "__main__":

    df = pd.read_csv(
        "outputs/risk_results.csv"
    )

    result = run_investigation_agent(df)

    os.makedirs("outputs", exist_ok=True)

    result.to_csv(
        "outputs/investigation_results.csv",
        index=False
    )

    print("=" * 50)
    print("INVESTIGATION AGENT")
    print("=" * 50)

    print(
        result[
            [
                "payment_id",
                "risk_level",
                "investigation_summary",
                "investigation_recommendation"
            ]
        ].head(20).to_string(index=False)
    )