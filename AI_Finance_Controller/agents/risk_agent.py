import pandas as pd


class RiskAgent:

    def calculate_risk(self, data):

        result = data.copy()

        # Financial impact
        result["financial_impact"] = (
            result["expected_settlement"]
            - result["actual_settlement"]
        ).abs()

        # Base risk score
        result["risk_score"] = 0

        # --------------------------------------------------
        # Exception based scoring
        # --------------------------------------------------

        result.loc[
            result["scenario"] == "amount_discrepancy",
            "risk_score"
        ] = 20

        result.loc[
            result["scenario"] == "wrong_settlement",
            "risk_score"
        ] = 30

        result.loc[
            result["scenario"] == "unexplained_difference",
            "risk_score"
        ] = 35

        result.loc[
            result["scenario"] == "duplicate_transaction",
            "risk_score"
        ] = 45

        result.loc[
            result["scenario"] == "missing_settlement",
            "risk_score"
        ] = 50

        # --------------------------------------------------
        # Increase risk based on financial impact
        # --------------------------------------------------

        result.loc[
            result["financial_impact"] >= 500,
            "risk_score"
        ] += 20

        result.loc[
            (result["financial_impact"] >= 250) &
            (result["financial_impact"] < 500),
            "risk_score"
        ] += 10

        # --------------------------------------------------
        # Cap score at 100
        # --------------------------------------------------

        result["risk_score"] = result["risk_score"].clip(upper=100)

        # --------------------------------------------------
        # Convert score to risk level
        # --------------------------------------------------

        def get_risk_level(score):

            if score >= 70:
                return "CRITICAL"

            elif score >= 40:
                return "HIGH"

            elif score >= 20:
                return "MEDIUM"

            else:
                return "LOW"

        result["risk_level"] = result["risk_score"].apply(
            get_risk_level
        )

        return result


def run_risk_agent(df):

    agent = RiskAgent()

    return agent.calculate_risk(df)


def main():

    print("\n========================================")
    print("AI FINANCE CONTROLLER")
    print("RISK AGENT")
    print("========================================")

    data = pd.read_csv(
        "data/ground_truth.csv"
    )

    transactions = pd.read_csv(
        "data/finance_controller_dataset.csv"
    )

    data = data.merge(
        transactions,
        on="payment_id"
    )

    agent = RiskAgent()

    result = agent.calculate_risk(data)

    exceptions = result[
        result["scenario"] != "normal"
    ]

    print(f"Total transactions : {len(result)}")
    print(f"Financial exceptions : {len(exceptions)}")

    print("\n========================================")
    print("RISK SUMMARY")
    print("========================================")

    print(
        exceptions["risk_level"].value_counts()
    )

    print("\n========================================")
    print("HIGH PRIORITY TRANSACTIONS")
    print("========================================")

    priority = exceptions[
        exceptions["risk_level"].isin(
            ["HIGH", "CRITICAL"]
        )
    ].sort_values(
        "risk_score",
        ascending=False
    )

    columns = [
        "payment_id",
        "scenario",
        "expected_settlement",
        "actual_settlement",
        "financial_impact",
        "risk_score",
        "risk_level"
    ]

    print(
        priority[columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()