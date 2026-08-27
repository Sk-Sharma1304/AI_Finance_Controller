import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyAgent:

    def __init__(self):
        self.model = IsolationForest(
            contamination=0.25,
            random_state=42
        )

    def detect(self, data):

        data = data.copy()

        features = [
            "settlement_ratio",
            "deviation_ratio",
            "difference_to_payment_ratio"
        ]

        X = data[features]

        # Train Isolation Forest
        self.model.fit(X)

        # Predict
        predictions = self.model.predict(X)

        data["ml_anomaly"] = predictions

        data["anomaly_status"] = data[
            "ml_anomaly"
        ].apply(
            lambda x:
            "ANOMALY" if x == -1 else "NORMAL"
        )

        return data


def run_anomaly_agent(data):

    agent = AnomalyAgent()

    return agent.detect(data)


def main():

    print("\n========================================")
    print("AI FINANCE CONTROLLER")
    print("ANOMALY AGENT")
    print("========================================")

    data = pd.read_csv(
        "data/reconciliation_features.csv"
    )

    result = run_anomaly_agent(data)

    anomalies = result[
        result["anomaly_status"] == "ANOMALY"
    ]

    print(f"Total transactions : {len(result)}")
    print(f"ML anomalies       : {len(anomalies)}")

    print("\n========================================")
    print("DETECTED ANOMALIES")
    print("========================================")

    columns = [
        "payment_id",
        "scenario",
        "settlement_ratio",
        "deviation_ratio",
        "difference_to_payment_ratio",
        "anomaly_status"
    ]

    print(
        anomalies[columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()