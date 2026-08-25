import pandas as pd

from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# 1. Load transaction dataset
# --------------------------------------------------

data = pd.read_csv(
    "finance_controller_dataset.csv"
)


# --------------------------------------------------
# 2. Create financial features
# --------------------------------------------------

data["expected_settlement"] = (
    data["payment_amount"]
    - data["fee"]
    - data["tax"]
    - data["refund"]
    + data["adjustment"]
)


data["settlement_difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


# --------------------------------------------------
# 3. Select features for ML
# --------------------------------------------------

features = [
    "payment_amount",
    "fee",
    "tax",
    "refund",
    "adjustment",
    "actual_settlement",
    "settlement_difference"
]


X = data[features]


# --------------------------------------------------
# 4. Create Isolation Forest model
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.25,
    random_state=42
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model.fit(X)


# --------------------------------------------------
# 6. Predict anomalies
# --------------------------------------------------

data["anomaly_prediction"] = model.predict(X)


# --------------------------------------------------
# 7. Convert prediction into readable format
# --------------------------------------------------

data["anomaly_status"] = data[
    "anomaly_prediction"
].apply(
    lambda x: "ANOMALY"
    if x == -1
    else "NORMAL"
)


# --------------------------------------------------
# 8. Get anomaly score
# --------------------------------------------------

data["anomaly_score"] = (
    model.decision_function(X)
)


# --------------------------------------------------
# 9. Select anomalies
# --------------------------------------------------

anomalies = data[
    data["anomaly_status"] == "ANOMALY"
].copy()


# --------------------------------------------------
# 10. Sort by anomaly score
# --------------------------------------------------

anomalies = anomalies.sort_values(
    by="anomaly_score"
)


# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

print("\n========================================")
print("AI FINANCE CONTROLLER")
print("ANOMALY DETECTION")
print("========================================")

print(
    f"Total transactions : {len(data)}"
)

print(
    f"AI anomalies       : {len(anomalies)}"
)


# --------------------------------------------------
# 12. Display anomaly records
# --------------------------------------------------

print("\n========================================")
print("DETECTED ANOMALIES")
print("========================================")

columns_to_show = [
    "payment_id",
    "payment_amount",
    "actual_settlement",
    "expected_settlement",
    "settlement_difference",
    "anomaly_score",
    "anomaly_status"
]


print(
    anomalies[columns_to_show]
    .to_string(index=False)
)