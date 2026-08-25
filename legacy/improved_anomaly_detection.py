import pandas as pd

from sklearn.ensemble import IsolationForest


# ==================================================
# 1. LOAD FEATURE DATASET
# ==================================================

data = pd.read_csv(
    "finance_features.csv"
)


# ==================================================
# 2. ML FEATURES
# ==================================================

features = [
    "fee_ratio",
    "tax_ratio",
    "refund_ratio",
    "adjustment_ratio",
    "settlement_ratio",
    "settlement_deviation_ratio",
    "absolute_deviation_ratio"
]


X = data[features]


# ==================================================
# 3. TRAIN ONLY ON NORMAL TRANSACTIONS
# ==================================================

normal_data = data[
    data["scenario"] == "normal"
]

X_train = normal_data[features]


print("\n========================================")
print("AI FINANCE CONTROLLER")
print("IMPROVED ANOMALY DETECTION")
print("========================================")

print(
    f"Total transactions : {len(data)}"
)

print(
    f"Normal training data : {len(X_train)}"
)


# ==================================================
# 4. CREATE MODEL
# ==================================================

model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42
)


# ==================================================
# 5. TRAIN
# ==================================================

model.fit(X_train)


# ==================================================
# 6. PREDICT
# ==================================================

data["ml_prediction"] = model.predict(
    X
)


# ==================================================
# 7. ANOMALY STATUS
# ==================================================

data["ml_anomaly"] = data[
    "ml_prediction"
].apply(
    lambda x:
        "ANOMALY"
        if x == -1
        else "NORMAL"
)


# ==================================================
# 8. ANOMALY SCORE
# ==================================================

data["anomaly_score"] = (
    -model.decision_function(X)
)


# ==================================================
# 9. GET ANOMALIES
# ==================================================

anomalies = data[
    data["ml_anomaly"] == "ANOMALY"
].copy()


anomalies = anomalies.sort_values(
    by="anomaly_score",
    ascending=False
)


# ==================================================
# 10. DISPLAY
# ==================================================

print(
    f"ML anomalies : {len(anomalies)}"
)


print("\n========================================")
print("DETECTED ANOMALIES")
print("========================================")


columns = [
    "payment_id",
    "scenario",
    "settlement_ratio",
    "settlement_deviation_ratio",
    "absolute_deviation_ratio",
    "anomaly_score",
    "ml_anomaly"
]


print(
    anomalies[columns]
    .to_string(index=False)
)