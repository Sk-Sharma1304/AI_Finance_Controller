import pandas as pd

from sklearn.ensemble import IsolationForest


# ==================================================
# 1. LOAD DATA
# ==================================================

transactions = pd.read_csv(
    "finance_controller_dataset.csv"
)

ground_truth = pd.read_csv(
    "ground_truth.csv"
)


# ==================================================
# 2. MERGE DATA
# ==================================================

data = transactions.merge(
    ground_truth,
    on="payment_id"
)


# ==================================================
# 3. CALCULATE EXPECTED SETTLEMENT
# ==================================================

data["calculated_expected"] = (
    data["payment_amount"]
    - data["fee"]
    - data["tax"]
    - data["refund"]
    + data["adjustment"]
)


# ==================================================
# 4. CALCULATE SETTLEMENT DIFFERENCE
# ==================================================

data["settlement_difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


data["financial_impact"] = (
    data["settlement_difference"].abs()
)


# ==================================================
# 5. FINANCIAL EXCEPTION CLASSIFICATION
# ==================================================

def classify_exception(row):

    scenario = row["scenario"]

    if scenario == "normal":
        return "NORMAL"

    elif scenario == "refund":
        return "REFUND"

    elif scenario == "adjustment":
        return "ADJUSTMENT"

    elif scenario == "amount_discrepancy":
        return "AMOUNT_DISCREPANCY"

    elif scenario == "missing_settlement":
        return "MISSING_SETTLEMENT"

    elif scenario == "duplicate_transaction":
        return "DUPLICATE_TRANSACTION"

    elif scenario == "wrong_settlement":
        return "WRONG_SETTLEMENT"

    elif scenario == "unexplained_difference":
        return "UNEXPLAINED_DIFFERENCE"

    return "UNKNOWN"


data["exception_type"] = data.apply(
    classify_exception,
    axis=1
)


# ==================================================
# 6. SELECT ML FEATURES
# ==================================================

features = [
    "payment_amount",
    "fee",
    "tax",
    "refund",
    "adjustment",
    "actual_settlement",
    "settlement_difference"
]


# ==================================================
# 7. TRAIN ML MODEL ONLY ON NORMAL TRANSACTIONS
# ==================================================

normal_data = data[
    data["scenario"] == "normal"
]


X_train = normal_data[features]


model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)


model.fit(X_train)


# ==================================================
# 8. PREDICT ALL TRANSACTIONS
# ==================================================

X_all = data[features]


data["ml_prediction"] = model.predict(
    X_all
)


data["ml_anomaly_score"] = (
    model.decision_function(X_all)
)


data["ml_anomaly"] = data[
    "ml_prediction"
].apply(
    lambda x: "ANOMALY"
    if x == -1
    else "NORMAL"
)


# ==================================================
# 9. FINANCIAL EXCEPTION FLAG
# ==================================================

financial_exception_types = [
    "AMOUNT_DISCREPANCY",
    "MISSING_SETTLEMENT",
    "DUPLICATE_TRANSACTION",
    "WRONG_SETTLEMENT",
    "UNEXPLAINED_DIFFERENCE"
]


data["financial_exception"] = (
    data["exception_type"]
    .isin(financial_exception_types)
)


# ==================================================
# 10. HYBRID DECISION
# ==================================================

def final_decision(row):

    financial_exception = row["financial_exception"]
    ml_anomaly = row["ml_anomaly"]

    if financial_exception and ml_anomaly == "ANOMALY":
        return "CONFIRMED_HIGH_PRIORITY"

    elif financial_exception:
        return "FINANCIAL_EXCEPTION"

    elif ml_anomaly == "ANOMALY":
        return "ML_REVIEW"

    else:
        return "NORMAL"


data["final_decision"] = data.apply(
    final_decision,
    axis=1
)


# ==================================================
# 11. DISPLAY SUMMARY
# ==================================================

print("\n========================================")
print("AI FINANCE CONTROLLER")
print("HYBRID ANALYSIS")
print("========================================")

print(
    f"Total transactions : {len(data)}"
)


print(
    f"Financial exceptions : "
    f"{data['financial_exception'].sum()}"
)


print(
    f"ML anomalies : "
    f"{(data['ml_anomaly'] == 'ANOMALY').sum()}"
)


# ==================================================
# 12. FINAL DECISION SUMMARY
# ==================================================

print("\n========================================")
print("FINAL DECISION SUMMARY")
print("========================================")

print(
    data["final_decision"]
    .value_counts()
)


# ==================================================
# 13. HIGH PRIORITY TRANSACTIONS
# ==================================================

print("\n========================================")
print("HIGH PRIORITY TRANSACTIONS")
print("========================================")


priority = data[
    data["final_decision"] ==
    "CONFIRMED_HIGH_PRIORITY"
].copy()


priority = priority.sort_values(
    by="financial_impact",
    ascending=False
)


columns = [
    "payment_id",
    "exception_type",
    "expected_settlement",
    "actual_settlement",
    "settlement_difference",
    "financial_impact",
    "ml_anomaly_score",
    "final_decision"
]


print(
    priority[columns]
    .to_string(index=False)
)