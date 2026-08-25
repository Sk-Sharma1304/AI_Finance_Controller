import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# 1. LOAD FEATURE DATASET
# ==================================================

data = pd.read_csv(
    "reconciliation_features.csv"
)


# ==================================================
# 2. FEATURES
# ==================================================

features = [
    "settlement_ratio",
    "deviation_ratio",
    "deviation_direction",
    "is_missing_settlement",
    "is_over_settlement",
    "is_under_settlement",
    "difference_to_payment_ratio"
]


# ==================================================
# 3. CREATE GROUND-TRUTH LABEL
# ==================================================

# normal = 0
# anything else = anomaly

data["actual_label"] = (
    data["scenario"] != "normal"
).astype(int)


# ==================================================
# 4. TRAIN / TEST SPLIT
# ==================================================

# First 48 normal transactions → training
normal_data = data[
    data["scenario"] == "normal"
]

train_data = normal_data.iloc[:48]

# Remaining 12 normal transactions → testing
normal_test = normal_data.iloc[48:]

# All 40 anomalies → testing
anomaly_test = data[
    data["scenario"] != "normal"
]

test_data = pd.concat(
    [normal_test, anomaly_test],
    ignore_index=True
)


X_train = train_data[features]

X_test = test_data[features]

y_test = test_data["actual_label"]


# ==================================================
# 5. TRAIN ISOLATION FOREST
# ==================================================

model = IsolationForest(
    n_estimators=300,
    contamination=0.40,
    random_state=42
)


model.fit(X_train)


# ==================================================
# 6. PREDICTION
# ==================================================

predictions = model.predict(
    X_test
)


# Isolation Forest:
# -1 = anomaly
#  1 = normal

y_pred = (
    predictions == -1
).astype(int)


# ==================================================
# 7. ANOMALY SCORE
# ==================================================

# Larger score = more anomalous

anomaly_scores = (
    -model.decision_function(X_test)
)


# ==================================================
# 8. METRICS
# ==================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    anomaly_scores
)


# ==================================================
# 9. DISPLAY
# ==================================================

print("\n========================================")
print("RECONCILIATION FEATURE MODEL")
print("EVALUATION")
print("========================================")

print(
    f"Training normal transactions : {len(train_data)}"
)

print(
    f"Testing normal transactions  : {len(normal_test)}"
)

print(
    f"Testing anomalies            : {len(anomaly_test)}"
)

print(
    f"Total test transactions      : {len(test_data)}"
)


# ==================================================
# 10. CONFUSION MATRIX
# ==================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==================================================
# 11. METRICS
# ==================================================

print("\n========================================")
print("MODEL METRICS")
print("========================================")

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ==================================================
# 12. CLASSIFICATION REPORT
# ==================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "ANOMALY"
        ],
        zero_division=0
    )
)


# ==================================================
# 13. SCENARIO-LEVEL ANALYSIS
# ==================================================

test_data = test_data.copy()

test_data["predicted_anomaly"] = y_pred

test_data["anomaly_score"] = anomaly_scores


print("\n========================================")
print("SCENARIO LEVEL DETECTION")
print("========================================")

scenario_summary = (
    test_data
    .groupby("scenario")
    .agg(
        total=("payment_id", "count"),
        detected_anomalies=(
            "predicted_anomaly",
            "sum"
        )
    )
)

scenario_summary[
    "detection_rate"
] = (
    scenario_summary["detected_anomalies"]
    / scenario_summary["total"]
)

print(
    scenario_summary
)