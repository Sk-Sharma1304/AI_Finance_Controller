import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


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
# 3. CREATE FEATURES
# ==================================================

data["settlement_difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


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
# 4. CREATE GROUND-TRUTH LABEL
# ==================================================

# NORMAL = 0
# ANOMALY = 1

data["true_label"] = (
    data["scenario"] != "normal"
).astype(int)


# ==================================================
# 5. SPLIT NORMAL DATA
# ==================================================

normal_data = data[
    data["scenario"] == "normal"
].copy()


normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.20,
    random_state=42
)


# ==================================================
# 6. ALL ANOMALOUS DATA
# ==================================================

anomaly_data = data[
    data["scenario"] != "normal"
].copy()


# ==================================================
# 7. TRAINING DATA
# ==================================================

X_train = normal_train[features]


# ==================================================
# 8. TEST DATA
# ==================================================

test_data = pd.concat(
    [
        normal_test,
        anomaly_data
    ],
    ignore_index=True
)


X_test = test_data[features]

y_test = test_data["true_label"]


# ==================================================
# 9. TRAIN ISOLATION FOREST
# ==================================================

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)


model.fit(X_train)


# ==================================================
# 10. PREDICTION
# ==================================================

predictions = model.predict(X_test)


# Isolation Forest:
# -1 = anomaly
# +1 = normal

y_pred = (
    predictions == -1
).astype(int)


# ==================================================
# 11. ANOMALY SCORE
# ==================================================

scores = -model.decision_function(
    X_test
)


# ==================================================
# 12. EVALUATION
# ==================================================

print("\n========================================")
print("ANOMALY MODEL EVALUATION")
print("========================================")

print(
    f"Training normal transactions : "
    f"{len(normal_train)}"
)

print(
    f"Testing normal transactions  : "
    f"{len(normal_test)}"
)

print(
    f"Testing anomalies            : "
    f"{len(anomaly_data)}"
)

print(
    f"Total test transactions      : "
    f"{len(test_data)}"
)


# ==================================================
# 13. CONFUSION MATRIX
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
# 14. PRECISION
# ==================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print(
    f"\nPrecision : {precision:.4f}"
)


# ==================================================
# 15. RECALL
# ==================================================

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print(
    f"Recall    : {recall:.4f}"
)


# ==================================================
# 16. F1 SCORE
# ==================================================

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ==================================================
# 17. ROC-AUC
# ==================================================

try:

    auc = roc_auc_score(
        y_test,
        scores
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

except ValueError:

    print(
        "ROC-AUC   : Could not be calculated"
    )


# ==================================================
# 18. CLASSIFICATION REPORT
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