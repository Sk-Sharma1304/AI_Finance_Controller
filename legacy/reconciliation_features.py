import pandas as pd


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
# 3. EXPECTED SETTLEMENT
# ==================================================

data["calculated_expected"] = (
    data["payment_amount"]
    - data["fee"]
    - data["tax"]
    - data["refund"]
    + data["adjustment"]
)


# ==================================================
# 4. SETTLEMENT DIFFERENCE
# ==================================================

data["settlement_difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


# ==================================================
# 5. RECONCILIATION FEATURES
# ==================================================

# How much of the expected amount was actually settled
data["settlement_ratio"] = (
    data["actual_settlement"]
    / data["expected_settlement"]
)


# Percentage deviation from expected settlement
data["deviation_ratio"] = (
    data["settlement_difference"].abs()
    / data["expected_settlement"]
)


# Direction of discrepancy
data["deviation_direction"] = (
    data["settlement_difference"]
    .apply(
        lambda x:
        1 if x > 0
        else (-1 if x < 0 else 0)
    )
)


# Whether settlement is completely missing
data["is_missing_settlement"] = (
    data["actual_settlement"] == 0
).astype(int)


# Whether settlement is greater than expected
data["is_over_settlement"] = (
    data["actual_settlement"]
    > data["expected_settlement"]
).astype(int)


# Whether settlement is lower than expected
data["is_under_settlement"] = (
    data["actual_settlement"]
    < data["expected_settlement"]
).astype(int)


# Size of discrepancy relative to payment
data["difference_to_payment_ratio"] = (
    data["settlement_difference"].abs()
    / data["payment_amount"]
)


# ==================================================
# 6. DISPLAY FEATURES
# ==================================================

print("\n========================================")
print("RECONCILIATION FEATURES")
print("========================================")

columns = [
    "payment_id",
    "scenario",
    "payment_amount",
    "expected_settlement",
    "actual_settlement",
    "settlement_ratio",
    "deviation_ratio",
    "deviation_direction",
    "is_missing_settlement",
    "is_over_settlement",
    "is_under_settlement",
    "difference_to_payment_ratio"
]


print(
    data[columns]
    .to_string(index=False)
)


# ==================================================
# 7. SAVE
# ==================================================

data.to_csv(
    "reconciliation_features.csv",
    index=False
)


print("\n========================================")
print("FEATURE DATASET CREATED")
print("========================================")

print(
    "Saved as: reconciliation_features.csv"
)