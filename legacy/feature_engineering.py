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
# 5. FINANCIAL FEATURES
# ==================================================

# Fee as percentage of payment
data["fee_ratio"] = (
    data["fee"] /
    data["payment_amount"]
)


# Tax as percentage of payment
data["tax_ratio"] = (
    data["tax"] /
    data["payment_amount"]
)


# Refund as percentage of payment
data["refund_ratio"] = (
    data["refund"] /
    data["payment_amount"]
)


# Adjustment as percentage of payment
data["adjustment_ratio"] = (
    data["adjustment"] /
    data["payment_amount"]
)


# Actual settlement as percentage of payment
data["settlement_ratio"] = (
    data["actual_settlement"] /
    data["payment_amount"]
)


# Difference as percentage of expected settlement
data["settlement_deviation_ratio"] = (
    data["settlement_difference"] /
    data["expected_settlement"]
)


# Absolute deviation
data["absolute_deviation_ratio"] = (
    data["settlement_difference"].abs() /
    data["expected_settlement"]
)


# ==================================================
# 6. DISPLAY
# ==================================================

print("\n========================================")
print("FINANCIAL FEATURE ENGINEERING")
print("========================================")

columns_to_show = [
    "payment_id",
    "payment_amount",
    "fee_ratio",
    "tax_ratio",
    "refund_ratio",
    "adjustment_ratio",
    "settlement_ratio",
    "settlement_deviation_ratio",
    "absolute_deviation_ratio"
]

print(
    data[columns_to_show]
    .head(15)
    .to_string(index=False)
)


# ==================================================
# 7. SAVE FEATURE DATASET
# ==================================================

data.to_csv(
    "finance_features.csv",
    index=False
)

print("\nFeature dataset saved as:")
print("finance_features.csv")