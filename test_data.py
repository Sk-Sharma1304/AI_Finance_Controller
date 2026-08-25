import pandas as pd

# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------

data = pd.read_csv("finance_controller_dataset.csv")


# ---------------------------------------
# 2. Calculate expected settlement
# ---------------------------------------

data["expected_settlement"] = (
    data["payment_amount"]
    - data["fee"]
    - data["tax"]
    - data["refund"]
    + data["adjustment"]
)


# ---------------------------------------
# 3. Compare expected vs actual
# ---------------------------------------

data["difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


# ---------------------------------------
# 4. Determine reconciliation status
# ---------------------------------------

data["reconciliation_status"] = data["difference"].apply(
    lambda x: "RECONCILED" if abs(x) < 0.01 else "EXCEPTION"
)


# ---------------------------------------
# 5. Display results
# ---------------------------------------

print(data[
    [
        "payment_id",
        "expected_settlement",
        "actual_settlement",
        "difference",
        "reconciliation_status"
    ]
].head(10))


# ---------------------------------------
# 6. Summary
# ---------------------------------------

print("\nReconciliation Summary:")
print(data["reconciliation_status"].value_counts())