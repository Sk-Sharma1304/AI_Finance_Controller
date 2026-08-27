import pandas as pd


# --------------------------------------------------
# 1. Load our transaction dataset
# --------------------------------------------------

transactions = pd.read_csv("finance_controller_dataset.csv")

# Load the ground truth generated for our dataset
ground_truth = pd.read_csv("ground_truth.csv")


# --------------------------------------------------
# 2. Combine both datasets
# --------------------------------------------------

data = transactions.merge(
    ground_truth,
    on="payment_id"
)


# --------------------------------------------------
# 3. Calculate reconciliation difference
# --------------------------------------------------

data["difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


# --------------------------------------------------
# 4. Determine reconciliation status
# --------------------------------------------------

data["reconciliation_status"] = data["difference"].apply(
    lambda x: "MATCHED" if x == 0 else "EXCEPTION"
)


# --------------------------------------------------
# 5. Select only exceptions
# --------------------------------------------------

exceptions = data[
    data["reconciliation_status"] == "EXCEPTION"
].copy()


# --------------------------------------------------
# 6. Display number of exceptions
# --------------------------------------------------

print("\n========================================")
print("EXCEPTION ANALYSIS")
print("========================================")

print(f"Total transactions : {len(data)}")
print(f"Total exceptions   : {len(exceptions)}")


# --------------------------------------------------
# 7. Display the exception records
# --------------------------------------------------

print("\n========================================")
print("EXCEPTION RECORDS")
print("========================================")

columns_to_show = [
    "payment_id",
    "order_id",
    "payment_amount",
    "fee",
    "tax",
    "refund",
    "adjustment",
    "expected_settlement",
    "actual_settlement",
    "difference"
]

print(
    exceptions[columns_to_show].to_string(index=False)
)