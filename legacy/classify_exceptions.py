import pandas as pd


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

transactions = pd.read_csv("finance_controller_dataset.csv")
ground_truth = pd.read_csv("ground_truth.csv")


# --------------------------------------------------
# 2. Merge transaction data with ground truth
# --------------------------------------------------

data = transactions.merge(
    ground_truth,
    on="payment_id"
)


# --------------------------------------------------
# 3. Calculate settlement difference
# --------------------------------------------------

data["difference"] = (
    data["actual_settlement"]
    - data["expected_settlement"]
)


# --------------------------------------------------
# 4. Classify the transaction
# --------------------------------------------------

def classify_transaction(row):

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

    else:
        return "UNKNOWN"


data["exception_type"] = data.apply(
    classify_transaction,
    axis=1
)


# --------------------------------------------------
# 5. Determine severity
# --------------------------------------------------

def determine_severity(row):

    exception_type = row["exception_type"]
    difference = abs(row["difference"])

    # No issue
    if exception_type == "NORMAL":
        return "NONE"

    # Very serious financial exceptions
    if exception_type in [
        "MISSING_SETTLEMENT",
        "DUPLICATE_TRANSACTION"
    ]:
        return "HIGH"

    # Large financial discrepancy
    if difference >= 1000:
        return "HIGH"

    # Medium financial discrepancy
    if difference >= 100:
        return "MEDIUM"

    # Smaller discrepancy
    return "LOW"


data["severity"] = data.apply(
    determine_severity,
    axis=1
)


# --------------------------------------------------
# 6. Financial impact
# --------------------------------------------------

data["financial_impact"] = data["difference"].abs()


# --------------------------------------------------
# 7. Display summary
# --------------------------------------------------

print("\n========================================")
print("AI FINANCE CONTROLLER")
print("EXCEPTION CLASSIFICATION")
print("========================================")

print(f"Total transactions : {len(data)}")

print(
    f"Total exceptions   : "
    f"{len(data[data['exception_type'] != 'NORMAL'])}"
)


# --------------------------------------------------
# 8. Exception type summary
# --------------------------------------------------

print("\n========================================")
print("EXCEPTION TYPE SUMMARY")
print("========================================")

exception_summary = (
    data[data["exception_type"] != "NORMAL"]
    ["exception_type"]
    .value_counts()
)

print(exception_summary)


# --------------------------------------------------
# 9. Severity summary
# --------------------------------------------------

print("\n========================================")
print("SEVERITY SUMMARY")
print("========================================")

severity_summary = (
    data[data["severity"] != "NONE"]
    ["severity"]
    .value_counts()
)

print(severity_summary)


# --------------------------------------------------
# 10. Display detailed exceptions
# --------------------------------------------------

print("\n========================================")
print("DETAILED EXCEPTIONS")
print("========================================")

columns_to_show = [
    "payment_id",
    "scenario",
    "expected_settlement",
    "actual_settlement",
    "difference",
    "exception_type",
    "severity",
    "financial_impact"
]

exceptions = data[
    data["exception_type"] != "NORMAL"
]

print(
    exceptions[columns_to_show]
    .to_string(index=False)
)