import pandas as pd


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

transactions = pd.read_csv("finance_controller_dataset.csv")
ground_truth = pd.read_csv("ground_truth.csv")


# --------------------------------------------------
# 2. Merge datasets
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
# 4. Classify transaction
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

    return "UNKNOWN"


data["exception_type"] = data.apply(
    classify_transaction,
    axis=1
)


# --------------------------------------------------
# 5. Calculate financial impact
# --------------------------------------------------

data["financial_impact"] = (
    data["difference"].abs()
)


# --------------------------------------------------
# 6. Calculate risk score
# --------------------------------------------------

def calculate_risk_score(row):

    exception_type = row["exception_type"]
    impact = row["financial_impact"]

    score = 0

    # ----------------------------------------------
    # Exception type score
    # ----------------------------------------------

    type_scores = {
        "MISSING_SETTLEMENT": 40,
        "DUPLICATE_TRANSACTION": 35,
        "UNEXPLAINED_DIFFERENCE": 25,
        "WRONG_SETTLEMENT": 20,
        "AMOUNT_DISCREPANCY": 15
    }

    score += type_scores.get(exception_type, 0)


    # ----------------------------------------------
    # Financial impact score
    # ----------------------------------------------

    if impact >= 1000:
        score += 30

    elif impact >= 500:
        score += 20

    elif impact >= 100:
        score += 10

    elif impact > 0:
        score += 5


    # ----------------------------------------------
    # Cap score at 100
    # ----------------------------------------------

    return min(score, 100)


data["risk_score"] = data.apply(
    calculate_risk_score,
    axis=1
)


# --------------------------------------------------
# 7. Convert score into risk level
# --------------------------------------------------

def get_risk_level(score):

    if score >= 70:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    elif score > 0:
        return "LOW"

    return "NONE"


data["risk_level"] = data["risk_score"].apply(
    get_risk_level
)


# --------------------------------------------------
# 8. Select actual financial exceptions
# --------------------------------------------------

exceptions = data[
    data["exception_type"].isin([
        "AMOUNT_DISCREPANCY",
        "MISSING_SETTLEMENT",
        "DUPLICATE_TRANSACTION",
        "WRONG_SETTLEMENT",
        "UNEXPLAINED_DIFFERENCE"
    ])
].copy()


# --------------------------------------------------
# 9. Sort by highest risk
# --------------------------------------------------

exceptions = exceptions.sort_values(
    by=["risk_score", "financial_impact"],
    ascending=False
)


# --------------------------------------------------
# 10. Display summary
# --------------------------------------------------

print("\n========================================")
print("AI FINANCE CONTROLLER")
print("RISK ANALYSIS")
print("========================================")

print(f"Total transactions : {len(data)}")
print(f"Financial exceptions : {len(exceptions)}")


# --------------------------------------------------
# 11. Risk distribution
# --------------------------------------------------

print("\n========================================")
print("RISK LEVEL SUMMARY")
print("========================================")

print(
    exceptions["risk_level"]
    .value_counts()
)


# --------------------------------------------------
# 12. Total financial impact
# --------------------------------------------------

print("\n========================================")
print("FINANCIAL IMPACT")
print("========================================")

print(
    f"Total financial impact : "
    f"₹{exceptions['financial_impact'].sum():,.2f}"
)


# --------------------------------------------------
# 13. Priority exceptions
# --------------------------------------------------

print("\n========================================")
print("PRIORITY EXCEPTIONS")
print("========================================")

columns_to_show = [
    "payment_id",
    "exception_type",
    "expected_settlement",
    "actual_settlement",
    "difference",
    "financial_impact",
    "risk_score",
    "risk_level"
]

print(
    exceptions[columns_to_show]
    .to_string(index=False)
)