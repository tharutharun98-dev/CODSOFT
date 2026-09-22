"""
make_dataset.py
----------------
Generates a sample customer-churn dataset for Task 3, modeled after the
well-known "Bank Customer Churn" style dataset referenced by CodSoft
(Kaggle: shubh0799/churn-modelling), with the same kind of columns:
demographics, account info, usage behavior, and a binary Exited/Churn
label.

NOTE: The real Kaggle dataset requires a login and cannot be fetched
automatically here, so this script builds a synthetic dataset with a
believable relationship between features and churn so the pipeline in
train_model.py can be run and reproduced end-to-end.

TO USE THE REAL DATASET INSTEAD:
1. Download Churn_Modelling.csv from the CodSoft task link (Kaggle).
2. Place it in dataset/churn.csv (keep the same column names, or
   update FEATURE_COLS in train_model.py to match).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 3000

geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.5, 0.25, 0.25])
gender = rng.choice(["Male", "Female"], size=n)
age = rng.integers(18, 75, size=n)
tenure = rng.integers(0, 11, size=n)  # years with bank
balance = np.round(np.abs(rng.normal(70000, 60000, size=n)), 2)
num_products = rng.integers(1, 5, size=n)
has_credit_card = rng.integers(0, 2, size=n)
is_active_member = rng.integers(0, 2, size=n)
estimated_salary = np.round(rng.uniform(10000, 150000, size=n), 2)
credit_score = rng.integers(350, 850, size=n)

# Construct churn probability from a believable combination of factors:
# older customers, low activity, high balance w/ few products, and
# Germany geography (arbitrary pattern) increase churn risk.
churn_logit = (
    -3.0
    + 0.04 * (age - 35)
    - 0.35 * is_active_member
    + 0.25 * (num_products == 1).astype(int)
    + 0.6 * (num_products >= 3).astype(int)
    + 0.000006 * balance
    - 0.15 * tenure / 5
    + 0.5 * (geography == "Germany").astype(int)
    - 0.002 * (credit_score - 650)
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
exited = (rng.uniform(0, 1, size=n) < churn_prob).astype(int)

df = pd.DataFrame({
    "CustomerId": np.arange(10000, 10000 + n),
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_credit_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Exited": exited,
})

df.to_csv("/home/claude/CODSOFT_ML_Internship/Task3_Customer_Churn_Prediction/dataset/churn.csv", index=False)
print(df["Exited"].value_counts(normalize=True))
print(f"Saved {len(df)} rows to dataset/churn.csv")
