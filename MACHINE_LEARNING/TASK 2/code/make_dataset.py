"""
make_dataset.py
----------------
Generates a sample, imbalanced credit-card-transaction dataset for Task 2.

NOTE: The official CodSoft dataset link points to the Kaggle
"Credit Card Fraud Detection" dataset, which requires a Kaggle login
and is far too large (150k+ rows, 30 PCA-anonymized columns) to fetch
in this environment. This script builds a smaller SYNTHETIC dataset
with the same shape/spirit (anonymized numeric features V1..V10,
Amount, Time, and a highly imbalanced binary Class label) so the full
pipeline in train_model.py runs end-to-end and is easy to reproduce.

TO USE THE REAL DATASET INSTEAD:
1. Download creditcard.csv from the CodSoft task link (Kaggle).
2. Place it in dataset/creditcard.csv (same column names: Time, V1-V28,
   Amount, Class).
3. Update FEATURE_COLS in train_model.py to V1-V28 and re-run.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n_legit = 4000
n_fraud = 80  # ~2% fraud rate, mimicking real-world class imbalance

# Legitimate transactions: features drawn from a "normal" distribution
legit = pd.DataFrame({
    f"V{i}": rng.normal(0, 1, n_legit) for i in range(1, 11)
})
legit["Time"] = rng.integers(0, 172800, n_legit)  # seconds over 2 days
legit["Amount"] = np.round(np.abs(rng.normal(60, 50, n_legit)), 2)
legit["Class"] = 0

# Fraudulent transactions: shifted distribution + different amount pattern
fraud = pd.DataFrame({
    f"V{i}": rng.normal(2.5, 1.5, n_fraud) for i in range(1, 11)
})
fraud["Time"] = rng.integers(0, 172800, n_fraud)
fraud["Amount"] = np.round(np.abs(rng.normal(300, 200, n_fraud)), 2)
fraud["Class"] = 1

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/CODSOFT_ML_Internship/Task2_Credit_Card_Fraud_Detection/dataset/creditcard.csv", index=False)

print(df["Class"].value_counts())
print(f"Saved {len(df)} rows to dataset/creditcard.csv "
      f"({fraud.shape[0]/df.shape[0]*100:.2f}% fraud)")
