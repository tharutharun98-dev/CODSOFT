"""
train_model.py
----------------
TASK 3: CUSTOMER CHURN PREDICTION

Predicts whether a customer will churn (Exited=1) based on demographic
and account/usage features. Compares Logistic Regression, Random
Forest, and Gradient Boosting, and reports feature importance.

Run:
    python code/train_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "churn.csv")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows | Churn rate: {df['Exited'].mean()*100:.2f}%")

NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT_COLS = ["Geography", "Gender"]

X = df[NUM_COLS + CAT_COLS]
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(drop="first"), CAT_COLS),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results, reports, pipelines = {}, {}, {}
best_name, best_f1, best_pipe = None, -1, None

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    reports[name] = classification_report(y_test, preds, zero_division=0, target_names=["Stayed", "Churned"])
    pipelines[name] = pipe

    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} ROC-AUC={auc:.4f}")
    print(reports[name])

    if f1 > best_f1:
        best_f1, best_name, best_pipe = f1, name, pipe

print(f"\nBest model (by F1-score): {best_name}")

# Save metrics
with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as f:
    f.write("TASK 3: CUSTOMER CHURN PREDICTION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        for k, v in r.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n" + reports[name] + "\n")
    f.write(f"\nBest model (highest F1-score): {best_name}\n")

# Metric comparison chart
metrics_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
metrics_df.plot(kind="bar", figsize=(10, 5), colormap="tab10")
plt.title("Task 3: Model Comparison (Customer Churn Prediction)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_metric_comparison.png"), dpi=150)
plt.close()

# Confusion matrix for best model
best_preds = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Purples")
plt.title(f"Confusion Matrix - {best_name} (Best Model)")
plt.colorbar()
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.yticks([0, 1], ["Stayed", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Feature importance (Random Forest, if not the only tree-free model)
rf_pipe = pipelines["Random Forest"]
feature_names = (NUM_COLS +
                  list(rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_COLS)))
importances = rf_pipe.named_steps["clf"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
imp_series.plot(kind="barh", color="#4C72B0")
plt.title("Task 3: Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

joblib.dump(best_pipe, os.path.join(OUT_DIR, "best_model_pipeline.pkl"))

print("\nAll outputs saved to the 'output' folder.")
