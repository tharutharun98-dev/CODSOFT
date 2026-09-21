"""
train_model.py
----------------
TASK 2: CREDIT CARD FRAUD DETECTION

Classifies transactions as fraudulent (1) or legitimate (0).
Compares Logistic Regression, Decision Tree, and Random Forest, with
special attention to precision/recall since fraud is a rare-event
(highly imbalanced) classification problem where plain accuracy is
misleading.

Run:
    python code/train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "creditcard.csv")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 11)] + ["Time", "Amount"]

# 1. Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows | Fraud rate: {df['Class'].mean()*100:.2f}%")

X = df[FEATURE_COLS]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Scale Amount/Time (V-features are already roughly standardized)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train & evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}

results = {}
reports = {}
roc_data = {}
best_name, best_f1, best_model = None, -1, None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    reports[name] = classification_report(y_test, preds, zero_division=0, target_names=["Legit", "Fraud"])
    roc_data[name] = roc_curve(y_test, probs)

    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    print(reports[name])

    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, model

print(f"\nBest model (by F1-score, most meaningful for imbalanced fraud data): {best_name}")

# 4. Save metrics report
with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as f:
    f.write("TASK 2: CREDIT CARD FRAUD DETECTION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        for k, v in r.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n" + reports[name] + "\n")
    f.write(f"\nBest model (highest F1-score): {best_name}\n")
    f.write("\nNote: Accuracy alone is misleading on imbalanced fraud data "
             "(predicting 'legit' for everyone would still score ~98% accuracy). "
             "Precision, Recall, F1 and ROC-AUC are the metrics that matter here.\n")

# 5. Metric comparison bar chart
metrics_df = pd.DataFrame(results).T[["precision", "recall", "f1", "roc_auc"]]
metrics_df.plot(kind="bar", figsize=(9, 5), colormap="viridis")
plt.title("Task 2: Model Comparison (Fraud Detection)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_metric_comparison.png"), dpi=150)
plt.close()

# 6. Confusion matrix for best model
best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Reds")
plt.title(f"Confusion Matrix - {best_name} (Best Model)")
plt.colorbar()
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.yticks([0, 1], ["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 7. ROC curve comparison
plt.figure(figsize=(6, 6))
for name, (fpr, tpr, _) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Task 2: ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
plt.close()

# 8. Save model artifacts
joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

print("\nAll outputs saved to the 'output' folder.")
