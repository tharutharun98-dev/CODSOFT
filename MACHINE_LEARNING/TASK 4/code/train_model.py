"""
train_model.py
----------------
TASK 4: SPAM SMS DETECTION

Classifies SMS messages as spam or legitimate (ham) using TF-IDF text
features and compares Naive Bayes, Logistic Regression, and Linear SVM.

Run:
    python code/train_model.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "spam.csv")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows | class balance:\n{df['label'].value_counts()}")

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results, reports = {}, {}
best_name, best_f1, best_model = None, -1, None

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label="spam", zero_division=0)
    rec = recall_score(y_test, preds, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, preds, pos_label="spam", zero_division=0)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    reports[name] = classification_report(y_test, preds, zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(reports[name])

    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, model

print(f"\nBest model (by F1-score for spam class): {best_name}")

with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as f:
    f.write("TASK 4: SPAM SMS DETECTION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        for k, v in r.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n" + reports[name] + "\n")
    f.write(f"\nBest model: {best_name}\n")

metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind="bar", figsize=(8, 5), colormap="Set2")
plt.title("Task 4: Model Comparison (Spam SMS Detection)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_metric_comparison.png"), dpi=150)
plt.close()

best_preds = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, best_preds, labels=["ham", "spam"])
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Oranges")
plt.title(f"Confusion Matrix - {best_name} (Best Model)")
plt.colorbar()
plt.xticks([0, 1], ["ham", "spam"])
plt.yticks([0, 1], ["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.pkl"))
joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))

# Demo predictions
sample_msgs = [
    "Congratulations! You've been chosen to win a brand new car. Click now!",
    "Hey, are you free to grab dinner tonight around 7?",
    "URGENT: verify your bank account now or it will be suspended today.",
]
sample_tfidf = vectorizer.transform(sample_msgs)
sample_preds = best_model.predict(sample_tfidf)
with open(os.path.join(OUT_DIR, "sample_predictions.txt"), "w") as f:
    f.write("Sample predictions on unseen SMS messages:\n\n")
    for msg, pred in zip(sample_msgs, sample_preds):
        line = f"Message: {msg}\nPrediction: {pred}\n\n"
        f.write(line)
        print(line)

print("\nAll outputs saved to the 'output' folder.")
