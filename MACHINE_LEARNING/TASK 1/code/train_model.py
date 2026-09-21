"""
train_model.py
----------------
TASK 1: MOVIE GENRE CLASSIFICATION

Predicts a movie's genre from its plot summary using TF-IDF features
and compares three classifiers: Multinomial Naive Bayes, Logistic
Regression, and a Linear Support Vector Machine.

Run:
    python code/train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "movies.csv")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, genres: {sorted(df['genre'].unique())}")

X = df["plot"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. TF-IDF feature extraction
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 3. Train & evaluate multiple models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
reports = {}
best_name, best_acc, best_model = None, -1, None

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    reports[name] = classification_report(y_test, preds, zero_division=0)
    print(f"\n=== {name} ===\nAccuracy: {acc:.4f}\n{reports[name]}")
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

# 4. Save metrics report
with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as f:
    f.write("TASK 1: MOVIE GENRE CLASSIFICATION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, acc in results.items():
        f.write(f"{name}: Accuracy = {acc:.4f}\n")
        f.write(reports[name] + "\n")
    f.write(f"\nBest performing model: {best_name} ({best_acc:.4f} accuracy)\n")

# 5. Accuracy comparison bar chart
plt.figure(figsize=(7, 5))
bars = plt.bar(results.keys(), results.values(), color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylabel("Accuracy")
plt.title("Task 1: Model Accuracy Comparison (Movie Genre Classification)")
plt.ylim(0, 1)
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_accuracy_comparison.png"), dpi=150)
plt.close()

# 6. Confusion matrix for best model
best_preds = best_model.predict(X_test_tfidf)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels)

plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name} (Best Model)")
plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 7. Save the trained model + vectorizer
joblib.dump(best_model, os.path.join(OUT_DIR, "best_model.pkl"))
joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))

# 8. Demo predictions on new/unseen plot text
sample_plots = [
    "A detective hunts a serial killer leaving riddles at every murder scene in a rain-soaked city.",
    "Two best friends fall in love while running a struggling bakery together.",
    "A spaceship crew wakes from cryosleep to find their ship overrun by an alien organism.",
]
sample_tfidf = vectorizer.transform(sample_plots)
sample_preds = best_model.predict(sample_tfidf)

with open(os.path.join(OUT_DIR, "sample_predictions.txt"), "w") as f:
    f.write("Sample predictions on unseen plot summaries:\n\n")
    for plot, pred in zip(sample_plots, sample_preds):
        line = f"Plot: {plot}\nPredicted Genre: {pred}\n\n"
        f.write(line)
        print(line)

print("\nAll outputs saved to the 'output' folder.")
