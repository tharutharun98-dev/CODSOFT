# Task 4 — Spam SMS Detection

## Problem statement
Classify SMS text messages as **spam** or **legitimate (ham)**.

## Approach
1. **Text vectorization:** `TfidfVectorizer` (unigrams + bigrams, English
   stop-words removed, top 3,000 features).
2. **Models compared:**
   - Multinomial Naive Bayes (classic, fast baseline for text classification)
   - Logistic Regression
   - Linear SVM
3. **Evaluation:** 80/20 stratified split; accuracy, precision, recall,
   and F1-score computed with `spam` as the positive class (since
   catching spam without flagging real messages as spam is the actual
   goal). Best model selected by F1-score.
4. **Demo:** 3 brand-new SMS messages (never seen during training) are
   classified to sanity-check real-world behavior.

## Files
```
dataset/spam.csv               -> label (spam/ham), message  (750 rows)
code/make_dataset.py             -> generates the sample dataset
code/train_model.py              -> full training/evaluation pipeline
output/model_metric_comparison.png
output/confusion_matrix.png
output/metrics_report.txt
output/sample_predictions.txt
output/best_model.pkl, tfidf_vectorizer.pkl
```

## How to run
```bash
python code/make_dataset.py
python code/train_model.py
```

## Results
See `output/metrics_report.txt`. Spam messages in this sample dataset
use recognizable patterns (urgency, "click here", prize/reward language,
phone-number call-to-action) which TF-IDF + linear models pick up on
very reliably — see `output/sample_predictions.txt` for live examples.

## Note on the dataset
The official CodSoft link points to the Kaggle "SMS Spam Collection
Dataset" (5,574 real UK SMS messages, ~13% spam), which requires a
Kaggle login. `make_dataset.py` builds a smaller **synthetic** dataset
with realistic spam/ham message patterns and the same `label, message`
columns, so the pipeline runs end-to-end. **To use the real dataset:**
download `spam.csv` from the task link, place it in `dataset/spam.csv`
with the same two columns, and re-run `train_model.py` — no code
changes needed. Expect somewhat different (likely still high) accuracy
on the real, more varied dataset.

## Possible extensions
- Add character n-grams to catch obfuscated spam ("fr33 c@sh").
- Try word embeddings or a small transformer for comparison.
- Deploy as a simple Flask/FastAPI endpoint for real-time filtering.
