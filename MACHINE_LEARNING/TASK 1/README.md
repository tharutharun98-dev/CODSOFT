# Task 1 — Movie Genre Classification

## Problem statement
Predict a movie's **genre** from its **plot summary** (text-only input).

## Approach
1. **Text vectorization:** `TfidfVectorizer` (unigrams + bigrams, English
   stop-words removed, top 5,000 features) converts each plot summary into
   a numeric feature vector.
2. **Models compared:**
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear Support Vector Machine (SVM)
3. **Evaluation:** 80/20 stratified train/test split, accuracy +
   per-class precision/recall/F1 via `classification_report`, plus a
   confusion matrix for the best model.
4. **Demo:** the trained model is run on 3 brand-new plot summaries it
   never saw during training to sanity-check real-world behavior.

## Files
```
dataset/movies.csv          -> title, genre, plot (400 rows, 10 genres)
code/make_dataset.py         -> generates the sample dataset
code/train_model.py          -> full training/evaluation pipeline
output/model_accuracy_comparison.png
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
All three models reach very high accuracy on the sample dataset (the
synthetic plots are template-based and therefore linearly very
separable — see note below). The **best model is selected automatically
by test accuracy** and is the one used for the confusion matrix and demo
predictions. See `output/metrics_report.txt` for the full breakdown.

## Note on the dataset
The official CodSoft link points to a Kaggle "Genre Classification
Dataset (IMDb)" that requires a Kaggle login and can't be auto-downloaded
in this environment. `make_dataset.py` builds a smaller **synthetic**
dataset with the same `title, genre, plot` columns so the pipeline is
fully runnable. **To use the real dataset:** download it from the task
link, save it as `dataset/movies.csv` with those same column names, and
re-run `train_model.py` — no code changes needed. Expect somewhat lower
(but more realistic) accuracy on the real, noisier IMDb data.

## Possible extensions
- Try word embeddings (Word2Vec / GloVe) instead of TF-IDF.
- Handle multi-label genres (many real movies have more than one genre).
- Add a deep learning baseline (e.g. an LSTM or fine-tuned transformer).
