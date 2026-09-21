# Task 2 — Credit Card Fraud Detection

## Problem Statement

Classify credit-card transactions as **fraudulent** or **legitimate**.

This is a classic **rare-event / imbalanced classification** problem — fraud makes up only a small fraction of all transactions.

## Approach

**Features:** 10 anonymized numeric features (`V1`–`V10`, standing in for the PCA-transformed features in the real dataset), plus `Time` and `Amount`.

**Scaling:** `StandardScaler` applied to all features.

**Models compared** (all trained with `class_weight="balanced"` to counter the class imbalance):

1. Logistic Regression
2. Decision Tree
3. Random Forest

**Evaluation:** Accuracy is misleading on imbalanced data — a model that predicts "legit" for every transaction would already score ~98% accuracy. The pipeline therefore reports:

- Precision
- Recall
- F1-score
- ROC-AUC

The best model is selected by **F1-score**, not accuracy.

## Project Structure

```
.
├── dataset/
│   └── creditcard.csv              # Time, V1–V10, Amount, Class (0 = legit, 1 = fraud)
├── code/
│   ├── make_dataset.py             # Generates the sample imbalanced dataset
│   └── train_model.py              # Full training / evaluation pipeline
└── output/
    ├── model_metric_comparison.png # Metric comparison across models
    ├── confusion_matrix.png        # Confusion matrix of the best model
    ├── roc_curves.png              # ROC curves for all models
    ├── metrics_report.txt          # Exact numeric results
    ├── best_model.pkl              # Saved best model
    └── scaler.pkl                  # Fitted StandardScaler
```

## How to Run

```bash
python code/make_dataset.py
python code/train_model.py
```

The first command generates `dataset/creditcard.csv`; the second trains all models, evaluates them, and writes the plots, report, and saved model/scaler to `output/`.

## Results

See `output/metrics_report.txt` for the exact numbers.

**Key takeaway:** The models differ far more in their **recall and precision on the fraud class** than in raw accuracy. These are the metrics that actually matter for a fraud detection system, since both kinds of error carry real costs:

- **False negatives** — missed fraud
- **False positives** — legitimate customers being blocked

## Note on the Dataset

The official CodSoft link points to the Kaggle **Credit Card Fraud Detection** dataset (~285,000 real anonymized transactions, 0.17% fraud rate), which requires a Kaggle login.

`make_dataset.py` instead builds a smaller **synthetic** dataset with an analogous structure and class imbalance, so the pipeline is fully runnable end-to-end without any downloads.

### Using the real dataset

1. Download `creditcard.csv` from the task link.
2. Place it at `dataset/creditcard.csv`.
3. In `code/train_model.py`, update `FEATURE_COLS` to `V1` … `V28` (the real dataset has 28 PCA features instead of 10).

Everything else works unchanged.

## Possible Extensions

- Try **SMOTE** or **undersampling** instead of `class_weight="balanced"`.
- Add **XGBoost** / **LightGBM** for comparison.
- **Tune the classification threshold** instead of using the default 0.5.
