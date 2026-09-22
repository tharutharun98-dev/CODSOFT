# Task 3 — Customer Churn Prediction

## Problem statement
Predict whether a subscription/bank customer will **churn** (`Exited=1`)
based on demographic and account/usage features.

## Approach
1. **Features:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`,
   `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`,
   `EstimatedSalary`.
2. **Preprocessing:** `ColumnTransformer` — `StandardScaler` for numeric
   columns, `OneHotEncoder` for `Geography`/`Gender` — wrapped in an
   sklearn `Pipeline` so preprocessing + model train/predict together.
3. **Models compared** (all `class_weight="balanced"` where supported):
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
4. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC; best model
   selected by F1-score. Also reports **feature importance** from the
   Random Forest model to show which factors drive churn.

## Files
```
dataset/churn.csv              -> CustomerId, CreditScore, Geography, Gender, Age,
                                    Tenure, Balance, NumOfProducts, HasCrCard,
                                    IsActiveMember, EstimatedSalary, Exited
code/make_dataset.py             -> generates the sample dataset
code/train_model.py              -> full training/evaluation pipeline
output/model_metric_comparison.png
output/confusion_matrix.png
output/feature_importance.png
output/metrics_report.txt
output/best_model_pipeline.pkl
```

## How to run
```bash
python code/make_dataset.py
python code/train_model.py
```

## Results
See `output/metrics_report.txt`. `output/feature_importance.png` shows
which features the Random Forest model relies on most — in this sample
data, **age, activity level, and number of products** are the strongest
churn signals, echoing common patterns seen in real churn datasets.

## Note on the dataset
The official CodSoft link points to a Kaggle bank-churn dataset
(`Churn_Modelling.csv`) that requires a Kaggle login. `make_dataset.py`
builds a synthetic dataset with the **same column names and a believable
relationship** between features and churn (older, less-active customers
with more products are more likely to churn) so the pipeline runs
end-to-end. **To use the real dataset:** download `Churn_Modelling.csv`
from the task link, place it in `dataset/churn.csv`, and re-run
`train_model.py` — the column names already match.

## Possible extensions
- Add SHAP values for per-customer explainability.
- Try XGBoost / LightGBM.
- Build a simple "at-risk customer" dashboard from the predicted
  probabilities.
