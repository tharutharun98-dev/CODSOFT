Task 2 — Credit Card Fraud Detection
Problem statement
Classify credit-card transactions as fraudulent or legitimate.
This is a classic rare-event / imbalanced classification problem —
fraud is a small fraction of all transactions.
Approach
Features: 10 anonymized numeric features (`V1`–`V10`, standing in
for the PCA-transformed features in the real dataset), plus `Time`
and `Amount`.
Scaling: `StandardScaler` applied to all features.
Models compared (all with `class_weight="balanced"` to counter
the imbalance):
Logistic Regression
Decision Tree
Random Forest
Evaluation: because accuracy is misleading on imbalanced data
(predicting "legit" for every transaction would already score
~98%), the pipeline reports precision, recall, F1-score, and
ROC-AUC, and selects the best model by F1-score, not accuracy.
Files
```
dataset/creditcard.csv        -> Time, V1-V10, Amount, Class (0=legit, 1=fraud)
code/make_dataset.py            -> generates the sample imbalanced dataset
code/train_model.py             -> full training/evaluation pipeline
output/model_metric_comparison.png
output/confusion_matrix.png
output/roc_curves.png
output/metrics_report.txt
output/best_model.pkl, scaler.pkl
```
How to run
```bash
python code/make_dataset.py
python code/train_model.py
```
Results
See `output/metrics_report.txt` for exact numbers. Key takeaway: models
differ far more on recall/precision for the fraud class than on raw
accuracy — this is the metric that actually matters for a fraud
detection system, since missing fraud (false negatives) and blocking
legitimate customers (false positives) both carry real costs.
Note on the dataset
The official CodSoft link points to the Kaggle "Credit Card Fraud
Detection" dataset (~285,000 real anonymized transactions, 0.17% fraud
rate) which requires a Kaggle login. `make_dataset.py` builds a smaller
synthetic dataset with an analogous structure and imbalance so the
pipeline is fully runnable end-to-end. To use the real dataset:
download `creditcard.csv` from the task link, place it in
`dataset/creditcard.csv`, and update `FEATURE_COLS` in
`train_model.py` to `V1`...`V28` (the real dataset has 28 PCA
features instead of 10) — everything else works unchanged.
Possible extensions
Try SMOTE / undersampling instead of `class_weight="balanced"`.
Add XGBoost / LightGBM for comparison.
Tune the classification threshold instead of using the default 0.5.
