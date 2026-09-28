# Customer Loan Approval Prediction

My second machine learning project — a model that predicts whether a customer's loan application
will be approved or rejected, based on their financial profile and CIBIL (credit) score.

## Overview

Given an applicant's income, existing assets, loan amount/term, dependents, education, and CIBIL
score, the model classifies the loan as **Approved** or **Rejected**. `predict.py` runs this as an
interactive CLI — enter an applicant's details and it prints the predicted outcome.

## Project structure

```
2-customer_loan_approval_prediction/
├── data/
│   ├── loan_approval_dataset.csv   # raw dataset
│   └── clean_dataset.csv           # cleaned dataset used for training
├── models/
│   └── loan_status_model.pkl       # saved sklearn pipeline (preprocessing + model)
├── notebooks/
│   ├── EDA.ipynb                   # data cleaning & exploratory analysis
│   └── model_train.ipynb           # preprocessing, training and evaluation
├── src/
│   ├── train.py                    # script version of the training pipeline
│   └── predict.py                  # CLI script for making a prediction
└── requirements.txt
```

## Dataset

Each row is one loan application, with columns such as:

- **Applicant**: `no_of_dependents`, `education`, `self_employed`
- **Financials**: `income_annum`, `loan_amount`, `loan_term`, `cibil_score`
- **Assets**: `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- **Target**: `loan_status` — `Approved` or `Rejected`

`loan_approval_dataset.csv` is the raw data; in `EDA.ipynb` the `loan_id` column is dropped, column
names and text values are stripped of stray whitespace, and `education`/`self_employed` are encoded
to 0/1 before the result is saved as `clean_dataset.csv`, which is what the model is trained on.

## Approach

1. **EDA** (`notebooks/EDA.ipynb`) — checked for nulls/duplicates, checked how balanced `education`
   and `self_employed` are, encoded them to binary, and visualized the data: histograms, a
   correlation heatmap against `loan_status`, and count/box plots of `loan_status` against
   `cibil_score`, `income_annum`, `loan_amount`, education and dependents. The notebook's own
   conclusion: **`cibil_score` is by far the strongest driver of approval**, with income, education,
   employment type and dependents mattering much less.
2. **Preprocessing + training** (`notebooks/model_train.ipynb`, mirrored in `src/train.py`) —
   a `ColumnTransformer` that standard-scales the numeric columns and passes the binary
   `education`/`self_employed` flags straight through, wrapped in a scikit-learn `Pipeline` with a
   `LogisticRegression` classifier. Data is split 80/20 for train/test.
3. **Evaluation** — accuracy, a classification report, and a confusion matrix (plotted as a heatmap
   in the notebook) are used to check performance for both classes, not just overall accuracy.
4. **Prediction** (`src/predict.py`) — asks for an applicant's details, builds a single-row
   DataFrame matching the training schema, and runs it through the saved pipeline
   (`models/loan_status_model.pkl`) to print an Approved/Rejected result.

## Getting started

```bash
pip install -r requirements.txt

# retrain the model from clean_dataset.csv
python src/train.py

# make a prediction for a new applicant
python src/predict.py
```

## Model Evaluation

- **Accuracy:** 90.40%

- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Approved | 0.86 | 0.88 | 0.87 | 312 |
| Rejected | 0.93 | 0.92 | 0.92 | 542 |
| **Accuracy** | | | **0.90** | **854** |
| **Macro Avg** | **0.90** | **0.90** | **0.90** | **854** |
| **Weighted Avg** | **0.90** | **0.90** | **0.90** | **854** |

## Notes / what's next

This is a baseline logistic regression classifier built to practice a full classification
workflow (cleaning → EDA → preprocessing → training → evaluation → inference). Since `cibil_score`
dominates the correlation with the outcome, an interesting next step would be checking how the
model performs with it removed, or trying a tree-based model (Random Forest / XGBoost) to see if
it picks up on interactions between the weaker features.
