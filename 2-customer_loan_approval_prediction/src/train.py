from pathlib import Path
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_dir = Path(__file__).resolve().parent.parent/'data'
model_dir =  Path(__file__).resolve().parent.parent/'models'

df = pd.read_csv(data_dir/'clean_dataset.csv')

numeric_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

binary_cols = ['education', 'self_employed']

x = df.drop('loan_status', axis = 1)
y = df['loan_status'].map({"Approved":1, "Rejected":0})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("binary", "passthrough", binary_cols)
    ]
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

model_pipeline.fit(x_train, y_train)
print("trained model successfully")


y_pred = model_pipeline.predict(x_test)

# evaluation
accuracy = accuracy_score(y_pred, y_test)
report = classification_report(y_pred, y_test, target_names=['Approved', 'Rejected'])
matrix = confusion_matrix(y_pred, y_test)

print("---------Model evaluation----------")
print(f"accuracy -->{accuracy*100:.2f}%")
print("classification report -->")
print(report)
print("confusion matrix -->")
print(matrix)


joblib.dump(model_pipeline, model_dir/'loan_status_model.pkl')
print("successfully saved model")