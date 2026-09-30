import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data_dir = Path(__file__).resolve().parent.parent/'data'
model_dir = Path(__file__).resolve().parent.parent/'models'

df = pd.read_csv(data_dir/'clean_data.csv')
print("loaded data")

x = df.drop('Label', axis = 1)
y = df['Label']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("splitted train test")


model = DecisionTreeClassifier(random_state=42)
print("loaded decisontree")

param_grid = {
    'criterion':['gini', 'entropy'],
    'max_depth':[15, 20, 25],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 5],

}

grid_search = GridSearchCV(
    model,
    param_grid,
    n_jobs=-1,
    cv= 5,
    scoring='f1_micro'
)
print("created a gridSearchCV")

grid_search.fit(x_train, y_train)
print("model trained successfully")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

joblib.dump(best_model, model_dir/'network_intrusion_detection_model.pkl')



accuracy = accuracy_score(y_pred, y_test)
report = classification_report(y_pred, y_test, target_names=best_model.classes_, digits = 4)
matrix = confusion_matrix(y_pred, y_test, labels=best_model.classes_)

print("-----------evaluation-----------------")
print(f"accuracy --> {accuracy}")
print(f"report --> \n {report}")
print(f"confusion matrix --> \n {matrix}")

