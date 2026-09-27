import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

data_dir = BASE_DIR / "data"
model_dir = BASE_DIR / "models"

df = pd.read_csv(data_dir/'clean_train.csv')

df = df.drop("date", axis=1)

x = df.drop("kwh", axis=1)
y = df["kwh"]

x = x.drop("household_id", axis = 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# numeric_cols = df.select_dtypes(include="number").drop("kwh").columns
# categoric_cols = df.select_dtypes(include = ["object", "string"]).drop("household_id")

numeric_cols = ["num_residents", "home_sqft", "hvac_age_years", "temp_avg_c", "temp_min_c", "temp_max_c", "humidity_pct", "wind_kph", "precip_mm", "solar_index", "prior_day_kwh", "prior_week_avg_kwh"]

categoric_cols = ["heating_type"]

binary_cols = ["has_ev", "has_solar", "has_pool",  "is_weekend", "is_holiday"]



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoric_cols),
        ("binary", "passthrough", binary_cols)
    ]
)


model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


model_pipeline.fit(x_train, y_train)

y_pred = model_pipeline.predict(x_test)

# evalutaion of model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = (1 - mape)*100


print("Model Evaluation")
print("----------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)
print("accuracy  :", accuracy)
print("----------------")


joblib.dump(model_pipeline, model_dir/"energy_model.pkl")
print("successfully saved the model")
