# Energy Consumption Predictor

My first machine learning project — a model that predicts a household's daily electricity
consumption (in kWh) based on household characteristics, HVAC info, and weather conditions.

## Overview

Given details about a household (size, number of residents, HVAC age, whether it has an EV/solar/pool)
along with the day's weather, the model estimates how much electricity that household will consume.
The weather data isn't manually entered — `predict.py` fetches it live from the
[Open-Meteo](https://open-meteo.com/) API based on the city you enter.

## Project structure

```
energy_consumption_predictor/
├── data/
│   ├── train.csv          # raw dataset (household-day records)
│   └── clean_train.csv    # cleaned dataset used for training
├── models/
│   └── energy_model.pkl   # saved sklearn pipeline (preprocessing + model)
├── notebooks/
│   ├── EDA.ipynb          # data cleaning & exploratory analysis
│   └── model_train.ipynb  # preprocessing, training and evaluation
├── src/
│   ├── train.py           # script version of the training pipeline
│   └── predict.py         # CLI script for making a live prediction
└── requirements.txt
```

## Dataset

Each row represents one household on one day, with columns such as:

- **Household**: `num_residents`, `home_sqft`, `has_ev`, `has_solar`, `has_pool`, `heating_type`, `hvac_age_years`
- **Weather**: `temp_avg_c`, `temp_min_c`, `temp_max_c`, `humidity_pct`, `wind_kph`, `precip_mm`, `solar_index`
- **Calendar**: `is_weekend`, `is_holiday`
- **History**: `prior_day_kwh`, `prior_week_avg_kwh`
- **Target**: `kwh` — the day's electricity consumption

`train.csv` is the raw data; in `EDA.ipynb` it's sorted by date/household, checked for nulls,
duplicates, and outliers in `kwh`, and cast to sensible dtypes before being saved as `clean_train.csv`,
which is what the model is actually trained on.

## Approach

1. **EDA** (`notebooks/EDA.ipynb`) — inspected distributions, checked for class imbalance in
   `heating_type`, confirmed each household has a consistent number of records, and looked at
   correlations between features and `kwh` (a correlation heatmap and scatter plots against things
   like `home_sqft` and `prior_day_kwh`) to get a feel for what drives consumption.
2. **Preprocessing + training** (`notebooks/model_train.ipynb`, mirrored in `src/train.py`) —
   a `ColumnTransformer` that standard-scales numeric features, one-hot encodes `heating_type`, and
   passes binary flags straight through, wrapped in a scikit-learn `Pipeline` with a
   `LinearRegression` model. The data is split 80/20 for train/test.
3. **Evaluation** — MAE, MSE, RMSE, R², and a MAPE-based accuracy score are printed after training.
   Residual plots (actual vs. predicted, residuals vs. predicted, residuals vs. home size) are used
   in the notebook to sanity-check the model rather than just trusting a single metric.
4. **Prediction** (`src/predict.py`) — asks for a city and household details, geocodes the city and
   pulls current weather from Open-Meteo, assembles a single-row DataFrame matching the training
   schema, and runs it through the saved pipeline (`models/energy_model.pkl`).

## Getting started

```bash
pip install -r requirements.txt

# retrain the model from clean_train.csv
python src/train.py

# make a live prediction (needs internet access for the weather API)
python src/predict.py
```

## model evaluation
- Mean absolute error --> 3.262427312408058
- Mean squared error --> 19.168306468030533
- Root Mean squared error --> 4.378162453362202
- R2 score --> 0.6684647817630265
- accuracy --> 87.05681502878937

## Notes / what's next

This is a baseline linear regression model built while learning the end-to-end ML workflow
(cleaning → EDA → preprocessing → training → evaluation → inference). The residual plots show the
error isn't perfectly constant across the range of predictions, so trying a non-linear model
(e.g. Random Forest or Gradient Boosting) and adding engineered features like month/day-of-week
are natural next steps.

