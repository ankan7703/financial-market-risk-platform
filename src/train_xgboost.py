
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

features = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "volatility_5d",
    "volatility_20d",
    "price_range",
    "volume_change",
]

target = "target_next_day_volatility"

# Reserve the final 20% for final evaluation.
split_idx = int(len(df) * 0.80)
development = df.iloc[:split_idx].copy()

X = development[features]
y = development[target]

tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X), start=1
):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    results.append({
        "fold": fold,
        "mae": mae,
        "rmse": rmse,
    })

results_df = pd.DataFrame(results)

print("\nXGBoost time-series validation results")
print(results_df.to_string(index=False))

print("\nAverage validation metrics:")
print(results_df[["mae", "rmse"]].mean())

print("\nDevelopment period:")
print(development.index.min(), "to", development.index.max())
