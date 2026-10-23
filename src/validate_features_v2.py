
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"
)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

features = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "abs_return_lag_1",
    "abs_return_lag_2",
    "abs_return_lag_3",
    "abs_return_lag_5",
    "volatility_5d",
    "mean_abs_return_5d",
    "volatility_10d",
    "mean_abs_return_10d",
    "volatility_20d",
    "mean_abs_return_20d",
    "price_range",
    "volume_change",
]

target = "target_next_day_volatility"

# Keep the final 20% reserved for final evaluation.
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

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    # Baseline: mean absolute return over the previous
    # 20 trading days, including the current day's return.
    baseline = (
        development["daily_return"].abs()
        .rolling(20)
        .mean()
        .iloc[val_idx]
    )

    results.append({
        "fold": fold,
        "rf_mae": mean_absolute_error(y_val, predictions),
        "rf_rmse": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "baseline_mae": mean_absolute_error(y_val, baseline),
        "baseline_rmse": np.sqrt(
            mean_squared_error(y_val, baseline)
        ),
    })

results_df = pd.DataFrame(results)

print("\nExpanded-feature validation results")
print(results_df.to_string(index=False))

print("\nAverage metrics:")
print(results_df.drop(columns="fold").mean())

print("\nFolds where Random Forest beats baseline on MAE:")
print((results_df["rf_mae"] < results_df["baseline_mae"]).sum(), "out of 5")