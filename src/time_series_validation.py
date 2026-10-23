
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

# Reserve the final 20% as the final evaluation period.
split_idx = int(len(df) * 0.80)
development = df.iloc[:split_idx].copy()

X = development[features]
y = development[target]

# A rolling baseline using only past and current-day returns.
baseline_predictions = (
    development["daily_return"].abs().rolling(20).mean()
)

tscv = TimeSeriesSplit(n_splits=5)

results = []

for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X), start=1
):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    baseline = baseline_predictions.iloc[val_idx]

    model_mae = mean_absolute_error(y_val, predictions)
    model_rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    baseline_mae = mean_absolute_error(y_val, baseline)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_val, baseline)
    )

    results.append({
        "fold": fold,
        "validation_start": X_val.index.min().date(),
        "validation_end": X_val.index.max().date(),
        "rf_mae": model_mae,
        "rf_rmse": model_rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
    })

results_df = pd.DataFrame(results)

print("\nTime-Series Cross-Validation Results")
print(results_df.to_string(index=False))

print("\nAverage metrics across folds:")
print(
    results_df[
        ["rf_mae", "rf_rmse", "baseline_mae", "baseline_rmse"]
    ].mean()
)

print("\nDevelopment period:")
print(development.index.min(), "to", development.index.max())

print("\nFinal evaluation period remains reserved:")
print(df.index[split_idx], "to", df.index[-1])
