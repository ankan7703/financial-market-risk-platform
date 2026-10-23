
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

# Reserve the last 20% from model selection.
split_idx = int(len(df) * 0.80)
development = df.iloc[:split_idx]

X = development[features]
y = development[target]

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    baseline = (
        development["daily_return"].abs()
        .rolling(20).mean().iloc[val_idx]
    )

    for name, model_template in models.items():
        model = model_template.__class__(**model_template.get_params())
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        results.append({
            "fold": fold,
            "model": name,
            "mae": mean_absolute_error(y_val, predictions),
            "rmse": np.sqrt(mean_squared_error(y_val, predictions)),
        })

    results.append({
        "fold": fold,
        "model": "Rolling baseline",
        "mae": mean_absolute_error(y_val, baseline),
        "rmse": np.sqrt(mean_squared_error(y_val, baseline)),
    })

results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("model")[["mae", "rmse"]]
    .agg(["mean", "std"])
    .round(6)
)

print("\nModel comparison (mean and standard deviation across folds):")
print(summary)

results_path = REPORT_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print(f"\nDetailed fold results saved to: {results_path}")
