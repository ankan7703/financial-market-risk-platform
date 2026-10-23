
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

X = df[features]
y = df[target]

# Preserve chronological order
split_idx = int(len(df) * 0.80)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate on the held-out test period
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Random Forest Results")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Inspect which features the model used most
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

# Save the trained model and feature names
artifact = {
    "model": model,
    "features": features,
    "target": target,
}

model_path = MODEL_DIR / "random_forest.joblib"
joblib.dump(artifact, model_path)

print(f"\nModel saved to: {model_path}")
