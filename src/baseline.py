
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

target = "target_next_day_volatility"

# Baseline 1: Predict the training-set mean absolute return
split_idx = int(len(df) * 0.80)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:].copy()

constant_prediction = train["daily_return"].abs().mean()

test["baseline_constant"] = constant_prediction

# Baseline 2: Predict using the previous 20 trading days'
# mean absolute daily return, including the current day's return.
rolling_prediction = (
    df["daily_return"].abs().rolling(window=20).mean()
)

test["baseline_20d"] = rolling_prediction.iloc[split_idx:]

# Calculate evaluation metrics
y_test = test[target]

def evaluate(name, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n{name}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

evaluate(
    "Baseline 1 — Training mean",
    test["baseline_constant"]
)

evaluate(
    "Baseline 2 — Rolling 20-day mean absolute return",
    test["baseline_20d"]
)

print("\nTest observations:", len(test))
print("Test period:", test.index.min(), "to", test.index.max())
