
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

target = "target_next_day_volatility"

# Features available at the end of each trading day
features = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "volatility_5d",
    "volatility_20d",
    "price_range",
    "volume_change",
]

X = df[features]
y = df[target]

# Chronological 80/20 split: no random shuffling
split_idx = int(len(df) * 0.80)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTesting period:")
print(X_test.index.min(), "to", X_test.index.max())

print("\nMissing feature values:", X.isna().sum().sum())
print("Missing target values:", y.isna().sum())
