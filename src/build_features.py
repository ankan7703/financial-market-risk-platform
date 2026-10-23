
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")
df.columns = df.columns.get_level_values(0)

df = df.apply(pd.to_numeric, errors="coerce")
df = df.loc[~df.index.isna()].sort_index()
df = df.loc[~df.index.duplicated(keep="first")]

# Daily returns
df["daily_return"] = df["Close"].pct_change()

# Lagged returns
for lag in [1, 2, 3, 5]:
    df[f"return_lag_{lag}"] = df["daily_return"].shift(lag - 1)

# Lagged absolute returns
for lag in [1, 2, 3, 5]:
    df[f"abs_return_lag_{lag}"] = (
        df["daily_return"].abs().shift(lag - 1)
    )

# Historical volatility and mean absolute return
for window in [5, 10, 20]:
    df[f"volatility_{window}d"] = (
        df["daily_return"].rolling(window).std()
    )
    df[f"mean_abs_return_{window}d"] = (
        df["daily_return"].abs().rolling(window).mean()
    )

# Intraday range and volume changes
df["price_range"] = (
    (df["High"] - df["Low"]) / df["Close"]
)
df["volume_change"] = df["Volume"].pct_change()

# Predict the absolute return on the next trading day
df["target_next_day_volatility"] = (
    df["daily_return"].shift(-1).abs()
)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

output_path = OUTPUT_DIR / "qqq_features_v2.csv"
df.to_csv(output_path)

print("Updated feature engineering complete!")
print("Dataset shape:", df.shape)
print("\nFeatures:")
print(df.columns.tolist())
print("\nMissing values:", df.isna().sum().sum())
print("Saved to:", output_path)
