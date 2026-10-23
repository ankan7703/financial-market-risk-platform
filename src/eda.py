
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"

df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")

# Flatten the yfinance column names
df.columns = df.columns.get_level_values(0)

# Convert price and volume columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_index()
df = df.loc[~df.index.isna()]

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

# Plot adjusted closing price
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"])
plt.title("QQQ Adjusted Closing Price (2015–2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate daily returns
df["Daily_Return"] = df["Close"].pct_change()

# Plot daily returns
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Daily_Return"])
plt.title("QQQ Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nDaily return statistics:")
print(df["Daily_Return"].describe())
