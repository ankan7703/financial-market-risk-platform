
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"

# Load the downloaded dataset
df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)

# Convert the index to dates
df.index = pd.to_datetime(df.index, errors="coerce")

print("===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isna().sum())

print("\n===== DUPLICATE DATES =====")
print(df.index.duplicated().sum())

print("\n===== DATE RANGE =====")
print("Start:", df.index.min())
print("End:", df.index.max())

print("\n===== BASIC STATISTICS =====")
print(df.describe())
