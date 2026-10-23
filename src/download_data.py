
from pathlib import Path

import pandas as pd
import yfinance as yf

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download Nasdaq-100 ETF data
ticker = "QQQ"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False,
)

if df.empty:
    raise RuntimeError(f"No market data downloaded for {ticker}")

# Save the downloaded data
output_path = DATA_DIR / "qqq_daily.csv"
df.to_csv(output_path)

print("Download successful")
print("Shape:", df.shape)
print("Saved to:", output_path)
print("\nFirst five rows:")
print(df.head())
