
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "qqq_features_v2.csv"

# Target retained for compatibility with the current training/API code.
TARGET = "target_next_day_abs_return"
# This model forecasts the next trading day's absolute return.
TARGET_DESCRIPTION = "Next trading day's absolute close-to-close return"


def load_price_data(path: Path) -> pd.DataFrame:
    """Load and clean daily OHLCV data."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=0)

    # Flatten multi-level columns, as produced by some data providers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Invalid prices and negative volumes are not valid OHLCV observations.
    valid_prices = (
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
        & (df["High"] >= df[["Open", "Close"]].max(axis=1))
        & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
        & (df["Volume"] >= 0)
    )

    df = df.loc[valid_prices].copy()

    if df.empty:
        raise ValueError("No valid OHLCV observations remain after cleaning.")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features available after a trading day's close."""
    df = df.copy()

    # Return observed at the current day's close.
    df["daily_return"] = df["Close"].pct_change()

    # Lag 1 means the most recently observed daily return.
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag - 1)
        df[f"abs_return_lag_{lag}"] = (
            df["daily_return"].abs().shift(lag - 1)
        )

    # Historical volatility and average absolute returns.
    # Rolling windows include the current close's return.
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = (
            df["daily_return"].rolling(window).std()
        )
        df[f"mean_abs_return_{window}d"] = (
            df["daily_return"].abs().rolling(window).mean()
        )

    # Intraday range and change in volume.
    df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = df["Volume"].pct_change()

    # Target: absolute close-to-close return on the next trading day.
    # shift(-1) intentionally moves tomorrow's return into today's target.
    df[TARGET] = df["daily_return"].shift(-1).abs()
    # Replace invalid infinite values and remove incomplete observations.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if df.empty:
        raise ValueError("No complete rows remain after feature engineering.")

    return df


def main() -> None:
    prices = load_price_data(DATA_PATH)
    features = build_features(prices)

    features.to_csv(OUTPUT_PATH, index_label="Date")

    print("Feature engineering complete.")
    print(f"Dataset shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Target column: {TARGET}")
    print(f"Target meaning: {TARGET_DESCRIPTION}")
    print(f"Missing values: {features.isna().sum().sum()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
