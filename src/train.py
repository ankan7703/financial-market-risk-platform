
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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

TARGET = "target_next_day_volatility"


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Keep the final 20% out of this training run.
    split_idx = int(len(df) * 0.80)
    development = df.iloc[:split_idx]

    X = development[FEATURES]
    y = development[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    # Store MLflow tracking data inside the project
    # Use a SQLite database for local experiment tracking
    MLFLOW_DIR = PROJECT_ROOT / "mlruns"
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    MLFLOW_DB = MLFLOW_DIR / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment("financial-market-risk")
    
    with mlflow.start_run():
        model.fit(X, y)
        predictions = model.predict(X)

        # These are training metrics, NOT generalization metrics.
        train_mae = mean_absolute_error(y, predictions)
        train_rmse = np.sqrt(mean_squared_error(y, predictions))

        mlflow.log_params(model.get_params())
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_rmse", train_rmse)

        artifact = {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "model_name": "RandomForestRegressor",
        }

        model_path = MODEL_DIR / "risk_model.joblib"
        joblib.dump(artifact, model_path)

        mlflow.log_artifact(str(model_path))

        print("Training complete.")
        print(f"Training MAE: {train_mae:.6f}")
        print(f"Training RMSE: {train_rmse:.6f}")
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
