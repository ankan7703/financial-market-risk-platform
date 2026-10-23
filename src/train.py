
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

TARGET = "target_next_day_abs_return"
VALIDATION_FRACTION = 0.20


def calculate_metrics(y_true, predictions):
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Remove rows with missing model inputs or target.
    df = df.dropna(subset=FEATURES + [TARGET])

    # Reserve the final 20% for final evaluation.
    final_split_idx = int(len(df) * 0.80)
    development = df.iloc[:final_split_idx]

    # Use the latest 20% of development data for validation.
    validation_split_idx = int(len(development) * 0.80)

    train = development.iloc[:validation_split_idx]
    validation = development.iloc[validation_split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    mlflow_dir = PROJECT_ROOT / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow_db = mlflow_dir / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db.as_posix()}"
    )
    mlflow.set_experiment("financial-market-risk")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        validation_predictions = model.predict(X_validation)

        train_metrics = calculate_metrics(
            y_train, train_predictions
        )
        validation_metrics = calculate_metrics(
            y_validation, validation_predictions
        )

        mlflow.log_params(model.get_params())
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_param("training_rows", len(train))
        mlflow.log_param("validation_rows", len(validation))
        mlflow.log_param("target", TARGET)

        mlflow.log_metrics({
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
        })

        # Save this validation-stage model for now.
        # We will finalize model selection before deployment.
        artifact = {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "model_name": "RandomForestRegressor",
        }

        model_path = MODEL_DIR / "risk_model.joblib"
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path))

        print("Training and validation complete.")
        print(f"Training period: {train.index.min()} to {train.index.max()}")
        print(
            f"Validation period: "
            f"{validation.index.min()} to {validation.index.max()}"
        )
        print(f"Training MAE: {train_metrics['mae']:.6f}")
        print(f"Training RMSE: {train_metrics['rmse']:.6f}")
        print(f"Validation MAE: {validation_metrics['mae']:.6f}")
        print(f"Validation RMSE: {validation_metrics['rmse']:.6f}")
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
