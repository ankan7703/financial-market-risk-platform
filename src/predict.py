
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"


def load_model():
    """Load and validate the saved model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = {"model", "features", "target", "model_name"}
    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise ValueError(
            f"Model artifact is missing keys: {sorted(missing_keys)}"
        )

    if artifact["target"] != "target_next_day_abs_return":
        raise ValueError(
            "Unexpected target in model artifact: "
            f"{artifact['target']!r}. Retrain the model with the corrected target."
        )

    return artifact


def predict_latest():
    """Predict the next trading day's absolute return."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}. "
            "Run src/build_features.py first."
        )

    artifact = load_model()
    model = artifact["model"]
    features = artifact["features"]

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    if df.empty:
        raise ValueError("The processed dataset contains no observations.")

    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Dataset is missing model features: {sorted(missing_features)}"
        )

    latest = df[features].iloc[[-1]]

    if latest.isna().any().any():
        raise ValueError("The latest observation contains missing features.")

    prediction = float(model.predict(latest)[0])

    if prediction < 0:
        raise ValueError(
            f"Model predicted a negative absolute return: {prediction}"
        )

    return {
        "as_of_date": latest.index[0].date().isoformat(),
        "predicted_absolute_return": prediction,
        "predicted_absolute_return_pct": prediction * 100,
        "target": artifact["target"],
        "model_name": artifact["model_name"],
    }


if __name__ == "__main__":
    result = predict_latest()

    print("Latest prediction")
    print("-----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
