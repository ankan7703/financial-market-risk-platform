
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"


def load_model():
    """Load the saved model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_latest():
    """Predict the next-day absolute return using the latest features."""
    artifact = load_model()

    model = artifact["model"]
    features = artifact["features"]

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    latest = df[features].iloc[[-1]]
    prediction = float(model.predict(latest)[0])

    return {
        "as_of_date": latest.index[0].date().isoformat(),
        "predicted_absolute_return": prediction,
        "predicted_absolute_return_pct": prediction * 100,
        "model_name": artifact["model_name"],
    }


if __name__ == "__main__":
    result = predict_latest()

    print("Latest prediction")
    print("-----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
