
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"


def test_saved_model_can_predict_latest_observation():
    assert MODEL_PATH.exists(), "Run src/train.py first"
    assert DATA_PATH.exists(), "Run src/build_features.py first"

    artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    features = artifact["features"]
    assert all(feature in df.columns for feature in features)

    latest = df.sort_index()[features].iloc[[-1]]
    prediction = float(artifact["model"].predict(latest)[0])

    assert np.isfinite(prediction)
    assert prediction >= 0
