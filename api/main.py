
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run src/train.py first."
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
        f"Unexpected model target: {artifact['target']!r}. "
        "Run src/train.py using the corrected target."
    )

model = artifact["model"]
features = artifact["features"]

app = FastAPI(
    title="Financial Market Risk API",
    description=(
        "Research API for estimating next-day absolute returns "
        "using historical market features."
    ),
    version="1.1.0",
)


class PredictionInput(BaseModel):
    return_lag_1: float
    return_lag_2: float
    return_lag_3: float
    return_lag_5: float
    abs_return_lag_1: float = Field(ge=0)
    abs_return_lag_2: float = Field(ge=0)
    abs_return_lag_3: float = Field(ge=0)
    abs_return_lag_5: float = Field(ge=0)
    volatility_5d: float = Field(ge=0)
    mean_abs_return_5d: float = Field(ge=0)
    volatility_10d: float = Field(ge=0)
    mean_abs_return_10d: float = Field(ge=0)
    volatility_20d: float = Field(ge=0)
    mean_abs_return_20d: float = Field(ge=0)
    price_range: float = Field(ge=0)
    volume_change: float


@app.get("/")
def home():
    return {
        "message": "Financial Market Risk API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": artifact["model_name"],
        "target": artifact["target"],
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        row = pd.DataFrame(
            [data.model_dump()],
            columns=features,
        )

        prediction = float(model.predict(row)[0])

        if not pd.notna(prediction) or prediction < 0:
            raise ValueError("Model returned an invalid absolute-return estimate.")

        return {
            "predicted_absolute_return": prediction,
            "predicted_absolute_return_pct": prediction * 100,
            "target": artifact["target"],
            "model_name": artifact["model_name"],
            "interpretation": (
                "Model-estimated next-day absolute return, "
                "not a directional forecast."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
