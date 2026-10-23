
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"

# Ensure the saved model exists.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run src/train.py first."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

app = FastAPI(
    title="Financial Market Risk API",
    description="Research API for next-day absolute-return predictions.",
    version="1.0.0",
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
    return {"status": "healthy", "model": artifact["model_name"]}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        row = pd.DataFrame(
            [data.model_dump()],
            columns=features,
        )
        prediction = float(model.predict(row)[0])

        return {
            "predicted_absolute_return": prediction,
            "predicted_absolute_return_pct": prediction * 100,
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
