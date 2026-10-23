
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "return_lag_1": 0.002,
        "return_lag_2": -0.004,
        "return_lag_3": 0.001,
        "return_lag_5": 0.003,
        "abs_return_lag_1": 0.002,
        "abs_return_lag_2": 0.004,
        "abs_return_lag_3": 0.001,
        "abs_return_lag_5": 0.003,
        "volatility_5d": 0.012,
        "mean_abs_return_5d": 0.008,
        "volatility_10d": 0.014,
        "mean_abs_return_10d": 0.009,
        "volatility_20d": 0.015,
        "mean_abs_return_20d": 0.010,
        "price_range": 0.020,
        "volume_change": 0.10,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_absolute_return" in result
    assert "predicted_absolute_return_pct" in result
    assert result["predicted_absolute_return"] >= 0
