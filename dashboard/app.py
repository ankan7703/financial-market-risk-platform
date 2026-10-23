
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import os
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Financial Market Risk Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Financial Market Risk Prediction")
st.caption(
    "Research dashboard for historical QQQ market data "
    "and next-day absolute-return estimates."
)

if not DATA_PATH.exists():
    st.error("Processed dataset not found. Run the feature pipeline first.")
    st.stop()

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

# Sidebar controls
st.sidebar.header("Settings")
lookback = st.sidebar.slider(
    "Historical window (trading days)",
    min_value=60,
    max_value=min(500, len(df)),
    value=min(252, len(df)),
)

recent = df.tail(lookback)

# Summary metrics
latest = df.iloc[-1]
col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest available date",
    df.index[-1].strftime("%Y-%m-%d"),
)

col2.metric(
    "Latest adjusted close",
    f"${latest['Close']:.2f}",
)

col3.metric(
    "20-day historical volatility",
    f"{latest['volatility_20d'] * 100:.2f}%",
)

st.subheader("QQQ Adjusted Closing Price")
price_fig = px.line(
    recent,
    x=recent.index,
    y="Close",
    labels={"x": "Date", "Close": "Adjusted Close (USD)"},
)
st.plotly_chart(price_fig, use_container_width=True)

st.subheader("Historical Volatility")
vol_fig = px.line(
    recent,
    x=recent.index,
    y="volatility_20d",
    labels={"x": "Date", "volatility_20d": "20-day volatility"},
)
st.plotly_chart(vol_fig, use_container_width=True)

st.subheader("Next-Day Absolute-Return Estimate")

st.warning(
    "Research demonstration only. This model does not predict market "
    "direction and is not investment advice."
)

if st.button("Generate Prediction"):
    try:
        # Use the actual latest feature row from the dataset.
        features = [
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

        payload = {
            feature: float(latest[feature])
            for feature in features
        }

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.metric(
            "Predicted absolute return",
            f"{result['predicted_absolute_return_pct']:.3f}%",
        )

        st.caption(
            f"Features from the latest available historical row: "
            f"{df.index[-1].strftime('%Y-%m-%d')}"
        )

    except requests.RequestException as exc:
        st.error(
            "Could not reach the prediction API. "
            "Check that FastAPI is running on port 8000."
        )
