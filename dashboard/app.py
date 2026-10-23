
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features_v2.csv"

# Local default; configure API_URL when deploying the dashboard separately.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

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
    st.error(
        "Processed dataset not found. "
        "Run the feature pipeline before starting the dashboard."
    )
    st.stop()

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()
df = df.loc[~df.index.isna()]

if df.empty:
    st.error("The processed dataset contains no valid observations.")
    st.stop()

missing_features = sorted(set(FEATURES) - set(df.columns))
required_columns = {"Close", "volatility_20d"}
missing_columns = sorted(required_columns - set(df.columns))

if missing_features or missing_columns:
    st.error(
        "The processed dataset is missing required columns. "
        "Rebuild the feature dataset before continuing."
    )
    if missing_features:
        st.write("Missing prediction features:", missing_features)
    if missing_columns:
        st.write("Missing display columns:", missing_columns)
    st.stop()

if df[["Close", "volatility_20d"]].iloc[-1].isna().any():
    st.error("The latest observation has missing display values.")
    st.stop()

# Sidebar controls
st.sidebar.header("Settings")

lookback = st.sidebar.slider(
    "Historical window (trading days)",
    min_value=min(60, len(df)),
    max_value=len(df),
    value=min(252, len(df)),
)

recent = df.tail(lookback)
latest = df.iloc[-1]

# Summary metrics
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

# Price chart
st.subheader("QQQ Adjusted Closing Price")

price_fig = px.line(
    recent,
    x=recent.index,
    y="Close",
    labels={
        "x": "Date",
        "Close": "Adjusted Close (USD)",
    },
)
price_fig.update_layout(xaxis_title="Date")
st.plotly_chart(price_fig, use_container_width=True)

# Volatility chart
st.subheader("Historical Volatility")

vol_fig = px.line(
    recent,
    x=recent.index,
    y="volatility_20d",
    labels={
        "x": "Date",
        "volatility_20d": "20-day volatility",
    },
)
vol_fig.update_layout(xaxis_title="Date")
st.plotly_chart(vol_fig, use_container_width=True)

# Prediction
st.subheader("Next-Day Absolute-Return Estimate")

st.warning(
    "Research demonstration only. This model estimates the magnitude "
    "of the next day's absolute return, not market direction. "
    "It is not investment advice."
)

if st.button("Generate Prediction", type="primary"):
    payload = {
        feature: float(latest[feature])
        for feature in FEATURES
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        prediction_pct = result.get("predicted_absolute_return_pct")

        if not isinstance(prediction_pct, (int, float)):
            raise ValueError(
                "API response is missing a valid prediction percentage."
            )

        if not 0 <= prediction_pct < float("inf"):
            raise ValueError("API returned an invalid prediction percentage.")

        st.metric(
            "Predicted absolute return",
            f"{prediction_pct:.3f}%",
        )

        st.caption(
            f"Feature observation date: {df.index[-1].strftime('%Y-%m-%d')}"
        )

        st.caption(
            f"Model: {result.get('model_name', 'Unknown')} | "
            f"Target: {result.get('target', 'Not reported')}"
        )

        st.info(
            result.get(
                "interpretation",
                "Model-estimated next-day absolute return; "
                "not a directional forecast.",
            )
        )

    except requests.RequestException as exc:
        st.error(
            f"Could not reach the prediction API at {API_URL}. "
            "Check that the API is running and API_URL is configured correctly."
        )
        with st.expander("Technical details"):
            st.code(str(exc))

    except (ValueError, TypeError) as exc:
        st.error(f"The prediction response could not be validated: {exc}")
