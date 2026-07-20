import pandas as pd
import requests
import streamlit as st

from src.streamlit_app.api_client import DEFAULT_API_URL, SentimentApiClient
from src.streamlit_app.components import render_metrics, render_prediction


st.set_page_config(
    page_title="Sentiment Analysis",
    layout="wide",
)

st.title("Sentiment Analysis")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    client = SentimentApiClient(api_url=api_url)

    if st.button("Refresh info", use_container_width=True):
        st.session_state["model_info"] = None

try:
    model_info = st.session_state.get("model_info")
    if model_info is None:
        model_info = client.info()
        st.session_state["model_info"] = model_info

    st.sidebar.success("API connected")
    st.sidebar.json(model_info)
except requests.RequestException as exc:
    st.sidebar.error(f"API unavailable: {exc}")
    st.stop()

single_tab, batch_tab, metrics_tab = st.tabs(["Predict", "Batch", "Metrics"])

with single_tab:
    text = st.text_area(
        "Review text",
        height=160,
        placeholder="Enter a review text...",
    )

    if st.button("Predict", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter text before prediction.")
        else:
            try:
                result = client.predict(text.strip())
                render_prediction(result)
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")

with batch_tab:
    batch_text = st.text_area(
        "One review per line",
        height=220,
        placeholder="First review\nSecond review\nThird review",
    )

    if st.button("Predict batch", type="primary", use_container_width=True):
        texts = [line.strip() for line in batch_text.splitlines() if line.strip()]

        if not texts:
            st.warning("Add at least one review.")
        else:
            try:
                response = client.predict_batch(texts)
                rows = response.get("predictions", [])
                table = pd.DataFrame(rows)
                st.dataframe(table, use_container_width=True)

                with st.expander("Raw response"):
                    st.json(response)
            except requests.RequestException as exc:
                st.error(f"Batch prediction failed: {exc}")

with metrics_tab:
    if st.button("Load metrics", use_container_width=True):
        try:
            metrics = client.metrics()
            render_metrics(metrics)
        except requests.RequestException as exc:
            st.error(f"Metrics loading failed: {exc}")
