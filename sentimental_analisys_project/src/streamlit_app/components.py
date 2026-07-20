from typing import Any

import pandas as pd
import streamlit as st


def render_prediction(result: dict[str, Any]) -> None:
    label = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)

    cols = st.columns(2)
    cols[0].metric("Prediction", label)
    cols[1].metric("Confidence", f"{confidence:.3f}")

    probabilities = result.get("probabilities", {})
    if probabilities:
        data = pd.DataFrame(
            {
                "class": list(probabilities.keys()),
                "probability": list(probabilities.values()),
            }
        ).set_index("class")
        st.bar_chart(data)

    with st.expander("Raw response"):
        st.json(result)


def render_metrics(metrics: dict[str, Any]) -> None:
    summary = metrics.get("summary") or {}

    if summary:
        cols = st.columns(4)
        cols[0].metric("Accuracy", _format_metric(summary.get("accuracy")))
        cols[1].metric("Macro F1", _format_metric(summary.get("macro_f1")))
        cols[2].metric("Weighted F1", _format_metric(summary.get("weighted_f1")))
        cols[3].metric("Best epoch", summary.get("best_epoch") or "-")

    with st.expander("Full metrics"):
        st.json(metrics)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
