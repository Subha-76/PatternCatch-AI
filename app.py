"""Streamlit dashboard for ThinkPattern AI.

This file is source code only; it is not executed during project packaging.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import load_behavior_data, validate_behavior_data

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sample_behavior.csv"

st.set_page_config(page_title="ThinkPattern AI", page_icon="🧠", layout="wide")
st.title("ThinkPattern AI")
st.subheader("Behavioral Pattern Prediction Using Sequential ML Models")
st.write(
    "Explore chronological activity data and, after training a model, use "
    "sequential ML to estimate a defined future outcome."
)
st.info(
    "The bundled CSV is synthetic demonstration data. Predictions are estimates, "
    "not guarantees about a person's future behavior."
)

uploaded_file = st.file_uploader("Optionally upload a behavioral CSV", type=["csv"])

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        issues = validate_behavior_data(data)
        if issues:
            st.error("Dataset validation needs attention:\n\n- " + "\n- ".join(issues))
            st.stop()
    else:
        data = load_behavior_data(DEFAULT_DATA)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.markdown("### Dataset overview")
a, b, c = st.columns(3)
a.metric("Rows", f"{len(data):,}")
b.metric("Participants", f"{data['user_id'].nunique():,}")
c.metric("Date range", f"{data['timestamp'].min().date()} – {data['timestamp'].max().date()}")

st.markdown("### Activity records")
st.dataframe(data, use_container_width=True, hide_index=True)

st.markdown("### Study duration over time")
chart_data = data.sort_values(["user_id", "timestamp"])
st.line_chart(chart_data.set_index("timestamp")[["study_minutes"]])

st.markdown("### Prediction")
st.write(
    "Model training and inference are intentionally separate from this overview. "
    "After you choose to train a model, the training script saves an artifact under "
    "`models/`. The prediction module provides the loading and inference functions."
)
st.caption("ThinkPattern AI — educational prototype.")
