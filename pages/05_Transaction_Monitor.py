"""
FraudGuard AI ? Page 5: Transaction Monitor

Allows users to:
1. Upload a batch CSV with all 30 feature columns.
2. View scored transactions with risk scores, predicted class, risk band.
3. Filter by risk band or predicted class.
4. Search by row index.
5. Export scored results as a CSV.

Row ordering is preserved from the input file.
No retraining or refitting occurs here.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHAMPION_ARTIFACT_PATH, THRESHOLD_PATH
from src.inference import (
    load_feature_schema,
    load_pipeline,
    load_threshold,
    score_dataframe,
    validate_input_schema,
)

st.set_page_config(page_title="Transaction Monitor ? FraudGuard AI", layout="wide")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #1a3a5c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Transaction Monitor")
st.markdown(
    "Upload a CSV file of transactions to score in batch. "
    "The model assigns a risk score and risk band to each row. "
    "Filter, search, and export the results."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if not CHAMPION_ARTIFACT_PATH.exists():
    st.error(
        "Trained model not found. Run the training script first:\n\n"
        "```\npython scripts/train.py\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_scorer():
    pipeline = load_pipeline(CHAMPION_ARTIFACT_PATH)
    threshold = load_threshold(THRESHOLD_PATH)
    feature_names = load_feature_schema()
    return pipeline, threshold, feature_names


try:
    pipeline, threshold, feature_names = load_scorer()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.subheader("Upload Transactions")

uploaded = st.file_uploader(
    "Upload a CSV file with the 30 feature columns (Time, Amount, V1-V28). "
    "Any additional columns (including Class) are preserved in the output.",
    type=["csv"],
    help="Maximum file size: 50 MB. Larger files may be slow to process.",
)

if uploaded is None:
    st.info(
        "No file uploaded. Upload a CSV to begin batch scoring.\n\n"
        "The file must contain these columns: **Time**, **Amount**, **V1-V28**."
    )
    st.markdown("---")
    st.subheader("Expected Column Format")
    example = pd.DataFrame(
        [[0.0, 149.62] + [0.0] * 28],
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
    )
    st.dataframe(example, use_container_width=True, hide_index=True)
    st.stop()

# ---------------------------------------------------------------------------
# Parse and validate
# ---------------------------------------------------------------------------
try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

st.success(f"File loaded: **{len(raw_df):,} rows**, **{len(raw_df.columns)} columns**.")

try:
    validate_input_schema(raw_df, feature_names)
except ValueError as e:
    st.error(f"Schema error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
with st.spinner("Scoring transactions..."):
    try:
        scored_df = score_dataframe(pipeline, raw_df, threshold, feature_names)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

n_flagged = int((scored_df["predicted_class"] == 1).sum())
n_total = len(scored_df)
flag_rate = n_flagged / n_total * 100 if n_total > 0 else 0.0

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Batch Scoring Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{n_total:,}")
c2.metric("Flagged as Fraud", f"{n_flagged:,}")
c3.metric(
    "Flag Rate",
    f"{flag_rate:.2f}%",
    help=f"Fraction flagged at threshold {threshold:.4f}",
)
c4.metric("Decision Threshold", f"{threshold:.4f}", help="F1-optimal, validated on held-out split.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.subheader("Filter & Explore Results")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    class_filter = st.selectbox(
        "Filter by Predicted Class",
        ["All", "Fraud (1)", "Legitimate (0)"],
    )
with col_f2:
    band_options = ["All"] + sorted(scored_df["risk_band"].unique().tolist())
    band_filter = st.selectbox("Filter by Risk Band", band_options)
with col_f3:
    score_min, score_max = st.slider(
        "Filter by Risk Score Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )

# Row ID search
search_input = st.text_input(
    "Search by Row Index (comma-separated, e.g. 0,5,12):",
    placeholder="Leave empty to show all filtered rows.",
)

# Apply filters
filtered = scored_df.copy()
if class_filter == "Fraud (1)":
    filtered = filtered[filtered["predicted_class"] == 1]
elif class_filter == "Legitimate (0)":
    filtered = filtered[filtered["predicted_class"] == 0]

if band_filter != "All":
    filtered = filtered[filtered["risk_band"] == band_filter]

filtered = filtered[
    (filtered["risk_score"] >= score_min) & (filtered["risk_score"] <= score_max)
]

if search_input.strip():
    try:
        search_ids = [int(x.strip()) for x in search_input.split(",")]
        filtered = filtered[filtered.index.isin(search_ids)]
    except ValueError:
        st.warning("Invalid row indices. Enter comma-separated integers.")

st.markdown(f"Showing **{len(filtered):,}** of **{n_total:,}** rows.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
# Highlight predicted fraud rows
def highlight_fraud(row):
    if row.get("predicted_class") == 1:
        return ["background-color: #fdf0f0"] * len(row)
    return [""] * len(row)


display_cols = ["risk_score", "predicted_class", "risk_band"] + [
    c for c in filtered.columns if c not in ["risk_score", "predicted_class", "risk_band"]
]
display_cols = [c for c in display_cols if c in filtered.columns]

try:
    styled = filtered[display_cols].style.apply(highlight_fraud, axis=1)
    st.dataframe(styled, use_container_width=True, height=400)
except Exception:
    st.dataframe(filtered[display_cols], use_container_width=True, height=400)

st.markdown("---")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.subheader("Export Scored Results")
csv_bytes = scored_df.to_csv(index=True).encode("utf-8")
st.download_button(
    label="Download Full Scored CSV",
    data=csv_bytes,
    file_name="fraudguard_scored_transactions.csv",
    mime="text/csv",
)

filtered_csv = filtered.to_csv(index=True).encode("utf-8")
st.download_button(
    label="Download Filtered Scored CSV",
    data=filtered_csv,
    file_name="fraudguard_scored_filtered.csv",
    mime="text/csv",
)

st.caption(
    "Exported CSV includes: risk_score, predicted_class, risk_band, "
    "and all original columns. Row order matches the input file. "
    "Risk scores are model outputs, not calibrated probabilities."
)
