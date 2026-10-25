"""
FraudGuard AI ? Page 4: Explainable AI

Displays:
- Global SHAP feature importance bar chart (from results/shap/)
- Global importance table (top-N features)
- Per-transaction SHAP waterfall for a selected demo transaction
- SHAP beeswarm summary plot if available

V1-V28 are anonymized PCA components. Their original meanings are not known
and are not invented or inferred here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALL_FEATURES,
    CHAMPION_ARTIFACT_PATH,
    DEMO_CSV_PATH,
    SHAP_RESULTS_DIR,
    TARGET_COL,
    THRESHOLD_PATH,
)
from src.inference import load_feature_schema, load_pipeline, load_threshold

st.set_page_config(page_title="Explainable AI ? FraudGuard AI", layout="wide")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #1a3a5c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Explainable AI")
st.markdown(
    "SHAP (SHapley Additive exPlanations) values quantify each feature's "
    "contribution to the model's output for individual predictions and across "
    "the dataset."
)
st.info(
    "V1-V28 are PCA-transformed, anonymized components. "
    "Their original feature meanings are confidential and are not displayed or inferred.",
    icon=None,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Global importance chart
# ---------------------------------------------------------------------------
st.subheader("Global Feature Importance")
st.markdown(
    "The chart below shows the **mean absolute SHAP value** per feature, "
    "averaged over a random validation sample. Higher values indicate greater "
    "influence on the model's fraud score."
)

global_imp_chart = SHAP_RESULTS_DIR / "global_importance.png"
global_imp_csv = SHAP_RESULTS_DIR / "global_importance.csv"

if global_imp_chart.exists():
    st.image(str(global_imp_chart), caption="Global SHAP Feature Importance", use_container_width=True)
else:
    st.warning(
        "SHAP importance chart not found. Generate it with:\n\n"
        "```\npython scripts/run_shap.py\n```"
    )

if global_imp_csv.exists():
    with st.expander("Importance Table (all features)", expanded=False):
        imp_df = pd.read_csv(global_imp_csv)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
        st.caption(
            "importance = mean |SHAP value| over the validation sample. "
            "Features with higher values had greater average impact on predictions."
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-transaction explanation
# ---------------------------------------------------------------------------
st.subheader("Per-Transaction Explanation")
st.markdown(
    "Select a demo transaction to see how each feature contributed to that "
    "specific prediction."
)

# Check prerequisites
model_ok = CHAMPION_ARTIFACT_PATH.exists()
explainer_ok = (SHAP_RESULTS_DIR / "shap_explainer.joblib").exists()
demo_ok = DEMO_CSV_PATH.exists()

missing = []
if not model_ok:
    missing.append("Trained model (`python scripts/train.py`)")
if not explainer_ok:
    missing.append("SHAP explainer (`python scripts/run_shap.py`)")
if not demo_ok:
    missing.append("Demo CSV (`python scripts/create_demo_csv.py`)")

if missing:
    st.warning("Missing artifacts ? run the following scripts first:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

# Load artifacts
@st.cache_resource(show_spinner="Loading model and SHAP explainer...")
def load_artifacts():
    import joblib
    pipeline = load_pipeline(CHAMPION_ARTIFACT_PATH)
    threshold = load_threshold(THRESHOLD_PATH)
    feature_names = load_feature_schema()
    explainer = joblib.load(SHAP_RESULTS_DIR / "shap_explainer.joblib")
    return pipeline, threshold, feature_names, explainer

try:
    pipeline, threshold, feature_names, explainer = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Demo transaction selector
demo_df = pd.read_csv(DEMO_CSV_PATH)
if TARGET_COL in demo_df.columns:
    demo_df["_display"] = demo_df.index.map(
        lambda i: (
            f"Row {i:03d} | "
            + ("Fraud" if demo_df.loc[i, TARGET_COL] == 1 else "Legitimate")
            + f" | Amount: ?{demo_df.loc[i, 'Amount']:.2f}"
        )
    )
else:
    demo_df["_display"] = [f"Row {i:03d}" for i in demo_df.index]

selected_label = st.selectbox(
    "Choose a transaction to explain:",
    demo_df["_display"].tolist(),
)
row_idx = int(selected_label.split("|")[0].replace("Row", "").strip())
selected_row = demo_df.loc[[row_idx], feature_names]
ground_truth = int(demo_df.loc[row_idx, TARGET_COL]) if TARGET_COL in demo_df.columns else None

# Prediction
y_prob = pipeline.predict_proba(selected_row)[:, 1][0]
y_pred = int(y_prob >= threshold)

col1, col2, col3 = st.columns(3)
col1.metric("Risk Score", f"{y_prob:.4f}")
col2.metric("Predicted Class", "Fraud" if y_pred == 1 else "Legitimate")
if ground_truth is not None:
    col3.metric("Ground Truth", "Fraud (1)" if ground_truth == 1 else "Legitimate (0)")

# SHAP waterfall
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from src.explainability import compute_shap_values, get_preprocessed_feature_names

    feature_names_pp = get_preprocessed_feature_names(pipeline)
    shap_vals = compute_shap_values(explainer, pipeline, selected_row)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    shap_explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=float(expected_value),
        data=shap_vals[0],
        feature_names=feature_names_pp,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation, max_display=18, show=False)
    plt.title(
        f"SHAP Waterfall ? Row {row_idx:03d}  "
        f"(Score: {y_prob:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "E[f(x)] = model expected output (baseline). "
        "f(x) = model output for this transaction. "
        "Red bars push the score toward fraud; blue bars push it toward legitimate. "
        "V1-V28 are anonymized PCA components."
    )

    # Bar chart of SHAP magnitudes for this transaction
    with st.expander("Feature Contribution Values (tabular)", expanded=False):
        contrib_df = pd.DataFrame({
            "Feature": feature_names_pp,
            "SHAP Value": shap_vals[0].tolist(),
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"SHAP waterfall failed: {e}")

st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown(
    """
    | Term | Meaning |
    |------|---------|
    | **Risk Score** | Model output between 0 and 1. Higher = more likely fraud according to the model. Not a calibrated probability. |
    | **E[f(x)]** | Baseline expected score across background training samples. |
    | **SHAP value > 0** | Feature pushed the score toward fraud (relative to baseline). |
    | **SHAP value < 0** | Feature pushed the score toward legitimate. |
    | **Mean |SHAP|** | Average magnitude of a feature's effect across many transactions (global importance). |
    | **V1-V28** | Anonymized PCA components. Original feature names are confidential. |
    """
)
