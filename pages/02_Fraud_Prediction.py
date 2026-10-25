"""
FraudGuard AI ? Page 2: Fraud Prediction

Allows the user to:
1. Select a demo transaction from data/demo_transactions.csv
2. Upload a single-row CSV with all 30 feature columns

Outputs:
- Risk score (model probability, NOT a calibrated probability)
- Risk band classification (Low / Medium / High / Critical)
- Predicted class at the validated threshold
- SHAP waterfall explanation for the selected transaction

IMPORTANT DISCLAIMERS shown in the UI:
- Risk scores are model outputs, not calibrated probabilities.
- V1-V28 are anonymized PCA components; feature names are not invented.
"""

import json
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
from src.inference import (
    load_feature_schema,
    load_pipeline,
    load_threshold,
    score_to_risk_band,
    validate_input_schema,
)

st.set_page_config(page_title="Fraud Prediction ? FraudGuard AI", layout="wide")

# CSS
st.markdown(
    """
    <style>
    h1, h2, h3 { color: #1a3a5c; }
    .risk-card {
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NAVY = "#1a3a5c"
RED = "#c0392b"


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained model...")
def load_model_and_threshold():
    """Load pipeline and threshold once; cache for the session."""
    pipeline = load_pipeline(CHAMPION_ARTIFACT_PATH)
    threshold = load_threshold(THRESHOLD_PATH)
    feature_names = load_feature_schema()
    return pipeline, threshold, feature_names


@st.cache_resource(show_spinner="Loading SHAP explainer...")
def load_explainer_and_background():
    """Load pre-fitted SHAP explainer and background data."""
    import joblib
    explainer_path = SHAP_RESULTS_DIR / "shap_explainer.joblib"
    bg_path = SHAP_RESULTS_DIR / "shap_background.joblib"
    if not explainer_path.exists() or not bg_path.exists():
        return None, None
    return joblib.load(explainer_path), joblib.load(bg_path)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Fraud Prediction")
st.markdown(
    "Score an individual transaction using the trained champion model. "
    "The model outputs a **risk score** (not a calibrated probability) and "
    "assigns a risk band based on validated thresholds."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_missing = not CHAMPION_ARTIFACT_PATH.exists()
if model_missing:
    st.error(
        "Trained model not found. Please run the training script first:\n\n"
        "```\npython scripts/train.py\n```\n\n"
        "The app cannot produce predictions without a trained model artifact.",
    )
    st.stop()

try:
    pipeline, threshold, feature_names = load_model_and_threshold()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# SHAP (optional)
explainer, X_background = load_explainer_and_background()
shap_available = explainer is not None

# ---------------------------------------------------------------------------
# Input: Demo or Upload
# ---------------------------------------------------------------------------
st.subheader("Select Transaction Input")
input_mode = st.radio(
    "Input source",
    ["Select a demo transaction", "Upload a single-row CSV"],
    horizontal=True,
)

selected_row: pd.DataFrame | None = None
ground_truth_label: int | None = None

# --- Demo transaction ---
if input_mode == "Select a demo transaction":
    if not DEMO_CSV_PATH.exists():
        st.warning(
            "Demo transaction file not found. Generate it with:\n\n"
            "```\npython scripts/create_demo_csv.py\n```"
        )
        st.stop()

    demo_df = pd.read_csv(DEMO_CSV_PATH)
    # Build display label for selector
    if TARGET_COL in demo_df.columns:
        demo_df["_display"] = demo_df.index.map(
            lambda i: (
                f"Row {i:03d} | "
                + ("Fraud" if demo_df.loc[i, TARGET_COL] == 1 else "Legitimate")
                + f" | Amount: ?{demo_df.loc[i, 'Amount']:.2f}"
            )
        )
        options = demo_df["_display"].tolist()
    else:
        options = [f"Row {i:03d} | Amount: ?{demo_df.loc[i,'Amount']:.2f}"
                   for i in demo_df.index]

    selected_label = st.selectbox("Choose a transaction:", options)
    row_idx = int(selected_label.split("|")[0].replace("Row", "").strip())
    selected_row = demo_df.loc[[row_idx], feature_names]
    if TARGET_COL in demo_df.columns:
        ground_truth_label = int(demo_df.loc[row_idx, TARGET_COL])

# --- CSV upload ---
else:
    uploaded = st.file_uploader(
        "Upload a CSV file with exactly the 30 feature columns "
        "(Time, Amount, V1-V28). One row will be scored.",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            upload_df = pd.read_csv(uploaded)
            validate_input_schema(upload_df, feature_names)
            selected_row = upload_df[feature_names].head(1)
            if len(upload_df) > 1:
                st.info(
                    f"Uploaded file has {len(upload_df)} rows. "
                    "Scoring the first row here. "
                    "Use the Transaction Monitor page to score all rows."
                )
        except ValueError as e:
            st.error(f"Schema error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Upload a CSV file to score a transaction.")
        st.stop()

if selected_row is None:
    st.stop()

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
y_prob = pipeline.predict_proba(selected_row)[:, 1][0]
y_pred = int(y_prob >= threshold)
risk_band = score_to_risk_band(y_prob)

# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Prediction Result")

risk_colors = {
    "Low Risk": "#27ae60",
    "Medium Risk": "#e67e22",
    "High Risk": "#c0392b",
    "Critical Risk": "#7b241c",
}
band_color = risk_colors.get(risk_band, NAVY)

col1, col2, col3 = st.columns(3)
col1.metric("Risk Score", f"{y_prob:.4f}", help="Model output score (0?1). Not a calibrated probability.")
col2.metric("Risk Band", risk_band, help="Assigned based on validated score thresholds.")
col3.metric("Predicted Class", "Fraud" if y_pred == 1 else "Legitimate",
            help=f"Classification at threshold {threshold:.4f} (selected on validation set).")

# Ground truth (only for demo transactions)
if ground_truth_label is not None:
    st.markdown(
        f"**Ground Truth Label** (from dataset): "
        f"{'Fraud (1)' if ground_truth_label == 1 else 'Legitimate (0)'}  "
        f"? this is the actual label from the ULB dataset, not a model prediction."
    )
    if y_pred == ground_truth_label:
        st.success("Prediction matches ground truth label.")
    else:
        st.warning("Prediction does not match ground truth label.")

# Risk band explanation
st.markdown(
    f"""
    <div style="
        background: {band_color}18;
        border-left: 4px solid {band_color};
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 12px;
    ">
        <strong style="color:{band_color};">{risk_band}</strong><br>
        Score: <strong>{y_prob:.4f}</strong> &nbsp;|&nbsp;
        Threshold: <strong>{threshold:.4f}</strong> (F1-optimal, validated on held-out validation set)<br>
        <em style="font-size:0.82rem; color:#666;">
        Risk score is a model-generated output and has not been probability-calibrated.
        Do not interpret as a precise percentage likelihood of fraud.
        </em>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Transaction feature values
# ---------------------------------------------------------------------------
with st.expander("Input Feature Values", expanded=False):
    st.dataframe(selected_row.T.rename(columns={selected_row.index[0]: "Value"}), use_container_width=True)

# ---------------------------------------------------------------------------
# SHAP waterfall explanation
# ---------------------------------------------------------------------------
st.subheader("Model Explanation (SHAP)")

if not shap_available:
    st.info(
        "SHAP explainer not found. Generate it with:\n\n"
        "```\npython scripts/run_shap.py\n```"
    )
else:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        from src.explainability import (
            compute_shap_values,
            get_preprocessed_feature_names,
        )

        feature_names_pp = get_preprocessed_feature_names(pipeline)
        shap_vals = compute_shap_values(explainer, pipeline, selected_row)
        expected_value = explainer.expected_value

        # For binary classifiers, expected_value may be a list
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

        # Waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=float(expected_value),
            data=shap_vals[0],
            feature_names=feature_names_pp,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall ? Feature Contributions to Risk Score", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.caption(
            "Each bar shows a feature's contribution to pushing the risk score "
            "above or below the baseline (expected model output across background samples). "
            "Red bars increase the score; blue bars decrease it. "
            "V1-V28 are anonymized PCA components ? their original meanings are not known."
        )
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
