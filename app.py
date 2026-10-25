"""
FraudGuard AI ? Main Streamlit Application Entry Point.

Run with:
    streamlit run app.py

Pages are located in pages/ and auto-discovered by Streamlit's multi-page
feature. Models are loaded once and cached.
"""

import sys
from pathlib import Path

import streamlit as st

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="assets/icon.png" if Path("assets/icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "FraudGuard AI ? Academic ML Project\n"
            "Dataset: ULB Credit Card Fraud Detection (Kaggle)\n"
            "Built with scikit-learn, XGBoost, SHAP, and Streamlit."
        ),
    },
)

# ---------------------------------------------------------------------------
# Global CSS overrides
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #1c2535;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a3a5c;
    }
    [data-testid="stSidebar"] * {
        color: #e8edf4 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e8edf4 !important;
        font-size: 0.95rem;
    }

    /* KPI metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dde3ee;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Headers */
    h1, h2, h3 { color: #1a3a5c; }

    /* Info / warning / error boxes */
    .stAlert { border-radius: 6px; }

    /* Tables */
    [data-testid="stDataFrame"] { border-radius: 6px; }

    /* Buttons */
    .stButton > button {
        background-color: #1a3a5c;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2c5282;
    }

    /* Dividers */
    hr { border-color: #dde3ee; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar header
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 16px 0 8px 0;">
            <h2 style="color: #e8edf4; margin: 0; font-size: 1.4rem; font-weight: 700;">
                FraudGuard AI
            </h2>
            <p style="color: #adbdce; font-size: 0.78rem; margin-top: 4px;">
                Credit Card Fraud Detection<br>Academic ML Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        <p style="color: #adbdce; font-size: 0.72rem; margin-top: 8px;">
        Dataset: ULB Credit Card Fraud Detection<br>
        Source: Kaggle / Machine Learning Group ? ULB<br>
        284,807 transactions | 492 fraud cases
        </p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Landing / redirect to Overview page
# ---------------------------------------------------------------------------
st.title("FraudGuard AI")
st.markdown(
    "**Academic Machine Learning Project** ? Credit Card Fraud Detection System"
)
st.markdown(
    """
    Use the **sidebar** to navigate between pages:

    | Page | Description |
    |------|-------------|
    | Overview | Dataset statistics, class distribution, EDA charts |
    | Fraud Prediction | Score a transaction and explain the model's decision |
    | Model Comparison | Compare all three classifiers with full evaluation metrics |
    | Explainable AI | Global SHAP importance and per-transaction contributions |
    | Transaction Monitor | Score a batch CSV, filter and export results |

    ---

    > **Dataset**: ULB Credit Card Fraud Detection ? 284,807 transactions,
    > 492 fraud cases (0.172%). Features V1-V28 are PCA-transformed
    > components; Time and Amount are the only raw features.
    >
    > **Citation**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and
    > Gianluca Bontempi. *Calibrating Probability with Undersampling for
    > Unbalanced Classification.* IEEE SSCI 2015.
    """
)

st.info(
    "Navigate to a page using the sidebar on the left. "
    "The app requires trained model artifacts ? run `python scripts/train.py` first "
    "if models are not yet available.",
    icon=None,
)
