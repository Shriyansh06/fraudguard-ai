"""
FraudGuard AI ? Page 3: Model Comparison

Displays:
- Metrics table for all three models (test set)
- Champion baseline vs tuned comparison
- ROC curves
- Precision-Recall curves
- Confusion matrices

All data is loaded from results/metrics.json and results/evaluation/*.png.
No models are re-loaded or re-evaluated here.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DASHBOARD_METRICS_PATH, EVAL_RESULTS_DIR, MODEL_NAMES
from src.evaluation import load_metrics

st.set_page_config(page_title="Model Comparison ? FraudGuard AI", layout="wide")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #1a3a5c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Model Comparison")
st.markdown(
    "Evaluation results for **Logistic Regression**, **Random Forest**, and **XGBoost** "
    "on identical stratified splits (70/15/15, seed 42). "
    "All metrics are computed on the held-out test set."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
metrics_path = DASHBOARD_METRICS_PATH
if not metrics_path.exists():
    st.error(
        "Metrics file not found. Run the training script first:\n\n"
        "```\npython scripts/train.py\n```"
    )
    st.stop()

try:
    metrics = load_metrics(path=metrics_path)
except Exception as e:
    st.error(f"Error loading metrics: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
st.subheader("Test Set Metrics (threshold = 0.5)")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

rows = []
for name in MODEL_NAMES:
    if name not in metrics:
        continue
    m = metrics[name]["baseline"]["test"]["metrics"]
    rows.append({
        "Model": MODEL_LABELS.get(name, name),
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1": f"{m['f1']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
        "Avg Precision": f"{m['average_precision']:.4f}",
        "N Test": f"{m['n_samples']:,}",
        "N Fraud": f"{m['n_positives']:,}",
    })

if rows:
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.info("No model metrics found in the saved file.")

st.caption(
    "Threshold = 0.5 for the three baseline models. "
    "The champion model uses a validation-optimised threshold ? see below."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Champion info
# ---------------------------------------------------------------------------
if "champion" in metrics:
    champ = metrics["champion"]
    champ_name = champ.get("name", "unknown")
    threshold = champ.get("threshold", 0.5)
    baseline_ap = champ.get("baseline_val_ap", None)
    tuned_ap = champ.get("tuned_val_ap", None)
    best_params = champ.get("tuning_best_params", {})

    st.subheader(f"Champion Model: {MODEL_LABELS.get(champ_name, champ_name)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Champion Selected By", "Validation Average Precision")
    c2.metric("Decision Threshold", f"{threshold:.4f}", help="F1-optimal threshold on validation set.")
    if baseline_ap and tuned_ap:
        delta = tuned_ap - baseline_ap
        c3.metric("Tuning Improvement (Val AP)", f"{tuned_ap:.4f}",
                  delta=f"{delta:+.4f} vs baseline")

    if best_params:
        with st.expander("Best Hyperparameters (RandomizedSearchCV)", expanded=False):
            for k, v in best_params.items():
                st.markdown(f"- `{k}` = `{v}`")

    # Champion test metrics
    champ_test_m = champ.get("test", {}).get("metrics", {})
    if champ_test_m:
        st.markdown(
            f"**Champion Test Metrics** (at threshold {threshold:.4f}):"
        )
        ccols = st.columns(6)
        metric_pairs = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("ROC-AUC", "roc_auc"),
            ("Avg Precision", "average_precision"),
        ]
        for col, (label, key) in zip(ccols, metric_pairs):
            col.metric(label, f"{champ_test_m.get(key, 0):.4f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Evaluation charts
# ---------------------------------------------------------------------------
st.subheader("Evaluation Charts")


def show_chart(path: Path, caption: str, col=None):
    target = col or st
    if path.exists():
        target.image(str(path), caption=caption, use_container_width=True)
    else:
        target.info(
            f"Chart not found: {path.name}. Run `python scripts/evaluate.py`."
        )


# Metrics comparison bar chart
show_chart(
    EVAL_RESULTS_DIR / "metrics_comparison.png",
    "Model Comparison ? Key Test-Set Metrics",
)
st.markdown("")

# ROC and PR curves
col_roc, col_pr = st.columns(2)
show_chart(EVAL_RESULTS_DIR / "roc_curves.png", "ROC Curves ? Test Set", col_roc)
show_chart(EVAL_RESULTS_DIR / "pr_curves.png", "Precision-Recall Curves ? Test Set", col_pr)
st.markdown("")

# Confusion matrices
show_chart(
    EVAL_RESULTS_DIR / "confusion_matrices.png",
    "Confusion Matrices ? Test Set (threshold=0.5)",
)
st.markdown("")

# Baseline vs validation-optimized threshold
show_chart(
    EVAL_RESULTS_DIR / "baseline_vs_threshold.png",
    "Baseline vs Champion Threshold ? Validation Set Metrics",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Methodology notes
# ---------------------------------------------------------------------------
st.subheader("Evaluation Methodology")
st.markdown(
    """
    - **Splits**: Stratified 70/15/15, seed 42. Splits are identical for all models.
    - **Champion selection**: Average Precision on the validation set (not the test set).
    - **Threshold tuning**: Decision threshold selected by maximising F1 on the
      validation set only. The test set was never used for model selection.
    - **Tuning**: RandomizedSearchCV with 5-fold StratifiedKFold, applied to the
      champion model on training data only.
    - **Test set evaluation**: Performed exactly once after all decisions were frozen.
    - **Class imbalance handling**: LR uses `class_weight='balanced'`;
      RF uses `class_weight='balanced_subsample'`;
      XGBoost uses `scale_pos_weight` computed from training data.
    - **Metric priority**: For this highly imbalanced dataset, Recall and Average
      Precision are more informative than Accuracy.
    """
)
