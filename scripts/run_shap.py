"""
FraudGuard AI ? SHAP explanation generation script.

Usage:
    python scripts/run_shap.py

Outputs:
    results/shap/global_importance.png   ? global mean|SHAP| bar chart
    results/shap/global_importance.csv   ? tabular global importance
    results/shap/shap_background.joblib  ? small background dataset for app

Requires:
    - models/champion_pipeline.joblib    (from train.py)
    - data/creditcard.csv                (to extract background samples)

SHAP values are computed in the TRANSFORMED feature space (after preprocessing).
V1-V28 are anonymized PCA components; we do NOT invent original feature meanings.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHAMPION_ARTIFACT_PATH, RANDOM_SEED, SHAP_RESULTS_DIR
from src.data_loader import get_X_y, make_splits, require_dataset
from src.explainability import (
    compute_shap_values,
    get_explainer,
    get_preprocessed_feature_names,
    global_feature_importance,
)
from src.inference import load_pipeline

NAVY = "#1a3a5c"
RED = "#c0392b"


def main() -> None:
    print("\nFraudGuard AI ? SHAP Explanation Generation")
    print("=" * 50)

    SHAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset and champion model...")
    df = require_dataset()
    df = df.drop_duplicates()

    pipeline = load_pipeline(CHAMPION_ARTIFACT_PATH)
    print("  Model loaded.")

    # ------------------------------------------------------------------
    # Get validation split (same seed = same split as training)
    # ------------------------------------------------------------------
    print("\n[2/5] Reconstructing data splits...")
    train_df, val_df, test_df = make_splits(df)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # ------------------------------------------------------------------
    # Background sample for SHAP (training data, balanced)
    # ------------------------------------------------------------------
    print("\n[3/5] Preparing SHAP background sample...")
    n_background = 200
    bg_fraud = X_train[y_train == 1].sample(
        min(n_background // 2, (y_train == 1).sum()), random_state=RANDOM_SEED
    )
    bg_legit = X_train[y_train == 0].sample(
        n_background - len(bg_fraud), random_state=RANDOM_SEED
    )
    X_background = pd.concat([bg_fraud, bg_legit]).sample(frac=1, random_state=RANDOM_SEED)
    print(f"  Background: {len(X_background)} rows "
          f"(fraud={len(bg_fraud)}, legit={len(bg_legit)})")

    # Save background for Streamlit app
    bg_path = SHAP_RESULTS_DIR / "shap_background.joblib"
    joblib.dump(X_background, bg_path)
    print(f"  Saved background: {bg_path}")

    # ------------------------------------------------------------------
    # Fit SHAP explainer
    # ------------------------------------------------------------------
    print("\n[4/5] Fitting SHAP explainer...")
    explainer = get_explainer(pipeline, X_background)
    print(f"  Explainer type: {type(explainer).__name__}")

    # ------------------------------------------------------------------
    # Compute SHAP values on a validation sample
    # ------------------------------------------------------------------
    print("\n[5/5] Computing SHAP values and plotting global importance...")
    n_shap_samples = min(500, len(X_val))
    X_shap_sample = X_val.sample(n_shap_samples, random_state=RANDOM_SEED)
    feature_names = get_preprocessed_feature_names(pipeline)

    shap_vals = compute_shap_values(explainer, pipeline, X_shap_sample)
    print(f"  SHAP values computed: shape={shap_vals.shape}")

    # Global importance
    importance_df = global_feature_importance(shap_vals, feature_names)
    imp_csv = SHAP_RESULTS_DIR / "global_importance.csv"
    importance_df.to_csv(imp_csv, index=False)
    print(f"  Saved: {imp_csv}")

    # Plot
    top_n = 20
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [RED if "V" in f else NAVY for f in top["feature"]]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Global Feature Importance (Top {top_n})\n"
        "Mean absolute SHAP value ? positive class (fraud)",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(labelsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RED, label="PCA component (V1-V28, anonymized)"),
        Patch(facecolor=NAVY, label="Raw feature (Time, Amount)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    imp_path = SHAP_RESULTS_DIR / "global_importance.png"
    plt.savefig(imp_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {imp_path}")

    # Save explainer for the app
    explainer_path = SHAP_RESULTS_DIR / "shap_explainer.joblib"
    joblib.dump(explainer, explainer_path)
    print(f"  Saved explainer: {explainer_path}")

    print("\nSHAP generation complete.\n")


if __name__ == "__main__":
    main()
