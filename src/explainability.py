"""
FraudGuard AI ? SHAP Explainability helpers.

Selects the correct SHAP explainer based on the underlying classifier type:
- TreeExplainer  ? RandomForest, XGBoost (fast, exact tree SHAP)
- LinearExplainer ? LogisticRegression (linear approximation)

Feature names returned correspond to the preprocessed feature space
(Time_scaled, Amount_scaled, V1-V28).

Note: V1-V28 are PCA-transformed, anonymized components of the original
transaction data. We do NOT invent original feature meanings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Explainer selection
# ---------------------------------------------------------------------------

def get_explainer(pipeline: Pipeline, X_background: pd.DataFrame) -> Any:
    """
    Select and return an appropriate SHAP Explainer for the given pipeline.

    The explainer operates on the *transformed* feature space (output of the
    preprocessor step), so X_background is preprocessed before fitting.

    Parameters
    ----------
    pipeline      : Fitted sklearn Pipeline with 'preprocessor' and
                    'classifier' named steps.
    X_background  : A small representative background sample (e.g., 200 rows
                    from the training set) used to estimate SHAP expectations.

    Returns
    -------
    A fitted SHAP Explainer instance.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform background data through the preprocessor only.
    X_bg_transformed = preprocessor.transform(X_background)
    # Convert to dense numpy array for SHAP compatibility
    if hasattr(X_bg_transformed, "toarray"):
        X_bg_transformed = X_bg_transformed.toarray()
    X_bg_transformed = np.asarray(X_bg_transformed, dtype=np.float64)

    classifier_type = type(classifier).__name__

    if classifier_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(
            classifier,
            data=X_bg_transformed,
            feature_perturbation="interventional",
        )
    elif classifier_type == "LogisticRegression":
        # LinearExplainer: pass background as the masker/data
        explainer = shap.LinearExplainer(
            classifier,
            X_bg_transformed,
        )
    else:
        # Fallback: kernel explainer (slow but universal)
        predict_fn = lambda x: classifier.predict_proba(x)[:, 1]  # noqa: E731
        explainer = shap.KernelExplainer(predict_fn, X_bg_transformed[:50])

    return explainer


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: Any,
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for input rows X.

    X is preprocessed via the pipeline's preprocessor step before SHAP
    values are computed, so the explainer and the SHAP values align.

    Returns
    -------
    2-D array of SHAP values, shape (n_samples, n_features).
    For binary classifiers, we return values for the positive class.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=np.float64)

    classifier = pipeline.named_steps["classifier"]
    classifier_type = type(classifier).__name__

    if classifier_type == "XGBClassifier":
        # XGBoost 2.x TreeExplainer returns 2-D array for binary classification
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list):
            # Older format: [negative_class, positive_class]
            return np.array(shap_vals[1])
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            # Shape (n_samples, n_features, n_classes) ? take positive class
            return arr[:, :, 1]
        # Already 2-D (XGBoost 2.x binary returns positive class directly)
        return arr

    elif classifier_type == "RandomForestClassifier":
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            return np.array(shap_vals[1])
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr

    else:  # LogisticRegression / KernelExplainer
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list):
            return np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
        return np.array(shap_vals)


def get_preprocessed_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Return the feature names after preprocessing (as used by SHAP).

    V1-V28 are PCA anonymized features; Time and Amount are standard scaled.
    """
    from src.config import PCA_FEATURES, TIME_COL, AMOUNT_COL
    # ColumnTransformer outputs scale columns first (Time, Amount), then passthrough
    return [f"{TIME_COL}_scaled", f"{AMOUNT_COL}_scaled"] + PCA_FEATURES


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------

def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature (global importance).

    Returns
    -------
    DataFrame sorted by importance descending, columns: feature, importance.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
