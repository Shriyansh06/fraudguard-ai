"""
FraudGuard AI ? Inference module.

Loads the saved champion pipeline and threshold; provides single-row and
batch prediction with consistent feature ordering.

The app calls these functions; it never retrains or re-fits any estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CHAMPION_ARTIFACT_PATH,
    FEATURE_SCHEMA_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
    THRESHOLD_PATH,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pipeline(path: Path = CHAMPION_ARTIFACT_PATH) -> Pipeline:
    """
    Load the saved sklearn Pipeline from disk.

    Raises FileNotFoundError with a helpful message if the model is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {path}\n\n"
            "Run the training script first:\n"
            "    python scripts/train.py\n"
        )
    return joblib.load(path)


def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    """Load the decision threshold chosen on the validation set."""
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found at: {path}\n"
            "Run scripts/train.py to generate it."
        )
    data = json.loads(path.read_text())
    return float(data["threshold"])


def load_model_metadata(path: Path = MODEL_META_PATH) -> dict:
    """Load model version and training metadata."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_feature_schema(path: Path = FEATURE_SCHEMA_PATH) -> list[str]:
    """Load the expected input feature list."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature schema not found at: {path}\n"
            "Run scripts/train.py to generate it."
        )
    return json.loads(path.read_text())["features"]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_input_schema(df: pd.DataFrame, expected_features: list[str]) -> None:
    """
    Verify that df contains all expected features in the correct order.

    Raises ValueError with a descriptive message on mismatch.
    """
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required features: {sorted(missing)}\n"
            f"Expected features: {expected_features}"
        )

    extra = set(df.columns) - set(expected_features)
    if extra:
        # Extra columns are silently dropped after reordering.
        pass


def reorder_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return df with only the expected features in the expected order."""
    return df[feature_names]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_proba_batch(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return predicted fraud probability for each row in X.

    Feature ordering is preserved (input row order = output row order).

    Returns
    -------
    1-D array of shape (n_rows,) with probabilities in [0, 1].
    """
    return pipeline.predict_proba(X)[:, 1]


def predict_batch(
    pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (probabilities, binary_predictions) for each row.

    Binary prediction: 1 if probability >= threshold else 0.
    """
    proba = predict_proba_batch(pipeline, X)
    preds = (proba >= threshold).astype(int)
    return proba, preds


def score_to_risk_band(score: float) -> str:
    """
    Map a model risk score (0-1) to a human-readable risk band.

    Risk bands are illustrative thresholds for display purposes;
    they are NOT calibrated probability estimates.
    """
    if score < 0.3:
        return "Low Risk"
    elif score < 0.6:
        return "Medium Risk"
    elif score < 0.85:
        return "High Risk"
    else:
        return "Critical Risk"


def score_dataframe(
    pipeline: Pipeline,
    df: pd.DataFrame,
    threshold: float,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Score a DataFrame of transactions and return results with predictions.

    Parameters
    ----------
    pipeline      : Fitted inference pipeline
    df            : Input DataFrame (must contain all feature columns)
    threshold     : Decision threshold
    feature_names : Expected feature names in expected order

    Returns
    -------
    Original df + columns: risk_score, predicted_class, risk_band.
    Row order is preserved.
    """
    validate_input_schema(df, feature_names)
    X = reorder_features(df, feature_names)

    proba, preds = predict_batch(pipeline, X, threshold)

    result = df.copy().reset_index(drop=True)
    result["risk_score"] = np.round(proba, 6)
    result["predicted_class"] = preds
    result["risk_band"] = [score_to_risk_band(p) for p in proba]

    return result
