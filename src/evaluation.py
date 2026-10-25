"""
FraudGuard AI ? Evaluation helpers.

Provides:
- A dictionary of all required metrics for a single model/split.
- Helpers to compute confusion matrices and curve data for plotting.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DASHBOARD_METRICS_PATH, RESULTS_DIR


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : array of 0/1 ground-truth labels
    y_prob : predicted probabilities for the positive class
    threshold : decision threshold (default 0.5)

    Returns
    -------
    Dictionary with: accuracy, precision, recall, f1, roc_auc,
                     average_precision, threshold used, prevalence.
    """
    y_pred = (y_prob >= threshold).astype(int)
    prevalence = float(y_true.mean())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "prevalence": prevalence,
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "n_negatives": int((1 - y_true).sum()),
    }


def get_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# Curve data helpers (for ROC and Precision-Recall plots)
# ---------------------------------------------------------------------------

def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute FPR, TPR, thresholds and AUC for a ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist(), "auc": auc}


def get_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute precision, recall, thresholds and AP for a PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": ap,
    }


# ---------------------------------------------------------------------------
# Threshold selection on validation set
# ---------------------------------------------------------------------------

def select_threshold_max_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Choose the decision threshold that maximises F1 on validation data.

    Returns
    -------
    Optimal threshold as a float.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision and recall have one more entry than thresholds
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_metrics(all_metrics: dict[str, Any], path: Path | None = None) -> None:
    """Save full metrics and a compact copy for the Streamlit dashboard."""
    save_path = path or (RESULTS_DIR / "metrics.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(all_metrics, indent=2))
    dashboard_path = (
        DASHBOARD_METRICS_PATH
        if path is None
        else path.with_name("metrics_dashboard.json")
    )
    dashboard_path.write_text(json.dumps(_without_curve_data(all_metrics), indent=2))


def _without_curve_data(value: Any) -> Any:
    """Drop large precomputed curve arrays; the dashboard uses saved chart images."""
    if isinstance(value, dict):
        return {
            key: _without_curve_data(item)
            for key, item in value.items()
            if key not in {"roc_curve", "pr_curve"}
        }
    if isinstance(value, list):
        return [_without_curve_data(item) for item in value]
    return value


def load_metrics(path: Path | None = None) -> dict[str, Any]:
    """Load previously saved metrics from JSON."""
    load_path = path or (RESULTS_DIR / "metrics.json")
    if not load_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {load_path}. Run scripts/train.py first."
        )
    return json.loads(load_path.read_text())
