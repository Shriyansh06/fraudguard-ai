"""
Tests for src/evaluation.py

Covers:
- compute_metrics returns all required keys.
- Metrics are in valid ranges.
- select_threshold_max_f1 returns a float in [0, 1].
- save_metrics and load_metrics round-trip correctly.
- get_confusion_matrix returns correct shape.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    compute_metrics,
    get_confusion_matrix,
    get_pr_curve_data,
    get_roc_curve_data,
    save_metrics,
    load_metrics,
    select_threshold_max_f1,
)


def _make_predictions(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = np.array([1] * 20 + [0] * 180)
    y_prob = rng.beta(2, 5, n)
    # Bump fraud rows to have higher scores
    y_prob[:20] = rng.beta(5, 2, 20)
    return y_true, y_prob


class TestComputeMetrics:
    REQUIRED_KEYS = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "average_precision", "threshold",
        "prevalence", "n_samples", "n_positives", "n_negatives",
    ]

    def test_all_keys_present(self):
        y_true, y_prob = _make_predictions()
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_valid_range(self):
        y_true, y_prob = _make_predictions()
        result = compute_metrics(y_true, y_prob)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_n_samples_correct(self):
        y_true, y_prob = _make_predictions(200)
        result = compute_metrics(y_true, y_prob)
        assert result["n_samples"] == 200

    def test_n_positives_n_negatives_sum(self):
        y_true, y_prob = _make_predictions(200)
        result = compute_metrics(y_true, y_prob)
        assert result["n_positives"] + result["n_negatives"] == result["n_samples"]

    def test_threshold_stored(self):
        y_true, y_prob = _make_predictions()
        result = compute_metrics(y_true, y_prob, threshold=0.3)
        assert result["threshold"] == 0.3

    def test_perfect_classifier(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestConfusionMatrix:
    def test_shape_is_2x2(self):
        y_true, y_prob = _make_predictions()
        cm = get_confusion_matrix(y_true, y_prob)
        assert cm.shape == (2, 2)

    def test_entries_sum_to_n(self):
        y_true, y_prob = _make_predictions(200)
        cm = get_confusion_matrix(y_true, y_prob)
        assert cm.sum() == 200


class TestCurveData:
    def test_roc_curve_has_required_keys(self):
        y_true, y_prob = _make_predictions()
        result = get_roc_curve_data(y_true, y_prob)
        for key in ["fpr", "tpr", "thresholds", "auc"]:
            assert key in result

    def test_pr_curve_has_required_keys(self):
        y_true, y_prob = _make_predictions()
        result = get_pr_curve_data(y_true, y_prob)
        for key in ["precision", "recall", "thresholds", "average_precision"]:
            assert key in result

    def test_auc_in_range(self):
        y_true, y_prob = _make_predictions()
        result = get_roc_curve_data(y_true, y_prob)
        assert 0.0 <= result["auc"] <= 1.0


class TestThresholdSelection:
    def test_returns_float(self):
        y_true, y_prob = _make_predictions()
        t = select_threshold_max_f1(y_true, y_prob)
        assert isinstance(t, float)

    def test_threshold_in_range(self):
        y_true, y_prob = _make_predictions()
        t = select_threshold_max_f1(y_true, y_prob)
        assert 0.0 <= t <= 1.0


class TestMetricsPersistence:
    def test_save_load_roundtrip(self):
        y_true, y_prob = _make_predictions()
        metrics_data = {"logistic_regression": {"test": compute_metrics(y_true, y_prob)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics_data, path=path)
            loaded = load_metrics(path=path)
            dashboard = json.loads(path.with_name("metrics_dashboard.json").read_text())
        assert "logistic_regression" in loaded
        assert dashboard == metrics_data

    def test_dashboard_metrics_omit_curve_data(self):
        metrics_data = {
            "model": {
                "test": {
                    "metrics": {"accuracy": 0.9},
                    "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                    "pr_curve": {"precision": [1.0], "recall": [0.5]},
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics_data, path=path)
            dashboard = json.loads(path.with_name("metrics_dashboard.json").read_text())
            full = json.loads(path.read_text())

        assert dashboard["model"]["test"]["metrics"] == {"accuracy": 0.9}
        assert "roc_curve" not in dashboard["model"]["test"]
        assert "pr_curve" not in dashboard["model"]["test"]
        assert "roc_curve" in full["model"]["test"]

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_metrics(path=Path("/tmp/does_not_exist_metrics.json"))
