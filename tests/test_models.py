"""
Tests for src/models.py

Covers:
- Each factory returns the correct estimator type.
- class_weight and scale_pos_weight are correctly configured.
- compute_scale_pos_weight returns correct ratio.
- A fitted toy pipeline produces probabilities in [0, 1].
- Model artifacts can be saved and loaded with joblib.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL, RANDOM_SEED
from src.models import (
    compute_scale_pos_weight,
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
)
from src.preprocessing import build_lr_pipeline, build_tree_pipeline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_toy_df(n: int = 150, seed: int = 0):
    """
    Return (X, y) with the correct 30-feature schema and imbalanced labels.
    Labels are shuffled so fraud rows are not all at the top.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        TIME_COL: rng.uniform(0, 172800, n),
        AMOUNT_COL: rng.exponential(100, n),
    })
    for v in PCA_FEATURES:
        df[v] = rng.standard_normal(n)
    df = df[ALL_FEATURES]

    n_fraud = max(2, int(n * 0.06))   # at least 2 fraud rows so both classes present
    n_legit = n - n_fraud
    y = np.array([1] * n_fraud + [0] * n_legit)
    # Shuffle so fraud labels are not contiguous
    shuffle_idx = rng.permutation(n)
    y = y[shuffle_idx]
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    return df, y


# ---------------------------------------------------------------------------
# Factory type tests
# ---------------------------------------------------------------------------

class TestFactoryTypes:
    def test_logistic_regression_type(self):
        from sklearn.linear_model import LogisticRegression
        lr = make_logistic_regression()
        assert isinstance(lr, LogisticRegression)

    def test_random_forest_type(self):
        from sklearn.ensemble import RandomForestClassifier
        rf = make_random_forest()
        assert isinstance(rf, RandomForestClassifier)

    def test_xgboost_type(self):
        from xgboost import XGBClassifier
        xgb = make_xgboost()
        assert isinstance(xgb, XGBClassifier)


# ---------------------------------------------------------------------------
# Class weight / imbalance configuration tests
# ---------------------------------------------------------------------------

class TestClassWeights:
    def test_lr_balanced_class_weight(self):
        lr = make_logistic_regression()
        assert lr.class_weight == "balanced"

    def test_rf_balanced_subsample_class_weight(self):
        rf = make_random_forest()
        assert rf.class_weight == "balanced_subsample"

    def test_xgb_scale_pos_weight_applied(self):
        xgb = make_xgboost(scale_pos_weight=50.0)
        # Use pytest.approx to handle int/float storage differences
        assert xgb.scale_pos_weight == pytest.approx(50.0)

    def test_xgb_default_scale_pos_weight_is_one(self):
        """When scale_pos_weight=None, factory should default to 1."""
        xgb = make_xgboost()
        assert xgb.scale_pos_weight == pytest.approx(1.0)

    def test_lr_random_state_matches_config(self):
        lr = make_logistic_regression()
        assert lr.random_state == RANDOM_SEED

    def test_rf_random_state_matches_config(self):
        rf = make_random_forest()
        assert rf.random_state == RANDOM_SEED

    def test_rf_n_estimators_custom(self):
        rf = make_random_forest(n_estimators=50)
        assert rf.n_estimators == 50

    def test_lr_max_iter(self):
        lr = make_logistic_regression()
        assert lr.max_iter == 1000

    def test_lr_solver(self):
        lr = make_logistic_regression()
        assert lr.solver == "lbfgs"


# ---------------------------------------------------------------------------
# compute_scale_pos_weight tests
# ---------------------------------------------------------------------------

class TestScalePosWeight:
    def test_correct_ratio(self):
        y = np.array([0] * 578 + [1] * 1)
        spw = compute_scale_pos_weight(y)
        assert spw == pytest.approx(578.0, rel=1e-3)

    def test_balanced_dataset_returns_one(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        spw = compute_scale_pos_weight(y)
        assert spw == pytest.approx(1.0, rel=1e-3)

    def test_returns_float(self):
        y = np.array([1] * 10 + [0] * 100)
        spw = compute_scale_pos_weight(y)
        assert isinstance(spw, float)

    def test_high_imbalance(self):
        """Typical ULB ratio ~578:1 should give large scale_pos_weight."""
        y = np.array([0] * 284315 + [1] * 492)
        spw = compute_scale_pos_weight(y)
        assert spw > 100.0

    def test_uses_training_labels(self):
        """Result must change when label balance changes."""
        y1 = np.array([0] * 100 + [1] * 10)
        y2 = np.array([0] * 100 + [1] * 50)
        assert compute_scale_pos_weight(y1) > compute_scale_pos_weight(y2)


# ---------------------------------------------------------------------------
# Pipeline fit + predict tests
# ---------------------------------------------------------------------------

class TestLRPipelineFit:
    def test_predict_proba_shape(self):
        X, y = _make_toy_df(150)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (150, 2), f"Expected (150, 2), got {proba.shape}"

    def test_probabilities_sum_to_one(self):
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_probabilities_in_range(self):
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        pos_proba = pipe.predict_proba(X)[:, 1]
        assert (pos_proba >= 0).all() and (pos_proba <= 1).all()

    def test_predict_returns_binary(self):
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


class TestRFPipelineFit:
    def test_predict_proba_shape(self):
        X, y = _make_toy_df(150)
        pipe = build_tree_pipeline(make_random_forest(n_estimators=10))
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (150, 2)

    def test_predict_returns_binary(self):
        X, y = _make_toy_df(100)
        pipe = build_tree_pipeline(make_random_forest(n_estimators=5))
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_in_range(self):
        X, y = _make_toy_df(120)
        pipe = build_tree_pipeline(make_random_forest(n_estimators=5))
        pipe.fit(X, y)
        pos_proba = pipe.predict_proba(X)[:, 1]
        assert (pos_proba >= 0).all() and (pos_proba <= 1).all()


class TestXGBPipelineFit:
    def test_predict_proba_shape(self):
        X, y = _make_toy_df(150)
        spw = compute_scale_pos_weight(y)
        pipe = build_tree_pipeline(make_xgboost(scale_pos_weight=spw))
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (150, 2)

    def test_probabilities_in_range(self):
        X, y = _make_toy_df(150)
        pipe = build_tree_pipeline(make_xgboost(scale_pos_weight=10.0))
        pipe.fit(X, y)
        pos_proba = pipe.predict_proba(X)[:, 1]
        assert (pos_proba >= 0).all() and (pos_proba <= 1).all()

    def test_probabilities_sum_to_one(self):
        X, y = _make_toy_df(150)
        pipe = build_tree_pipeline(make_xgboost(scale_pos_weight=5.0))
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Artifact save / load integrity
# ---------------------------------------------------------------------------

class TestModelArtifacts:
    def test_joblib_save_and_load_lr(self):
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lr_pipeline.joblib"
            joblib.dump(pipe, path)
            assert path.exists(), "Saved file not found"
            loaded = joblib.load(path)
            np.testing.assert_array_almost_equal(
                pipe.predict_proba(X),
                loaded.predict_proba(X),
                decimal=6,
            )

    def test_joblib_save_and_load_rf(self):
        X, y = _make_toy_df(100)
        pipe = build_tree_pipeline(make_random_forest(n_estimators=5))
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rf_pipeline.joblib"
            joblib.dump(pipe, path)
            loaded = joblib.load(path)
            np.testing.assert_array_almost_equal(
                pipe.predict_proba(X),
                loaded.predict_proba(X),
                decimal=6,
            )

    def test_loaded_model_accepts_30_features(self):
        """Loaded pipeline must accept exactly 30 input features."""
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipe.joblib"
            joblib.dump(pipe, path)
            loaded = joblib.load(path)
            proba = loaded.predict_proba(X)
            assert proba.shape == (len(X), 2)

    def test_loaded_model_output_row_count(self):
        """Output row count must match input row count after loading."""
        X, y = _make_toy_df(80)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipe.joblib"
            joblib.dump(pipe, path)
            loaded = joblib.load(path)
            assert loaded.predict_proba(X).shape[0] == 80

    def test_wrong_feature_raises(self):
        """Pipeline should raise when a required feature column is missing."""
        X, y = _make_toy_df(100)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        X_bad = X.drop(columns=["V14"])  # remove one column
        with pytest.raises(Exception):
            pipe.predict_proba(X_bad)
