"""
Tests for src/preprocessing.py

Covers:
- Preprocessor can be fitted on training data and transforms correctly.
- No leakage: scaler fitted on train should not be re-fitted on test.
- Output shape matches expected number of features.
- Feature order matches config.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, AMOUNT_COL, PCA_FEATURES, TIME_COL
from src.models import make_logistic_regression, make_random_forest
from src.preprocessing import (
    build_lr_pipeline,
    build_preprocessor,
    build_tree_pipeline,
    get_feature_names_out,
)


def _make_toy_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic DataFrame with the expected feature columns."""
    rng = np.random.default_rng(seed)
    data = {TIME_COL: rng.uniform(0, 172800, n),
            AMOUNT_COL: rng.exponential(100, n)}
    for v in PCA_FEATURES:
        data[v] = rng.standard_normal(n)
    return pd.DataFrame(data)[ALL_FEATURES]


class TestPreprocessor:
    def test_preprocessor_output_shape(self):
        """Preprocessor should output the correct number of features."""
        df = _make_toy_df(50)
        pre = build_preprocessor()
        out = pre.fit_transform(df)
        assert out.shape == (50, len(ALL_FEATURES)), (
            f"Expected {len(ALL_FEATURES)} features, got {out.shape[1]}"
        )

    def test_scale_columns_standardised(self):
        """After fitting, Time and Amount columns should have approx mean=0, std=1."""
        df = _make_toy_df(500)
        pre = build_preprocessor()
        out = pre.fit_transform(df)
        # Columns 0 and 1 are scaled (Time, Amount)
        mean_time = out[:, 0].mean()
        std_time = out[:, 0].std()
        assert abs(mean_time) < 0.1, f"Scaled Time mean too large: {mean_time}"
        assert abs(std_time - 1.0) < 0.1, f"Scaled Time std off: {std_time}"

    def test_pca_columns_passed_through(self):
        """V1-V28 should be unchanged after the ColumnTransformer."""
        df = _make_toy_df(50)
        pre = build_preprocessor()
        out = pre.fit_transform(df)
        # Columns 2 onwards should equal the original PCA values
        original_pca = df[PCA_FEATURES].values
        assert np.allclose(out[:, 2:], original_pca), "PCA columns should be passed through unchanged"

    def test_no_leakage_transform_only(self):
        """Transform on unseen data should NOT refit the scaler."""
        train_df = _make_toy_df(200, seed=0)
        test_df = _make_toy_df(50, seed=1) * 100  # very different scale
        pre = build_preprocessor()
        pre.fit(train_df)
        # This should just apply the TRAIN scaler to the test data
        out_test = pre.transform(test_df)
        # If scaler were refitted on test, test would have mean~0; with train scaler it won't
        # We verify the scaler params come from train
        scaler = pre.named_transformers_["scale"]
        assert np.allclose(scaler.mean_, train_df[[TIME_COL, AMOUNT_COL]].mean().values, rtol=0.01)

    def test_feature_names_out(self):
        """get_feature_names_out should return a list of the right length."""
        df = _make_toy_df(20)
        lr_pipe = build_lr_pipeline(make_logistic_regression())
        y = np.array([1] * 2 + [0] * 18)  # at least 2 classes required by sklearn
        lr_pipe.fit(df, y)
        names = get_feature_names_out(lr_pipe)
        assert len(names) == len(ALL_FEATURES), (
            f"Expected {len(ALL_FEATURES)} feature names, got {len(names)}"
        )


class TestPipelineBuilders:
    def test_lr_pipeline_structure(self):
        """LR pipeline should have 'preprocessor' and 'classifier' steps."""
        pipe = build_lr_pipeline(make_logistic_regression())
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_tree_pipeline_structure(self):
        """Tree pipeline should have 'preprocessor' and 'classifier' steps."""
        pipe = build_tree_pipeline(make_random_forest(n_estimators=10))
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_lr_pipeline_fit_predict(self):
        """LR pipeline should fit and produce probability outputs."""
        df = _make_toy_df(100)
        y = np.array([1] * 5 + [0] * 95)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(df, y)
        proba = pipe.predict_proba(df)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
