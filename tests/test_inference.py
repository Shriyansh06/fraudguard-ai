"""
Tests for src/inference.py

Covers:
- validate_input_schema raises on missing features.
- reorder_features returns correct column order.
- score_to_risk_band returns correct bands.
- predict_batch output length matches input length.
- score_dataframe preserves row order.
- load_pipeline raises FileNotFoundError when model missing.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
from src.inference import (
    reorder_features,
    score_to_risk_band,
    validate_input_schema,
    load_pipeline,
)


def _make_feature_df(n: int = 10) -> pd.DataFrame:
    """Minimal valid feature DataFrame."""
    rng = np.random.default_rng(0)
    data = {TIME_COL: rng.uniform(0, 1e5, n), AMOUNT_COL: rng.exponential(100, n)}
    for v in PCA_FEATURES:
        data[v] = rng.standard_normal(n)
    return pd.DataFrame(data)


class TestSchemaValidation:
    def test_valid_input_passes(self):
        df = _make_feature_df()
        validate_input_schema(df, ALL_FEATURES)  # should not raise

    def test_missing_feature_raises(self):
        df = _make_feature_df()
        df = df.drop(columns=["V14"])
        with pytest.raises(ValueError, match="missing required features"):
            validate_input_schema(df, ALL_FEATURES)

    def test_extra_column_allowed(self):
        """Extra columns should not raise ? they are dropped by reorder_features."""
        df = _make_feature_df()
        df["extra"] = 999
        validate_input_schema(df, ALL_FEATURES)  # should not raise


class TestReorderFeatures:
    def test_output_columns_match_expected(self):
        df = _make_feature_df()
        df["extra"] = 999
        result = reorder_features(df, ALL_FEATURES)
        assert list(result.columns) == ALL_FEATURES

    def test_row_count_preserved(self):
        df = _make_feature_df(25)
        result = reorder_features(df, ALL_FEATURES)
        assert len(result) == 25


class TestRiskBand:
    def test_low_risk(self):
        assert score_to_risk_band(0.1) == "Low Risk"
        assert score_to_risk_band(0.0) == "Low Risk"

    def test_medium_risk(self):
        assert score_to_risk_band(0.45) == "Medium Risk"

    def test_high_risk(self):
        assert score_to_risk_band(0.7) == "High Risk"

    def test_critical_risk(self):
        assert score_to_risk_band(0.95) == "Critical Risk"
        assert score_to_risk_band(1.0) == "Critical Risk"

    def test_boundary_values(self):
        # Exact boundary 0.3 ? Medium
        assert score_to_risk_band(0.3) == "Medium Risk"
        # Exact boundary 0.6 ? High
        assert score_to_risk_band(0.6) == "High Risk"
        # Exact boundary 0.85 ? Critical
        assert score_to_risk_band(0.85) == "Critical Risk"


class TestLoadPipeline:
    def test_missing_model_raises_file_not_found(self):
        non_existent = Path("/tmp/does_not_exist.joblib")
        with pytest.raises(FileNotFoundError, match="Trained model not found"):
            load_pipeline(non_existent)


class TestScoreDataframe:
    """Integration tests for score_dataframe using a toy fitted pipeline."""

    @pytest.fixture
    def toy_pipeline(self):
        """Fit a tiny LR pipeline on toy data for testing inference."""
        from src.models import make_logistic_regression
        from src.preprocessing import build_lr_pipeline
        rng = np.random.default_rng(0)
        n = 200
        X = _make_feature_df(n)
        y = np.array([1] * 10 + [0] * 190)
        pipe = build_lr_pipeline(make_logistic_regression())
        pipe.fit(X, y)
        return pipe

    def test_output_length_matches_input(self, toy_pipeline):
        from src.inference import score_dataframe
        df = _make_feature_df(30)
        result = score_dataframe(toy_pipeline, df, threshold=0.5, feature_names=ALL_FEATURES)
        assert len(result) == 30

    def test_output_has_required_columns(self, toy_pipeline):
        from src.inference import score_dataframe
        df = _make_feature_df(10)
        result = score_dataframe(toy_pipeline, df, threshold=0.5, feature_names=ALL_FEATURES)
        assert "risk_score" in result.columns
        assert "predicted_class" in result.columns
        assert "risk_band" in result.columns

    def test_scores_in_range(self, toy_pipeline):
        from src.inference import score_dataframe
        df = _make_feature_df(50)
        result = score_dataframe(toy_pipeline, df, threshold=0.5, feature_names=ALL_FEATURES)
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 1).all()

    def test_predicted_class_is_binary(self, toy_pipeline):
        from src.inference import score_dataframe
        df = _make_feature_df(50)
        result = score_dataframe(toy_pipeline, df, threshold=0.5, feature_names=ALL_FEATURES)
        assert set(result["predicted_class"].unique()).issubset({0, 1})

    def test_row_order_preserved(self, toy_pipeline):
        """Row ordering must match input ordering."""
        from src.inference import score_dataframe
        df = _make_feature_df(20)
        df_shuffled = df.sample(frac=1, random_state=5).reset_index(drop=True)
        result = score_dataframe(toy_pipeline, df_shuffled, threshold=0.5, feature_names=ALL_FEATURES)
        # Verify the input rows are unchanged
        pd.testing.assert_frame_equal(
            result[ALL_FEATURES].reset_index(drop=True),
            df_shuffled[ALL_FEATURES].reset_index(drop=True),
        )
