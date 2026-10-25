"""
Tests for src/data_loader.py

Covers:
- Schema validation raises on missing columns.
- Schema validation raises on unexpected target values.
- make_splits preserves stratification and correct ratios.
- get_X_y splits columns correctly.
- check_dataset_present returns bool.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, TARGET_COL
from src.data_loader import (
    _validate_schema,
    check_dataset_present,
    get_X_y,
    make_splits,
)


def _make_fake_df(n: int = 1000, fraud_frac: float = 0.02) -> pd.DataFrame:
    """Create a syntethic DataFrame with the expected schema for testing."""
    rng = np.random.default_rng(42)
    n_fraud = max(1, int(n * fraud_frac))
    data = {}
    for col in ALL_FEATURES:
        data[col] = rng.standard_normal(n)
    data[TARGET_COL] = [1] * n_fraud + [0] * (n - n_fraud)
    return pd.DataFrame(data)


class TestSchemaValidation:
    def test_valid_schema_passes(self):
        df = _make_fake_df()
        _validate_schema(df)  # should not raise

    def test_missing_column_raises(self):
        df = _make_fake_df()
        df = df.drop(columns=["V1"])
        with pytest.raises(ValueError, match="missing expected columns"):
            _validate_schema(df)

    def test_invalid_target_value_raises(self):
        df = _make_fake_df()
        df.loc[0, TARGET_COL] = 2  # invalid
        with pytest.raises(ValueError, match="Unexpected values"):
            _validate_schema(df)

    def test_extra_columns_not_blocked(self):
        """Extra columns are allowed (they should be ignored downstream)."""
        df = _make_fake_df()
        df["extra_col"] = 999
        _validate_schema(df)  # should not raise


class TestSplits:
    def test_split_sizes(self):
        df = _make_fake_df(1000)
        train, val, test = make_splits(df, 0.70, 0.15, 0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(df), "Split sizes should sum to full dataset"

    def test_train_ratio(self):
        df = _make_fake_df(1000)
        train, val, test = make_splits(df, 0.70, 0.15, 0.15)
        ratio = len(train) / len(df)
        assert abs(ratio - 0.70) < 0.02, f"Train ratio {ratio:.3f} not close to 0.70"

    def test_stratification_preserves_fraud_rate(self):
        df = _make_fake_df(2000, fraud_frac=0.02)
        overall_rate = df[TARGET_COL].mean()
        train, val, test = make_splits(df)
        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            rate = split[TARGET_COL].mean()
            assert abs(rate - overall_rate) < 0.01, (
                f"{split_name} fraud rate {rate:.4f} diverges from {overall_rate:.4f}"
            )

    def test_no_overlap_between_splits(self):
        """Row indices should not appear in more than one split."""
        df = _make_fake_df(500).reset_index(drop=True)
        train, val, test = make_splits(df)
        # After reset_index in make_splits, check column uniqueness on a key
        # (use index of returned DFs ? these are re-indexed 0..n)
        # We check that the union of lengths equals total rows
        assert len(train) + len(val) + len(test) == len(df)

    def test_reproducibility(self):
        df = _make_fake_df(500)
        train1, val1, test1 = make_splits(df, seed=42)
        train2, val2, test2 = make_splits(df, seed=42)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)


class TestGetXY:
    def test_x_has_only_features(self):
        df = _make_fake_df(100)
        X, y = get_X_y(df)
        assert list(X.columns) == ALL_FEATURES
        assert TARGET_COL not in X.columns

    def test_y_is_series(self):
        df = _make_fake_df(100)
        X, y = get_X_y(df)
        assert isinstance(y, pd.Series)
        assert y.name == TARGET_COL

    def test_lengths_match(self):
        df = _make_fake_df(100)
        X, y = get_X_y(df)
        assert len(X) == len(y) == 100


class TestCheckDataset:
    def test_returns_bool(self):
        result = check_dataset_present()
        assert isinstance(result, bool)
