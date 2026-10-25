"""
Tests for CSV upload error handling (as used in Streamlit pages).

Covers:
- Empty CSV raises or returns empty DataFrame gracefully.
- CSV missing required columns raises ValueError.
- CSV with extra columns is accepted after reordering.
- Malformed CSV is caught.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
from src.inference import validate_input_schema, reorder_features


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _make_valid_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {TIME_COL: rng.uniform(0, 1e5, n), AMOUNT_COL: rng.exponential(100, n)}
    for v in PCA_FEATURES:
        data[v] = rng.standard_normal(n)
    return pd.DataFrame(data)


class TestCSVUploadHandling:
    def test_valid_csv_accepted(self):
        df = _make_valid_df()
        validate_input_schema(df, ALL_FEATURES)  # should not raise
        reordered = reorder_features(df, ALL_FEATURES)
        assert list(reordered.columns) == ALL_FEATURES

    def test_missing_column_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=["Amount"])
        with pytest.raises(ValueError, match="missing required features"):
            validate_input_schema(df, ALL_FEATURES)

    def test_extra_column_accepted(self):
        df = _make_valid_df()
        df["Class"] = 0  # ground truth column should be accepted
        validate_input_schema(df, ALL_FEATURES)  # no raise
        reordered = reorder_features(df, ALL_FEATURES)
        assert "Class" not in reordered.columns

    def test_wrong_dtype_is_accepted(self):
        """String values in numeric columns should not crash schema validation.
        (Downstream inference may fail, but schema check is column-name based.)"""
        df = _make_valid_df()
        df["Amount"] = df["Amount"].astype(str)
        validate_input_schema(df, ALL_FEATURES)  # schema is column-name based

    def test_empty_dataframe_has_no_rows(self):
        df = pd.DataFrame(columns=ALL_FEATURES)
        validate_input_schema(df, ALL_FEATURES)  # zero rows is valid schema
        assert len(df) == 0

    def test_malformed_csv_raises_parse_error(self):
        """Simulate a malformed CSV (unparseable)."""
        bad_bytes = b"this is not a csv\x00\x01\x02"
        buf = io.BytesIO(bad_bytes)
        try:
            df = pd.read_csv(buf)
            # If pandas doesn't raise, schema validation should catch missing columns
            with pytest.raises(ValueError):
                validate_input_schema(df, ALL_FEATURES)
        except Exception:
            pass  # parse error caught ? expected

    def test_single_row_csv(self):
        df = _make_valid_df(1)
        validate_input_schema(df, ALL_FEATURES)
        assert len(df) == 1

    def test_large_csv_accepted(self):
        df = _make_valid_df(10_000)
        validate_input_schema(df, ALL_FEATURES)
        reordered = reorder_features(df, ALL_FEATURES)
        assert len(reordered) == 10_000
