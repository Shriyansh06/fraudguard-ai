"""
FraudGuard AI ? Data loading and schema validation.

Responsibilities:
- Locate the dataset file.
- Validate column names and target values.
- Provide reproducible stratified splits (train / val / test).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES,
    DATASET_PATH,
    EXPECTED_COLUMNS,
    FEATURE_SCHEMA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def check_dataset_present() -> bool:
    """Return True if the dataset CSV exists on disk."""
    return DATASET_PATH.exists()


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the ULB Credit Card Fraud dataset from *path*.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found.
    ValueError
        If column names or target values do not match expected schema.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n\n"
            "Download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Then place the file at: data/creditcard.csv"
        )

    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if column names or target values are unexpected."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_cols)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    unexpected_targets = set(df[TARGET_COL].unique()) - {0, 1}
    if unexpected_targets:
        raise ValueError(
            f"Unexpected values in '{TARGET_COL}' column: {unexpected_targets}. "
            "Expected only 0 (legitimate) and 1 (fraud)."
        )


# ---------------------------------------------------------------------------
# Stratified splits ? 70 / 15 / 15
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce reproducible stratified train / validation / test splits.

    The test set is returned last and should remain UNTOUCHED until final
    evaluation.  Validation and test fractions are computed relative to the
    full dataset size, not relative to remaining data.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Split ratios must sum to 1.0"
    )

    # First split: train vs. (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df[TARGET_COL],
        random_state=seed,
    )

    # Second split: val vs. test (relative sizes within val+test portion)
    relative_test = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test,
        stratify=val_test_df[TARGET_COL],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and label vector y."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ---------------------------------------------------------------------------
# Feature schema persistence
# ---------------------------------------------------------------------------

def save_feature_schema(feature_names: list[str]) -> None:
    """Persist the ordered feature list used during training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    schema = {"features": feature_names, "n_features": len(feature_names)}
    FEATURE_SCHEMA_PATH.write_text(json.dumps(schema, indent=2))


def load_feature_schema() -> list[str]:
    """Load the saved feature ordering."""
    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {FEATURE_SCHEMA_PATH}. "
            "Run scripts/train.py first."
        )
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text())
    return schema["features"]


# ---------------------------------------------------------------------------
# CLI helper: abort if dataset missing
# ---------------------------------------------------------------------------

def require_dataset() -> pd.DataFrame:
    """
    Load the dataset or print a helpful message and exit.

    Use this at the top of training scripts.
    """
    if not check_dataset_present():
        print(
            "\n[ERROR] Dataset not found.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place the file at: data/creditcard.csv\n"
        )
        sys.exit(1)
    return load_dataset()
