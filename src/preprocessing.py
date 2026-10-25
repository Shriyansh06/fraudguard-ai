"""
FraudGuard AI ? Preprocessing pipelines.

Design principles:
- Pipelines are fitted ONLY on training data (no leakage).
- StandardScaler applied to Time and Amount (PCA features V1-V28 are
  already approximately standardized by the dataset authors).
- SMOTE (optional) is applied inside cross-validation or after train split;
  never on val/test.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AMOUNT_COL, ALL_FEATURES, PCA_FEATURES, TIME_COL


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

SCALE_COLS: list[str] = [TIME_COL, AMOUNT_COL]
PASSTHROUGH_COLS: list[str] = PCA_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - StandardScales Time and Amount.
    - Passes V1-V28 through unchanged.

    The transformer must be fitted on TRAINING data only.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_lr_pipeline(estimator) -> Pipeline:
    """
    Full sklearn Pipeline for Logistic Regression:
    preprocessing -> classifier.

    Parameters
    ----------
    estimator : LogisticRegression instance
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_tree_pipeline(estimator) -> Pipeline:
    """
    Minimal Pipeline for tree-based models (RF, XGBoost).

    Tree models do not require feature scaling; however, we still apply
    the ColumnTransformer so that all pipelines share a consistent
    preprocessing interface and identical output feature ordering.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def get_feature_names_out(pipeline: Pipeline) -> list[str]:
    """
    Retrieve output feature names from a fitted pipeline's preprocessor step.

    Useful for SHAP explanations where feature names must align with the
    transformed matrix.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())
