"""
FraudGuard AI ? Model factory functions.

Each factory returns a configured (unfitted) sklearn-compatible estimator.
Imbalance handling is done via class_weight or scale_pos_weight ? no SMOTE
is applied at this layer (SMOTE comparison is in the training script).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_SEED


def make_logistic_regression(class_weight: str = "balanced") -> LogisticRegression:
    """
    Logistic Regression with L2 regularisation.

    class_weight='balanced' adjusts weights inversely proportional to
    class frequencies in the training data ? appropriate for high imbalance.
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
        C=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def make_random_forest(
    n_estimators: int = 200,
    class_weight: str = "balanced_subsample",
) -> RandomForestClassifier:
    """
    Random Forest with balanced class weights per bootstrap sample.

    'balanced_subsample' recomputes class weights for each bootstrap sample,
    which tends to work better than 'balanced' for heavily imbalanced data.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def make_xgboost(scale_pos_weight: float | None = None) -> XGBClassifier:
    """
    XGBoost classifier.

    scale_pos_weight is set to (n_negatives / n_positives) on the training
    set; this compensates for class imbalance in a principled way.
    Pass None to compute it dynamically during training.
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        device="cpu",
    )


def compute_scale_pos_weight(y_train) -> float:
    """
    Compute XGBoost's scale_pos_weight as n_negatives / n_positives.

    This is the recommended approach from the XGBoost documentation for
    imbalanced binary classification.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / n_pos)
