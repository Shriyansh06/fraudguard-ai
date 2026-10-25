"""
FraudGuard AI ? Central configuration: paths, constants, feature lists, seeds.

All scripts and notebooks import from this module to ensure consistency.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (the directory containing this src/ folder)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_PATH: Path = DATA_DIR / "creditcard.csv"
DEMO_CSV_PATH: Path = DATA_DIR / "demo_transactions.csv"

EXPECTED_ROWS: int = 284_807
EXPECTED_COLS: int = 31

TARGET_COL: str = "Class"
TIME_COL: str = "Time"
AMOUNT_COL: str = "Amount"

PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]  # V1 ? V28
ALL_FEATURES: list[str] = [TIME_COL, AMOUNT_COL] + PCA_FEATURES  # 30 features

EXPECTED_COLUMNS: list[str] = ALL_FEATURES + [TARGET_COL]

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15  # must equal 1 - TRAIN_RATIO - VAL_RATIO

# ---------------------------------------------------------------------------
# Training / CV
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5
TUNING_N_ITER: int = 30  # RandomizedSearchCV iterations

# ---------------------------------------------------------------------------
# Model artifact filenames
# ---------------------------------------------------------------------------
MODEL_NAMES: list[str] = ["logistic_regression", "random_forest", "xgboost"]
CHAMPION_MODEL_NAME: str = "champion"  # resolved during training

ARTIFACT_PATHS: dict[str, Path] = {
    name: MODELS_DIR / f"{name}_pipeline.joblib" for name in MODEL_NAMES
}
CHAMPION_ARTIFACT_PATH: Path = MODELS_DIR / f"{CHAMPION_MODEL_NAME}_pipeline.joblib"
METRICS_PATH: Path = RESULTS_DIR / "metrics.json"
DASHBOARD_METRICS_PATH: Path = RESULTS_DIR / "metrics_dashboard.json"
THRESHOLD_PATH: Path = MODELS_DIR / "champion_threshold.json"
FEATURE_SCHEMA_PATH: Path = MODELS_DIR / "feature_schema.json"
MODEL_META_PATH: Path = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Results sub-paths
# ---------------------------------------------------------------------------
EDA_RESULTS_DIR: Path = RESULTS_DIR / "eda"
EVAL_RESULTS_DIR: Path = RESULTS_DIR / "evaluation"
SHAP_RESULTS_DIR: Path = RESULTS_DIR / "shap"

# ---------------------------------------------------------------------------
# Quick-dev mode: use a 5 % stratified sample ? clearly non-final
# ---------------------------------------------------------------------------
QUICK_DEV_SAMPLE_FRAC: float = 0.05
QUICK_DEV_LABEL: str = "[QUICK-DEV ? non-final results]"
