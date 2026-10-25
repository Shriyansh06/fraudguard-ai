"""
FraudGuard AI ? Standalone test runner.
Run this script directly to check if all source modules import correctly
and core logic works without the full dataset.

Usage:
    python scripts/run_tests.py

This tests the same logic as pytest but without requiring pytest to be
installed separately from requirements.txt.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


print("\nFraudGuard AI ? Module Smoke Tests")
print("=" * 55)

# ------------------------------------------------------------------
# 1. Config imports
# ------------------------------------------------------------------
print("\n[1] Config module")

def test_config_imports():
    from src.config import (
        ALL_FEATURES, TARGET_COL, RANDOM_SEED,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        MODEL_NAMES, CHAMPION_ARTIFACT_PATH,
    )
    assert len(ALL_FEATURES) == 30
    assert TARGET_COL == "Class"
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
    assert len(MODEL_NAMES) == 3

check("Config imports and constants", test_config_imports)

# ------------------------------------------------------------------
# 2. Data loader
# ------------------------------------------------------------------
print("\n[2] Data loader")

def test_schema_validation_pass():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, TARGET_COL
    from src.data_loader import _validate_schema
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.standard_normal(100) for col in ALL_FEATURES})
    df[TARGET_COL] = np.random.choice([0, 1], 100)
    _validate_schema(df)  # should not raise

check("Schema validation - valid", test_schema_validation_pass)

def test_schema_validation_missing():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, TARGET_COL
    from src.data_loader import _validate_schema
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.standard_normal(100) for col in ALL_FEATURES[:-2]})
    df[TARGET_COL] = 0
    try:
        _validate_schema(df)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass  # expected

check("Schema validation - missing column raises", test_schema_validation_missing)

def test_splits():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, TARGET_COL
    from src.data_loader import make_splits
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.standard_normal(1000) for col in ALL_FEATURES})
    df[TARGET_COL] = [1] * 20 + [0] * 980
    train, val, test = make_splits(df)
    assert len(train) + len(val) + len(test) == 1000
    assert abs(len(train) / 1000 - 0.70) < 0.02

check("Stratified 70/15/15 splits", test_splits)

# ------------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------------
print("\n[3] Preprocessing")

def test_preprocessor():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.preprocessing import build_preprocessor
    rng = np.random.default_rng(0)
    df = pd.DataFrame({TIME_COL: rng.uniform(0, 1e5, 200),
                       AMOUNT_COL: rng.exponential(100, 200)})
    for v in PCA_FEATURES:
        df[v] = rng.standard_normal(200)
    df = df[ALL_FEATURES]
    pre = build_preprocessor()
    out = pre.fit_transform(df)
    assert out.shape == (200, 30), f"Expected (200,30) got {out.shape}"
    # Time should be approx standardized
    assert abs(out[:, 0].mean()) < 0.2

check("Preprocessor output shape and scaling", test_preprocessor)

def test_no_leakage():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.preprocessing import build_preprocessor
    rng = np.random.default_rng(0)
    def make_df(n, seed):
        rng2 = np.random.default_rng(seed)
        df = pd.DataFrame({TIME_COL: rng2.uniform(0, 1e5, n),
                           AMOUNT_COL: rng2.exponential(100, n)})
        for v in PCA_FEATURES:
            df[v] = rng2.standard_normal(n)
        return df[ALL_FEATURES]
    train_df = make_df(300, 0)
    test_df = make_df(100, 1)
    pre = build_preprocessor()
    pre.fit(train_df)
    scaler = pre.named_transformers_["scale"]
    # Scaler mean should match train mean
    assert abs(scaler.mean_[0] - train_df[TIME_COL].mean()) < 1.0

check("No leakage: scaler fitted on train only", test_no_leakage)

# ------------------------------------------------------------------
# 4. Model factories
# ------------------------------------------------------------------
print("\n[4] Model factories")

def test_lr_factory():
    from src.models import make_logistic_regression
    lr = make_logistic_regression()
    assert lr.class_weight == "balanced"
    assert lr.max_iter == 1000

check("LR factory", test_lr_factory)

def test_rf_factory():
    from src.models import make_random_forest
    rf = make_random_forest(n_estimators=10)
    assert rf.class_weight == "balanced_subsample"
    assert rf.n_estimators == 10

check("RF factory", test_rf_factory)

def test_xgb_factory():
    from src.models import make_xgboost
    xgb = make_xgboost(scale_pos_weight=10.0)
    assert xgb.scale_pos_weight == 10.0

check("XGBoost factory", test_xgb_factory)

def test_scale_pos_weight():
    import numpy as np
    from src.models import compute_scale_pos_weight
    y = np.array([0] * 100 + [1] * 2)
    spw = compute_scale_pos_weight(y)
    assert abs(spw - 50.0) < 0.1

check("scale_pos_weight computation", test_scale_pos_weight)

# ------------------------------------------------------------------
# 5. Evaluation metrics
# ------------------------------------------------------------------
print("\n[5] Evaluation")

def test_compute_metrics():
    import numpy as np
    from src.evaluation import compute_metrics
    y_true = np.array([1] * 20 + [0] * 180)
    y_prob = np.random.default_rng(0).beta(2, 5, 200)
    result = compute_metrics(y_true, y_prob)
    required = ["accuracy", "precision", "recall", "f1",
                "roc_auc", "average_precision", "threshold",
                "prevalence", "n_samples", "n_positives", "n_negatives"]
    for k in required:
        assert k in result, f"Missing key: {k}"
    assert result["n_samples"] == 200
    assert 0.0 <= result["roc_auc"] <= 1.0

check("compute_metrics all keys and ranges", test_compute_metrics)

def test_threshold_selection():
    import numpy as np
    from src.evaluation import select_threshold_max_f1
    y_true = np.array([1] * 20 + [0] * 180)
    rng = np.random.default_rng(0)
    y_prob = rng.beta(2, 5, 200)
    t = select_threshold_max_f1(y_true, y_prob)
    assert 0.0 <= t <= 1.0

check("Threshold selection returns valid float", test_threshold_selection)

# ------------------------------------------------------------------
# 6. Inference schema
# ------------------------------------------------------------------
print("\n[6] Inference")

def test_validate_schema_ok():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.inference import validate_input_schema
    rng = np.random.default_rng(0)
    df = pd.DataFrame({TIME_COL: rng.standard_normal(5),
                       AMOUNT_COL: rng.standard_normal(5)})
    for v in PCA_FEATURES:
        df[v] = rng.standard_normal(5)
    validate_input_schema(df, ALL_FEATURES)

check("validate_input_schema - valid", test_validate_schema_ok)

def test_validate_schema_missing():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.inference import validate_input_schema
    rng = np.random.default_rng(0)
    df = pd.DataFrame({TIME_COL: rng.standard_normal(5),
                       AMOUNT_COL: rng.standard_normal(5)})
    # Missing V-features
    try:
        validate_input_schema(df, ALL_FEATURES)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

check("validate_input_schema - missing cols raises", test_validate_schema_missing)

def test_risk_bands():
    from src.inference import score_to_risk_band
    assert score_to_risk_band(0.1) == "Low Risk"
    assert score_to_risk_band(0.45) == "Medium Risk"
    assert score_to_risk_band(0.7) == "High Risk"
    assert score_to_risk_band(0.95) == "Critical Risk"

check("Risk bands", test_risk_bands)

def test_load_pipeline_missing():
    from pathlib import Path
    from src.inference import load_pipeline
    try:
        load_pipeline(Path("/nonexistent/model.joblib"))
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass

check("load_pipeline raises on missing file", test_load_pipeline_missing)

# ------------------------------------------------------------------
# 7. Full pipeline (toy end-to-end)
# ------------------------------------------------------------------
print("\n[7] End-to-end pipeline (toy data)")

def test_e2e_lr():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.models import make_logistic_regression
    from src.preprocessing import build_lr_pipeline
    from src.inference import score_dataframe

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({TIME_COL: rng.uniform(0, 1e5, n),
                       AMOUNT_COL: rng.exponential(100, n)})
    for v in PCA_FEATURES:
        df[v] = rng.standard_normal(n)
    df = df[ALL_FEATURES]
    y = np.array([1] * 10 + [0] * 190)

    pipe = build_lr_pipeline(make_logistic_regression())
    pipe.fit(df, y)

    result = score_dataframe(pipe, df, threshold=0.5, feature_names=ALL_FEATURES)
    assert len(result) == n
    assert "risk_score" in result.columns
    assert "predicted_class" in result.columns
    assert "risk_band" in result.columns
    assert (result["risk_score"] >= 0).all()
    assert (result["risk_score"] <= 1).all()
    assert set(result["predicted_class"].unique()).issubset({0, 1})

check("End-to-end LR pipeline fit + score_dataframe", test_e2e_lr)

def test_e2e_rf():
    import numpy as np
    import pandas as pd
    from src.config import ALL_FEATURES, PCA_FEATURES, TIME_COL, AMOUNT_COL
    from src.models import make_random_forest
    from src.preprocessing import build_tree_pipeline
    from src.inference import score_dataframe

    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({TIME_COL: rng.uniform(0, 1e5, n),
                       AMOUNT_COL: rng.exponential(100, n)})
    for v in PCA_FEATURES:
        df[v] = rng.standard_normal(n)
    df = df[ALL_FEATURES]
    y = np.array([1] * 8 + [0] * 142)

    pipe = build_tree_pipeline(make_random_forest(n_estimators=10))
    pipe.fit(df, y)
    result = score_dataframe(pipe, df, threshold=0.5, feature_names=ALL_FEATURES)
    assert len(result) == n

check("End-to-end RF pipeline fit + score_dataframe", test_e2e_rf)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} checks")
if FAIL == 0:
    print("All checks passed!")
else:
    print("Some checks failed ? see errors above.")
print()
sys.exit(0 if FAIL == 0 else 1)
