"""
FraudGuard AI ? Full training pipeline.

Usage:
    python scripts/train.py              # full dataset
    python scripts/train.py --quick      # 5 % stratified sample (non-final)

Pipeline:
    1. Load and validate dataset.
    2. Stratified 70/15/15 split.
    3. Train LR, RF, XGBoost on training set.
    4. Evaluate all three on validation set.
    5. Select champion by Average Precision on validation set.
    6. Tune champion with RandomizedSearchCV (training folds only).
    7. Select decision threshold on validation set.
    8. Final evaluation on untouched test set.
    9. Save pipelines, metrics, threshold, feature schema, metadata.

IMPORTANT:
  - Preprocessing (scaler) is fitted only on training data.
  - SMOTE comparison uses imblearn Pipeline; fitted inside CV folds only.
  - Threshold tuning uses validation data only.
  - Test set is used exactly once at the end for final reporting.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALL_FEATURES,
    ARTIFACT_PATHS,
    CHAMPION_ARTIFACT_PATH,
    CV_FOLDS,
    EVAL_RESULTS_DIR,
    FEATURE_SCHEMA_PATH,
    MODEL_NAMES,
    MODELS_DIR,
    MODEL_META_PATH,
    QUICK_DEV_LABEL,
    QUICK_DEV_SAMPLE_FRAC,
    RANDOM_SEED,
    RESULTS_DIR,
    THRESHOLD_PATH,
    TUNING_N_ITER,
)
from src.data_loader import get_X_y, make_splits, require_dataset, save_feature_schema
from src.evaluation import (
    compute_metrics,
    get_confusion_matrix,
    get_pr_curve_data,
    get_roc_curve_data,
    save_metrics,
    select_threshold_max_f1,
)
from src.models import (
    compute_scale_pos_weight,
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
)
from src.preprocessing import build_lr_pipeline, build_tree_pipeline


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

LR_PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "classifier__solver": ["lbfgs", "saga"],
    "classifier__max_iter": [500, 1000, 2000],
    "classifier__class_weight": ["balanced"],
}

RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced_subsample"],
}

XGB_PARAM_GRID = {
    "classifier__n_estimators": [200, 300, 500],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__max_depth": [4, 6, 8],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}

PARAM_GRIDS = {
    "logistic_regression": LR_PARAM_GRID,
    "random_forest": RF_PARAM_GRID,
    "xgboost": XGB_PARAM_GRID,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline(name: str, y_train: pd.Series):
    """Build the unfitted pipeline for a given model name."""
    if name == "logistic_regression":
        return build_lr_pipeline(make_logistic_regression())
    elif name == "random_forest":
        return build_tree_pipeline(make_random_forest())
    elif name == "xgboost":
        spw = compute_scale_pos_weight(y_train)
        return build_tree_pipeline(make_xgboost(scale_pos_weight=spw))
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_model(name: str, pipeline, X_train, y_train) -> object:
    """Fit the pipeline on training data."""
    print(f"    Training {name}...", end="", flush=True)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f" done in {elapsed:.1f}s")
    return pipeline


def evaluate_on_split(
    name: str,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> dict:
    y_prob = pipeline.predict_proba(X)[:, 1]
    metrics = compute_metrics(np.array(y), y_prob, threshold=threshold)
    cm = get_confusion_matrix(np.array(y), y_prob, threshold=threshold).tolist()
    roc = get_roc_curve_data(np.array(y), y_prob)
    pr = get_pr_curve_data(np.array(y), y_prob)
    return {
        "model": name,
        "split": split_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": roc,
        "pr_curve": pr,
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    mode_label = QUICK_DEV_LABEL if quick else "FULL DATASET"
    print(f"\nFraudGuard AI ? Training Pipeline  [{mode_label}]")
    print("=" * 70)

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading dataset...")
    df = require_dataset()
    df = df.drop_duplicates()
    print(f"  Loaded {len(df):,} rows after de-duplication.")

    if quick:
        fraud_sample = df[df["Class"] == 1].sample(
            frac=QUICK_DEV_SAMPLE_FRAC, random_state=RANDOM_SEED
        )
        legit_sample = df[df["Class"] == 0].sample(
            frac=QUICK_DEV_SAMPLE_FRAC, random_state=RANDOM_SEED
        )
        df = pd.concat([fraud_sample, legit_sample]).sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        print(f"  [QUICK-DEV] Using {len(df):,} rows ({QUICK_DEV_SAMPLE_FRAC*100:.0f}% sample).")

    # ------------------------------------------------------------------
    # 2. Splits
    # ------------------------------------------------------------------
    print("\n[2/9] Creating stratified splits (70/15/15)...")
    train_df, val_df, test_df = make_splits(df)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)

    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"  Train fraud rate : {y_train.mean()*100:.3f}%")
    print(f"  Val   fraud rate : {y_val.mean()*100:.3f}%")
    print(f"  Test  fraud rate : {y_test.mean()*100:.3f}%")

    # Save feature schema
    save_feature_schema(ALL_FEATURES)
    print(f"  Feature schema saved ({len(ALL_FEATURES)} features).")

    # ------------------------------------------------------------------
    # 3. Train baseline models
    # ------------------------------------------------------------------
    print("\n[3/9] Training baseline models...")
    fitted_pipelines = {}
    for name in MODEL_NAMES:
        pipeline = make_pipeline(name, y_train)
        fitted_pipelines[name] = train_model(name, pipeline, X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Validate all models (default threshold 0.5)
    # ------------------------------------------------------------------
    print("\n[4/9] Evaluating on validation set (threshold=0.5)...")
    val_results = {}
    for name, pipeline in fitted_pipelines.items():
        result = evaluate_on_split(name, pipeline, X_val, y_val, 0.5, "validation")
        val_results[name] = result
        m = result["metrics"]
        print(
            f"  {name:<22} | "
            f"AP={m['average_precision']:.4f}  "
            f"ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"Recall={m['recall']:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Select champion by Average Precision on validation set
    # ------------------------------------------------------------------
    print("\n[5/9] Selecting champion model...")
    champion_name = max(
        val_results,
        key=lambda n: val_results[n]["metrics"]["average_precision"],
    )
    champion_pipeline = fitted_pipelines[champion_name]
    print(f"  Champion: {champion_name}  "
          f"(Val AP = {val_results[champion_name]['metrics']['average_precision']:.4f})")

    # ------------------------------------------------------------------
    # 6. Tune champion with RandomizedSearchCV
    # ------------------------------------------------------------------
    print(f"\n[6/9] Tuning {champion_name} with RandomizedSearchCV ({CV_FOLDS}-fold CV)...")
    tuned_pipeline = make_pipeline(champion_name, y_train)

    search = RandomizedSearchCV(
        tuned_pipeline,
        param_distributions=PARAM_GRIDS[champion_name],
        n_iter=TUNING_N_ITER,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_champion = search.best_estimator_
    print(f"  Best CV AP: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    # Compare baseline vs tuned on validation
    baseline_ap = val_results[champion_name]["metrics"]["average_precision"]
    y_val_prob_tuned = tuned_champion.predict_proba(X_val)[:, 1]
    tuned_val_ap = float(average_precision_score(y_val, y_val_prob_tuned))
    print(f"\n  Baseline Val AP : {baseline_ap:.4f}")
    print(f"  Tuned    Val AP : {tuned_val_ap:.4f}")

    # Use tuned model if it improves validation AP
    used_tuned_champion = tuned_val_ap >= baseline_ap
    if used_tuned_champion:
        champion_pipeline = tuned_champion
        print("  Using TUNED champion.")
    else:
        print("  Tuning did not improve validation AP; keeping baseline champion.")

    # ------------------------------------------------------------------
    # 7. Select decision threshold on validation set
    # ------------------------------------------------------------------
    print("\n[7/9] Selecting decision threshold on validation set...")
    y_val_prob = champion_pipeline.predict_proba(X_val)[:, 1]
    best_threshold = select_threshold_max_f1(np.array(y_val), y_val_prob)
    print(f"  Optimal threshold (max-F1 on val): {best_threshold:.4f}")

    # Save threshold
    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": best_threshold,
        "method": "max_f1_on_validation",
        "champion_model": champion_name,
    }, indent=2))

    # ------------------------------------------------------------------
    # 8. Final evaluation on test set (ONCE, for reporting only)
    # ------------------------------------------------------------------
    print("\n[8/9] Final test-set evaluation (untouched test split)...")
    all_results = {}

    # Evaluate ALL models on test set (for comparison table)
    for name, pipeline in fitted_pipelines.items():
        test_result = evaluate_on_split(name, pipeline, X_test, y_test, 0.5, "test")
        val_result_for_name = val_results[name]
        all_results[name] = {
            "baseline": {
                "validation": val_result_for_name,
                "test": test_result,
            }
        }

    # Champion (tuned) test results at optimal threshold
    champion_variant = "tuned" if used_tuned_champion else "baseline"
    champ_test = evaluate_on_split(
        f"{champion_name}_{champion_variant}",
        champion_pipeline,
        X_test,
        y_test,
        best_threshold,
        "test",
    )
    champ_val = evaluate_on_split(
        f"{champion_name}_{champion_variant}",
        champion_pipeline,
        X_val,
        y_val,
        best_threshold,
        "validation",
    )
    all_results["champion"] = {
        "name": champion_name,
        "validation": champ_val,
        "test": champ_test,
        "threshold": best_threshold,
        "used_tuned_model": used_tuned_champion,
        "tuning_best_params": search.best_params_,
        "baseline_val_ap": baseline_ap,
        "tuned_val_ap": tuned_val_ap,
    }

    m = champ_test["metrics"]
    print(f"  Champion ({champion_name}) Test Results @ threshold={best_threshold:.4f}:")
    print(f"    Accuracy   : {m['accuracy']:.4f}")
    print(f"    Precision  : {m['precision']:.4f}")
    print(f"    Recall     : {m['recall']:.4f}")
    print(f"    F1         : {m['f1']:.4f}")
    print(f"    ROC-AUC    : {m['roc_auc']:.4f}")
    print(f"    Avg Prec.  : {m['average_precision']:.4f}")
    print(f"    N Test     : {m['n_samples']:,}  "
          f"(fraud={m['n_positives']}, legit={m['n_negatives']})")

    # ------------------------------------------------------------------
    # 9. Save all artifacts
    # ------------------------------------------------------------------
    print("\n[9/9] Saving artifacts...")

    # Save individual baseline models
    for name, pipeline in fitted_pipelines.items():
        path = ARTIFACT_PATHS[name]
        joblib.dump(pipeline, path)
        print(f"  Saved: {path}")

    # Save champion (tuned)
    joblib.dump(champion_pipeline, CHAMPION_ARTIFACT_PATH)
    print(f"  Saved champion: {CHAMPION_ARTIFACT_PATH}")

    # Save metrics
    save_metrics(all_results)
    print(f"  Saved metrics: {RESULTS_DIR / 'metrics.json'}")

    # Save model metadata
    metadata = {
        "champion_model": champion_name,
        "threshold": best_threshold,
        "used_tuned_model": used_tuned_champion,
        "training_mode": "quick_dev" if quick else "full",
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "features": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "version": "1.0.0",
        "framework_versions": {
            "sklearn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
        "quick_dev": quick,
        "quick_dev_note": QUICK_DEV_LABEL if quick else None,
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved metadata: {MODEL_META_PATH}")

    print(f"\n{'='*70}")
    print("Training complete!")
    if quick:
        print(f"NOTE: {QUICK_DEV_LABEL}")
        print("Run without --quick flag for final, publishable results.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGuard AI Training Script")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Use {QUICK_DEV_SAMPLE_FRAC*100:.0f}%% stratified sample for "
            "rapid development. Results are NON-FINAL."
        ),
    )
    args = parser.parse_args()
    main(quick=args.quick)
