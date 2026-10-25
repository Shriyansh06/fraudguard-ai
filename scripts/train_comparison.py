"""Train and compare fraud classifiers with a stratified 80/20 split."""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH, RANDOM_SEED, ROOT_DIR

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = ROOT_DIR / "results" / "model_comparison_80_20"
MODEL_DIR = ROOT_DIR / "models" / "comparison_80_20"
TARGET = "Class"
METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mae",
    "mse",
    "rmse",
    "r2",
]


def build_models(positive_weight: float) -> dict[str, object]:
    """Create the requested classification estimators."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=10000,
                random_state=RANDOM_SEED,
            ),
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            ),
        ),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0,
            device="cpu",
        ),
    }


def get_risk_scores(model: object, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities or sigmoid-transformed SVM margins."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]

    margins = np.asarray(model.decision_function(features))
    margins = np.clip(margins, -50, 50)
    return 1.0 / (1.0 + np.exp(-margins))


def evaluate_model(
    model: object, features: pd.DataFrame, labels: pd.Series
) -> tuple[dict[str, float | int | list[list[int]]], np.ndarray]:
    """Calculate classification and risk-score regression metrics."""
    predictions = np.asarray(model.predict(features))
    scores = get_risk_scores(model, features)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    return (
        {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "mae": float(mean_absolute_error(labels, scores)),
            "mse": float(mean_squared_error(labels, scores)),
            "rmse": float(np.sqrt(mean_squared_error(labels, scores))),
            "r2": float(r2_score(labels, scores)),
            "confusion_matrix": matrix.astype(int).tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        scores,
    )


def save_eda(data: pd.DataFrame, duplicate_count: int) -> dict[str, object]:
    """Write dataset summary tables and exploratory plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fraud = data.loc[data[TARGET] == 1]
    legitimate = data.loc[data[TARGET] == 0]
    correlations = (
        data.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    summary = {
        "dataset_path": str(DATASET_PATH),
        "rows_after_deduplication": int(len(data)),
        "columns": int(data.shape[1]),
        "features": [column for column in data.columns if column != TARGET],
        "missing_values_by_column": {
            column: int(count)
            for column, count in data.isna().sum().items()
        },
        "duplicate_rows_removed": duplicate_count,
        "class_distribution": {
            "legitimate": {
                "count": int(counts.loc[0]),
                "percent": float(counts.loc[0] / len(data) * 100),
            },
            "fraud": {
                "count": int(counts.loc[1]),
                "percent": float(counts.loc[1] / len(data) * 100),
            },
        },
        "amount_describe_by_class": {
            "all": data["Amount"].describe().to_dict(),
            "legitimate": legitimate["Amount"].describe().to_dict(),
            "fraud": fraud["Amount"].describe().to_dict(),
        },
        "time_seconds": {
            "min": float(data["Time"].min()),
            "max": float(data["Time"].max()),
            "mean": float(data["Time"].mean()),
        },
        "top_10_fraud_correlations": {
            feature: float(value) for feature, value in correlations.head(10).items()
        },
    }
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    correlations.rename("correlation_with_class").to_csv(
        OUTPUT_DIR / "feature_correlations.csv", header=True
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[int(counts.loc[0]), int(counts.loc[1])],
        hue=["Legitimate", "Fraud"],
        palette=["#2563eb", "#dc2626"],
        legend=False,
        ax=ax,
    )
    ax.set(title="Transaction Class Distribution", xlabel="Class", ylabel="Transactions")
    for index, count in enumerate(counts.tolist()):
        ax.text(index, count, f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=data,
        x="Amount",
        hue=TARGET,
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        log_scale=(False, True),
        ax=ax,
    )
    ax.set(title="Transaction Amount Distribution by Class", ylabel="Density (log scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_amount_distribution.png", dpi=160)
    plt.close(fig)

    top_correlations = correlations.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=top_correlations.values,
        y=top_correlations.index,
        hue=top_correlations.index,
        palette="vlag",
        legend=False,
        ax=ax,
    )
    ax.set(title="Top Features Correlated with Fraud", xlabel="Pearson correlation", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_feature_correlations.png", dpi=160)
    plt.close(fig)
    return summary


def save_metric_plots(
    results: dict[str, dict[str, dict[str, object]]],
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> None:
    """Write metric comparison, ROC, and confusion matrix plots."""
    names = list(results)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        values = [float(results[name]["test"][metric]) for name in names]
        sns.barplot(x=names, y=values, hue=names, legend=False, ax=ax, palette="deep")
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
    fig.suptitle("80/20 Test-Set Model Metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_metrics_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (false_positive_rate, true_positive_rate, auc) in roc_curves.items():
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        title="ROC Curves ? 20% Held-Out Test Set",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_roc_curves.png", dpi=160)
    plt.close(fig)

    for split in ("train", "test"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, name in zip(axes.flat, names):
            matrix = results[name][split]["confusion_matrix"]
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                ax=ax,
            )
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        fig.suptitle(f"{split.title()} Confusion Matrices")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{split}_confusion_matrices.png", dpi=160)
        plt.close(fig)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading dataset and performing EDA...")
    raw_data = pd.read_csv(DATASET_PATH)
    required_columns = {"Class", "Amount", "Time"}
    if not required_columns.issubset(raw_data.columns):
        raise ValueError(
            f"Dataset is missing required columns: {sorted(required_columns - set(raw_data.columns))}"
        )
    if not set(raw_data[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class column must contain only binary labels 0 and 1.")

    duplicate_count = int(raw_data.duplicated().sum())
    data = raw_data.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    summary = save_eda(data, duplicate_count)
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    feature_columns = [column for column in data.columns if column != TARGET]
    X = data[feature_columns]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = build_models(positive_weight)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rows after de-duplication: {len(data):,}")
    print(f"Train: {len(X_train):,} ({len(X_train) / len(data):.0%})")
    print(f"Test : {len(X_test):,} ({len(X_test) / len(data):.0%})")
    print(f"Fraud in train/test: {int(y_train.sum()):,} / {int(y_test.sum()):,}")

    results: dict[str, dict[str, dict[str, object]]] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    rows: list[dict[str, object]] = []
    confusion_rows: dict[str, list[dict[str, object]]] = {"train": [], "test": []}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        results[name] = {}

        for split_name, features, labels in (
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ):
            metrics, scores = evaluate_model(model, features, labels)
            results[name][split_name] = metrics
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "fit_time_seconds": round(fit_seconds, 3),
                    **{metric: metrics[metric] for metric in METRIC_NAMES},
                }
            )
            confusion_rows[split_name].append(
                {
                    "model": name,
                    "TN": metrics["tn"],
                    "FP": metrics["fp"],
                    "FN": metrics["fn"],
                    "TP": metrics["tp"],
                }
            )
            if split_name == "test":
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(labels, scores)
                roc_curves[name] = (fpr, tpr, float(metrics["roc_auc"]))

        print(
            f"  Fit {fit_seconds:.1f}s | "
            f"test accuracy={results[name]['test']['accuracy']:.4f}, "
            f"precision={results[name]['test']['precision']:.4f}, "
            f"recall={results[name]['test']['recall']:.4f}, "
            f"F1={results[name]['test']['f1']:.4f}, "
            f"ROC-AUC={results[name]['test']['roc_auc']:.4f}"
        )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    for split_name, matrix_rows in confusion_rows.items():
        pd.DataFrame(matrix_rows).to_csv(
            OUTPUT_DIR / f"{split_name}_confusion_matrices.csv", index=False
        )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "split": {
                    "train_fraction": 0.8,
                    "test_fraction": 0.2,
                    "random_state": RANDOM_SEED,
                    "stratified": True,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                },
                "regression_metric_note": (
                    "MAE, MSE, RMSE, and R2 compare each classifier's fraud-risk "
                    "score with the binary Class label; they are not from a "
                    "separate linear regression model. SVM margins are sigmoid-transformed."
                ),
                "results": results,
            },
            indent=2,
        )
    )
    save_metric_plots(results, roc_curves)

    comparison = pd.DataFrame(rows)
    test_comparison = comparison.loc[comparison["split"] == "test"].set_index("model")
    print("\nTest-set comparison")
    print(
        test_comparison[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "mae", "mse", "rmse", "r2"]
        ].to_string(float_format=lambda value: f"{value:.4f}")
    )
    print("\nTest confusion matrices (rows: actual 0/1; columns: predicted 0/1)")
    for row in confusion_rows["test"]:
        print(
            f"{row['model']:<22} "
            f"[[TN={row['TN']:,}, FP={row['FP']:,}], "
            f"[FN={row['FN']:,}, TP={row['TP']:,}]]"
        )
    print(f"\nEDA, metric tables, and graphs: {OUTPUT_DIR}")
    print(f"Trained model artifacts: {MODEL_DIR}")


if __name__ == "__main__":
    main()
