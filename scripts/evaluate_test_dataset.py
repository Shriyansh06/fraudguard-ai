"""Train the comparison classifiers without test-row overlap and evaluate a CSV."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_comparison as comparison
from src.config import ALL_FEATURES, DATASET_PATH, RANDOM_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all comparison models on a labeled CSV test set."
    )
    parser.add_argument("test_csv", type=Path, help="Path to the labeled test CSV.")
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=DATASET_PATH,
        help=f"Training dataset CSV (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "test_evaluation",
        help="Directory for tables, plots, and predictions.",
    )
    return parser.parse_args()


def load_labeled_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} CSV not found: {path}")

    frame = pd.read_csv(path)
    required = set(ALL_FEATURES) | {"Class"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} CSV is missing required columns: {missing}")

    frame = frame.loc[:, ALL_FEATURES + ["Class"]].copy()
    for column in ALL_FEATURES + ["Class"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any():
        raise ValueError(f"{name} CSV contains missing feature or label values.")
    if not set(frame["Class"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} Class column must contain only 0 and 1.")
    if frame.empty:
        raise ValueError(f"{name} CSV contains no rows.")
    return frame


def make_plots(
    output_dir: Path,
    metrics: dict[str, dict[str, object]],
    labels: pd.Series,
    scores_by_model: dict[str, np.ndarray],
) -> None:
    sns.set_theme(style="whitegrid")
    model_names = list(metrics)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, metric in zip(axes.flat, comparison.METRIC_NAMES):
        values = [float(metrics[name][metric]) for name in model_names]
        sns.barplot(
            x=model_names,
            y=values,
            hue=model_names,
            legend=False,
            palette="deep",
            ax=ax,
        )
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
    fig.suptitle("Model Metrics ? Supplied Test Set", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "test_metrics_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, scores in scores_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        title="ROC Curves ? Supplied Test Set",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "test_roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP={metrics[name]['average_precision']:.3f})",
        )
    ax.set(
        title="Precision?Recall Curves ? Supplied Test Set",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "test_precision_recall_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, name in zip(axes.flat, model_names):
        matrix = metrics[name]["confusion_matrix"]
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
    fig.suptitle("Test-Set Confusion Matrices")
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrices.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_labeled_csv(args.test_csv, "Test")
    training_data = load_labeled_csv(args.training_csv, "Training")

    duplicate_test_rows = int(test_data.duplicated().sum())
    duplicate_training_rows = int(training_data.duplicated().sum())
    training_data = training_data.drop_duplicates().reset_index(drop=True)

    test_keys = pd.util.hash_pandas_object(
        test_data[ALL_FEATURES], index=False
    )
    training_keys = pd.util.hash_pandas_object(
        training_data[ALL_FEATURES], index=False
    )
    overlap = training_keys.isin(set(test_keys.tolist()))
    overlap_rows_removed = int(overlap.sum())
    training_data = training_data.loc[~overlap].reset_index(drop=True)
    if training_data.empty or training_data["Class"].nunique() != 2:
        raise ValueError(
            "Training data must retain examples from both classes after "
            "removing rows that match the test set."
        )
    if test_data["Class"].nunique() != 2:
        raise ValueError("Test data must contain both classes to calculate ROC metrics.")

    comparison.OUTPUT_DIR = output_dir
    eda_summary = comparison.save_eda(test_data, duplicate_test_rows)
    eda_summary["dataset_path"] = str(args.test_csv.resolve())
    eda_summary["rows_in_test_csv"] = eda_summary.pop("rows_after_deduplication")
    eda_summary["duplicate_rows_present"] = duplicate_test_rows
    eda_summary.pop("duplicate_rows_removed", None)
    (output_dir / "eda_summary.json").write_text(
        json.dumps(eda_summary, indent=2)
    )

    X_train = training_data[ALL_FEATURES]
    y_train = training_data["Class"].astype(int)
    X_test = test_data[ALL_FEATURES]
    y_test = test_data["Class"].astype(int)
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = comparison.build_models(positive_weight)

    results: dict[str, dict[str, object]] = {}
    scores_by_model: dict[str, np.ndarray] = {}
    scored = test_data.copy()
    report_rows: list[dict[str, object]] = []
    class_reports: dict[str, dict[str, object]] = {}

    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        metrics, scores = comparison.evaluate_model(model, X_test, y_test)
        metrics["average_precision"] = float(average_precision_score(y_test, scores))
        results[name] = metrics
        scores_by_model[name] = scores
        class_reports[name] = classification_report(
            y_test,
            model.predict(X_test),
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            output_dict=True,
            zero_division=0,
        )

        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        scored[f"{slug}_risk_score"] = np.round(scores, 6)
        scored[f"{slug}_predicted_class"] = np.asarray(model.predict(X_test), dtype=int)
        report_rows.append(
            {
                "model": name,
                "fit_time_seconds": round(fit_seconds, 3),
                **{
                    metric: metrics[metric]
                    for metric in comparison.METRIC_NAMES
                },
                "average_precision": metrics["average_precision"],
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TP": metrics["tp"],
            }
        )
        print(
            f"  Fit {fit_seconds:.1f}s | precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"ROC-AUC={metrics['roc_auc']:.4f} "
            f"AP={metrics['average_precision']:.4f}",
            flush=True,
        )

    pd.DataFrame(report_rows).to_csv(output_dir / "metrics.csv", index=False)
    scored.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": name,
                "TN": values["tn"],
                "FP": values["fp"],
                "FN": values["fn"],
                "TP": values["tp"],
            }
            for name, values in results.items()
        ]
    ).to_csv(output_dir / "confusion_matrices.csv", index=False)
    (output_dir / "classification_reports.json").write_text(
        json.dumps(class_reports, indent=2)
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_csv": str(args.test_csv.resolve()),
                "training_csv": str(args.training_csv.resolve()),
                "test_rows": len(test_data),
                "training_rows_after_exclusions": len(training_data),
                "test_duplicate_rows": duplicate_test_rows,
                "training_duplicate_rows_removed": duplicate_training_rows,
                "overlapping_training_rows_removed": overlap_rows_removed,
                "random_state": RANDOM_SEED,
                "metric_note": (
                    "Risk scores are model outputs and are not necessarily "
                    "calibrated probabilities. MAE, MSE, RMSE, and R2 compare "
                    "those scores against the binary class label."
                ),
                "results": results,
            },
            indent=2,
        )
    )
    make_plots(output_dir, results, y_test, scores_by_model)

    comparison_table = pd.DataFrame(report_rows).set_index("model")
    display_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "mae",
        "mse",
        "rmse",
        "r2",
    ]
    print("\nTest-set comparison:")
    print(
        comparison_table[display_columns].to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print(
        f"\nTest rows: {len(test_data):,}; "
        f"training rows: {len(training_data):,}; "
        f"overlapping training rows excluded: {overlap_rows_removed:,}"
    )
    print(f"All outputs: {output_dir}")


if __name__ == "__main__":
    main()
