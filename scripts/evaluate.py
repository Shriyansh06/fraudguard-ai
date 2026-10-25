"""
FraudGuard AI ? Post-training evaluation script.

Usage:
    python scripts/evaluate.py

Loads saved models and computes evaluation charts saved to results/evaluation/.
This script does NOT retrain any model ? it only loads and evaluates.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ARTIFACT_PATHS,
    EVAL_RESULTS_DIR,
    MODEL_NAMES,
    RESULTS_DIR,
)
from src.evaluation import load_metrics

NAVY = "#1a3a5c"
RED = "#c0392b"
BLUE = "#2980b9"
GREEN = "#27ae60"
ORANGE = "#e67e22"

COLORS = {
    "logistic_regression": BLUE,
    "random_forest": GREEN,
    "xgboost": ORANGE,
}
LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def plot_roc_curves(metrics: dict, out_dir: Path) -> None:
    """Plot all three ROC curves on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

    for name in MODEL_NAMES:
        if name not in metrics:
            continue
        roc = metrics[name]["baseline"]["test"]["roc_curve"]
        auc = roc["auc"]
        ax.plot(
            roc["fpr"], roc["tpr"],
            color=COLORS.get(name, NAVY),
            linewidth=2,
            label=f"{LABELS.get(name, name)} (AUC={auc:.4f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves ? Test Set")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = out_dir / "roc_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pr_curves(metrics: dict, out_dir: Path) -> None:
    """Plot all three Precision-Recall curves on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name in MODEL_NAMES:
        if name not in metrics:
            continue
        pr = metrics[name]["baseline"]["test"]["pr_curve"]
        ap = pr["average_precision"]
        ax.plot(
            pr["recall"], pr["precision"],
            color=COLORS.get(name, NAVY),
            linewidth=2,
            label=f"{LABELS.get(name, name)} (AP={ap:.4f})",
        )

    # Baseline: random classifier at class prevalence
    if MODEL_NAMES[0] in metrics:
        prev = metrics[MODEL_NAMES[0]]["baseline"]["test"]["metrics"]["prevalence"]
        ax.axhline(prev, color="gray", linestyle="--", linewidth=1,
                   label=f"No-skill (prevalence={prev:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves ? Test Set")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = out_dir / "pr_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(metrics: dict, out_dir: Path) -> None:
    """Plot confusion matrices for all models."""
    n = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, MODEL_NAMES):
        if name not in metrics:
            ax.set_visible(False)
            continue
        cm = np.array(metrics[name]["baseline"]["test"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred Legit", "Pred Fraud"],
            yticklabels=["True Legit", "True Fraud"],
            cbar=False,
        )
        ap = metrics[name]["baseline"]["test"]["metrics"]["average_precision"]
        ax.set_title(f"{LABELS.get(name, name)}\n(AP={ap:.4f})", fontsize=11)

    plt.suptitle("Confusion Matrices ? Test Set (threshold=0.5)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "confusion_matrices.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_comparison(metrics: dict, out_dir: Path) -> None:
    """Bar chart comparing key metrics across all models."""
    metric_keys = ["average_precision", "roc_auc", "f1", "recall", "precision"]
    metric_labels = ["Avg Precision", "ROC-AUC", "F1", "Recall", "Precision"]

    model_vals = {}
    for name in MODEL_NAMES:
        if name not in metrics:
            continue
        m = metrics[name]["baseline"]["test"]["metrics"]
        model_vals[name] = [m.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (name, vals) in enumerate(model_vals.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS.get(name, name),
                      color=COLORS.get(name, NAVY), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison ? Test Set Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "metrics_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_baseline_vs_threshold(metrics: dict, out_dir: Path) -> None:
    """Compare baseline and validation-threshold metrics on validation data."""
    if "champion" not in metrics:
        print("  No champion data found; skipping baseline vs tuned plot.")
        return

    champ = metrics["champion"]
    name = champ.get("name", "champion")

    baseline_val = metrics.get(name, {}).get("baseline", {}).get("validation", {}).get("metrics", {})
    tuned_val = champ.get("validation", {}).get("metrics", {})

    if not baseline_val or not tuned_val:
        print("  Insufficient data for baseline vs tuned comparison plot.")
        return

    metric_keys = ["average_precision", "roc_auc", "f1", "recall", "precision"]
    metric_labels = ["Avg Precision", "ROC-AUC", "F1", "Recall", "Precision"]

    baseline_vals = [baseline_val.get(k, 0) for k in metric_keys]
    champion_vals = [tuned_val.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline (0.5 threshold)",
        color=BLUE,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        champion_vals,
        width,
        label=f"Champion threshold ({champ['threshold']:.4f})",
        color=NAVY,
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Baseline vs Thresholded {LABELS.get(name, name)} ? Validation Set",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "baseline_vs_threshold.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main() -> None:
    print("\nFraudGuard AI ? Post-Training Evaluation")
    print("=" * 50)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading saved metrics...")
    metrics = load_metrics()
    print("  Metrics loaded successfully.")

    print("\n[2/5] Plotting ROC curves...")
    plot_roc_curves(metrics, EVAL_RESULTS_DIR)

    print("\n[3/5] Plotting Precision-Recall curves...")
    plot_pr_curves(metrics, EVAL_RESULTS_DIR)

    print("\n[4/5] Plotting confusion matrices...")
    plot_confusion_matrices(metrics, EVAL_RESULTS_DIR)

    print("\n[5/5] Plotting metrics comparison and threshold comparison...")
    plot_metrics_comparison(metrics, EVAL_RESULTS_DIR)
    plot_baseline_vs_threshold(metrics, EVAL_RESULTS_DIR)

    print("\nEvaluation complete. Charts saved to:", EVAL_RESULTS_DIR)


if __name__ == "__main__":
    main()
