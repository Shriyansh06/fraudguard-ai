"""
FraudGuard AI ? Exploratory Data Analysis script.

Usage:
    python scripts/run_eda.py

Outputs:
    results/eda/eda_summary.json   ? key statistics
    results/eda/*.png              ? saved charts (used by Streamlit dashboard)

Observations are printed to stdout and also stored in eda_summary.json.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# Allow importing src/ modules from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AMOUNT_COL,
    ALL_FEATURES,
    EDA_RESULTS_DIR,
    PCA_FEATURES,
    TARGET_COL,
    TIME_COL,
)
from src.data_loader import require_dataset

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
})
NAVY = "#1a3a5c"
RED = "#c0392b"
BLUE = "#2980b9"


def setup_output_dir() -> Path:
    EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_RESULTS_DIR


# ---------------------------------------------------------------------------
# 1. Basic info
# ---------------------------------------------------------------------------

def run_basic_info(df: pd.DataFrame) -> dict:
    """Print and return basic dataset statistics."""
    n_rows, n_cols = df.shape
    n_dup = df.duplicated().sum()
    n_missing = df.isnull().sum().sum()
    fraud_count = int(df[TARGET_COL].sum())
    legit_count = int((df[TARGET_COL] == 0).sum())
    fraud_pct = fraud_count / n_rows * 100

    print(f"\n{'='*60}")
    print(" DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"  Rows              : {n_rows:,}")
    print(f"  Columns           : {n_cols}")
    print(f"  Duplicates        : {n_dup}")
    print(f"  Missing values    : {n_missing}")
    print(f"  Fraud (Class=1)   : {fraud_count:,}  ({fraud_pct:.3f}%)")
    print(f"  Legit (Class=0)   : {legit_count:,}")
    print(f"  Imbalance ratio   : 1 : {legit_count // fraud_count}")

    print("\n-- First 5 rows --")
    print(df.head().to_string())
    print("\n-- dtypes & non-null counts --")
    df.info()
    print("\n-- Descriptive statistics --")
    print(df.describe().round(4).to_string())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": int(n_dup),
        "n_missing": int(n_missing),
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": round(fraud_pct, 4),
        "imbalance_ratio": round(legit_count / fraud_count, 1),
        "features": ALL_FEATURES,
        "target": TARGET_COL,
        "observations": {
            "class_imbalance": (
                f"The dataset is highly imbalanced: only {fraud_pct:.3f}% of "
                f"transactions are fraudulent ({fraud_count:,} out of {n_rows:,}). "
                "Standard accuracy is a misleading metric; Area under the "
                "Precision-Recall curve (AUPRC) is more informative."
            ),
            "missing_data": (
                "No missing values detected. The dataset is complete, which "
                "simplifies preprocessing."
            ),
            "duplicates": (
                f"Found {n_dup} duplicate rows. Duplicates are removed before "
                "modelling to avoid data leakage between splits."
                if n_dup > 0
                else "No duplicate rows detected."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 2. Class distribution chart
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]
    colors = [BLUE, RED]

    axes[0].bar(labels, counts.values, color=colors, width=0.4, edgecolor="white")
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

    wedge_colors = [BLUE, RED]
    axes[1].pie(
        counts.values,
        labels=[f"{l}\n{v:,}" for l, v in zip(labels, counts.values)],
        colors=wedge_colors,
        autopct="%1.2f%%",
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Distribution (%)")

    plt.suptitle("Class Imbalance ? ULB Credit Card Fraud Dataset", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "class_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 3. Amount and Time distributions
# ---------------------------------------------------------------------------

def plot_amount_time_distributions(df: pd.DataFrame, out_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Amount distribution (log scale) ? all
    axes[0, 0].hist(
        np.log1p(df[AMOUNT_COL]), bins=100, color=BLUE, alpha=0.8, edgecolor="none"
    )
    axes[0, 0].set_title("Transaction Amount (log scale, all classes)")
    axes[0, 0].set_xlabel("log(Amount + 1)")
    axes[0, 0].set_ylabel("Count")

    # Amount by class
    for cls, color, label in [(0, BLUE, "Legit"), (1, RED, "Fraud")]:
        axes[0, 1].hist(
            np.log1p(df.loc[df[TARGET_COL] == cls, AMOUNT_COL]),
            bins=80, alpha=0.6, color=color, label=label, edgecolor="none",
        )
    axes[0, 1].set_title("Amount by Class (log scale)")
    axes[0, 1].set_xlabel("log(Amount + 1)")
    axes[0, 1].legend()

    # Time (seconds from first transaction)
    axes[1, 0].hist(df[TIME_COL], bins=100, color=BLUE, alpha=0.8, edgecolor="none")
    axes[1, 0].set_title("Time Distribution (seconds)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Count")

    # Time by class
    for cls, color, label in [(0, BLUE, "Legit"), (1, RED, "Fraud")]:
        axes[1, 1].hist(
            df.loc[df[TARGET_COL] == cls, TIME_COL],
            bins=80, alpha=0.6, color=color, label=label, edgecolor="none",
        )
    axes[1, 1].set_title("Time by Class")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()

    plt.suptitle("Amount and Time Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "amount_time_distributions.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> str:
    # Use a sample for speed if dataset is large
    sample = df.sample(min(10_000, len(df)), random_state=42)
    corr = sample[ALL_FEATURES + [TARGET_COL]].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Heatmap (lower triangle)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 5. Feature vs target boxplots (top PCA features)
# ---------------------------------------------------------------------------

def plot_feature_vs_target_boxplots(df: pd.DataFrame, out_dir: Path) -> str:
    """Boxplots of top 12 PCA features by mean absolute difference between classes."""
    fraud = df[df[TARGET_COL] == 1]
    legit = df[df[TARGET_COL] == 0]
    diffs = {
        col: abs(fraud[col].mean() - legit[col].mean()) for col in PCA_FEATURES
    }
    top_features = sorted(diffs, key=diffs.get, reverse=True)[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        data_legit = df.loc[df[TARGET_COL] == 0, feat].sample(
            min(2000, legit.shape[0]), random_state=42
        )
        data_fraud = df.loc[df[TARGET_COL] == 1, feat]
        axes[i].boxplot(
            [data_legit, data_fraud],
            tick_labels=["Legit", "Fraud"],
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            boxprops={"facecolor": BLUE},
            flierprops={"markerfacecolor": RED, "markersize": 2, "alpha": 0.3},
            widths=0.5,
        )
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_ylabel("Value")

    plt.suptitle("Top 12 PCA Features by Class Separation (Boxplots)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "feature_vs_target_boxplots.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 6. Amount statistics by class
# ---------------------------------------------------------------------------

def compute_amount_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for cls, name in [(0, "legitimate"), (1, "fraud")]:
        subset = df.loc[df[TARGET_COL] == cls, AMOUNT_COL]
        stats[name] = {
            "mean": round(float(subset.mean()), 2),
            "median": round(float(subset.median()), 2),
            "std": round(float(subset.std()), 2),
            "max": round(float(subset.max()), 2),
        }
    print(f"\n  Amount stats ? Legit  : {stats['legitimate']}")
    print(f"  Amount stats ? Fraud  : {stats['fraud']}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nFraudGuard AI ? Exploratory Data Analysis")
    out_dir = setup_output_dir()

    print("\n[1/6] Loading dataset...")
    df = require_dataset()

    print("\n[2/6] Basic info...")
    summary = run_basic_info(df)

    print("\n[3/6] Plotting class distribution...")
    plot_class_distribution(df, out_dir)

    print("\n[4/6] Plotting amount/time distributions...")
    plot_amount_time_distributions(df, out_dir)

    print("\n[5/6] Plotting correlation heatmap...")
    plot_correlation_heatmap(df, out_dir)

    print("\n[6/6] Plotting feature-vs-target boxplots...")
    plot_feature_vs_target_boxplots(df, out_dir)

    # Amount stats
    amount_stats = compute_amount_stats(df)
    summary["amount_stats"] = amount_stats

    # Save summary JSON
    summary_path = out_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  EDA summary saved to: {summary_path}")
    print("\nEDA complete.\n")


if __name__ == "__main__":
    main()
