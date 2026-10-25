"""
FraudGuard AI ? Notebook 04: Hyperparameter Tuning
====================================================
Demonstrates RandomizedSearchCV tuning.
Run AFTER scripts/train.py.
This notebook loads saved metrics ? it does NOT retrain models.
"""

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.config import RESULTS_DIR
from src.evaluation import load_metrics
import matplotlib.pyplot as plt
import numpy as np

# %%
# ?? Cell 1: Load saved metrics ?????????????????????????????????????????????
metrics_path = RESULTS_DIR / "metrics.json"
if not metrics_path.exists():
    print("No metrics found. Run: python scripts/train.py")
else:
    metrics = load_metrics()
    if "champion" in metrics:
        champ = metrics["champion"]
        print(f"Champion model  : {champ['name']}")
        print(f"Baseline Val AP : {champ.get('baseline_val_ap', 'N/A')}")
        print(f"Tuned Val AP    : {champ.get('tuned_val_ap', 'N/A')}")
        print(f"Decision threshold: {champ.get('threshold', 0.5):.4f}")
        print(f"\nBest params:")
        for k, v in champ.get("tuning_best_params", {}).items():
            print(f"  {k}: {v}")
    else:
        print("No champion data found in metrics.")

# %%
# ?? Cell 2: Validation metrics ? baseline vs tuned ?????????????????????????
if "champion" in metrics:
    champ = metrics["champion"]
    keys = ["average_precision", "roc_auc", "f1", "recall", "precision"]
    labels = ["Avg Precision", "ROC-AUC", "F1", "Recall", "Precision"]

    base_vals = [champ.get("validation", {}).get("metrics", {}).get(k, 0) for k in keys]
    name = champ.get("name", "champion")
    # Baseline from the un-tuned model
    baseline_metrics = metrics.get(name, {}).get("baseline", {}).get("validation", {}).get("metrics", {})
    base_untuned = [baseline_metrics.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, base_untuned, 0.35, label="Baseline", color="#2980b9", alpha=0.8)
    ax.bar(x + 0.18, base_vals, 0.35, label="Tuned", color="#1a3a5c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Baseline vs Tuned {name} ? Validation Set", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()
