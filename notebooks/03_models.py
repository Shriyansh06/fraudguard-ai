"""
Notebook 03 ? Model Training and Comparison
============================================
Demonstrates training and comparing all three models.
Run AFTER scripts/train.py has been executed.

This notebook loads saved metrics ? it does NOT retrain models.
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_NAMES, RESULTS_DIR
from src.evaluation import load_metrics

# %%
# ?? Cell 1: Load saved metrics ?????????????????????????????????????????????
metrics_path = RESULTS_DIR / "metrics.json"
if not metrics_path.exists():
    print("Metrics not found. Run: python scripts/train.py")
else:
    metrics = load_metrics()
    print("Metrics loaded for:", list(metrics.keys()))

# %%
# ?? Cell 2: Comparison table ???????????????????????????????????????????????
LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

rows = []
for name in MODEL_NAMES:
    if name not in metrics:
        continue
    m = metrics[name]["baseline"]["test"]["metrics"]
    rows.append({
        "Model": LABELS.get(name, name),
        "Avg Precision": round(m["average_precision"], 4),
        "ROC-AUC": round(m["roc_auc"], 4),
        "F1": round(m["f1"], 4),
        "Recall": round(m["recall"], 4),
        "Precision": round(m["precision"], 4),
        "Accuracy": round(m["accuracy"], 4),
    })

df_compare = pd.DataFrame(rows)
print(df_compare.to_string(index=False))

# %%
# ?? Cell 3: ROC curves ?????????????????????????????????????????????????????
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"logistic_regression": "#2980b9", "random_forest": "#27ae60", "xgboost": "#e67e22"}
ax.plot([0, 1], [0, 1], "k--", label="Random")

for name in MODEL_NAMES:
    if name not in metrics:
        continue
    roc = metrics[name]["baseline"]["test"]["roc_curve"]
    ax.plot(roc["fpr"], roc["tpr"], color=colors[name], linewidth=2,
            label=f"{LABELS[name]} (AUC={roc['auc']:.4f})")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves ? Test Set")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ?? Cell 4: Champion model summary ?????????????????????????????????????????
if "champion" in metrics:
    champ = metrics["champion"]
    print(f"Champion: {champ['name']}")
    print(f"Decision threshold: {champ['threshold']:.4f}")
    print(f"Baseline Val AP: {champ.get('baseline_val_ap',0):.4f}")
    print(f"Tuned    Val AP: {champ.get('tuned_val_ap',0):.4f}")
    m = champ.get("test", {}).get("metrics", {})
    if m:
        print(f"\nChampion Test Results @ threshold={champ['threshold']:.4f}:")
        for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"]:
            print(f"  {k:<20}: {m.get(k,0):.4f}")
