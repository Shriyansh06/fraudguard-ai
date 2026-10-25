"""
Notebook 05 ? SHAP Explainability
====================================
Demonstrates global and local SHAP explanations.
Run AFTER scripts/train.py and scripts/run_shap.py.
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from src.config import (
    CHAMPION_ARTIFACT_PATH, DEMO_CSV_PATH,
    SHAP_RESULTS_DIR, TARGET_COL, ALL_FEATURES,
)
from src.inference import load_pipeline, load_threshold, load_feature_schema
from src.explainability import (
    compute_shap_values, get_explainer,
    get_preprocessed_feature_names, global_feature_importance,
)

# %%
# ?? Cell 1: Load artifacts ?????????????????????????????????????????????????
missing = []
if not CHAMPION_ARTIFACT_PATH.exists():
    missing.append("Champion model (run scripts/train.py)")
explainer_path = SHAP_RESULTS_DIR / "shap_explainer.joblib"
bg_path = SHAP_RESULTS_DIR / "shap_background.joblib"
if not explainer_path.exists():
    missing.append("SHAP explainer (run scripts/run_shap.py)")

if missing:
    print("Missing:\n" + "\n".join(f"  - {m}" for m in missing))
else:
    pipeline = load_pipeline()
    threshold = load_threshold()
    feature_names = load_feature_schema()
    explainer = joblib.load(explainer_path)
    X_background = joblib.load(bg_path)
    print("All artifacts loaded.")

# %%
# ?? Cell 2: Load demo transactions ?????????????????????????????????????????
if not DEMO_CSV_PATH.exists():
    print("Demo CSV not found. Run: python scripts/create_demo_csv.py")
else:
    demo_df = pd.read_csv(DEMO_CSV_PATH)
    X_demo = demo_df[feature_names]
    print(f"Demo: {len(demo_df)} rows")

# %%
# ?? Cell 3: Global SHAP importance ?????????????????????????????????????????
feature_names_pp = get_preprocessed_feature_names(pipeline)
n_sample = min(100, len(X_demo))
X_sample = X_demo.sample(n_sample, random_state=42)

shap_vals = compute_shap_values(explainer, pipeline, X_sample)
importance = global_feature_importance(shap_vals, feature_names_pp)

print("Top 10 features by mean |SHAP|:")
print(importance.head(10).to_string(index=False))

# %%
# ?? Cell 4: Plot global importance ?????????????????????????????????????????
top = importance.head(15)
plt.figure(figsize=(8, 6))
plt.barh(top["feature"][::-1], top["importance"][::-1], color="#1a3a5c")
plt.xlabel("Mean |SHAP value|")
plt.title("Global Feature Importance (SHAP)", fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# ?? Cell 5: Per-transaction waterfall ??????????????????????????????????????
idx = 0  # change to any demo row index
row = X_demo.iloc[[idx]]
sv = compute_shap_values(explainer, pipeline, row)
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[-1]

exp = shap.Explanation(
    values=sv[0],
    base_values=float(expected_value),
    data=sv[0],
    feature_names=feature_names_pp,
)

shap.plots.waterfall(exp, max_display=15, show=True)

score = pipeline.predict_proba(row)[:, 1][0]
pred = int(score >= threshold)
gt = int(demo_df.iloc[idx][TARGET_COL]) if TARGET_COL in demo_df.columns else "N/A"
print(f"\nRow {idx} | Score: {score:.4f} | Predicted: {'Fraud' if pred else 'Legit'} | GT: {gt}")
