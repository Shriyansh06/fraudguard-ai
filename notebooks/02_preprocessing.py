"""
Notebook 02 ? Preprocessing Pipeline Demo
==========================================
Demonstrates the preprocessing pipeline: what it does, why, and
how it avoids data leakage.

Run AFTER notebook 01_eda.py.
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ALL_FEATURES, TARGET_COL, TIME_COL, AMOUNT_COL, PCA_FEATURES
from src.data_loader import load_dataset, make_splits, get_X_y, check_dataset_present
from src.preprocessing import build_preprocessor, build_lr_pipeline, build_tree_pipeline
from src.models import make_logistic_regression, make_random_forest

# %%
# ?? Cell 1: Load and split ?????????????????????????????????????????????????
if not check_dataset_present():
    print("Dataset not found. See notebooks/01_eda.py for instructions.")
else:
    df = load_dataset().drop_duplicates()
    train_df, val_df, test_df = make_splits(df)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# %%
# ?? Cell 2: Preprocessor output ????????????????????????????????????????????
pre = build_preprocessor()
X_train_transformed = pre.fit_transform(X_train)
print(f"Input shape : {X_train.shape}")
print(f"Output shape: {X_train_transformed.shape}")
print(f"\nTime (before): mean={X_train[TIME_COL].mean():.1f}  std={X_train[TIME_COL].std():.1f}")
print(f"Time (after) : mean={X_train_transformed[:,0].mean():.4f}  std={X_train_transformed[:,0].std():.4f}")

# %%
# ?? Cell 3: Leakage demonstration ??????????????????????????????????????????
print("Scaler fitted on TRAIN only:")
print(f"  Scaler mean (Time): {pre.named_transformers_['scale'].mean_[0]:.2f}")
print(f"  Train Time mean   : {X_train[TIME_COL].mean():.2f}")

# Transform val WITHOUT re-fitting (this is the correct approach)
X_val_transformed = pre.transform(X_val)
print(f"\nVal Time raw mean   : {X_val[TIME_COL].mean():.2f}")
print(f"Val Time scaled mean: {X_val_transformed[:,0].mean():.4f}  (not zero ? scaler from train)")

# %%
# ?? Cell 4: Full pipeline structure ????????????????????????????????????????
lr_pipe = build_lr_pipeline(make_logistic_regression())
print("LR Pipeline steps:")
for name, step in lr_pipe.steps:
    print(f"  {name}: {type(step).__name__}")

# %%
# ?? Cell 5: SMOTE note ?????????????????????????????????????????????????????
print("""
SMOTE Note:
-----------
SMOTE (Synthetic Minority Over-sampling Technique) is NOT applied in this
notebook because applying it outside of cross-validation folds would cause
data leakage ? synthetic samples from the training set would inform the
validation evaluation.

SMOTE can be tested using imblearn.pipeline.Pipeline inside CV folds only.
The main training script (scripts/train.py) uses class_weight and
scale_pos_weight instead, which are leakage-free alternatives.
""")
