"""
=============================================================
AI Fraud Detection & Risk Intelligence System
Training Script -- creditcard_train_80.csv
=============================================================
Trains 7 ML models and prints full metrics to terminal.
Saves trained models + scaler for later test evaluation.
=============================================================
"""
import io, sys
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, time, warnings, pickle, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# -- Check & install deps -----------------------------------------
def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

required = {"scikit-learn": "sklearn", "xgboost": "xgboost"}
for pkg, imp in required.items():
    try:
        __import__(imp)
    except ImportError:
        print(f"  Installing {pkg}...")
        install(pkg)

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report, average_precision_score)
from sklearn.model_selection import cross_val_score

# -- Paths --------------------------------------------------------
TRAIN_PATH   = r"C:\Users\Shriyansh\Downloads\creditcard_train_80.csv"
SAVE_DIR     = r"d:\minor\ml_dashboard\models"
RESULTS_JSON = r"d:\minor\ml_dashboard\train_results.json"
os.makedirs(SAVE_DIR, exist_ok=True)

D  = "=" * 70
SD = "-" * 70

def hdr(t):  print(f"\n{D}\n  {t}\n{D}")
def shdr(t): print(f"\n{SD}\n  {t}\n{SD}")

# =================================================================
# 1. LOAD DATA
# =================================================================
hdr("1. LOADING TRAINING DATASET")
print(f"  Path  : {TRAIN_PATH}")

t0 = time.time()
df = pd.read_csv(TRAIN_PATH)
load_time = time.time() - t0

print(f"  [OK] Loaded in {load_time:.2f}s")
print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Cols  : {list(df.columns)}")
print(f"  Nulls : {df.isnull().sum().sum()}")

# =================================================================
# 2. EDA
# =================================================================
hdr("2. EXPLORATORY DATA ANALYSIS (EDA)")

target_col = "Class"
fraud   = df[df[target_col] == 1]
legit   = df[df[target_col] == 0]
total   = len(df)
n_fraud = len(fraud)
n_legit = len(legit)

print(f"\n  CLASS DISTRIBUTION")
print(f"  {'Class':<14} {'Count':>10} {'Percent':>14}")
print(f"  {'-'*40}")
print(f"  {'Legit (0)':<14} {n_legit:>10,} {n_legit/total*100:>13.4f}%")
print(f"  {'Fraud (1)':<14} {n_fraud:>10,} {n_fraud/total*100:>13.4f}%")
print(f"  {'TOTAL':<14} {total:>10,} {'100.0000%':>14}")

print(f"\n  AMOUNT STATISTICS")
print(f"  {'Stat':<12} {'All Txns':>14} {'Legit':>14} {'Fraud':>14}")
print(f"  {'-'*56}")
for stat, fn in [("Mean","mean"),("Std","std"),("Min","min"),
                  ("25%","quantile"),("Median","median"),("75%","quantile"),("Max","max")]:
    if fn == "quantile":
        q = 0.25 if stat == "25%" else 0.75
        av = df['Amount'].quantile(q)
        lv = legit['Amount'].quantile(q)
        fv = fraud['Amount'].quantile(q)
    else:
        av = getattr(df['Amount'],    fn)()
        lv = getattr(legit['Amount'], fn)()
        fv = getattr(fraud['Amount'], fn)()
    print(f"  {stat:<12} {av:>14.4f} {lv:>14.4f} {fv:>14.4f}")

print(f"\n  TIME RANGE")
print(f"  Min: {df['Time'].min():,.0f}s  |  Max: {df['Time'].max():,.0f}s  |  Mean: {df['Time'].mean():,.1f}s")

print(f"\n  PCA FEATURE STATS (V1 - V10)")
print(f"  {'Feature':<10} {'Mean':>10} {'Std':>10} {'Min':>12} {'Max':>12}")
print(f"  {'-'*56}")
for col in [c for c in df.columns if c.startswith('V')][:10]:
    print(f"  {col:<10} {df[col].mean():>10.4f} {df[col].std():>10.4f} "
          f"{df[col].min():>12.4f} {df[col].max():>12.4f}")

print(f"\n  TOP 10 FEATURES CORRELATED WITH FRAUD (Pearson |r|)")
corr = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
print(f"  {'Feature':<12} {'|r|':>8}   Bar")
print(f"  {'-'*50}")
for feat, r in corr.head(10).items():
    bar = "#" * int(r * 40)
    print(f"  {feat:<12} {r:>8.4f}   {bar}")

# =================================================================
# 3. FEATURE PREPARATION
# =================================================================
hdr("3. FEATURE PREPARATION & SCALING")
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].values
y = df[target_col].values

print(f"  Feature matrix X : {X.shape}")
print(f"  Target vector  y : {y.shape}   Fraud rate = {y.mean()*100:.4f}%")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  [OK] StandardScaler fitted on {X.shape[0]:,} training samples")

# Save scaler + feature list
with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(SAVE_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)
print(f"  [OK] Scaler & feature list saved --> {SAVE_DIR}")

# =================================================================
# 4. MODEL DEFINITIONS
# =================================================================
hdr("4. MODEL CONFIGURATIONS")

models = {
    "Logistic Regression": LogisticRegression(
        C=0.01, max_iter=1000, class_weight='balanced',
        solver='saga', random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, class_weight='balanced',
        min_samples_leaf=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True,
        random_state=42, cache_size=800
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5, weights='distance',
        metric='euclidean', n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, subsample=0.8, random_state=42
    ),
}

for name, m in models.items():
    print(f"  [OK] {name:<28} {m.__class__.__name__}")

# =================================================================
# 5. TRAIN ALL MODELS
# =================================================================
hdr("5. TRAINING ALL MODELS")
print("  NOTE: SVM may take 10-20 min on 200k+ rows. Please wait...\n")

all_results = {}

for i, (name, model) in enumerate(models.items(), 1):
    shdr(f"[{i}/{len(models)}] {name}")
    t_start = time.time()

    print(f"  Fitting... ", end="", flush=True)
    model.fit(X_scaled, y)
    fit_time = time.time() - t_start
    print(f"done in {fit_time:.2f}s")

    # Save model
    model_path = os.path.join(SAVE_DIR, f"{name.replace(' ','_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [SAVED] {model_path}")

    # Predictions on training set
    print(f"  Predicting on training set... ", end="", flush=True)
    y_pred  = model.predict(X_scaled)
    y_prob  = (model.predict_proba(X_scaled)[:, 1]
               if hasattr(model, "predict_proba")
               else model.decision_function(X_scaled))
    print("done")

    acc   = accuracy_score(y, y_pred)
    prec  = precision_score(y, y_pred, zero_division=0)
    rec   = recall_score(y, y_pred, zero_division=0)
    f1    = f1_score(y, y_pred, zero_division=0)
    auc   = roc_auc_score(y, y_prob)
    prauc = average_precision_score(y, y_prob)
    cm    = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])

    print(f"\n  +---------- TRAINING SET METRICS ---------------------------+")
    print(f"  | Accuracy  : {acc*100:>10.4f}%                              |")
    print(f"  | Precision : {prec*100:>10.4f}%  (Fraud class)              |")
    print(f"  | Recall    : {rec*100:>10.4f}%  (Fraud class)              |")
    print(f"  | F1-Score  : {f1*100:>10.4f}%  (Fraud class)              |")
    print(f"  | ROC-AUC   : {auc:>14.6f}                             |")
    print(f"  | PR-AUC    : {prauc:>14.6f}                             |")
    print(f"  +------------------------------------------------------------+")

    print(f"\n  CONFUSION MATRIX (Training Set)")
    print(f"  {'':<24} Pred: Legit    Pred: Fraud")
    print(f"  {'Actual: Legit':<24} {tn:>12,}   {fp:>12,}")
    print(f"  {'Actual: Fraud':<24} {fn:>12,}   {tp:>12,}")
    print(f"\n  TN={tn:,}  |  FP={fp:,}  |  FN={fn:,}  |  TP={tp:,}")

    print(f"\n  CLASSIFICATION REPORT:")
    print(classification_report(y, y_pred, target_names=['Legit','Fraud'], digits=4))

    all_results[name] = {
        "fit_time_s": round(fit_time, 3),
        "train": {
            "accuracy":  round(float(acc),   6),
            "precision": round(float(prec),  6),
            "recall":    round(float(rec),   6),
            "f1":        round(float(f1),    6),
            "roc_auc":   round(float(auc),   6),
            "pr_auc":    round(float(prauc), 6),
            "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    }

# =================================================================
# 6. FULL SUMMARY TABLE
# =================================================================
hdr("6. COMPLETE TRAINING SUMMARY TABLE")

cols_w = [27, 11, 11, 11, 11, 10, 10, 9]
col_names = ["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC","Time(s)"]
print("  " + "".join(c.ljust(cols_w[i]) for i,c in enumerate(col_names)))
print("  " + "-" * sum(cols_w))

best = {
    "accuracy":  max(all_results, key=lambda n: all_results[n]["train"]["accuracy"]),
    "precision": max(all_results, key=lambda n: all_results[n]["train"]["precision"]),
    "recall":    max(all_results, key=lambda n: all_results[n]["train"]["recall"]),
    "f1":        max(all_results, key=lambda n: all_results[n]["train"]["f1"]),
    "roc_auc":   max(all_results, key=lambda n: all_results[n]["train"]["roc_auc"]),
    "pr_auc":    max(all_results, key=lambda n: all_results[n]["train"]["pr_auc"]),
}

for name, res in all_results.items():
    t = res["train"]
    row = [
        name,
        f"{t['accuracy']*100:.4f}%",
        f"{t['precision']*100:.4f}%",
        f"{t['recall']*100:.4f}%",
        f"{t['f1']*100:.4f}%",
        f"{t['roc_auc']:.4f}",
        f"{t['pr_auc']:.4f}",
        f"{res['fit_time_s']:.1f}s",
    ]
    line = "  " + "".join(str(row[i]).ljust(cols_w[i]) for i in range(len(col_names)))
    # Mark best models
    markers = []
    if name == best["f1"]:      markers.append("BEST-F1")
    if name == best["roc_auc"]: markers.append("BEST-AUC")
    if name == best["recall"]:  markers.append("BEST-RECALL")
    if markers:
        line += f"  << {' | '.join(markers)} >>"
    print(line)

# =================================================================
# 7. CROSS-VALIDATION (3-fold, subsample)
# =================================================================
hdr("7. CROSS-VALIDATION (3-Fold F1, 50k Subsample)")
print("  Running on a 50k subsample for speed...\n")

cv_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
sub_idx  = np.random.RandomState(42).choice(len(X_scaled), min(50000, len(X_scaled)), replace=False)
X_cv = X_scaled[sub_idx]; y_cv = y[sub_idx]

for name in cv_names:
    t_cv = time.time()
    scores = cross_val_score(models[name], X_cv, y_cv, cv=3, scoring='f1', n_jobs=-1)
    elapsed = time.time() - t_cv
    fold_str = ", ".join(f"{s:.4f}" for s in scores)
    print(f"  {name:<28} mean={scores.mean():.4f} +/-{scores.std():.4f}"
          f"   [{fold_str}]   ({elapsed:.1f}s)")

# =================================================================
# 8. SAVE JSON RESULTS
# =================================================================
hdr("8. SAVING ALL RESULTS")

meta = {
    "dataset":      TRAIN_PATH,
    "n_train":      total,
    "n_features":   len(feature_cols),
    "n_fraud":      int(n_fraud),
    "n_legit":      int(n_legit),
    "fraud_rate":   round(n_fraud/total, 6),
    "feature_cols": feature_cols,
    "target_col":   target_col,
    "results":      all_results,
}
with open(RESULTS_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print(f"  [OK] Results JSON --> {RESULTS_JSON}")
print(f"  [OK] Models dir   --> {SAVE_DIR}/")
for name in models:
    sz = os.path.getsize(os.path.join(SAVE_DIR, f"{name.replace(' ','_')}.pkl"))
    print(f"       {name.replace(' ','_')}.pkl  ({sz/1024:.0f} KB)")
scaler_sz = os.path.getsize(os.path.join(SAVE_DIR, "scaler.pkl"))
print(f"       scaler.pkl  ({scaler_sz/1024:.1f} KB)")

# =================================================================
# DONE
# =================================================================
hdr("TRAINING COMPLETE -- SEND TEST CSV TO EVALUATE")
print("""
  All 7 models trained and saved successfully!

  NEXT: Send your test CSV path (e.g. creditcard_test_20.csv)
  Then evaluation script will run on test set and report:
    - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
    - MAE, MSE, RMSE, R2
    - Confusion Matrix per model
    - Full comparison table

""")
print(D)
