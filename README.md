# FraudGuard AI

**Academic Machine Learning Project — Credit Card Fraud Detection**

A complete, reproducible ML system for detecting fraudulent credit card transactions. Implements three classifiers (Logistic Regression, Random Forest, XGBoost), rigorous evaluation, SHAP explainability, and a professional Streamlit dashboard.

---

## Dataset

**ULB Credit Card Fraud Detection**
- Source: [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: 284,807 rows × 31 columns
- Features: `Time`, `Amount`, `V1`–`V28` (PCA-transformed, anonymized), `Class`
- Target: `Class` (0 = legitimate, 1 = fraud)
- Fraud rate: 0.172% (492 fraud / 284,315 legitimate)

**Citation:**
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> IEEE Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

---

## Project Structure

```
fraudguard-ai/
├── app.py                    # Streamlit main entry point
├── pages/
│   ├── 01_Overview.py        # Dataset stats and EDA charts
│   ├── 02_Fraud_Prediction.py
│   ├── 03_Model_Comparison.py
│   ├── 04_Explainable_AI.py
│   └── 05_Transaction_Monitor.py
├── src/
│   ├── config.py             # Paths, constants, feature lists
│   ├── data_loader.py        # Load, validate, split dataset
│   ├── preprocessing.py      # sklearn ColumnTransformer pipelines
│   ├── models.py             # Model factory functions
│   ├── evaluation.py         # Metrics, curves, persistence
│   ├── explainability.py     # SHAP explainer selection
│   └── inference.py          # Load model, score transactions
├── scripts/
│   ├── run_eda.py            # EDA → results/eda/
│   ├── train.py              # Full training pipeline
│   ├── evaluate.py           # Post-training chart generation
│   ├── run_shap.py           # SHAP global importance
│   └── create_demo_csv.py    # Small demo CSV for Streamlit
├── notebooks/
│   ├── 01_eda.py             # Interactive EDA
│   ├── 02_preprocessing.py   # Pipeline demo and leakage proof
│   ├── 03_models.py          # Model comparison
│   └── 05_explainability.py  # SHAP walkthrough
├── data/
│   ├── creditcard.csv        # [NOT COMMITTED — download from Kaggle]
│   └── demo_transactions.csv # Small sample (committed)
├── models/                   # Saved pipeline artifacts (.joblib)
├── results/
│   ├── eda/                  # EDA charts + summary JSON
│   ├── evaluation/           # Evaluation charts
│   └── shap/                 # SHAP charts and explainer
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_loader.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_csv_upload.py
├── docs/
│   └── report_outline.md     # College report structure
├── requirements.txt
├── pyproject.toml
└── .streamlit/config.toml
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it at: `data/creditcard.csv`

---

## Running the Training Pipeline

### 80/20 model comparison and report

To train the requested classifiers on an 80% stratified training split and
evaluate them on the held-out 20% test split:

```bash
python scripts/train_comparison.py
```

The script trains Logistic Regression, Decision Tree, Random Forest, SVM,
K-Nearest Neighbors, Naive Bayes, Gradient Boosting, and XGBoost. It writes
EDA summaries and plots, train/test metric tables, confusion-matrix tables,
ROC curves, and comparison graphs to `results/model_comparison_80_20/`.
Trained pipelines are saved separately in `models/comparison_80_20/`.

MAE, MSE, RMSE, and R² are calculated by comparing each classifier's
fraud-risk score with the binary target; they are supplementary score-error
metrics, not results from a separate Linear Regression model. SVM uses a
linear support vector classifier for practical runtime on the full dataset.

### Evaluate a supplied labeled test CSV

To train the comparison classifiers on `data/creditcard.csv` and evaluate a
separate labeled CSV (with the same `Time`, `Amount`, `V1`–`V28`, and `Class`
columns), run:

```bash
python scripts/evaluate_test_dataset.py path/to/test.csv
```

Exact feature-row overlaps between the training CSV and test CSV are removed
from training before fitting. Tables, per-row predictions, and EDA/model
comparison graphs are written to `results/test_evaluation/` by default. Use
`--training-csv` or `--output-dir` to select different paths. Duplicate test
rows are retained and reported so every supplied row is included in scoring.

### Full training (recommended for final results)

```bash
python scripts/run_eda.py        # Step 1: EDA + charts
python scripts/train.py          # Step 2: Train all models, tune, evaluate
python scripts/evaluate.py       # Step 3: Generate evaluation charts
python scripts/run_shap.py       # Step 4: SHAP global importance
python scripts/create_demo_csv.py  # Step 5: Demo transactions for Streamlit
```

### Quick development mode (5% sample — non-final results)

```bash
python scripts/train.py --quick
```

> **Warning**: `--quick` results are labelled `[QUICK-DEV — non-final results]`
> and must NOT be used in the final submission.

---

## Running the Streamlit App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

**Prerequisites**: trained model artifacts must exist in `models/` and
results in `results/`. Run the training pipeline first.

### Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**.
3. Select the repository, branch, and `app.py` as the app's main file.
4. Deploy. The dependencies are installed from `requirements.txt`.

The repository must include the champion model files, `data/demo_transactions.csv`,
and the `results/` files used by the dashboard. Do not upload either copy of
`creditcard.csv`; the running app uses the saved artifacts and does not retrain.

---

## Running Tests

```bash
pytest
# With coverage
pytest --cov=src tests/
```

The test suite does NOT require the full dataset — it uses synthetic toy data.

---

## Deployment Notes

For deployment without the 150 MB dataset:

1. Commit to git:
   - `models/*.joblib` (trained pipelines)
   - `results/` (evaluation charts and metrics JSON)
   - `data/demo_transactions.csv` (small sample)

2. Do NOT commit:
   - `data/creditcard.csv` (too large; download from Kaggle)

3. The Streamlit app loads saved artifacts — it never retrains.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Stratified 70/15/15 split | Preserves class distribution; val set for model selection; test set untouched |
| Champion selected by Val Average Precision | Most informative metric for imbalanced datasets |
| Threshold via max-F1 on validation | Balances precision and recall without touching test set |
| class_weight / scale_pos_weight | Leakage-free imbalance handling; no SMOTE in final pipeline |
| TreeExplainer for RF/XGB | Exact, efficient SHAP values; LinearExplainer for LR |
| Scores displayed as risk scores | Model outputs are not calibrated probabilities |

---

## Limitations

1. **Anonymized features**: V1-V28 are PCA components; original feature meanings are unknown.
2. **Temporal leakage risk**: The dataset spans two days; real-world deployment would require time-aware splitting.
3. **No probability calibration**: Risk scores are model outputs. For calibrated probabilities, apply Platt scaling or isotonic regression evaluated on a separate calibration set.
4. **Dataset age**: The ULB dataset is from 2013; fraud patterns evolve over time.
5. **Quick-dev mode**: Results from `--quick` flag are non-final and must not be submitted.
6. **XGBoost version compatibility**: SHAP TreeExplainer output format may vary across XGBoost versions. Pinned in requirements.txt.

---

## Academic Integrity

All code is original. No results are fabricated. Model outputs, metrics, and charts are generated from the real ULB dataset. Observations in EDA scripts and comments are derived from actual data outputs.
