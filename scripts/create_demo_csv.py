"""
FraudGuard AI ? Demo CSV creation script.

Usage:
    python scripts/create_demo_csv.py

Creates data/demo_transactions.csv: a small (200-row) stratified sample
from the real dataset for use in the Streamlit Fraud Prediction page.

This file CAN be committed to git (it is small) and allows the Streamlit
app to demonstrate predictions without needing the 150 MB full dataset.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, DEMO_CSV_PATH, RANDOM_SEED, TARGET_COL
from src.data_loader import require_dataset

N_FRAUD = 100
N_LEGIT = 100


def main() -> None:
    print("\nFraudGuard AI ? Demo CSV Creation")
    print("=" * 50)

    print("\n[1/3] Loading dataset...")
    df = require_dataset()
    df = df.drop_duplicates()
    print(f"  Loaded {len(df):,} rows.")

    print("\n[2/3] Sampling demo transactions...")
    fraud_sample = df[df[TARGET_COL] == 1].sample(
        min(N_FRAUD, (df[TARGET_COL] == 1).sum()),
        random_state=RANDOM_SEED,
    )
    legit_sample = df[df[TARGET_COL] == 0].sample(
        N_LEGIT,
        random_state=RANDOM_SEED,
    )

    demo = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=RANDOM_SEED)
    demo = demo.reset_index(drop=True)

    print(f"  Fraud rows : {len(fraud_sample)}")
    print(f"  Legit rows : {len(legit_sample)}")
    print(f"  Total      : {len(demo)}")

    print("\n[3/3] Saving demo CSV...")
    DEMO_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(DEMO_CSV_PATH, index=False)
    print(f"  Saved: {DEMO_CSV_PATH}")
    print(f"  Columns: {list(demo.columns)}")
    print("\nDemo CSV creation complete.\n")


if __name__ == "__main__":
    main()
