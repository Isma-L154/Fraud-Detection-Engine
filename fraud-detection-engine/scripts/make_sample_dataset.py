"""Generate a small synthetic dataset with the real one's schema.

The real training data is a 150 MB Kaggle download behind an account. That is a
hard dependency for anyone who wants to check the pipeline runs, and impossible
for CI. This produces a file with the same columns, dtypes and class imbalance, so
`train.py` can be exercised end to end in seconds without it.

**It is not a substitute for the real data.** A model trained on this scores
nothing meaningful — the features are noise with a planted signal. It exists to
prove the pipeline works, not the model.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "creditcard_sample.csv"

# The real dataset's fraud rate, so the imbalance the pipeline has to cope with is
# represented rather than smoothed away.
FRAUD_RATE = 0.0017


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(rows * FRAUD_RATE))

    frame = pd.DataFrame(
        {f"V{i}": rng.normal(0, 1, rows) for i in range(1, 29)},
    )
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = np.round(np.abs(rng.lognormal(3, 1.2, rows)), 2).clip(0, 50_000)
    frame["Class"] = 0

    # A planted, learnable signal. Without one the classifier has nothing to fit and
    # the run tells you nothing about whether the pipeline works.
    fraud_rows = rng.choice(rows, size=n_fraud, replace=False)
    frame.loc[fraud_rows, "Class"] = 1
    frame.loc[fraud_rows, "V14"] -= 6.0
    frame.loc[fraud_rows, "V17"] -= 5.0

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = generate(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"Wrote {len(frame):,} rows to {args.output} ({size_mb:.1f} MB)")
    print(f"Fraud rate: {frame['Class'].mean() * 100:.3f}%")
    print()
    print("This is synthetic. A model trained on it scores nothing meaningful —")
    print("it exists to prove the pipeline runs. Use the real dataset for a real model:")
    print("  python scripts/fetch_dataset.py")


if __name__ == "__main__":
    main()
