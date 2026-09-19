"""Obtain the real training dataset.

The Credit Card Fraud Detection dataset is distributed through Kaggle, which
requires an account, so it cannot simply be downloaded in a script without
credentials. This checks what is present, uses the Kaggle CLI when it is
configured, and otherwise prints the exact manual steps rather than failing with
an HTTP error that explains nothing.
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "notebooks" / "creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

MANUAL_STEPS = f"""
The Kaggle CLI is not configured, so the dataset cannot be fetched automatically.

Either:

  1. Install and configure the CLI, then re-run this script:
       pip install kaggle
       # Put your API token at ~/.kaggle/kaggle.json (Kaggle > Settings > API)

  2. Or download it by hand:
       https://www.kaggle.com/datasets/{KAGGLE_DATASET}
       Unzip it and place creditcard.csv at:
         {TARGET}

To run the pipeline without the real data — the schema is the same, the model is
not meaningful — use:
    python scripts/make_sample_dataset.py
"""


def main() -> int:
    if TARGET.exists():
        size_mb = TARGET.stat().st_size / 1_000_000
        print(f"Already present: {TARGET} ({size_mb:.0f} MB)")
        return 0

    if shutil.which("kaggle") is None:
        print(MANUAL_STEPS)
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} to {TARGET.parent} ...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(TARGET.parent),
            "--unzip",
        ],
        check=False,
    )
    if result.returncode != 0 or not TARGET.exists():
        print(MANUAL_STEPS)
        return 1

    print(f"Done: {TARGET} ({TARGET.stat().st_size / 1_000_000:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
