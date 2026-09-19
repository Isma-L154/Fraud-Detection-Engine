# Trains a fraud detection pipeline and logs the run to MLflow.
# Output: models/fraud_model.pkl (scaler + classifier bundled together)

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# This script lives in a subdirectory but imports from app/, so the project root
# has to be importable. Without this, `python notebooks/train.py` — the command the
# README gives — fails with ModuleNotFoundError before doing anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.integrity import sha256_of
from app.ml.artifact import ArtifactMetadata, envelope

# Resolved from this file, not the working directory, so the script reads and writes
# the same places whether it is run from the project root or from notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "notebooks" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"

SAMPLE_PATH = PROJECT_ROOT / "notebooks" / "creditcard_sample.csv"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=None,
    help="CSV to train on. Defaults to the real dataset, falling back to the sample.",
)
args = parser.parse_args()


def resolve_dataset(explicit: Path | None) -> Path:
    """Pick the dataset, and say clearly what to do when there is none."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Dataset not found at {explicit}")
        return explicit
    if DATASET_PATH.exists():
        return DATASET_PATH
    if SAMPLE_PATH.exists():
        print(f"Real dataset absent; training on the synthetic sample at {SAMPLE_PATH}.")
        print("The resulting model is not meaningful. Run scripts/fetch_dataset.py for real data.")
        return SAMPLE_PATH
    raise FileNotFoundError(
        f"No dataset found.\n"
        f"  Real:   {DATASET_PATH}  -> python scripts/fetch_dataset.py\n"
        f"  Sample: {SAMPLE_PATH}   -> python scripts/make_sample_dataset.py"
    )


mlflow.set_experiment("fraud-detection")

dataset_path = resolve_dataset(args.dataset)
df = pd.read_csv(dataset_path)

print(f"Dataset: {df.shape[0]:,} rows | Fraud rate: {df['Class'].mean() * 100:.2f}%")

# Drop Time — it's a sequential counter with no predictive signal
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# stratify=y preserves the 0.17% fraud ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    # Bundle scaler + model so production inference never skips the transformation step
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",  # critical: compensates for 0.17% fraud rate
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # fraud probability score

    auc = roc_auc_score(y_test, y_proba)

    # Two reports. sklearn's predict() decides at 0.5; the service decides at
    # settings.decision_threshold. Measuring only the first is how the published
    # figures came to describe an operating point the service never used (#17).
    report_default = classification_report(y_test, y_pred, output_dict=True)
    y_pred_operating = (y_proba >= settings.decision_threshold).astype(int)
    report = classification_report(y_test, y_pred_operating, output_dict=True)

    mlflow.log_params(
        {
            "n_estimators": 100,
            "class_weight": "balanced",
            "decision_threshold": settings.decision_threshold,
            "sklearn_version": sklearn.__version__,
        }
    )
    metrics = {
        "auc_roc": auc,
        "precision_fraud": report["1"]["precision"],
        "recall_fraud": report["1"]["recall"],
        "f1_fraud": report["1"]["f1-score"],
        # Kept for comparison with the historical figures, measured at 0.5.
        "precision_fraud_at_0_5": report_default["1"]["precision"],
        "recall_fraud_at_0_5": report_default["1"]["recall"],
    }
    mlflow.log_metrics(metrics)

    # Provenance travels inside the artifact, so it cannot drift from the model it
    # describes and is covered by the same integrity digest (#15).
    metadata = ArtifactMetadata(
        run_id=mlflow.active_run().info.run_id,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__,
        dataset=dataset_path.name,
        metrics={k: float(v) for k, v in metrics.items()},
        decision_threshold=settings.decision_threshold,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(envelope(pipeline, metadata), MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH))

    # The digest the service must be configured with. Printed rather than written
    # beside the artifact on purpose: a checksum an artifact-writer can also rewrite
    # verifies nothing. Logged to MLflow so the run that produced it is the record.
    digest = sha256_of(MODEL_PATH)
    mlflow.log_param("model_sha256", digest)

    print(f"\nAUC-ROC: {auc:.4f}")
    print("\n--- configure the service with these ---")
    print(f"MODEL_VERSION={metadata.run_id}")
    print(f"MODEL_SHA256={digest}")
    print("The service refuses to start without MODEL_SHA256 outside development.")

    print("\nAt 0.5, sklearn default — the historically published figures:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
    print(f"\nAt {settings.decision_threshold}, the operating point the service uses:")
    print(classification_report(y_test, y_pred_operating, target_names=["legit", "fraud"]))
