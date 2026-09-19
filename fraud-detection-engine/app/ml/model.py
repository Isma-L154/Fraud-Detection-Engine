# Singleton model loader. The pipeline is loaded once at startup and reused
# across all requests. Never load from disk inside a request handler.
# A single shared instance keeps the pipeline in memory across requests instead of
# deserialising a 3.5 MB artifact on each one.

import logging
from typing import Any

import joblib
import pandas as pd

from app.core.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

# Canonical path to the model artifact, from configuration. Its default resolves
# from the package location rather than the process working directory, so the
# service starts from any directory (#19).
MODEL_PATH = settings.model_path

# Version tag injected into every prediction response.
MODEL_VERSION = "1.0.0"

# Thresholds that map a raw probability score to a human-readable risk level.
# These are business rules and change independently of the model. Decoupling the
# decision threshold from these buckets is issue #17.
RISK_THRESHOLDS = {
    "LOW": 0.30,
    "MEDIUM": 0.70,
    "HIGH": 1.01,  # catch-all upper bound
}


class FraudDetectionModel:
    """
    Wraps the trained sklearn Pipeline with production concerns:
    versioning, input validation, risk bucketing, and error handling.
    """

    def __init__(self) -> None:
        # Populated by load(); None until then. Annotated so the type checker can
        # narrow it and catch a predict() that runs before load().
        self._pipeline: Any | None = None

    def load(self) -> None:
        """Load the pipeline from disk. Call once at application startup."""
        if not MODEL_PATH.exists():
            # Name the absolute path that was actually checked. The previous message
            # printed a relative path, which told you nothing about where it looked.
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                f"Train it with: python {PROJECT_ROOT / 'notebooks' / 'train.py'}"
            )

        self._pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH} | version={MODEL_VERSION}")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def version(self) -> str:
        return MODEL_VERSION

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single transaction.

        Args:
            features: dict matching TransactionRequest fields (V1-V28 + Amount)

        Returns:
            dict with is_fraud, fraud_probability, risk_level, model_version
        """
        # Checked against _pipeline rather than the is_loaded property so the type
        # checker narrows it for the predict_proba call below.
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        # Build a single-row DataFrame preserving feature column order.
        feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X = pd.DataFrame([features])[feature_columns]

        # predict_proba returns [[prob_legit, prob_fraud]]
        # We only need the fraud probability — index 1
        fraud_probability = float(self._pipeline.predict_proba(X)[0][1])
        is_fraud = fraud_probability >= RISK_THRESHOLDS["LOW"]

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level": self._get_risk_level(fraud_probability),
            "model_version": MODEL_VERSION,
        }

    def _get_risk_level(self, probability: float) -> str:
        """Map a raw probability to a risk bucket."""
        if probability < RISK_THRESHOLDS["LOW"]:
            return "LOW"
        elif probability < RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "HIGH"


# Module-level singleton, imported directly by the route handlers. This is not
# dependency injection and cannot be substituted without patching — moving it
# behind a FastAPI dependency is issue #18.
fraud_model = FraudDetectionModel()
