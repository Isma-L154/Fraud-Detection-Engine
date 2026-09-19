# Loads the trained pipeline once and scores transactions with it. One instance is
# created during the application lifespan and shared across requests — never load
# from disk inside a request handler; deserialising a 3.5 MB artifact per request
# would dominate the latency.

import logging
from typing import Any

import joblib
import pandas as pd

from app.core.config import PROJECT_ROOT, settings
from app.core.integrity import sha256_of, verify
from app.core.risk import risk_level

logger = logging.getLogger(__name__)

# Canonical path to the model artifact, from configuration. Its default resolves
# from the package location rather than the process working directory, so the
# service starts from any directory (#19).
MODEL_PATH = settings.model_path

# Version tag injected into every prediction response.
MODEL_VERSION = "1.0.0"


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

        # Verify BEFORE loading. joblib.load is pickle underneath: by the time the
        # object exists, whatever the file contained has already run.
        if settings.model_sha256:
            verify(MODEL_PATH, settings.model_sha256)
            logger.info("Artifact digest verified")
        else:
            # Only reachable in development; Settings refuses to build otherwise.
            logger.warning(
                "Loading %s WITHOUT integrity verification — set MODEL_SHA256. "
                "Unpickling executes the file's contents. Digest is %s",
                MODEL_PATH,
                sha256_of(MODEL_PATH),
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

        return {
            "is_fraud": fraud_probability >= settings.decision_threshold,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level": risk_level(fraud_probability),
            "model_version": MODEL_VERSION,
            "decision_threshold": settings.decision_threshold,
        }
