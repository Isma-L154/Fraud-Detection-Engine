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
from app.ml.artifact import UNKNOWN_VERSION, ArtifactMetadata, unpack

logger = logging.getLogger(__name__)

# Column order is load-bearing: the pipeline was fitted on V1..V28 then Amount, and
# a frame built in any other order scores the wrong feature against the wrong
# weight — silently, with no exception.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Canonical path to the model artifact, from configuration. Its default resolves
# from the package location rather than the process working directory, so the
# service starts from any directory (#19).
MODEL_PATH = settings.model_path

# The version comes from the artifact itself (see app/ml/artifact.py), not from a
# constant here. A constant could not change when the file did, so the reported
# version described this source file rather than the model it claimed to describe.


class FraudDetectionModel:
    """
    Wraps the trained sklearn Pipeline with production concerns:
    versioning, input validation, risk bucketing, and error handling.
    """

    def __init__(self) -> None:
        # Populated by load(); None until then. Annotated so the type checker can
        # narrow it and catch a predict() that runs before load().
        self._pipeline: Any | None = None
        self._metadata = ArtifactMetadata()

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

        artifact = unpack(joblib.load(MODEL_PATH))
        self._pipeline = artifact.pipeline
        self._metadata = artifact.metadata

        if self._metadata.version == UNKNOWN_VERSION:
            # Not an error: an artifact trained before the envelope existed is
            # unlabelled, not invalid. Saying so beats inventing a version.
            logger.warning(
                "Loaded an artifact with no training metadata from %s — "
                "it will report version '%s'. Retrain to attach provenance.",
                MODEL_PATH,
                UNKNOWN_VERSION,
            )
        logger.info(
            "Model loaded",
            extra={
                "path": str(MODEL_PATH),
                "model_version": self._metadata.version,
                "trained_at": self._metadata.trained_at,
                "sklearn_version": self._metadata.sklearn_version,
            },
        )

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def version(self) -> str:
        """Identifies the training run that produced the loaded artifact."""
        return self._metadata.version

    @property
    def metadata(self) -> ArtifactMetadata:
        return self._metadata

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single transaction.

        Args:
            features: dict matching TransactionRequest fields (V1-V28 + Amount)

        Returns:
            dict with is_fraud, fraud_probability, risk_level, model_version
        """
        return self.predict_many([features])[0]

    def predict_many(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Score several transactions in one pass.

        sklearn's per-call overhead is large relative to scoring one row, so N rows
        in one array is materially cheaper than N separate calls. predict()
        delegates here, so there is exactly one place that builds the frame, runs
        inference and maps a probability to a result.

        Results are returned in input order.
        """
        # Checked against _pipeline rather than the is_loaded property so the type
        # checker narrows it for the predict_proba call below.
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if not rows:
            return []

        X = pd.DataFrame(rows)[FEATURE_COLUMNS]

        # predict_proba returns one [prob_legit, prob_fraud] pair per row; column 1
        # is the fraud probability.
        probabilities = self._pipeline.predict_proba(X)[:, 1]

        return [self._as_result(float(p)) for p in probabilities]

    def _as_result(self, fraud_probability: float) -> dict[str, Any]:
        return {
            "is_fraud": fraud_probability >= settings.decision_threshold,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level": risk_level(fraud_probability),
            "model_version": self._metadata.version,
            "decision_threshold": settings.decision_threshold,
        }
