# Singleton model loader. The pipeline is loaded once at startup and reused
# across all requests. Never load from disk inside a request handler. 
# (I used Singleton pattern here, helps me save memory and speed up inference by reusing the same model instance across requests.)

import joblib
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical path to the model artifact — relative to project root
MODEL_PATH = Path("models/fraud_model.pkl")

# Version tag injected into every prediction response. 
MODEL_VERSION = "1.0.0"

# Thresholds that map a raw probability score to a human-readable risk level.
#(I decided to put these, cause these are business rules that might change independently of the model. )
RISK_THRESHOLDS = {
    "LOW":    0.30,
    "MEDIUM": 0.70,
    "HIGH":   1.01,  # catch-all upper bound
}


class FraudDetectionModel:
    """
    Wraps the trained sklearn Pipeline with production concerns:
    versioning, input validation, risk bucketing, and error handling.
    """

    def __init__(self) -> None:
        self._pipeline = None  # loaded lazily on first call to load()

    def load(self) -> None:
        """Load the pipeline from disk. Call once at application startup."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run notebooks/train.py first."
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
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        # Build a single-row DataFrame preserving feature column order.
        feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X = pd.DataFrame([features])[feature_columns]

        # predict_proba returns [[prob_legit, prob_fraud]]
        # We only need the fraud probability — index 1
        fraud_probability = float(self._pipeline.predict_proba(X)[0][1])
        is_fraud = fraud_probability >= RISK_THRESHOLDS["LOW"]

        return {
            "is_fraud":          is_fraud,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level":        self._get_risk_level(fraud_probability),
            "model_version":     MODEL_VERSION,
        }

    def _get_risk_level(self, probability: float) -> str:
        """Map a raw probability to a risk bucket."""
        if probability < RISK_THRESHOLDS["LOW"]:
            return "LOW"
        elif probability < RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "HIGH"


# Module-level singleton — imported by FastAPI's dependency injection
fraud_model = FraudDetectionModel()