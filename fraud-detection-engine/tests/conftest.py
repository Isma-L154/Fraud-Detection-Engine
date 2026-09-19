"""Shared fixtures.

The model is a module-level global (`app.ml.model.fraud_model`) imported directly by
the route handlers, so there is no dependency to override — the only way to substitute
it is to patch the object. Moving it behind a FastAPI dependency is issue #18; when
that lands, these fixtures collapse into `app.dependency_overrides`.

No test here loads the real 3.5 MB artifact.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient


class FakePipeline:
    """Stands in for the trained sklearn Pipeline.

    Only `predict_proba` is used by FraudDetectionModel, and only column 1 of the
    first row is read. The probability is settable so a test can place a prediction
    in any risk bucket.
    """

    def __init__(self, fraud_probability: float = 0.0) -> None:
        self.fraud_probability = fraud_probability
        self.calls: list[Any] = []

    def predict_proba(self, X: Any) -> list[list[float]]:
        self.calls.append(X)
        return [[1.0 - self.fraud_probability, self.fraud_probability]]


class ExplodingPipeline:
    """Raises on inference, to exercise the 500 path in the predict handler."""

    def predict_proba(self, X: Any) -> list[list[float]]:
        raise ValueError("synthetic inference failure")


@pytest.fixture
def valid_transaction() -> dict[str, float]:
    """A request body that satisfies every constraint in TransactionRequest."""
    body: dict[str, float] = {f"V{i}": 0.0 for i in range(1, 29)}
    body["Amount"] = 149.62
    return body


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def loaded_model(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakePipeline
) -> Iterator[FakePipeline]:
    """Put the singleton into a loaded state backed by the fake pipeline.

    `load()` is also stubbed, because the FastAPI lifespan calls it on startup and
    would otherwise read the real artifact from disk.
    """
    from app.ml import model as model_module

    monkeypatch.setattr(model_module.fraud_model, "_pipeline", fake_pipeline)
    monkeypatch.setattr(model_module.fraud_model, "load", lambda: None)
    yield fake_pipeline


@pytest.fixture
def unloaded_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Leave the singleton unloaded, as it would be if startup had failed."""
    from app.ml import model as model_module

    monkeypatch.setattr(model_module.fraud_model, "_pipeline", None)
    monkeypatch.setattr(model_module.fraud_model, "load", lambda: None)
    yield


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Rate limiting is off by default.

    Without this, tests share one 30-per-minute budget keyed on the same client
    address, and whichever test happens to run thirty-first fails for reasons that
    have nothing to do with what it asserts. The test that covers the limit turns it
    back on explicitly.
    """
    from app.core.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", False)
    yield


@pytest.fixture
def client(loaded_model: FakePipeline) -> Iterator[TestClient]:
    """A client against an app whose model is loaded with the fake pipeline."""
    from app.api.main import app

    with TestClient(app) as test_client:
        yield test_client
