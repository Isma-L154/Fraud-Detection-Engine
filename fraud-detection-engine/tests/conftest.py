"""Shared fixtures.

The scorer reaches handlers through `get_scorer`, so tests replace it with
`app.dependency_overrides` — no module patching.

No test here loads the real 3.5 MB artifact.
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np

# Set before anything imports app.core.config, which instantiates Settings at import
# and would otherwise fail here exactly as it would in production with no ENV set.
# setdefault, not assignment, so a deliberate value from the environment still wins.
os.environ.setdefault("ENV", "development")
os.environ.setdefault("CORS_ORIGINS", '["http://localhost:3000"]')

import pytest
from fastapi.testclient import TestClient

from app.ml.model import FraudDetectionModel


class FakePipeline:
    """Stands in for the trained sklearn Pipeline.

    Only `predict_proba` is used by FraudDetectionModel, which reads column 1 of
    every row. The probability is settable so a test can place a prediction in any
    risk bucket.
    """

    def __init__(self, fraud_probability: float = 0.0) -> None:
        self.fraud_probability = fraud_probability
        self.calls: list[Any] = []

    def predict_proba(self, X: Any) -> "np.ndarray[Any, Any]":
        # One row out per row in, as an array — the real pipeline returns an ndarray
        # the caller slices with [:, 1]. A list-of-one-row stand-in passed while the
        # code only ever scored single rows and would have hidden the batch path.
        self.calls.append(X)
        p = self.fraud_probability
        return np.array([[1.0 - p, p]] * len(X))


class ExplodingPipeline:
    """Raises on inference, to exercise the 500 path in the predict handler."""

    def predict_proba(self, X: Any) -> "np.ndarray[Any, Any]":
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


def build_scorer(pipeline: Any | None) -> FraudDetectionModel:
    """A FraudDetectionModel backed by `pipeline`, or unloaded when None."""
    scorer = FraudDetectionModel()
    scorer._pipeline = pipeline
    return scorer


@pytest.fixture
def loaded_model(fake_pipeline: FakePipeline) -> FakePipeline:
    """The fake pipeline the `client` fixture scores with."""
    return fake_pipeline


@pytest.fixture(autouse=True)
def clear_auth_throttle() -> Iterator[None]:
    """Reset the failed-authentication budget between tests.

    It is module state shared by the whole process, so without this the 401s one
    test produces spend the budget of every test after it — and the failure lands
    somewhere unrelated to the cause.
    """
    from app.core.rate_limit import reset_auth_throttle

    reset_auth_throttle()
    yield
    reset_auth_throttle()


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
def make_client() -> Iterator[Any]:
    """Build a TestClient whose scorer is overridden with the given pipeline.

    The lifespan still runs — it just no longer decides what the handlers use, so
    the real artifact is never read.
    """
    from app.api.dependencies import get_scorer
    from app.api.main import app

    clients = []

    def _make(pipeline: Any | None) -> TestClient:
        scorer = build_scorer(pipeline)
        app.dependency_overrides[get_scorer] = lambda: scorer
        client = TestClient(app)
        clients.append(client)
        return client

    yield _make
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Any, fake_pipeline: FakePipeline) -> TestClient:
    """A client against an app whose scorer is the fake pipeline."""
    return make_client(fake_pipeline)


@pytest.fixture
def unloaded_client(make_client: Any) -> TestClient:
    """A client whose scorer has no pipeline, as after a failed startup."""
    return make_client(None)
