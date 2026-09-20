"""Application startup wiring.

The other API tests override `get_scorer`, which is what makes them fast and
artifact-free — but it also means they never exercise the lifespan or the
dependency itself. These do, so the wiring that connects them cannot rot unnoticed.
"""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.api.dependencies import get_scorer
from app.api.main import app
from app.ml.model import FraudDetectionModel
from app.ml.protocol import TransactionScorer

from .conftest import FakePipeline


@pytest.fixture
def stub_load(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    """Make load() install a fake pipeline instead of reading the real artifact."""
    pipeline = FakePipeline(0.5)

    def fake_load(self: FraudDetectionModel) -> None:
        self._pipeline = pipeline

    monkeypatch.setattr(FraudDetectionModel, "load", fake_load)
    return pipeline


def test_lifespan_builds_a_scorer_and_puts_it_on_app_state(stub_load: FakePipeline) -> None:
    """The scorer is created once at startup, not per request."""
    with TestClient(app):
        scorer = app.state.scorer

    assert isinstance(scorer, FraudDetectionModel)
    assert scorer.is_loaded is True


def test_get_scorer_returns_the_instance_from_app_state(stub_load: FakePipeline) -> None:
    """The dependency reads app.state rather than constructing anything."""
    with TestClient(app) as client:
        first = client.get("/api/v1/health")
        second = client.get("/api/v1/health")

    assert first.status_code == second.status_code == 200
    assert first.json()["model_loaded"] is True


def test_every_request_shares_one_scorer(stub_load: FakePipeline) -> None:
    """Deserialising the artifact per request would dominate latency, so the same
    instance must serve every request."""
    seen: list[int] = []

    def recording(request: Request) -> TransactionScorer:
        # Annotated as Request, not object: FastAPI resolves dependency parameters by
        # their annotation, and an unrecognised one becomes a query parameter instead
        # of the request.
        scorer = get_scorer(request)
        seen.append(id(scorer))
        return scorer

    app.dependency_overrides[get_scorer] = recording
    try:
        with TestClient(app) as client:
            for _ in range(3):
                client.get("/api/v1/health")
    finally:
        app.dependency_overrides.clear()

    assert len(seen) == 3
    assert len(set(seen)) == 1, "all three requests must share one scorer instance"


def test_middleware_order_is_what_the_comments_claim() -> None:
    """The order is load-bearing and was documented wrongly once already.

    Starlette inserts each new middleware at the front, so the stack is the reverse
    of registration order. Several behaviours depend on this: the size limit must
    sit outside anything that would hold the body, the batch counter must sit inside
    the size limit for its buffering to be bounded, and the security headers being
    innermost is why the 413 sets its own.
    """
    outermost_first = [m.cls.__name__ for m in app.user_middleware]

    assert outermost_first == [
        "CORSMiddleware",
        "BodySizeLimitMiddleware",
        "BatchCostMiddleware",
        "MetricsMiddleware",
        "RequestIdMiddleware",
        "SecurityHeadersMiddleware",
    ]


def test_the_size_limit_sits_outside_the_batch_counter() -> None:
    """Stated separately because it is the one that would be a vulnerability rather
    than an inconsistency: BatchCostMiddleware buffers the whole body, and is only
    safe because the size limit has already capped it."""
    order = [m.cls.__name__ for m in app.user_middleware]
    assert order.index("BodySizeLimitMiddleware") < order.index("BatchCostMiddleware")
