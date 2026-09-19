"""Endpoint behaviour, through the real FastAPI app."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from .conftest import ExplodingPipeline, FakePipeline


def test_health_reports_ok_when_the_model_is_loaded(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_health_reports_degraded_when_the_model_is_absent(unloaded_client: TestClient) -> None:
    """A health check that only proves the process is up is not a health check.

    This is the case a load balancer has to see: the server answers, but the service
    cannot do its job.
    """
    response = unloaded_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_predict_returns_an_assessment(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    response = client.post("/api/v1/predict", json=valid_transaction)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "is_fraud",
        "fraud_probability",
        "risk_level",
        "model_version",
        "decision_threshold",
    }
    assert isinstance(body["is_fraud"], bool)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_reflects_the_model_score(
    client: TestClient, loaded_model: FakePipeline, valid_transaction: dict[str, float]
) -> None:
    loaded_model.fraud_probability = 0.85
    body = client.post("/api/v1/predict", json=valid_transaction).json()
    assert body["fraud_probability"] == 0.85
    assert body["risk_level"] == "HIGH"
    assert body["is_fraud"] is True


def test_predict_rejects_a_malformed_body(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    valid_transaction["V1"] = 999.0
    response = client.post("/api/v1/predict", json=valid_transaction)
    assert response.status_code == 422


def test_predict_rejects_an_empty_body(client: TestClient) -> None:
    assert client.post("/api/v1/predict", json={}).status_code == 422


def test_predict_returns_503_when_the_model_is_absent(
    unloaded_client: TestClient, valid_transaction: dict[str, float]
) -> None:
    response = unloaded_client.post("/api/v1/predict", json=valid_transaction)

    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_predict_does_not_leak_exception_detail(
    make_client: Any, valid_transaction: dict[str, float]
) -> None:
    """An inference failure must not put the exception text in the response.

    The handler logs the traceback and returns a generic message; this asserts the
    synthetic failure message does not appear in the body.
    """
    client = make_client(ExplodingPipeline())
    response = client.post("/api/v1/predict", json=valid_transaction)

    assert response.status_code == 500
    assert "synthetic inference failure" not in response.text
    assert "ValueError" not in response.text


def test_retrain_reports_that_it_is_not_implemented(client: TestClient) -> None:
    """It previously returned 202 with "Retraining job queued" while queueing
    nothing, so a caller — or a monitor built on that response — had no way to know
    the work was never scheduled. 501 is the truthful answer."""
    response = client.post("/api/v1/retrain")
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/does-not-exist").status_code == 404


def test_rate_limit_rejects_the_thirty_first_request(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit, provoked rather than assumed.

    A configured limit that has never been triggered is unverified. The autouse
    fixture disables limiting for every other test; this one turns it back on and
    exhausts the budget.
    """
    from app.core.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    codes = [client.post("/api/v1/predict", json=valid_transaction).status_code for _ in range(31)]

    assert codes[:30] == [200] * 30, "the first 30 requests should be served"
    assert codes[30] == 429, "the 31st request should be rejected"


def test_error_responses_match_the_documented_schema(
    unloaded_client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """The OpenAPI schema must describe what the API actually returns.

    ErrorResponse is advertised on /predict for 422 and 503. It previously declared a
    `code` field that no handler ever populated, so the documented contract was a
    promise the service did not keep.
    """
    from app.schemas.transaction import ErrorResponse

    unavailable = unloaded_client.post("/api/v1/predict", json=valid_transaction)

    assert unavailable.status_code == 503
    # Every declared field is present in the real response.
    assert set(ErrorResponse.model_fields) <= set(unavailable.json())


def test_a_completely_different_scorer_can_be_substituted(
    valid_transaction: dict[str, float],
) -> None:
    """The handlers depend on the TransactionScorer protocol, not on
    FraudDetectionModel.

    This is what the dependency buys: a remote scoring service, a cache, or a
    different estimator can replace the model without editing a handler. The stand-in
    below shares no code with FraudDetectionModel — it is not a subclass, holds no
    sklearn pipeline, and never touches pandas.
    """
    from app.api.dependencies import get_scorer
    from app.api.main import app
    from app.ml.protocol import TransactionScorer

    class AlwaysFraud:
        is_loaded = True
        version = "stub-9.9.9"

        def predict(self, features: dict[str, float]) -> dict[str, object]:
            return {
                "is_fraud": True,
                "fraud_probability": 1.0,
                "risk_level": "HIGH",
                "model_version": self.version,
                "decision_threshold": 0.0,
            }

        def predict_many(self, rows: list[dict[str, float]]) -> list[dict[str, object]]:
            return [self.predict(row) for row in rows]

    stub = AlwaysFraud()
    assert isinstance(stub, TransactionScorer), "the stub must satisfy the protocol"

    app.dependency_overrides[get_scorer] = lambda: stub
    try:
        client = TestClient(app)
        body = client.post("/api/v1/predict", json=valid_transaction).json()
        health = client.get("/api/v1/health").json()
    finally:
        app.dependency_overrides.clear()

    assert body["is_fraud"] is True
    assert body["model_version"] == "stub-9.9.9"
    assert health["model_version"] == "stub-9.9.9"
