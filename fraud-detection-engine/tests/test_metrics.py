"""Prometheus metrics: what is exposed, and to whom."""

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import ValidationError

from app.core.config import Settings, settings


def _scrape(client: TestClient) -> str:
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    return response.text


def test_metrics_render_in_exposition_format(client: TestClient) -> None:
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert CONTENT_TYPE_LATEST.split(";")[0] in response.headers["content-type"]
    assert "# HELP" in response.text and "# TYPE" in response.text


def test_requests_are_counted_by_endpoint_and_status(client: TestClient) -> None:
    client.get("/api/v1/health")
    body = _scrape(client)
    assert 'fraud_http_requests_total{endpoint="/api/v1/health"' in body
    assert 'status="200"' in body


def test_prediction_duration_is_separate_from_request_duration(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """Model cost has to be distinguishable from framework overhead, otherwise a
    latency alert cannot say which one regressed."""
    client.post("/api/v1/predict", json=valid_transaction)
    body = _scrape(client)
    assert "fraud_prediction_duration_seconds_count" in body
    assert "fraud_http_request_duration_seconds_count" in body


def test_predictions_are_counted_by_risk_bucket(
    client: TestClient, loaded_model: object, valid_transaction: dict[str, float]
) -> None:
    """The risk distribution shifting is itself a drift signal (#23)."""
    loaded_model.fraud_probability = 0.9  # type: ignore[attr-defined]
    client.post("/api/v1/predict", json=valid_transaction)
    body = _scrape(client)
    assert 'fraud_predictions_total{is_fraud="true",risk_level="HIGH"}' in body


def test_model_state_is_exposed_as_gauges(client: TestClient) -> None:
    body = _scrape(client)
    assert "fraud_model_loaded" in body
    assert "fraud_model_info" in body


def test_unmatched_paths_do_not_create_unbounded_series(client: TestClient) -> None:
    """Labelling by raw path would let a caller grow the metric store without limit
    by hitting distinct URLs in a loop."""
    for i in range(5):
        client.get(f"/api/v1/does-not-exist-{i}")
    body = _scrape(client)
    assert "does-not-exist" not in body
    assert 'endpoint="unmatched"' in body


def test_metrics_are_absent_from_the_public_schema(client: TestClient) -> None:
    """Internal telemetry is not part of the published contract."""
    schema = client.get("/openapi.json").json()
    assert "/api/v1/metrics" not in schema["paths"]


def test_a_token_is_required_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "metrics_token", "s" * 32)

    assert client.get("/api/v1/metrics").status_code == 401

    wrong = client.get("/api/v1/metrics", headers={"Authorization": "Bearer wrong"})
    assert wrong.status_code == 401

    ok = client.get("/api/v1/metrics", headers={"Authorization": "Bearer " + "s" * 32})
    assert ok.status_code == 200


def test_the_401_does_not_leak_metrics(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "metrics_token", "s" * 32)
    response = client.get("/api/v1/metrics")
    assert "fraud_http_requests_total" not in response.text
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("env", ["staging", "production"])
def test_settings_require_a_metrics_token_outside_development(env: str) -> None:
    with pytest.raises(ValidationError, match="metrics_token is required"):
        Settings(
            env=env,
            cors_origins=["https://app.example.com"],
            model_sha256="b8" + "0" * 62,
            _env_file=None,
        )


def test_a_short_metrics_token_is_rejected() -> None:
    """A guessable token is not a control."""
    with pytest.raises(ValidationError):
        Settings(
            env="production",
            cors_origins=["https://app.example.com"],
            model_sha256="b8" + "0" * 62,
            metrics_token="short",
            _env_file=None,
        )
