"""API key authentication."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import API_KEY_HEADER
from app.core.auth import Consumer, identify, parse_keys
from app.core.config import settings

from .factories import API_KEY, CONSUMER, deployed_settings

KEYS = {"alpha": "a" * 32, "beta": "b" * 32}


@pytest.fixture
def authenticated(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Configure keys, and return headers that satisfy them."""
    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    return {API_KEY_HEADER: KEYS["alpha"]}


def test_identify_matches_a_configured_key() -> None:
    consumer = identify(KEYS["beta"], parse_keys(KEYS))
    assert consumer == Consumer(name="beta")


@pytest.mark.parametrize("presented", ["", "wrong", "a" * 31, "a" * 33, "A" * 32])
def test_identify_rejects_anything_else(presented: str) -> None:
    assert identify(presented, parse_keys(KEYS)) is None


def test_identify_returns_none_when_no_keys_are_configured() -> None:
    assert identify("anything", {}) is None


@pytest.mark.parametrize("path", ["/api/v1/predict", "/api/v1/predict/batch", "/api/v1/retrain"])
def test_protected_endpoints_refuse_without_a_key(
    client: TestClient, authenticated: dict[str, str], path: str
) -> None:
    assert client.post(path, json={}).status_code == 401


@pytest.mark.parametrize("presented", ["", "wrong-key", "a" * 31])
def test_a_bad_key_is_refused(
    client: TestClient, authenticated: dict[str, str], presented: str
) -> None:
    response = client.post("/api/v1/predict", json={}, headers={API_KEY_HEADER: presented})
    assert response.status_code == 401


def test_the_rejection_does_not_say_which_part_was_wrong(
    client: TestClient, authenticated: dict[str, str]
) -> None:
    """Absent, malformed and simply-wrong must be indistinguishable: telling them
    apart tells an attacker which part to work on."""
    absent = client.post("/api/v1/predict", json={})
    wrong = client.post("/api/v1/predict", json={}, headers={API_KEY_HEADER: "x" * 32})
    assert absent.status_code == wrong.status_code == 401
    assert absent.json() == wrong.json()


def test_a_valid_key_is_accepted(
    client: TestClient, authenticated: dict[str, str], valid_transaction: dict[str, float]
) -> None:
    response = client.post("/api/v1/predict", json=valid_transaction, headers=authenticated)
    assert response.status_code == 200


def test_authentication_runs_before_validation(
    client: TestClient, authenticated: dict[str, str]
) -> None:
    """An unauthenticated caller must not learn the shape of the schema by probing
    it. An invalid body with no key is 401, not 422."""
    assert client.post("/api/v1/predict", json={"nonsense": 1}).status_code == 401


def test_health_stays_unauthenticated(client: TestClient, authenticated: dict[str, str]) -> None:
    """A load balancer cannot present a credential, and liveness is not sensitive."""
    assert client.get("/api/v1/health").status_code == 200


def test_development_without_keys_serves_anonymously(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place this fails open, bounded by configuration: Settings refuses to
    build outside development without keys."""
    monkeypatch.setattr(settings, "api_keys", {})
    assert client.post("/api/v1/predict", json=valid_transaction).status_code == 200


@pytest.mark.parametrize("env", ["staging", "production"])
def test_settings_require_api_keys_outside_development(env: str) -> None:
    with pytest.raises(ValidationError, match="api_keys is required"):
        deployed_settings(env=env, api_keys={})


def test_a_short_api_key_is_rejected() -> None:
    """A guessable key is not a control."""
    with pytest.raises(ValidationError, match="shorter than 32"):
        deployed_settings(api_keys={"weak": "short"})


def test_duplicate_api_keys_are_rejected() -> None:
    """Two consumers sharing a key means neither can be revoked independently, and
    the rate limiter cannot tell them apart."""
    with pytest.raises(ValidationError, match="unique per consumer"):
        deployed_settings(api_keys={"one": API_KEY, "two": API_KEY})


def test_the_configured_consumer_is_usable() -> None:
    assert deployed_settings().api_keys[CONSUMER] == API_KEY
