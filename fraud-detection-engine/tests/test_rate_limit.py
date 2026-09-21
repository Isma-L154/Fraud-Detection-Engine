"""Rate limiting: who gets charged, and whether it can be fooled."""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import API_KEY_HEADER
from app.core.config import settings
from app.core.rate_limit import FORWARDED_FOR, limiter, rate_limit_key

KEYS = {"alpha": "a" * 32, "beta": "b" * 32}


class FakeRequest:
    """The two things rate_limit_key reads."""

    def __init__(self, peer: str = "10.0.0.1", headers: dict[str, str] | None = None) -> None:
        self.client = type("C", (), {"host": peer})()
        self.headers = headers or {}
        self.state = type("S", (), {})()


def test_an_authenticated_caller_is_keyed_by_account() -> None:
    """IPs are shared behind carrier NAT and rotate freely, so a per-IP limit
    punishes the wrong people and fails to stop the right ones."""
    request = FakeRequest()
    request.state.consumer = type("C", (), {"name": "alpha"})()
    assert rate_limit_key(request) == "consumer:alpha"  # type: ignore[arg-type]


def test_an_unauthenticated_caller_falls_back_to_the_peer_address() -> None:
    assert rate_limit_key(FakeRequest(peer="203.0.113.7")) == "ip:203.0.113.7"  # type: ignore[arg-type]


def test_a_forwarded_header_from_an_untrusted_peer_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is the attack the old code was open to.

    X-Forwarded-For is client-controlled. Trusting it unconditionally lets anyone
    send a different address on every request and never be limited at all — worse
    than not reading it, because the limit then looks like a control while being
    none.
    """
    monkeypatch.setattr(settings, "trusted_proxies", [])
    spoofed = FakeRequest(peer="203.0.113.7", headers={FORWARDED_FOR: "1.2.3.4"})
    assert rate_limit_key(spoofed) == "ip:203.0.113.7"  # type: ignore[arg-type]


def test_a_forwarded_header_from_a_trusted_proxy_is_believed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", ["10.0.0.1"])
    proxied = FakeRequest(peer="10.0.0.1", headers={FORWARDED_FOR: "198.51.100.9, 10.0.0.1"})
    assert rate_limit_key(proxied) == "ip:198.51.100.9"  # type: ignore[arg-type]


def test_a_trusted_proxy_sending_no_forwarded_header_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", ["10.0.0.1"])
    assert rate_limit_key(FakeRequest(peer="10.0.0.1")) == "ip:10.0.0.1"  # type: ignore[arg-type]


def test_a_missing_client_does_not_crash() -> None:
    request = FakeRequest()
    request.client = None  # type: ignore[assignment]
    assert rate_limit_key(request) == "ip:unknown"  # type: ignore[arg-type]


def test_one_consumer_cannot_exhaust_another_budget(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of keying per account: a noisy consumer must not deny service to a
    quiet one sharing an address."""
    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    alpha = {API_KEY_HEADER: KEYS["alpha"]}
    beta = {API_KEY_HEADER: KEYS["beta"]}

    for _ in range(30):
        assert (
            client.post("/api/v1/predict", json=valid_transaction, headers=alpha).status_code == 200
        )

    assert (
        client.post("/api/v1/predict", json=valid_transaction, headers=alpha).status_code == 429
    ), "alpha has spent its budget"
    assert (
        client.post("/api/v1/predict", json=valid_transaction, headers=beta).status_code == 200
    ), "beta's budget is its own"


def test_retrain_is_limited_far_more_strictly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retraining is the expensive path: unbounded CPU and a rewritten artifact."""
    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    alpha = {API_KEY_HEADER: KEYS["alpha"]}
    codes = [client.post("/api/v1/retrain", headers=alpha).status_code for _ in range(4)]

    assert codes[:2] == [501, 501], "the first two are served (and report 501 by design)"
    assert codes[2] == 429, "the third exceeds the 2/hour limit"


def test_failed_authentication_is_throttled(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, an attacker guesses keys for free.

    require_consumer raises during FastAPI's dependency resolution, before slowapi's
    wrapper around the handler runs, so a 401 never reached the limiter at all: 60
    wrong keys produced 60 401s and consumed nothing.
    """
    from app.core.rate_limit import reset_auth_throttle

    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(settings, "auth_failure_rate_limit", "5/minute")
    reset_auth_throttle()

    wrong = {API_KEY_HEADER: "w" * 32}
    codes = [
        client.post("/api/v1/predict", json=valid_transaction, headers=wrong).status_code
        for _ in range(7)
    ]

    assert codes[:5] == [401] * 5, "the first five attempts are refused normally"
    assert codes[5] == 429, "the sixth exceeds the attempt budget"
    assert codes[6] == 429


def test_throttling_failures_does_not_penalise_a_valid_consumer(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong key must not lock out the caller who has the right one."""
    from app.core.rate_limit import limiter, reset_auth_throttle

    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(settings, "auth_failure_rate_limit", "3/minute")
    reset_auth_throttle()
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    for _ in range(4):
        client.post("/api/v1/predict", json=valid_transaction, headers={API_KEY_HEADER: "w" * 32})

    ok = client.post(
        "/api/v1/predict", json=valid_transaction, headers={API_KEY_HEADER: KEYS["alpha"]}
    )
    assert ok.status_code == 200


def test_a_successful_request_does_not_consume_the_attempt_budget(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only failures are counted. Charging successes would throttle normal traffic
    twice — the per-consumer limit already covers that."""
    from app.core.rate_limit import reset_auth_throttle

    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(settings, "auth_failure_rate_limit", "2/minute")
    reset_auth_throttle()

    good = {API_KEY_HEADER: KEYS["alpha"]}
    for _ in range(5):
        assert (
            client.post("/api/v1/predict", json=valid_transaction, headers=good).status_code == 200
        )

    # The budget is untouched, so a wrong key still gets its normal 401.
    wrong = client.post(
        "/api/v1/predict", json=valid_transaction, headers={API_KEY_HEADER: "w" * 32}
    )
    assert wrong.status_code == 401


def test_the_throttled_response_says_when_to_retry(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.rate_limit import reset_auth_throttle

    monkeypatch.setattr(settings, "api_keys", dict(KEYS))
    monkeypatch.setattr(settings, "auth_failure_rate_limit", "1/minute")
    reset_auth_throttle()

    wrong = {API_KEY_HEADER: "w" * 32}
    client.post("/api/v1/predict", json=valid_transaction, headers=wrong)
    throttled = client.post("/api/v1/predict", json=valid_transaction, headers=wrong)

    assert throttled.status_code == 429
    assert "Retry-After" in throttled.headers
