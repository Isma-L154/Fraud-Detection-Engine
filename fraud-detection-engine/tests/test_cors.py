"""CORS behaviour, observed on real responses.

CORS restrains browsers, not `curl`, so none of this is authorisation — the point
of these tests is that the allowlist is actually an allowlist, not that it protects
the endpoint.
"""

from fastapi.testclient import TestClient

from app.core.config import settings

# Read from the running configuration rather than hardcoded. conftest.py uses
# setdefault, so a developer with CORS_ORIGINS already exported would otherwise see
# these fail for a reason that has nothing to do with the code.
ALLOWED = settings.cors_origins[0]
DISALLOWED = "https://evil.example.com"


def test_an_allowed_origin_is_echoed_back(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Origin": ALLOWED})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_a_disallowed_origin_gets_no_allow_origin_header(client: TestClient) -> None:
    """The header's absence is what makes the browser refuse the response.

    Note the request itself still succeeds — that is CORS working as designed, and
    exactly why it is not a substitute for authentication (#10).
    """
    response = client.get("/api/v1/health", headers={"Origin": DISALLOWED})
    assert "access-control-allow-origin" not in response.headers


def test_the_incoming_origin_is_never_reflected(client: TestClient) -> None:
    """Reflecting Origin back is equivalent to a wildcard, with the appearance of a
    policy."""
    for origin in (DISALLOWED, "null", "http://localhost:3001"):
        response = client.get("/api/v1/health", headers={"Origin": origin})
        assert response.headers.get("access-control-allow-origin") != origin


def test_preflight_from_an_allowed_origin_lists_only_the_expected_headers(
    client: TestClient,
) -> None:
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"]
    assert "*" not in allowed
    assert "content-type" in allowed.lower()


def test_preflight_from_a_disallowed_origin_is_refused(client: TestClient) -> None:
    response = client.options(
        "/api/v1/predict",
        headers={"Origin": DISALLOWED, "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_disallowed_methods_are_not_advertised(client: TestClient) -> None:
    response = client.options(
        "/api/v1/predict",
        headers={"Origin": ALLOWED, "Access-Control-Request-Method": "DELETE"},
    )
    assert "DELETE" not in response.headers.get("access-control-allow-methods", "")
