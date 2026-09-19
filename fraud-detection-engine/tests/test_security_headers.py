"""Security response headers.

Asserted on real responses rather than by reading the configuration, which is the
standard the baseline sets: a header that was never observed is not a control.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import API_CSP, BASE_HEADERS, DOCS_CSP


@pytest.mark.parametrize(("header", "expected"), list(BASE_HEADERS.items()))
def test_base_headers_are_present_on_a_successful_response(
    client: TestClient, header: str, expected: str
) -> None:
    response = client.get("/api/v1/health")
    assert response.headers[header] == expected


@pytest.mark.parametrize(("header", "expected"), list(BASE_HEADERS.items()))
def test_base_headers_are_present_on_an_error_response(
    client: TestClient, header: str, expected: str
) -> None:
    """An error response is still a response. Headers must not depend on the path
    a request happened to take."""
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422
    assert response.headers[header] == expected


def test_headers_are_present_on_a_404(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_json_responses_get_the_restrictive_policy(client: TestClient) -> None:
    csp = client.get("/api/v1/health").headers["Content-Security-Policy"]
    assert csp == API_CSP
    assert "default-src 'none'" in csp
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp


def test_the_api_policy_forbids_framing_and_base_rewriting(client: TestClient) -> None:
    csp = client.get("/api/v1/health").headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'none'" in csp


def test_docs_get_the_wider_policy_and_still_render(client: TestClient) -> None:
    """The docs pages need Swagger's CDN and inline bootstrap. They are served in
    development only, which is what makes that acceptable."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["Content-Security-Policy"] == DOCS_CSP
    assert "cdn.jsdelivr.net" in response.headers["Content-Security-Policy"]


def test_no_csp_anywhere_permits_eval(client: TestClient) -> None:
    for path in ("/api/v1/health", "/docs", "/openapi.json"):
        csp = client.get(path).headers["Content-Security-Policy"]
        assert "unsafe-eval" not in csp, f"{path} permits eval"


def test_hsts_is_absent_in_development(client: TestClient) -> None:
    """Sending HSTS from a plaintext dev server makes the browser refuse http://
    for this host, and it persists past the session."""
    assert "Strict-Transport-Security" not in client.get("/api/v1/health").headers


@pytest.mark.parametrize("env", ["staging", "production"])
def test_hsts_is_sent_outside_development(env: str) -> None:
    """Exercised against a minimal app rather than the real one, so the middleware's
    own branch is tested without rebuilding the service for each environment."""
    from fastapi import FastAPI

    from app.api.middleware import SecurityHeadersMiddleware
    from app.core.config import Settings

    settings = Settings(
        env=env,
        cors_origins=["https://app.example.com"],
        model_sha256="b8" + "0" * 62,
        metrics_token="t" * 32,
        _env_file=None,
    )
    probe = FastAPI()
    probe.add_middleware(SecurityHeadersMiddleware, settings=settings)

    @probe.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    response = TestClient(probe).get("/ping")
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
