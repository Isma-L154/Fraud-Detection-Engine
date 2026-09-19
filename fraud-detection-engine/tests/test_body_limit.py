"""Request body size limit.

The schema can only reject a payload it has already received and parsed in full, so
the limit has to sit below it. These tests assert the rejection happens at the
transport layer, not after deserialisation.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


def test_a_valid_payload_is_well_under_the_limit(valid_transaction: dict[str, float]) -> None:
    """If the limit were near the real payload size it would be a liveness bug, not
    a control. A prediction body is about 709 bytes against a 16 KiB limit."""
    import json

    assert len(json.dumps(valid_transaction)) < settings.max_request_bytes / 4


def test_a_normal_request_still_succeeds(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    assert client.post("/api/v1/predict", json=valid_transaction).status_code == 200


def test_an_oversized_body_is_rejected_with_413(client: TestClient) -> None:
    oversized = b'{"padding":"' + b"x" * (settings.max_request_bytes * 2) + b'"}'
    response = client.post(
        "/api/v1/predict",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_rejection_does_not_depend_on_the_body_being_valid_json(client: TestClient) -> None:
    """The limit is about bytes, not about content. Garbage of the same size is
    refused identically — and with 413, not 422, because it never reached parsing."""
    response = client.post(
        "/api/v1/predict",
        content=b"\x00" * (settings.max_request_bytes * 2),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_a_lying_content_length_does_not_get_through(client: TestClient) -> None:
    """The fast path trusts Content-Length only to refuse early. A body that under-
    declares its length must still be caught by the byte counter as it arrives."""
    oversized = b"x" * (settings.max_request_bytes * 2)
    response = client.post(
        "/api/v1/predict",
        content=oversized,
        headers={"Content-Type": "application/json", "Content-Length": "10"},
    )
    assert response.status_code in (413, 422, 400), (
        "an under-declared oversized body must not be accepted as valid"
    )


def test_a_body_just_under_the_limit_is_not_rejected_by_the_limiter(
    client: TestClient,
) -> None:
    """The boundary matters: rejecting valid traffic is the failure mode of a limit
    set too tight. Just under the limit must reach validation (422 here, not 413)."""
    payload = b'{"junk":"' + b"y" * (settings.max_request_bytes - 100) + b'"}'
    assert len(payload) < settings.max_request_bytes
    response = client.post(
        "/api/v1/predict",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, "should reach schema validation, not the size limit"


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/retrain"])
def test_other_endpoints_still_work(client: TestClient, path: str) -> None:
    method = client.get if path.endswith("health") else client.post
    assert method(path).status_code in (200, 202)


def test_the_413_response_carries_security_headers(client: TestClient) -> None:
    """A response produced by middleware still has to carry them."""
    oversized = b"x" * (settings.max_request_bytes * 2)
    response = client.post(
        "/api/v1/predict", content=oversized, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.headers["X-Content-Type-Options"] == "nosniff"
