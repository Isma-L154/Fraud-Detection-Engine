"""Batch prediction."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


def _batch(tx: dict[str, float], n: int) -> dict[str, object]:
    return {"transactions": [dict(tx) for _ in range(n)]}


def test_a_batch_is_scored_and_returned_in_order(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """Results must line up with the submitted transactions; a caller matches them
    by position."""
    transactions = []
    for amount in (10.0, 20.0, 30.0):
        tx = dict(valid_transaction)
        tx["Amount"] = amount
        transactions.append(tx)

    response = client.post("/api/v1/predict/batch", json={"transactions": transactions})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_batch_results_match_single_predictions(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """The two paths share one scoring implementation; if they ever disagree, one of
    them is wrong."""
    single = client.post("/api/v1/predict", json=valid_transaction).json()
    batched = client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 1)).json()
    assert batched["predictions"][0] == single


def test_an_empty_batch_is_rejected(client: TestClient) -> None:
    assert client.post("/api/v1/predict/batch", json={"transactions": []}).status_code == 422


def test_a_batch_over_the_cap_is_rejected(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """An unbounded batch is a denial-of-service vector: inference is CPU-bound, so
    one request could occupy a worker indefinitely."""
    oversized = _batch(valid_transaction, settings.max_batch_size + 1)
    assert client.post("/api/v1/predict/batch", json=oversized).status_code == 422


def test_a_batch_at_exactly_the_cap_is_accepted(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    at_cap = _batch(valid_transaction, settings.max_batch_size)
    response = client.post("/api/v1/predict/batch", json=at_cap)
    assert response.status_code == 200
    assert response.json()["count"] == settings.max_batch_size


def test_one_invalid_transaction_rejects_the_whole_batch(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """The documented contract. A partially-applied batch is harder to reason about
    than a rejected one, and the 422 names the offending index."""
    payload = _batch(valid_transaction, 3)
    payload["transactions"][1]["V1"] = 999.0  # type: ignore[index]

    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 422
    assert "1" in str(response.json()["detail"])


def test_the_batch_endpoint_returns_503_when_the_model_is_absent(
    unloaded_client: TestClient, valid_transaction: dict[str, float]
) -> None:
    response = unloaded_client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 2))
    assert response.status_code == 503


def test_batching_cannot_bypass_the_rate_limit(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the cost middleware exists.

    With a per-request cost, 30 batches of 100 would score 3,000 transactions
    against a 30-per-minute limit. Charging per transaction means a single batch of
    30 exhausts the same budget as 30 single requests.
    """
    from app.core.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    first = client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 30))
    assert first.status_code == 200, "a batch of exactly the budget is served"

    second = client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 1))
    assert second.status_code == 429, "the budget is spent; one more transaction is refused"


def test_a_small_batch_is_charged_proportionally(
    client: TestClient, valid_transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Charging every batch the maximum would be conservative but punish small ones."""
    from app.core.rate_limit import limiter

    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    # Six batches of five is exactly the 30-per-minute budget.
    for i in range(6):
        response = client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 5))
        assert response.status_code == 200, f"batch {i + 1} of 6 should fit the budget"

    # The seventh does not.
    assert (
        client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 1)).status_code == 429
    )


def test_batch_metrics_count_every_transaction(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 4))
    body = client.get("/api/v1/metrics").text
    assert "fraud_predictions_total" in body


def test_the_batch_error_path_does_not_leak_exception_detail(
    make_client: object, valid_transaction: dict[str, float]
) -> None:
    """Same containment as the single endpoint: the traceback is logged, not sent."""
    from .conftest import ExplodingPipeline

    client = make_client(ExplodingPipeline())  # type: ignore[operator]
    response = client.post("/api/v1/predict/batch", json=_batch(valid_transaction, 3))

    assert response.status_code == 500
    assert "synthetic inference failure" not in response.text


def test_predict_many_returns_nothing_for_an_empty_list(fake_pipeline: object) -> None:
    """Guarded so an empty list does not reach pandas, which would raise on the
    column selection rather than return nothing."""
    from app.ml.model import FraudDetectionModel

    model = FraudDetectionModel()
    model._pipeline = fake_pipeline
    assert model.predict_many([]) == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"transactions": [1, 2, 3]}', 3),
        (b'{"transactions": []}', 1),
        (b"not json at all", 1),
        (b"[1, 2, 3]", 1),
        (b"", 1),
        (b'{"transactions": "not a list"}', 1),
        (b"\xff\xfe invalid utf8", 1),
    ],
)
def test_batch_cost_counts_or_falls_back_to_one(body: bytes, expected: int) -> None:
    """A body that cannot be counted is charged one unit and then rejected by the
    schema anyway. Guessing higher would let malformed input consume someone
    else's quota."""
    from app.api.batch_cost import _count

    assert _count(body) == expected


def test_a_chunked_batch_body_is_reassembled(
    client: TestClient, valid_transaction: dict[str, float]
) -> None:
    """The middleware buffers across http.request messages; a body arriving in
    pieces must still be counted and replayed intact."""
    import json

    payload = json.dumps(_batch(valid_transaction, 3)).encode()

    def chunks() -> object:
        for i in range(0, len(payload), 64):
            yield payload[i : i + 64]

    response = client.post(
        "/api/v1/predict/batch",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_batch_cost_handles_a_disconnect_and_a_second_receive() -> None:
    """Two ASGI paths a TestClient cannot reach.

    A client that disconnects mid-body sends http.disconnect instead of a further
    http.request — the middleware must stop reading rather than block. And an
    application that calls receive() again after the replayed body must get the
    real channel, not a second copy of the body.
    """
    import asyncio

    from app.api.batch_cost import BatchCostMiddleware

    messages = [
        {"type": "http.request", "body": b'{"transactions": [1]}', "more_body": True},
        {"type": "http.disconnect"},
        {"type": "http.request", "body": b"after", "more_body": False},
    ]
    seen: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return messages.pop(0)

    async def app(scope: dict, recv: object, send: object) -> None:
        seen.append(await recv())  # the replayed body
        seen.append(await recv())  # falls through to the real channel

    async def send(message: dict) -> None:  # pragma: no cover - unused
        raise AssertionError("send should not be called")

    scope = {"type": "http", "path": "/api/v1/predict/batch"}
    asyncio.run(BatchCostMiddleware(app)(scope, receive, send))

    assert scope["state"]["batch_size"] == 1, "counted the body read before the disconnect"
    assert seen[0]["body"] == b'{"transactions": [1]}'
    assert seen[1]["body"] == b"after", "a second receive reaches the real channel"
