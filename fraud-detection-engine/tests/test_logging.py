"""Structured logging and request correlation."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.api.request_id import HEADER, _clean
from app.core.logging import JsonFormatter, RequestIdFilter, configure_logging
from app.core.request_context import request_id_var


def _record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_object_per_line() -> None:
    payload = json.loads(JsonFormatter().format(_record(request_id="abc123")))
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["request_id"] == "abc123"
    assert payload["timestamp"]


def test_json_formatter_keeps_structured_context() -> None:
    """Fields passed via `extra` must survive as fields, not be flattened into text."""
    payload = json.loads(
        JsonFormatter().format(_record(risk_level="HIGH", latency_ms=12.3, request_id="x"))
    )
    assert payload["risk_level"] == "HIGH"
    assert payload["latency_ms"] == 12.3


def test_json_formatter_includes_the_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("failed")
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_filter_attaches_the_current_request_id() -> None:
    token = request_id_var.set("req-42")
    try:
        record = _record()
        RequestIdFilter().filter(record)
        assert record.request_id == "req-42"
    finally:
        request_id_var.reset(token)


def test_filter_uses_a_placeholder_outside_a_request() -> None:
    """Startup and shutdown logs legitimately have no request."""
    record = _record()
    RequestIdFilter().filter(record)
    assert record.request_id == "-"


@pytest.mark.parametrize("value", ["abc123", "a" * 64, "with-dash_and_underscore"])
def test_a_safe_inbound_id_is_accepted(value: str) -> None:
    assert _clean(value) == value


@pytest.mark.parametrize(
    "value",
    ["a" * 65, "has space", "new\nline", "semi;colon", 'quote"', "<script>", ""],
)
def test_an_unsafe_inbound_id_is_discarded(value: str) -> None:
    """The id reaches log files. An unbounded, attacker-controlled string in a log
    line is how log injection and log flooding start — a rejected value is replaced
    with a generated one, never sanitised in place."""
    assert _clean(value) == ""


def test_the_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert len(response.headers[HEADER]) == 32  # uuid4().hex


def test_an_inbound_id_is_preserved_for_tracing(client: TestClient) -> None:
    """So a trace can span a proxy or an upstream service."""
    response = client.get("/api/v1/health", headers={HEADER: "upstream-trace-1"})
    assert response.headers[HEADER] == "upstream-trace-1"


def test_an_unsafe_inbound_id_is_replaced_not_echoed(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={HEADER: "bad value with spaces"})
    assert response.headers[HEADER] != "bad value with spaces"
    assert len(response.headers[HEADER]) == 32


def test_each_request_gets_its_own_id(client: TestClient) -> None:
    ids = {client.get("/api/v1/health").headers[HEADER] for _ in range(5)}
    assert len(ids) == 5


def test_the_prediction_log_never_carries_transaction_values(
    client: TestClient, valid_transaction: dict[str, float], caplog: pytest.LogCaptureFixture
) -> None:
    """V1-V28 are PCA components of real card transactions and the amount with a
    timestamp is identifying. The decision is loggable; the input is not."""
    distinctive = 12.34
    valid_transaction["Amount"] = distinctive
    valid_transaction["V7"] = -7.77

    with caplog.at_level(logging.INFO):
        assert client.post("/api/v1/predict", json=valid_transaction).status_code == 200

    assert "12.34" not in caplog.text
    assert "-7.77" not in caplog.text
    assert "prediction" in caplog.text


def test_the_prediction_log_carries_the_decision_as_fields(
    client: TestClient, valid_transaction: dict[str, float], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        client.post("/api/v1/predict", json=valid_transaction)

    entry = next(r for r in caplog.records if r.message == "prediction")
    assert entry.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(entry.latency_ms, float)
    assert hasattr(entry, "model_version")


def test_configure_logging_accepts_both_formats() -> None:
    """Text in development for a human, JSON elsewhere for an aggregator."""
    configure_logging("INFO", json_output=False)
    configure_logging("INFO", json_output=True)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)
