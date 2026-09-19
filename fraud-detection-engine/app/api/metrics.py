"""Prometheus metrics.

Exposes what an operator of a fraud scorer actually needs to alert on: how much
traffic arrives, how long the model takes as distinct from framework overhead, how
the risk distribution is moving, and whether the loaded model is the expected one.

**Single process by design.** prometheus_client keeps counters in process memory,
so two uvicorn workers would serve two different halves of the truth depending on
which one a scrape reached. The alternative is multiprocess mode, which needs a
shared directory and cannot observe a worker that died. One worker per container,
scaled with replicas, is both simpler and the usual container practice — the
Dockerfile says so.
"""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

REQUESTS = Counter(
    "fraud_http_requests_total",
    "HTTP requests handled.",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "fraud_http_request_duration_seconds",
    "Wall time to handle a request, including framework overhead.",
    ["method", "endpoint"],
)

PREDICTION_DURATION = Histogram(
    "fraud_prediction_duration_seconds",
    "Time inside the model only, excluding framework overhead.",
    # A RandomForest scores in low tens of milliseconds; the defaults start at 5ms
    # and jump to 10ms, which puts nearly every observation in one bucket.
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Predictions served, by risk bucket. A shift in this distribution is itself a drift signal.",
    ["risk_level", "is_fraud"],
)

RATE_LIMITED = Counter(
    "fraud_rate_limited_total",
    "Requests rejected by the rate limiter.",
    ["endpoint"],
)

MODEL_LOADED = Gauge(
    "fraud_model_loaded",
    "1 when a model is loaded and the service can score, 0 otherwise.",
)

MODEL_INFO = Gauge(
    "fraud_model_info",
    "Always 1. The labels carry which model is running.",
    ["version"],
)


def _endpoint(request: Request) -> str:
    """The route template, not the concrete path.

    Using the raw path would create a new time series per distinct URL, which is
    unbounded cardinality — a caller hitting /api/v1/does-not-exist in a loop would
    grow the metric store without limit.
    """
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and duration."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = _endpoint(request)
        REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
        REQUEST_DURATION.labels(request.method, endpoint).observe(elapsed)
        if response.status_code == 429:
            RATE_LIMITED.labels(endpoint).inc()
        return response


def render() -> tuple[bytes, str]:
    """The exposition payload and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
