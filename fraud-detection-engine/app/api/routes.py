# API route definitions. Each handler is intentionally thin: request validation and
# response formatting only. Validation lives in Pydantic, business logic lives in the
# model loader. This is the controller layer.

import logging
import secrets
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from app.api import metrics
from app.api.dependencies import get_scorer, require_consumer
from app.core.auth import Consumer
from app.core.config import settings
from app.core.rate_limit import PREDICT_RATE_LIMIT, RETRAIN_RATE_LIMIT, limiter
from app.ml.protocol import TransactionScorer
from app.schemas.transaction import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _batch_cost(request: Request) -> int:
    """One unit per transaction, counted by BatchCostMiddleware before parsing."""
    return int(request.scope.get("state", {}).get("batch_size", 1))


# ---------------------------------------------------------------------------------


# Called by the container HEALTHCHECK in the Dockerfile, so this is a production
# endpoint, not a debugging convenience.
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health_check(scorer: TransactionScorer = Depends(get_scorer)) -> HealthResponse:
    """
    Used by load balancers and monitoring tools to verify the service is up
    and the model is actually loaded in memory — not just that the server responds.
    """
    return HealthResponse(
        status="ok" if scorer.is_loaded else "degraded",
        model_loaded=scorer.is_loaded,
        model_version=scorer.version,
    )


# ---------------------------------------------------------------------------------


# The main fraud prediction endpoint. Expects a JSON body matching TransactionRequest.
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a transaction for fraud",
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
@limiter.limit(PREDICT_RATE_LIMIT)  # Rate limit to prevent abuse
def predict(
    request: Request,
    transaction: TransactionRequest,
    scorer: TransactionScorer = Depends(get_scorer),
    consumer: Consumer = Depends(require_consumer),
) -> PredictionResponse:
    """
    Receives a single transaction and returns a fraud assessment.
    Pydantic validates the input before this handler is ever called —
    any 422 errors are automatic and don't reach this function.

    `request` is unused by this handler but required: slowapi reads the client
    address off it to build the rate-limit key, and refuses to decorate a
    handler that does not accept one.
    """
    if not scorer.is_loaded:
        # Should not happen in normal operation, but guards the case where the model
        # failed to load at startup.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Try again later.",
        )
    start_time = time.perf_counter()

    try:
        with metrics.PREDICTION_DURATION.time():
            result = scorer.predict(transaction.model_dump())
    except Exception as e:
        # Log the full error internally but never expose raw exception messages to the client
        # (Because they could contain sensitive info or be exploited by attackers)
        logger.error("prediction failed", exc_info=True)
        # `from e` keeps the original traceback chained for the logs without putting
        # any of it in the response.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. The error has been logged.",
        ) from e

    latency_ms = (time.perf_counter() - start_time) * 1000
    # Structured fields rather than an interpolated string, so this is queryable
    # without a regular expression. Deliberately NOT logged: the V1-V28 features and
    # the amount. V1-V28 are PCA components of real card transactions and the amount
    # with a timestamp is identifying — the decision is what is useful here, and the
    # request id is what ties it back to the caller.
    logger.info(
        "prediction",
        extra={
            "risk_level": result["risk_level"],
            "fraud_probability": round(result["fraud_probability"], 4),
            "is_fraud": result["is_fraud"],
            "decision_threshold": result["decision_threshold"],
            "model_version": result["model_version"],
            "latency_ms": round(latency_ms, 1),
            "consumer": consumer.name,
        },
    )

    metrics.PREDICTIONS.labels(result["risk_level"], str(result["is_fraud"]).lower()).inc()

    return PredictionResponse(**result)


# ---------------------------------------------------------------------------------


# Scores several transactions in one pass. sklearn's per-call overhead dominates a
# single-row prediction, so this is materially cheaper per transaction than N calls.
@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate several transactions for fraud",
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
@limiter.limit(PREDICT_RATE_LIMIT, cost=lambda request: _batch_cost(request))
def predict_batch(
    request: Request,
    batch: BatchTransactionRequest,
    scorer: TransactionScorer = Depends(get_scorer),
    consumer: Consumer = Depends(require_consumer),
) -> BatchPredictionResponse:
    """Score a batch, returning results in the submitted order.

    The rate limit charges one unit per transaction, not one per request —
    otherwise batching would be a trivial way to multiply throughput past the
    per-request limit by the batch size.
    """
    if not scorer.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Try again later.",
        )

    start_time = time.perf_counter()

    try:
        with metrics.PREDICTION_DURATION.time():
            results = scorer.predict_many([t.model_dump() for t in batch.transactions])
    except Exception as e:
        logger.error("batch prediction failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. The error has been logged.",
        ) from e

    latency_ms = (time.perf_counter() - start_time) * 1000
    for result in results:
        metrics.PREDICTIONS.labels(result["risk_level"], str(result["is_fraud"]).lower()).inc()

    logger.info(
        "batch prediction",
        extra={
            "count": len(results),
            "latency_ms": round(latency_ms, 1),
            "latency_ms_per_transaction": round(latency_ms / len(results), 3),
            "model_version": results[0]["model_version"] if results else "",
            "consumer": consumer.name,
        },
    )

    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
        count=len(results),
    )


# ---------------------------------------------------------------------------------


# Retraining is not implemented. The endpoint remains so the contract and its
# protection exist before the behaviour does — an unauthenticated POST that starts
# a real training run would be unbounded CPU, a rewritten artifact, and a direct
# path to poisoning the model that scores production traffic.
@router.post(
    "/retrain",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Trigger model retraining (not implemented)",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        501: {"model": ErrorResponse, "description": "Retraining is not implemented"},
    },
)
@limiter.limit(RETRAIN_RATE_LIMIT)
def retrain(
    request: Request,
    consumer: Consumer = Depends(require_consumer),
) -> None:
    """Report honestly that retraining does not happen here.

    This previously returned 202 with "Retraining job queued" while queueing
    nothing. A caller — or a monitor built on it — had no way to know the work was
    never scheduled. 501 is the truthful answer: the endpoint is recognised and
    deliberately unimplemented.

    When it is implemented it must be asynchronous, publishing to a queue rather
    than training in the request, and idempotent enough that a repeated call does
    not start a second concurrent run.
    """
    logger.warning("retrain requested but not implemented", extra={"consumer": consumer.name})
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Retraining is not implemented. Train offline with notebooks/train.py.",
    )


# ---------------------------------------------------------------------------------


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    include_in_schema=False,  # internal telemetry, not part of the public contract
)
def prometheus_metrics(authorization: str = Header(default="")) -> Response:
    """Exposition endpoint for a scraper.

    Requires a bearer token outside development. Traffic volume, latency
    distribution and the fraud rate are all commercially sensitive, and a public
    /metrics hands them to anyone.
    """
    expected = settings.metrics_token
    if expected:
        presented = authorization.removeprefix("Bearer ").strip()
        # compare_digest rather than ==: a plain comparison on a secret leaks its
        # length and prefix through timing.
        if not secrets.compare_digest(presented, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authorised.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)
