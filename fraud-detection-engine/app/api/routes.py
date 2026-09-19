# API route definitions. Each handler is intentionally thin and only responsible for request validation and response formatting.
# Validation lives in Pydantic, business logic lives in the model loader. THIS IS THE "CONTROLLER" LAYER OF THE APPLICATION.

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from app.core.rate_limit import PREDICT_RATE_LIMIT, limiter
from app.ml.model import fraud_model
from app.schemas.transaction import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#---------------------------------------------------------------------------------

#Good for testing and debugging **It's not used in production**
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health_check():
    """
    Used by load balancers and monitoring tools to verify the service is up
    and the model is actually loaded in memory — not just that the server responds.
    """
    return HealthResponse(
        status="ok" if fraud_model.is_loaded else "degraded",
        model_loaded=fraud_model.is_loaded,
        model_version=fraud_model.version,
    )

#---------------------------------------------------------------------------------

# This is the main endpoint for fraud prediction. It expects a JSON payload that matches the TransactionRequest schema.
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
def predict(request: Request, transaction: TransactionRequest):
    """
    Receives a single transaction and returns a fraud assessment.
    Pydantic validates the input before this handler is ever called —
    any 422 errors are automatic and don't reach this function.

    `request` is unused by this handler but required: slowapi reads the client
    address off it to build the rate-limit key, and refuses to decorate a
    handler that does not accept one.
    """
    if not fraud_model.is_loaded:
        # This shouldn't happen in normal operation but guards against edge cases where the model failed to load at startup
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Try again later.",
        )
    start_time = time.perf_counter()

    try:
        result = fraud_model.predict(transaction.model_dump())
    except Exception as e:
        # Log the full error internally but never expose raw exception messages to the client
        # (Because they could contain sensitive info or be exploited by attackers)
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. The error has been logged.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"prediction | risk={result['risk_level']} "
        f"prob={result['fraud_probability']:.4f} "
        f"latency={latency_ms:.1f}ms"
    )

    return PredictionResponse(**result)
#---------------------------------------------------------------------------------

# This endpoint is a placeholder for triggering model retraining
# (This methods helps in the future when we want to implement a retraining pipeline. In production, this would likely publish a message to a queue rather than doing the work synchronously.)
@router.post(
    "/retrain",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger model retraining",
)
def retrain():
    """
    Placeholder for the retraining pipeline.
    In production this would publish a message to a queue (SQS, Celery)
    and return immediately — retraining is never synchronous in a live API.
    Returns 202 Accepted because the work happens asynchronously.
    """
    logger.info("Retraining requested")
    return {
        "message": "Retraining job queued.",
        "status": "accepted",
    }
