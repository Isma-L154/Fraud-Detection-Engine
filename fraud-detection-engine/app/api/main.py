# Application factory. Handles startup, shutdown, and global configuration.
# This is the entry point that uvicorn runs.

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.batch_cost import BatchCostMiddleware
from app.api.body_limit import BodySizeLimitMiddleware
from app.api.metrics import MODEL_INFO, MODEL_LOADED, MetricsMiddleware
from app.api.middleware import SecurityHeadersMiddleware
from app.api.request_id import RequestIdMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.rate_limit import limiter
from app.ml.model import FraudDetectionModel

configure_logging(settings.log_level, json_output=settings.json_logs)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Runs startup logic before the app accepts requests, and cleanup on shutdown.
    Loading the model here guarantees it's in memory before the first request
    arrives — never load it lazily inside a request handler.
    """
    # Startup. The scorer is built here and stored on app.state, so handlers reach
    # it through a dependency rather than importing a module global.
    logger.info("Starting up in %s — loading model...", settings.env)
    scorer = FraudDetectionModel()
    scorer.load()
    app.state.scorer = scorer
    MODEL_LOADED.set(1 if scorer.is_loaded else 0)
    MODEL_INFO.labels(version=scorer.version).set(1)
    logger.info("Model ready. Accepting requests.")

    yield  # app is running and serving requests here

    MODEL_LOADED.set(0)
    logger.info("Shutting down.")


app = FastAPI(
    title="Fraud Detection Engine",
    description="Real-time credit card fraud detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
)

# slowapi reads the limiter off app.state at request time, so this has to happen
# after the app exists — and it has to be the same instance the routes decorate.
app.state.limiter = limiter
# slowapi types its handler against RateLimitExceeded rather than Exception, which
# is narrower than Starlette's signature. The call is correct at runtime.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Middleware order, outermost first, is the REVERSE of registration order: Starlette
# inserts each new one at the front of the stack, so the last registered runs first.
# Verified by inspecting app.user_middleware and by observing which headers a 413
# actually carries — not inferred.
#
#   CORS -> BodySizeLimit -> BatchCost -> Metrics -> RequestId -> SecurityHeaders -> handler
#
# Security headers are innermost, so they apply to everything the router produces,
# including its 404s, 422s and 500s. A response short-circuited further out never
# reaches this middleware, which is why BodySizeLimitMiddleware sets the headers on
# its own 413.
app.add_middleware(SecurityHeadersMiddleware, settings=settings)

# Correlation id, inside the size limit but outside everything that logs, so every
# line emitted while handling a request carries the same id.
app.add_middleware(RequestIdMiddleware)

# Inside the size limit and outside the request id. It therefore counts every
# response the router produces, but not the 413 the size limit short-circuits.
app.add_middleware(MetricsMiddleware)

# Counts batch items so the rate limiter can charge per transaction. It sits inside
# the size limit, which has already capped what this may buffer — that ordering is
# what makes buffering the body here safe rather than a denial-of-service vector.
app.add_middleware(BatchCostMiddleware)

# Registered after the rest so it sits outside them: an oversized body is refused
# before metrics, correlation or the handler has had to hold it. Only CORS is
# further out, which is what puts its headers on this middleware's 413.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

# Origins come from configuration so staging and production differ without a code
# change. CORS restrains browsers, not curl — it is never authorisation (#12).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    # The API consumes a JSON body and nothing else. "*" accepted any request header,
    # which is wider than anything this service reads. Add to this list when an
    # endpoint genuinely needs a header, rather than reopening it.
    allow_headers=["Content-Type", "Accept"],
)
app.include_router(router, prefix="/api/v1")
