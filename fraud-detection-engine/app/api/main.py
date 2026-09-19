# Application factory. Handles startup, shutdown, and global configuration.
# This is the entry point that uvicorn runs.

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.middleware import SecurityHeadersMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.rate_limit import limiter
from app.ml.model import FraudDetectionModel

configure_logging(settings.log_level)
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
    logger.info("Model ready. Accepting requests.")

    yield  # app is running and serving requests here

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

# Security headers on every response, added before CORS so it runs outermost and
# cannot be skipped by a response short-circuited further in.
app.add_middleware(SecurityHeadersMiddleware, settings=settings)

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
