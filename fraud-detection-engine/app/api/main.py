# Application factory. Handles startup, shutdown, and global configuration.
# This is the entry point that uvicorn runs.

import logging
import logging.config
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.ml.model import fraud_model
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Structured logging config — outputs consistent format across all modules
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
})
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the app accepts requests, and cleanup on shutdown.
    Loading the model here guarantees it's in memory before the first request
    arrives — never load it lazily inside a request handler.
    """
    #Startup 
    logger.info("Starting up — loading model...")
    fraud_model.load()
    logger.info("Model ready. Accepting requests.")

    yield  # app is running and serving requests here

    logger.info("Shutting down.")

app = FastAPI(
    title="Fraud Detection Engine",
    description="Real-time credit card fraud detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENV") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENV") == "development" else None,
)

# CORS middleware — adjust origins as needed for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://98.90.203.9:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")