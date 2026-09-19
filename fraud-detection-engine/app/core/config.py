"""Typed application configuration.

Every environment-dependent value comes from here. Nothing in `app/` reads
`os.getenv` directly: an untyped lookup with an implicit default is how a
development setting reaches production unnoticed.

Validation happens at import. An invalid or missing required value stops the
process at startup rather than producing a service that runs with the wrong
settings and says nothing.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> app/core -> app -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings, read from the environment and from `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    env: Environment = Field(
        ...,
        description=(
            "Deployment environment. Required with no default: the previous code read "
            "os.getenv('ENV') and treated anything that was not exactly 'development' — "
            "including unset, and including a capitalisation difference — as production. "
            "Making it explicit means a misconfiguration is a startup failure, not a "
            "silent change of behaviour."
        ),
    )

    log_level: LogLevel = "INFO"

    cors_origins: list[str] = Field(
        ...,
        description=(
            "Origins allowed to call the API from a browser. Required, so staging and "
            "production must each state their own rather than inheriting a default."
        ),
    )

    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.pkl"

    predict_rate_limit: str = "30/minute"

    @field_validator("cors_origins")
    @classmethod
    def reject_wildcard_origin(cls, origins: list[str]) -> list[str]:
        """A wildcard origin is never correct for this service.

        CORS restrains browsers, not `curl`, so it is not authorisation — but
        widening it to `*` removes the only browser-side restriction there is, for
        an API that has no authentication yet (#10).
        """
        if not origins:
            raise ValueError("cors_origins must list at least one origin")
        if "*" in origins:
            raise ValueError("cors_origins must not contain '*' — list explicit origins")
        return origins

    @property
    def docs_enabled(self) -> bool:
        """Interactive API documentation is served outside production only.

        /docs and /redoc render HTML with CDN-loaded assets, which is a different
        exposure from the JSON API (#13).
        """
        return self.env == "development"


# Instantiated at import so an invalid configuration fails the process at startup.
settings = Settings()
