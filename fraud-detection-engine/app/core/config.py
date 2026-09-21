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

from pydantic import Field, field_validator, model_validator
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

    log_json: bool | None = Field(
        None,
        description=(
            "Emit logs as JSON. Defaults to plain text in development, where a human "
            "reads them, and JSON everywhere else, where an aggregator does. Set "
            "explicitly to override."
        ),
    )

    cors_origins: list[str] = Field(
        ...,
        description=(
            "Origins allowed to call the API from a browser. Required, so staging and "
            "production must each state their own rather than inheriting a default."
        ),
    )

    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.pkl"

    predict_rate_limit: str = "30/minute"

    retrain_rate_limit: str = Field(
        "2/hour",
        description=(
            "Much stricter than prediction. Retraining is the expensive path — "
            "unbounded CPU and a rewritten artifact — and nobody legitimately "
            "triggers it often."
        ),
    )

    auth_failure_rate_limit: str = Field(
        "10/minute",
        description=(
            "Budget for FAILED authentication attempts, per caller address. Only "
            "failures are counted: charging successes would throttle normal traffic "
            "twice, since predict_rate_limit already covers that. Without this an "
            "attacker guesses keys for free — the 401 is raised during dependency "
            "resolution, before the request limiter is reached."
        ),
    )

    trusted_proxies: list[str] = Field(
        default_factory=list,
        description=(
            "Peer addresses whose X-Forwarded-For header is believed. Empty by "
            "default: that header is client-controlled, so trusting it without "
            "knowing the request came through a real proxy lets anyone spoof an "
            "address per request and never be rate limited. Set this to the reverse "
            "proxy or load balancer addresses when one exists."
        ),
    )

    max_batch_size: int = Field(
        100,
        gt=0,
        le=1000,
        description=(
            "Largest batch accepted by /predict/batch. An unbounded batch is a "
            "denial-of-service vector: inference is CPU-bound, so one request could "
            "occupy a worker indefinitely. Raising it means raising "
            "max_request_bytes too — 100 transactions is roughly 70 KB."
        ),
    )

    max_request_bytes: int = Field(
        128 * 1024,
        gt=0,
        description=(
            "Largest request body accepted, in bytes. Default 128 KiB: a single "
            "/predict payload is about 709 bytes and a full batch of max_batch_size "
            "about 70 KB, so this leaves headroom for the larger of the two while "
            "keeping one request from costing more memory than the work it asks for. "
            "Raising max_batch_size means raising this with it."
        ),
    )

    decision_threshold: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Fraud probability at or above which a transaction is reported as fraud. "
            "Separate from the presentation buckets in app/core/risk.py: this is the "
            "operating point, they are labels. Measured on the 56,962-row holdout "
            "(98 frauds): at 0.30 precision 0.930 / recall 0.816 / F1 0.870, with 6 "
            "false positives and 18 missed frauds; at 0.50 precision 0.961 / recall "
            "0.755 / F1 0.846, with 3 false positives and 24 missed frauds. 0.30 "
            "catches six more frauds for three more reviews, which is the right trade "
            "when a review is cheap and a missed fraud is not."
        ),
    )

    model_sha256: str | None = Field(
        None,
        description=(
            "Expected SHA-256 of the model artifact, recorded out of band by whoever "
            "deploys it. Loading a .pkl executes its contents, so an unverified "
            "artifact is remote code execution at startup. Deliberately NOT read from "
            "a file beside the artifact: anyone able to rewrite the artifact could "
            "rewrite that too. `python notebooks/train.py` prints the value."
        ),
    )

    api_keys: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Consumer name -> API key. One key per consumer so a key can be revoked "
            "without affecting the others, and so the rate limiter has an identity "
            "to key on. Required outside development: the service would otherwise "
            "serve anyone who can reach the port. Example: "
            'API_KEYS={"settlement-batch":"<32+ chars>"}'
        ),
    )

    @field_validator("api_keys")
    @classmethod
    def reject_weak_api_keys(cls, keys: dict[str, str]) -> dict[str, str]:
        """A guessable key is not a control, and a shared one cannot be revoked."""
        for name, key in keys.items():
            if len(key) < 32:
                raise ValueError(f"api key for '{name}' is shorter than 32 characters")
        if len(set(keys.values())) != len(keys):
            raise ValueError("api keys must be unique per consumer")
        return keys

    @model_validator(mode="after")
    def require_api_keys_outside_development(self) -> "Settings":
        if self.env != "development" and not self.api_keys:
            raise ValueError(
                "api_keys is required when env is not 'development' — "
                "without it every endpoint is open to anyone who can reach the port"
            )
        return self

    metrics_token: str | None = Field(
        None,
        min_length=16,
        description=(
            "Bearer token for /metrics. Required outside development. Metrics are "
            "internal telemetry: published openly they leak traffic volume, latency "
            "and the fraud rate. This is deliberately a separate, narrow credential "
            "and does not pre-empt the API's authentication model (#10)."
        ),
    )

    @model_validator(mode="after")
    def require_metrics_token_outside_development(self) -> "Settings":
        if self.env != "development" and not self.metrics_token:
            raise ValueError(
                "metrics_token is required when env is not 'development' — "
                "/metrics leaks traffic volume, latency and the fraud rate"
            )
        return self

    @model_validator(mode="after")
    def require_artifact_digest_outside_development(self) -> "Settings":
        """The digest is mandatory anywhere the artifact is not built locally.

        Development is exempt because the artifact there is produced by the person
        running the service, on the machine running it, so there is no transport to
        protect. Everywhere else it arrives from somewhere and must be checked.
        """
        if self.env != "development" and not self.model_sha256:
            raise ValueError(
                "model_sha256 is required when env is not 'development' — "
                "loading an unverified artifact executes whatever it contains"
            )
        return self

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
    def json_logs(self) -> bool:
        """Resolved log format: explicit setting if given, else by environment."""
        if self.log_json is not None:
            return self.log_json
        return self.env != "development"

    @property
    def docs_enabled(self) -> bool:
        """Interactive API documentation is served outside production only.

        /docs and /redoc render HTML with CDN-loaded assets, which is a different
        exposure from the JSON API (#13).
        """
        return self.env == "development"


# Instantiated at import so an invalid configuration fails the process at startup.
settings = Settings()
