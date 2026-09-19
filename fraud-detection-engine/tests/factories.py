"""Builders for test fixtures.

`Settings` grows a required field every time a control is added — the artifact
digest, the metrics token, now the API keys — and each one broke every test that
constructed a non-development Settings by hand. Building them in one place means
the next required field is one edit rather than fifteen.

That the additions broke tests is the validator working; having to fix them in
fifteen places was the test suite's problem, not the validator's.
"""

from typing import Any

from app.core.config import Settings

# Values that satisfy the validators without meaning anything.
DIGEST = "b8" + "0" * 62
METRICS_TOKEN = "t" * 32
API_KEY = "k" * 32
CONSUMER = "test-consumer"


def deployed_settings(**overrides: Any) -> Settings:
    """A Settings for staging or production, with every required field supplied.

    `_env_file=None` stops pydantic-settings reading a developer's real .env, which
    would make results depend on the machine the tests run on.
    """
    values: dict[str, Any] = {
        "env": "production",
        "cors_origins": ["https://app.example.com"],
        "model_sha256": DIGEST,
        "metrics_token": METRICS_TOKEN,
        "api_keys": {CONSUMER: API_KEY},
    }
    values.update(overrides)
    return Settings(**values, _env_file=None)


def development_settings(**overrides: Any) -> Settings:
    """A Settings for development, where the deployed-only fields are exempt."""
    values: dict[str, Any] = {
        "env": "development",
        "cors_origins": ["http://localhost:3000"],
    }
    values.update(overrides)
    return Settings(**values, _env_file=None)
