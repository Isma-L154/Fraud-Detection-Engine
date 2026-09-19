"""Configuration validation.

The point of a typed Settings object is that a misconfiguration is a startup
failure. These tests assert it actually refuses, rather than accepting a value and
behaving unexpectedly later.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings

# A digest is required whenever env is not development, so the shared base carries
# one. The tests that assert that requirement build their Settings explicitly.
DIGEST = "b8" + "0" * 62
BASE = {
    "env": "production",
    "cors_origins": ["https://app.example.com"],
    "model_sha256": DIGEST,
}


def _settings(**overrides: object) -> Settings:
    """Build Settings from explicit values only.

    `_env_file=None` stops pydantic-settings reading a developer's real .env, which
    would make the result depend on the machine the tests run on.
    """
    return Settings(**{**BASE, **overrides}, _env_file=None)  # type: ignore[arg-type]


def test_accepts_a_valid_configuration() -> None:
    settings = _settings()
    assert settings.env == "production"
    assert settings.cors_origins == ["https://app.example.com"]
    assert settings.log_level == "INFO"


def test_fails_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV has no default. An unset value must stop the process, not be assumed."""
    monkeypatch.delenv("ENV", raising=False)
    with pytest.raises(ValidationError) as exc:
        Settings(cors_origins=["https://app.example.com"], model_sha256=DIGEST, _env_file=None)  # type: ignore[call-arg]
    assert "env" in str(exc.value).lower()


def test_fails_without_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    with pytest.raises(ValidationError):
        Settings(env="production", model_sha256=DIGEST, _env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["Development", "prod", "dev", "", "PRODUCTION"])
def test_rejects_an_unknown_environment(value: str) -> None:
    """The old code compared against the literal "development", so any other string —
    including a capitalisation difference — silently selected production behaviour.
    """
    with pytest.raises(ValidationError):
        _settings(env=value)


def test_rejects_wildcard_cors_origin() -> None:
    with pytest.raises(ValidationError, match="explicit origins"):
        _settings(cors_origins=["*"])


def test_rejects_empty_cors_origins() -> None:
    with pytest.raises(ValidationError, match="at least one origin"):
        _settings(cors_origins=[])


@pytest.mark.parametrize("value", ["TRACE", "verbose", "info"])
def test_rejects_an_unknown_log_level(value: str) -> None:
    with pytest.raises(ValidationError):
        _settings(log_level=value)


def test_rejects_an_unknown_setting() -> None:
    """extra="forbid" catches a typo'd variable name instead of ignoring it."""
    with pytest.raises(ValidationError):
        _settings(predict_rate_limitt="30/minute")


def test_docs_are_served_in_development_only() -> None:
    assert _settings(env="development", model_sha256=None).docs_enabled is True
    assert _settings(env="staging").docs_enabled is False
    assert _settings(env="production").docs_enabled is False


def test_model_path_defaults_inside_the_project() -> None:
    path = _settings().model_path
    assert path.is_absolute()
    assert path.parts[-2:] == ("models", "fraud_model.pkl")


def test_model_path_can_be_overridden() -> None:
    assert _settings(model_path=Path("/tmp/other.pkl")).model_path == Path("/tmp/other.pkl")


def test_log_format_defaults_by_environment() -> None:
    """Text where a human reads it, JSON where an aggregator does."""
    assert _settings(env="development", model_sha256=None).json_logs is False
    assert _settings(env="staging").json_logs is True
    assert _settings(env="production").json_logs is True


@pytest.mark.parametrize("explicit", [True, False])
def test_log_format_can_be_overridden_explicitly(explicit: bool) -> None:
    assert _settings(env="production", log_json=explicit).json_logs is explicit
    assert _settings(env="development", model_sha256=None, log_json=explicit).json_logs is explicit
