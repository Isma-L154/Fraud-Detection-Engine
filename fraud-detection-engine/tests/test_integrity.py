"""Artifact integrity verification.

`joblib.load` is `pickle`: loading an artifact executes its contents. These tests
assert the service refuses an artifact that is not the one it was configured for,
and that the refusal happens BEFORE the load.
"""

import os
from pathlib import Path

import joblib
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.integrity import ArtifactIntegrityError, sha256_of, verify
from app.ml import model as model_module
from app.ml.model import FraudDetectionModel

from .conftest import FakePipeline


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "fraud_model.pkl"
    joblib.dump(FakePipeline(0.4), path)
    return path


def test_sha256_matches_a_known_value(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    # SHA-256 of b"abc", a published test vector.
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_verify_accepts_a_matching_digest(artifact: Path) -> None:
    verify(artifact, sha256_of(artifact))


def test_verify_is_case_and_whitespace_tolerant(artifact: Path) -> None:
    """A digest copied out of a log or a CI variable often carries either."""
    verify(artifact, f"  {sha256_of(artifact).upper()}  ")


def test_verify_rejects_a_tampered_artifact(artifact: Path) -> None:
    original = sha256_of(artifact)
    artifact.write_bytes(artifact.read_bytes() + b"\x00")

    with pytest.raises(ArtifactIntegrityError) as exc:
        verify(artifact, original)
    assert "Refusing to load" in str(exc.value)


def test_load_refuses_a_tampered_artifact(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    """The service-level path: startup must fail, not warn and continue."""
    expected = sha256_of(artifact)
    artifact.write_bytes(b"different bytes entirely")

    monkeypatch.setattr(model_module, "MODEL_PATH", artifact)
    monkeypatch.setattr(model_module.settings, "model_sha256", expected)

    model = FraudDetectionModel()
    with pytest.raises(ArtifactIntegrityError):
        model.load()
    assert model.is_loaded is False, "nothing may be loaded after a failed check"


def test_verification_happens_before_the_file_is_unpickled(
    monkeypatch: pytest.MonkeyPatch, artifact: Path
) -> None:
    """The ordering IS the control.

    Verifying after the load would be pointless: unpickling has already executed
    whatever the file contained. This asserts joblib.load is never reached when the
    digest does not match.
    """
    expected = sha256_of(artifact)
    artifact.write_bytes(b"tampered")

    called = []
    monkeypatch.setattr(model_module, "MODEL_PATH", artifact)
    monkeypatch.setattr(model_module.settings, "model_sha256", expected)
    monkeypatch.setattr(model_module.joblib, "load", lambda p: called.append(p))

    with pytest.raises(ArtifactIntegrityError):
        FraudDetectionModel().load()

    assert called == [], "joblib.load must not be reached on a digest mismatch"


def test_load_succeeds_when_the_digest_matches(
    monkeypatch: pytest.MonkeyPatch, artifact: Path
) -> None:
    monkeypatch.setattr(model_module, "MODEL_PATH", artifact)
    monkeypatch.setattr(model_module.settings, "model_sha256", sha256_of(artifact))

    model = FraudDetectionModel()
    model.load()
    assert model.is_loaded is True


def test_development_loads_without_a_digest_but_warns(
    monkeypatch: pytest.MonkeyPatch, artifact: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Exempt in development only, and never silently."""
    monkeypatch.setattr(model_module, "MODEL_PATH", artifact)
    monkeypatch.setattr(model_module.settings, "model_sha256", None)

    with caplog.at_level("WARNING"):
        FraudDetectionModel().load()

    assert "WITHOUT integrity verification" in caplog.text
    assert sha256_of(artifact) in caplog.text, "the warning must give the digest to configure"


@pytest.mark.parametrize("env", ["staging", "production"])
def test_settings_refuse_to_build_without_a_digest_outside_development(env: str) -> None:
    with pytest.raises(ValidationError, match="model_sha256 is required"):
        Settings(env=env, cors_origins=["https://app.example.com"], _env_file=None)


def test_development_settings_build_without_a_digest() -> None:
    settings = Settings(env="development", cors_origins=["http://localhost:3000"], _env_file=None)
    assert settings.model_sha256 is None


def test_a_crafted_artifact_would_execute_on_load(tmp_path: Path) -> None:
    """Why this control exists, demonstrated rather than asserted.

    A pickle's __reduce__ runs on load. This one calls os.getenv, which is harmless;
    the point is that the callable is chosen by whoever wrote the file, and it runs
    before anything inspects the object.
    """

    class Payload:
        def __reduce__(self) -> tuple[object, tuple[str, str]]:
            return (os.getenv, ("PATH", "unset"))

    crafted = tmp_path / "crafted.pkl"
    joblib.dump(Payload(), crafted)

    assert joblib.load(crafted) == os.getenv("PATH", "unset")

    # And the control refuses it, because its digest is not the configured one.
    with pytest.raises(ArtifactIntegrityError):
        verify(crafted, "0" * 64)
