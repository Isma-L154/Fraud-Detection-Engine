"""The artifact envelope: provenance that travels with the model."""

from pathlib import Path

import joblib
import pytest

from app.ml import model as model_module
from app.ml.artifact import (
    UNKNOWN_VERSION,
    ArtifactMetadata,
    envelope,
    unpack,
)
from app.ml.model import FraudDetectionModel

from .conftest import FakePipeline


def test_metadata_reports_the_run_id_as_the_version() -> None:
    assert ArtifactMetadata(run_id="abc123").version == "abc123"


def test_metadata_without_a_run_id_is_unknown_not_invented() -> None:
    """Reporting a made-up version is worse than admitting there is none."""
    assert ArtifactMetadata().version == UNKNOWN_VERSION


def test_unpack_reads_the_envelope() -> None:
    pipeline = FakePipeline(0.3)
    meta = ArtifactMetadata(run_id="run-1", sklearn_version="1.8.0")
    artifact = unpack(envelope(pipeline, meta))
    assert artifact.pipeline is pipeline
    assert artifact.metadata.run_id == "run-1"


def test_unpack_accepts_a_bare_pipeline_from_before_the_envelope() -> None:
    """An artifact trained before this format is unlabelled, not invalid."""
    pipeline = FakePipeline(0.3)
    artifact = unpack(pipeline)
    assert artifact.pipeline is pipeline
    assert artifact.metadata.version == UNKNOWN_VERSION


def test_unpack_accepts_metadata_stored_as_a_dict() -> None:
    """joblib may return the mapping rather than the dataclass depending on how the
    artifact was written."""
    artifact = unpack(
        {"format_version": 1, "pipeline": FakePipeline(0.1), "metadata": {"run_id": "r9"}}
    )
    assert artifact.metadata.run_id == "r9"


def _write(path: Path, pipeline: object, meta: ArtifactMetadata | None = None) -> None:
    joblib.dump(envelope(pipeline, meta) if meta else pipeline, path)


def test_the_reported_version_comes_from_the_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The point of the issue: swapping the file must change the reported version,
    with no code edit."""
    first = tmp_path / "a.pkl"
    second = tmp_path / "b.pkl"
    _write(first, FakePipeline(0.1), ArtifactMetadata(run_id="run-aaa"))
    _write(second, FakePipeline(0.1), ArtifactMetadata(run_id="run-bbb"))
    monkeypatch.setattr(model_module.settings, "model_sha256", None)

    monkeypatch.setattr(model_module, "MODEL_PATH", first)
    model_a = FraudDetectionModel()
    model_a.load()

    monkeypatch.setattr(model_module, "MODEL_PATH", second)
    model_b = FraudDetectionModel()
    model_b.load()

    assert model_a.version == "run-aaa"
    assert model_b.version == "run-bbb"
    assert model_a.version != model_b.version


def test_predictions_carry_the_artifact_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid_transaction: dict[str, float]
) -> None:
    path = tmp_path / "m.pkl"
    _write(path, FakePipeline(0.2), ArtifactMetadata(run_id="run-xyz"))
    monkeypatch.setattr(model_module, "MODEL_PATH", path)
    monkeypatch.setattr(model_module.settings, "model_sha256", None)

    model = FraudDetectionModel()
    model.load()
    assert model.predict(valid_transaction)["model_version"] == "run-xyz"


def test_loading_an_unlabelled_artifact_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "legacy.pkl"
    _write(path, FakePipeline(0.2))
    monkeypatch.setattr(model_module, "MODEL_PATH", path)
    monkeypatch.setattr(model_module.settings, "model_sha256", None)

    with caplog.at_level("WARNING"):
        FraudDetectionModel().load()

    assert "no training metadata" in caplog.text
    assert "Retrain" in caplog.text


def test_metadata_records_the_sklearn_version_that_trained_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The old artifact carried no _sklearn_version, so unpickling it under a
    different scikit-learn produced no warning at all."""
    path = tmp_path / "m.pkl"
    _write(path, FakePipeline(0.2), ArtifactMetadata(run_id="r", sklearn_version="1.8.0"))
    monkeypatch.setattr(model_module, "MODEL_PATH", path)
    monkeypatch.setattr(model_module.settings, "model_sha256", None)

    model = FraudDetectionModel()
    model.load()
    assert model.metadata.sklearn_version == "1.8.0"
