"""The model wrapper: risk bucketing, feature ordering, and the unloaded guard."""

from pathlib import Path

import joblib
import pytest

from app.core import risk as risk_module
from app.ml import model as model_module
from app.ml.model import FraudDetectionModel

from .conftest import FakePipeline


def _model_returning(probability: float) -> FraudDetectionModel:
    model = FraudDetectionModel()
    model._pipeline = FakePipeline(probability)
    return model


def test_predict_raises_when_not_loaded() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        FraudDetectionModel().predict({})


def test_load_raises_when_the_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure must name the absolute path it checked, not a relative one."""
    missing = tmp_path / "absent" / "fraud_model.pkl"
    monkeypatch.setattr(model_module, "MODEL_PATH", missing)

    with pytest.raises(FileNotFoundError) as exc:
        FraudDetectionModel().load()

    assert str(missing) in str(exc.value)


def test_load_reads_the_artifact_from_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercises the real joblib round trip, with a trivial object rather than the
    3.5 MB production artifact."""
    artifact = tmp_path / "fraud_model.pkl"
    joblib.dump(FakePipeline(0.42), artifact)
    monkeypatch.setattr(model_module, "MODEL_PATH", artifact)

    model = FraudDetectionModel()
    assert model.is_loaded is False
    model.load()
    assert model.is_loaded is True


def test_is_loaded_reflects_state(fake_pipeline: FakePipeline) -> None:
    model = FraudDetectionModel()
    assert model.is_loaded is False
    model._pipeline = fake_pipeline
    assert model.is_loaded is True


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "LOW"),
        (0.29, "LOW"),
        (0.30, "MEDIUM"),  # LOW threshold is exclusive on the way up
        (0.69, "MEDIUM"),
        (0.70, "HIGH"),  # MEDIUM threshold is exclusive on the way up
        (1.0, "HIGH"),
    ],
)
def test_risk_level_boundaries(
    valid_transaction: dict[str, float], probability: float, expected: str
) -> None:
    """Every bucket edge, including the exact threshold values.

    The boundaries are where a change in the thresholds would silently reclassify
    transactions, so they are the values worth pinning.
    """
    result = _model_returning(probability).predict(valid_transaction)
    assert result["risk_level"] == expected


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, False), (0.2999, False), (0.30, True), (0.99, True)],
)
def test_is_fraud_uses_the_decision_threshold(
    valid_transaction: dict[str, float], probability: float, expected: bool
) -> None:
    """The decision boundary, at its configured default of 0.30."""
    result = _model_returning(probability).predict(valid_transaction)
    assert result["is_fraud"] is expected
    assert (probability >= model_module.settings.decision_threshold) is expected


def test_probability_is_rounded_to_four_decimals(valid_transaction: dict[str, float]) -> None:
    result = _model_returning(0.123456789).predict(valid_transaction)
    assert result["fraud_probability"] == 0.1235


def test_features_reach_the_pipeline_in_training_order(
    valid_transaction: dict[str, float],
) -> None:
    """Column order is load-bearing.

    The pipeline was fitted on V1..V28 then Amount. A DataFrame built from a dict in
    any other order scores the wrong feature against the wrong weight, silently — no
    exception, just a wrong answer. This asserts the ordering is preserved.
    """
    pipeline = FakePipeline(0.5)
    model = FraudDetectionModel()
    model._pipeline = pipeline

    # Feed the fields in reverse order to prove the wrapper reorders them.
    shuffled = dict(reversed(list(valid_transaction.items())))
    model.predict(shuffled)

    expected = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assert list(pipeline.calls[0].columns) == expected


def test_response_carries_the_model_version(valid_transaction: dict[str, float]) -> None:
    result = _model_returning(0.1).predict(valid_transaction)
    assert result["model_version"] == FraudDetectionModel().version


def test_decision_threshold_is_independent_of_the_risk_buckets(
    monkeypatch: pytest.MonkeyPatch, valid_transaction: dict[str, float]
) -> None:
    """The two used to be the same constant, so retuning one silently retuned the other.

    Moving the decision boundary must leave the presentation buckets exactly where
    they are.
    """
    monkeypatch.setattr(model_module.settings, "decision_threshold", 0.90)

    at_low = _model_returning(0.50).predict(valid_transaction)
    assert at_low["risk_level"] == "MEDIUM", "buckets must not move with the decision"
    assert at_low["is_fraud"] is False, "0.50 is below the new 0.90 decision boundary"

    at_high = _model_returning(0.95).predict(valid_transaction)
    assert at_high["risk_level"] == "HIGH"
    assert at_high["is_fraud"] is True


def test_risk_buckets_are_independent_of_the_decision_threshold(
    monkeypatch: pytest.MonkeyPatch, valid_transaction: dict[str, float]
) -> None:
    """And the reverse: moving a bucket edge must not move the fraud decision."""
    monkeypatch.setitem(risk_module.RISK_THRESHOLDS, "LOW", 0.05)

    result = _model_returning(0.20).predict(valid_transaction)
    assert result["risk_level"] == "MEDIUM", "0.20 is now above the moved LOW edge"
    assert result["is_fraud"] is False, "the decision boundary is still 0.30"


def test_response_states_the_threshold_that_produced_the_decision(
    valid_transaction: dict[str, float],
) -> None:
    """A fraud decision that cannot be attributed to a threshold cannot be audited."""
    result = _model_returning(0.4).predict(valid_transaction)
    assert result["decision_threshold"] == model_module.settings.decision_threshold


def test_risk_thresholds_contains_only_boundaries_that_are_read() -> None:
    """A threshold nobody reads invites someone to 'fix' the range by editing a
    number with no effect.

    RISK_THRESHOLDS carried a "HIGH": 1.01 entry described as a catch-all upper
    bound. risk_level checks LOW, then MEDIUM, then returns HIGH unconditionally —
    the entry was never read and the bound it documented did not exist.
    """
    assert set(risk_module.RISK_THRESHOLDS) == {"LOW", "MEDIUM"}


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.2999, 0.3, 0.5, 0.6999, 0.7, 0.9, 1.0])
def test_every_probability_maps_to_a_known_bucket(probability: float) -> None:
    """The buckets have to cover the whole domain a probability can take."""
    assert risk_module.risk_level(probability) in {"LOW", "MEDIUM", "HIGH"}
