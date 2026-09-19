"""The synthetic dataset generator.

It exists so that clone -> train -> serve works without a 150 MB Kaggle download,
which is what made this project unreproducible. These tests assert it produces
something the real pipeline can actually consume.
"""

import pandas as pd

from scripts.make_sample_dataset import FRAUD_RATE, generate

EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


def test_schema_matches_what_the_pipeline_expects() -> None:
    """train.py drops Class and Time and feeds V1-V28 plus Amount. A sample missing
    any of those fails at training rather than here, which is a worse place."""
    frame = generate(rows=2_000)
    assert set(frame.columns) == set(EXPECTED_COLUMNS)


def test_dtypes_are_numeric() -> None:
    frame = generate(rows=1_000)
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)


def test_the_class_imbalance_is_represented() -> None:
    """Smoothing the imbalance away would make the sample useless for exercising
    class_weight="balanced", which is the one thing the pipeline does about it."""
    frame = generate(rows=50_000)
    assert 0 < frame["Class"].mean() <= FRAUD_RATE * 2


def test_both_classes_are_present_even_in_a_tiny_sample() -> None:
    """A stratified split raises if a class has fewer than two members."""
    frame = generate(rows=100)
    assert frame["Class"].nunique() == 2
    assert frame["Class"].sum() >= 2


def test_amount_respects_the_schema_bounds() -> None:
    """TransactionRequest caps Amount at 50,000 and forbids negatives."""
    amount = generate(rows=20_000)["Amount"]
    assert amount.min() >= 0
    assert amount.max() <= 50_000


def test_the_planted_signal_is_learnable() -> None:
    """Without a signal the run tells you nothing about whether the pipeline works,
    only that it did not crash."""
    frame = generate(rows=20_000)
    fraud = frame[frame["Class"] == 1]
    legit = frame[frame["Class"] == 0]
    assert fraud["V14"].mean() < legit["V14"].mean() - 3


def test_generation_is_deterministic() -> None:
    assert generate(rows=500).equals(generate(rows=500))
