"""The request contract.

TransactionRequest is the only real input control this service has, so these tests
assert what it rejects rather than only what it accepts.
"""

import pytest
from pydantic import ValidationError

from app.schemas.transaction import TransactionRequest


def test_accepts_a_valid_transaction(valid_transaction: dict[str, float]) -> None:
    tx = TransactionRequest(**valid_transaction)
    assert tx.Amount == 149.62
    assert tx.V1 == 0.0


@pytest.mark.parametrize("field", ["V1", "V14", "V28"])
def test_rejects_value_above_upper_bound(valid_transaction: dict[str, float], field: str) -> None:
    valid_transaction[field] = 30.01
    with pytest.raises(ValidationError) as exc:
        TransactionRequest(**valid_transaction)
    assert field in str(exc.value)


@pytest.mark.parametrize("field", ["V1", "V14", "V28"])
def test_rejects_value_below_lower_bound(valid_transaction: dict[str, float], field: str) -> None:
    valid_transaction[field] = -30.01
    with pytest.raises(ValidationError):
        TransactionRequest(**valid_transaction)


@pytest.mark.parametrize("boundary", [-30.0, 30.0])
def test_accepts_the_bounds_themselves(
    valid_transaction: dict[str, float], boundary: float
) -> None:
    """ge/le are inclusive — the boundary value itself must not be rejected."""
    valid_transaction["V1"] = boundary
    assert boundary == TransactionRequest(**valid_transaction).V1


def test_rejects_negative_amount(valid_transaction: dict[str, float]) -> None:
    valid_transaction["Amount"] = -0.01
    with pytest.raises(ValidationError):
        TransactionRequest(**valid_transaction)


def test_rejects_amount_above_cap(valid_transaction: dict[str, float]) -> None:
    valid_transaction["Amount"] = 50_000.01
    with pytest.raises(ValidationError):
        TransactionRequest(**valid_transaction)


def test_accepts_zero_amount(valid_transaction: dict[str, float]) -> None:
    """Documented as allowed: ge=0, and the validator explicitly lets 0 through."""
    valid_transaction["Amount"] = 0.0
    assert TransactionRequest(**valid_transaction).Amount == 0.0


def test_rejects_a_missing_field(valid_transaction: dict[str, float]) -> None:
    del valid_transaction["V17"]
    with pytest.raises(ValidationError) as exc:
        TransactionRequest(**valid_transaction)
    assert "V17" in str(exc.value)


def test_rejects_a_non_numeric_value(valid_transaction: dict[str, float]) -> None:
    valid_transaction["V1"] = "not a number"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        TransactionRequest(**valid_transaction)


def test_rejects_an_unexpected_field(valid_transaction: dict[str, float]) -> None:
    """extra="forbid" is what stops parameter pollution. Assert it actually forbids."""
    valid_transaction["is_admin"] = 1.0
    with pytest.raises(ValidationError) as exc:
        TransactionRequest(**valid_transaction)
    assert "is_admin" in str(exc.value)


def test_amount_is_rounded_to_two_decimals(valid_transaction: dict[str, float]) -> None:
    """Characterises CURRENT behaviour, which is silent coercion.

    The validator rounds rather than rejecting, so a client sending 10.999 is scored
    on 11.0 and never told. Issue #20 decides whether that becomes a rejection; this
    test exists so that change is visible rather than silent.
    """
    valid_transaction["Amount"] = 10.999
    assert TransactionRequest(**valid_transaction).Amount == 11.0
