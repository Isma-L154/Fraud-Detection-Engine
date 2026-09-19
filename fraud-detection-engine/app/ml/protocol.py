"""The abstraction the API depends on.

Route handlers depend on this, never on `FraudDetectionModel`. That is what lets a
different estimator, a cached scorer, or a remote scoring service be substituted
without editing a handler — and what lets a test supply a fake without patching
module globals.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TransactionScorer(Protocol):
    """Something that can score a transaction for fraud."""

    @property
    def is_loaded(self) -> bool:
        """Whether this scorer is ready to serve predictions."""
        ...

    @property
    def version(self) -> str:
        """Identifier of the model behind this scorer."""
        ...

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Score one transaction."""
        ...
