"""FastAPI dependencies.

The scorer is created once during the application lifespan and stored on
`app.state`. Handlers receive it through `Depends`, which means a test replaces it
with `app.dependency_overrides` rather than patching a module attribute.
"""

from fastapi import Request

from app.ml.protocol import TransactionScorer


def get_scorer(request: Request) -> TransactionScorer:
    """Return the scorer this application was started with."""
    scorer: TransactionScorer = request.app.state.scorer
    return scorer
