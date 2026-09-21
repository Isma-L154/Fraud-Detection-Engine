"""FastAPI dependencies.

The scorer is created once during the application lifespan and stored on
`app.state`. Handlers receive it through `Depends`, which means a test replaces it
with `app.dependency_overrides` rather than patching a module attribute.
"""

from fastapi import Header, HTTPException, Request, status

from app.core.auth import Consumer, identify, parse_keys
from app.core.config import settings
from app.core.rate_limit import auth_retry_after, charge_failed_auth
from app.ml.protocol import TransactionScorer

API_KEY_HEADER = "X-API-Key"

# Development only. Named so it is obvious in a log line that nothing authenticated.
ANONYMOUS = Consumer(name="anonymous")


def get_scorer(request: Request) -> TransactionScorer:
    """Return the scorer this application was started with."""
    scorer: TransactionScorer = request.app.state.scorer
    return scorer


def require_consumer(
    request: Request, x_api_key: str = Header(default="", alias=API_KEY_HEADER)
) -> Consumer:
    """Identify the caller, or refuse.

    When no keys are configured — development only, since Settings refuses to build
    otherwise — every caller is ANONYMOUS. That is the one place this fails open,
    and it is bounded by configuration rather than by a runtime check that could be
    wrong.
    """
    keys = parse_keys(settings.api_keys)
    if not keys:
        request.state.consumer = ANONYMOUS
        return ANONYMOUS

    consumer = identify(x_api_key, keys)
    if consumer is None:
        # Charge the attempt before refusing. This raises before slowapi's wrapper
        # around the handler runs, so without a budget of its own a wrong key costs
        # the caller nothing and can be retried without limit.
        if not charge_failed_auth(request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts.",
                headers={"Retry-After": str(auth_retry_after(request))},
            )

        # The same response whether the header was absent, malformed or simply
        # wrong: distinguishing them tells an attacker which part to work on.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorised.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # The rate limiter reads this to charge per account rather than per IP (#16).
    request.state.consumer = consumer
    return consumer
