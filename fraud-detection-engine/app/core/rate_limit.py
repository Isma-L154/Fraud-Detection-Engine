# The application's single rate limiter.
#
# It lives here rather than in main.py or routes.py because both need the *same*
# instance: routes.py decorates handlers with it, main.py registers it on app.state
# so slowapi can find it at request time. Two separate Limiter objects — which is
# what this module replaced — decorate and enforce against different state.

import time

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
from slowapi import Limiter
from starlette.requests import Request

from app.core.config import settings

PREDICT_RATE_LIMIT = settings.predict_rate_limit

# Far stricter than prediction. Retraining is the expensive path — unbounded CPU
# and a rewritten artifact — and nobody legitimately triggers it often.
RETRAIN_RATE_LIMIT = settings.retrain_rate_limit

# Header a trusted reverse proxy uses to report the original client.
FORWARDED_FOR = "x-forwarded-for"


def rate_limit_key(request: Request) -> str:
    """Identify who to charge.

    Per account first. IPs are shared behind carrier NAT and rotate freely, so a
    per-IP limit punishes the wrong people and stops the wrong ones.

    Falling back to an address, X-Forwarded-For is honoured only when the immediate
    peer is a configured proxy. That header is client-controlled: trusting it
    unconditionally means anyone can spoof an address per request and never be
    limited at all — worse than not reading it, because the limit then looks like a
    control while being none.
    """
    consumer = getattr(request.state, "consumer", None)
    if consumer is not None:
        return f"consumer:{consumer.name}"

    peer = request.client.host if request.client else "unknown"
    if peer in settings.trusted_proxies:
        forwarded = request.headers.get(FORWARDED_FOR, "")
        client = forwarded.split(",")[0].strip()
        if client:
            # False positive: a Flask rule matching a function that is not a
            # route. This value is a rate-limit bucket key — it reaches
            # slowapi's storage, never a response body, so there is nothing
            # for XSS to land in. The rule id is one token and cannot be
            # wrapped, hence the noqa alongside it.
            # nosemgrep: python.flask.security.audit.directly-returned-format-string.directly-returned-format-string  # noqa: E501
            return f"ip:{client}"
    return f"ip:{peer}"


limiter = Limiter(key_func=rate_limit_key)


# Failed authentication needs its own budget, separate from slowapi's.
#
# require_consumer raises its 401 during FastAPI's dependency resolution, which
# happens before slowapi's wrapper around the handler ever runs — so a wrong key
# never reached the request limiter at all. Sixty wrong keys produced sixty 401s
# and consumed nothing.
#
# In-process storage, like the metrics registry: correct with one worker per
# container, which is what the Dockerfile runs and why.
_auth_storage = MemoryStorage()
_auth_throttle = FixedWindowRateLimiter(_auth_storage)
_AUTH_BUCKET = "auth-failure"


def charge_failed_auth(request: Request) -> bool:
    """Record a failed authentication attempt.

    Returns False once the budget for this caller is spent. Keyed by the same
    function as everything else, which for an unauthenticated request resolves to
    the caller's address — there is no account yet, which is the whole point.

    The limit is parsed per call rather than cached so that changing it in
    configuration takes effect; this runs only on the failure path.
    """
    return bool(
        _auth_throttle.hit(
            parse(settings.auth_failure_rate_limit), _AUTH_BUCKET, rate_limit_key(request)
        )
    )


def auth_retry_after(request: Request) -> int:
    """Seconds until this caller may try again."""
    reset_at, _ = _auth_throttle.get_window_stats(
        parse(settings.auth_failure_rate_limit), _AUTH_BUCKET, rate_limit_key(request)
    )
    return max(1, int(reset_at - time.time()))


def reset_auth_throttle() -> None:
    """Clear the attempt budget. For tests; nothing in the app calls it."""
    _auth_storage.reset()
