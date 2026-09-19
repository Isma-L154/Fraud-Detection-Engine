# The application's single rate limiter.
#
# It lives here rather than in main.py or routes.py because both need the *same*
# instance: routes.py decorates handlers with it, main.py registers it on app.state
# so slowapi can find it at request time. Two separate Limiter objects — which is
# what this module replaced — decorate and enforce against different state.

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
            return f"ip:{client}"
    return f"ip:{peer}"


limiter = Limiter(key_func=rate_limit_key)
