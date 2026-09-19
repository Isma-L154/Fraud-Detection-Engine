# The application's single rate limiter.
#
# It lives here rather than in main.py or routes.py because both need the *same*
# instance: routes.py decorates handlers with it, main.py registers it on app.state
# so slowapi can find it at request time. Two separate Limiter objects — which is
# what this module replaces — decorate and enforce against different state.

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

# get_remote_address keys on the socket peer, which is wrong behind a proxy —
# tracked separately in issue #16.
PREDICT_RATE_LIMIT = settings.predict_rate_limit

limiter = Limiter(key_func=get_remote_address)
