"""Request correlation.

Every log line emitted while handling a request carries the same id, and the id
comes back in the response. Without it, a reported bad decision can only be traced
by guessing from timestamps.
"""

import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.request_context import request_id_var

HEADER = "X-Request-ID"

# An inbound id is accepted so a trace can span a proxy or an upstream service, but
# it is bounded and sanitised: it reaches log files, and an unbounded attacker-
# controlled string in a log line is how log injection and log flooding start.
_MAX_LENGTH = 64
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _clean(value: str) -> str:
    """Return `value` if it is a safe, bounded identifier, else the empty string."""
    if not value or len(value) > _MAX_LENGTH:
        return ""
    return value if set(value) <= _ALLOWED else ""


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a correlation id to each request and echo it in the response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _clean(request.headers.get(HEADER, "")) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[HEADER] = request_id
        return response
