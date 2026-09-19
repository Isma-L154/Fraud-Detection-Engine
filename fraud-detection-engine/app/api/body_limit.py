"""Request body size limit.

Enforced at the ASGI layer, before the body is parsed, because the Pydantic schema
can only reject a payload it has already received and deserialised in full. A 500 MB
body validated as "invalid" has still cost 500 MB of memory.

Pure ASGI rather than BaseHTTPMiddleware: this has to intercept the `receive`
channel to count bytes as they arrive, which a request/response middleware cannot do.
"""

import logging

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.middleware import API_CSP, BASE_HEADERS

logger = logging.getLogger(__name__)

_TOO_LARGE_BODY = b'{"detail":"Request body too large."}'


class BodySizeLimitMiddleware:
    """Reject requests whose body exceeds `max_bytes` with 413."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: an honest Content-Length lets us refuse before reading anything.
        if self._declared_length(scope) > self.max_bytes:
            await self._reject(scope, send, reason="declared Content-Length")
            return

        # Slow path: a chunked request declares no length, and a dishonest one can
        # declare anything. Count what actually arrives.
        received = 0
        too_large = False

        async def counting_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    # Present an empty final chunk so the app stops reading rather
                    # than waiting for a body that will never complete.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: Message) -> None:
            if too_large and message["type"] == "http.response.start":
                await self._send_413(send)
                return
            if too_large and message["type"] == "http.response.body":
                return
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

    def _declared_length(self, scope: Scope) -> int:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    # A malformed Content-Length is not something to guess at.
                    return 0
        return 0

    async def _reject(self, scope: Scope, send: Send, reason: str) -> None:
        logger.warning("Rejected oversized request to %s (%s)", scope.get("path"), reason)
        await self._send_413(send)

    async def _send_413(self, send: Send) -> None:
        # This middleware sits outermost so an oversized body is refused before
        # anything else holds it — which means the security-headers middleware never
        # runs for this response. It sets them itself rather than being moved inward:
        # a response short-circuited here is still a response the client sees.
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(_TOO_LARGE_BODY)).encode()),
            (b"content-security-policy", API_CSP.encode()),
        ]
        headers += [(k.lower().encode(), v.encode()) for k, v in BASE_HEADERS.items()]

        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})
