"""Count batch items before the handler runs.

The rate limiter has to charge one unit per transaction, not one per request —
otherwise batching multiplies throughput past the per-request limit by the batch
size, which makes the limit meaningless.

slowapi evaluates its cost callable synchronously, before the body is parsed, so
the count has to be on the request by then. This middleware buffers the body,
counts the items, records the count, and replays the body downstream.

Buffering is safe here because BodySizeLimitMiddleware runs outside this one and
has already capped the body. Without that cap this would be the denial-of-service
vector it exists to prevent.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

BATCH_PATH = "/api/v1/predict/batch"


class BatchCostMiddleware:
    """Record how many transactions a batch request carries."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != BATCH_PATH:
            await self.app(scope, receive, send)
            return

        body = b""
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            more = message.get("more_body", False)

        scope.setdefault("state", {})["batch_size"] = _count(body)

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _count(body: bytes) -> int:
    """Number of transactions in the body, or 1 when it cannot be counted.

    A body that does not parse is charged one unit and then rejected by the schema
    anyway — guessing higher would let malformed input consume someone else's quota.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return 1

    transactions = payload.get("transactions") if isinstance(payload, dict) else None
    if isinstance(transactions, list) and transactions:
        return len(transactions)
    return 1
