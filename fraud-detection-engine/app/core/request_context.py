"""Per-request correlation id.

Held in a ContextVar so any logger in the request's call stack can reach it
without the id being threaded through every function signature.
"""

from contextvars import ContextVar

# Empty outside a request — startup and shutdown logs legitimately have no id.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
