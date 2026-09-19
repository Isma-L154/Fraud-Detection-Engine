"""Response middleware.

Security headers are set here rather than per-route so a new endpoint cannot be
added without them.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import Settings

# Applied to every response, JSON and HTML alike.
BASE_HEADERS = {
    # Stop a browser from second-guessing the declared content type. A JSON body
    # that a browser decides to treat as HTML is how a reflected value becomes a
    # script execution.
    "X-Content-Type-Options": "nosniff",
    # Do not leak the URL of this API to anything the user navigates to next.
    "Referrer-Policy": "no-referrer",
    # frame-ancestors is the modern control; X-Frame-Options covers clients that
    # predate CSP. Neither is redundant yet.
    "X-Frame-Options": "DENY",
    # Browsers should not attempt to guess a language or infer permissions.
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# For JSON responses. The API returns data, never markup, so nothing needs to load.
API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

# For the generated documentation pages, which render HTML and pull Swagger UI and
# ReDoc assets from jsDelivr. This is the narrowest policy those pages actually run
# under: they inline a bootstrap script and ReDoc builds styles at runtime, so
# 'unsafe-inline' cannot be removed without vendoring the assets. Served in
# development only (see Settings.docs_enabled), which is what keeps that acceptable.
DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' https://fastapi.tiangolo.com data:; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'"
)

DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json"})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers to every response."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)

        for header, value in BASE_HEADERS.items():
            response.headers[header] = value

        # The documentation pages are the only HTML this service serves, and they
        # need a materially wider policy than the JSON API.
        is_docs = request.url.path in DOCS_PATHS
        response.headers["Content-Security-Policy"] = DOCS_CSP if is_docs else API_CSP

        # HSTS only where the service is actually reachable over TLS. Sending it
        # from a plaintext development server teaches the browser to refuse http://
        # for this host, which then persists past the end of the session.
        if self._settings.env != "development":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
