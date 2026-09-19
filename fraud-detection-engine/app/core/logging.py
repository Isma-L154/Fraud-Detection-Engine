"""Logging setup.

Configured by an explicit call, not as an import side effect: the previous
`dictConfig` ran at module import in `app/api/main.py`, so importing that module
for any reason reconfigured logging for the whole process.

Two formats. JSON in deployed environments, because a format string in one handler
is not a schema and any aggregator would have to parse it back with a regular
expression. Plain text in development, because a human reads it.

**Never log transaction feature values or amounts.** V1-V28 are PCA components of
real card transactions and the amount plus a timestamp is identifying. Log the
decision and the correlation id, not the input.
"""

import json
import logging
import logging.config
from typing import Any

from app.core.config import LogLevel
from app.core.request_context import request_id_var

TEXT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record.

    A filter rather than an adapter, so third-party loggers — uvicorn, slowapi —
    carry the id too without being wrapped.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with stable field names."""

    # Everything the logging module puts on a record by default. Anything else was
    # passed by the caller as structured context and is worth keeping.
    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "asctime",
        "message",
        "request_id",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: LogLevel = "INFO", *, json_output: bool = True) -> None:
    """Install the application's log format and level. Call once, at startup."""
    formatter: dict[str, Any] = (
        {"()": JsonFormatter}
        if json_output
        else {"format": TEXT_FORMAT, "datefmt": "%Y-%m-%d %H:%M:%S"}
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {"default": formatter},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                }
            },
            # uvicorn installs its own handlers when it builds its Config, which is
            # before it imports the app — so its access and error logs would bypass
            # this format entirely. Clearing their handlers and letting them
            # propagate puts every line, ours and uvicorn's, through one formatter.
            "loggers": {
                name: {"handlers": [], "propagate": True, "level": level}
                for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )
