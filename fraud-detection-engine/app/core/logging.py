"""Logging setup.

Configured by an explicit call, not as an import side effect. The previous
`dictConfig` ran at module import in `app/api/main.py`, so importing that module
for any reason — a test, a script, a tool — reconfigured logging for the whole
process.
"""

import logging.config

from app.core.config import LogLevel


def configure_logging(level: LogLevel = "INFO") -> None:
    """Install the application's log format and level. Call once, at startup."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )
