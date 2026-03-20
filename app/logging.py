"""Logging helpers for the CLI."""

from __future__ import annotations

import logging as stdlib_logging

from rich.logging import RichHandler

from app.constants import APP_LOGGER_NAME


def configure_logging(level: str) -> None:
    """Configure process-wide logging once per command execution."""

    stdlib_logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def get_logger(name: str | None = None) -> stdlib_logging.Logger:
    """Return a configured application logger."""

    return stdlib_logging.getLogger(name or APP_LOGGER_NAME)

