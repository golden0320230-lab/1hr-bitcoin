"""Shared retry helpers for external network calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def retry_operation(
    *exception_types: type[BaseException],
    attempts: int = 3,
    min_wait_seconds: float = 1,
    max_wait_seconds: float = 4,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build a tenacity retry decorator with project defaults."""

    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception_type(exception_types),
    )
