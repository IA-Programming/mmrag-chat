"""Typed errors. `core` raises AppError; `app` renders `user_message` (never a traceback)."""

from __future__ import annotations

import uuid


def new_event_id() -> str:
    return uuid.uuid4().hex[:10]


class AppError(Exception):
    """An error with a plain-sentence message that is safe to show to a tester.

    `detail` goes to the logs only. `event_id` links the UI message to log events.
    """

    code: str = "error"

    def __init__(self, user_message: str, *, detail: str | None = None, event_id: str | None = None, code: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
        self.event_id = event_id or new_event_id()
        if code:
            self.code = code


class ValidationError(AppError):
    code = "validation"


class DoesNotFitError(AppError):
    """The governor cannot make room for a model (FR-005 AC3)."""

    code = "does_not_fit"


class QueueTimeoutError(AppError):
    code = "queue_timeout"


class CancelledError(AppError):
    code = "cancelled"


class RuntimeNotReadyError(AppError):
    code = "not_ready"


class BackendError(AppError):
    """llama-server / Gemini returned an error or was unreachable."""

    code = "backend"


class RateLimitedError(BackendError):
    code = "rate_limited"


class EgressDeniedError(AppError):
    code = "egress_denied"
