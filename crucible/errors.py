"""Every refusal in Crucible has a code and a named reason.

There is no error type that means "something went wrong"; if the server will not do
what was asked, it says which thing it will not do and why (DESIGN.md section 10).
"""

from __future__ import annotations

from typing import Any


class CrucibleError(Exception):
    """Base for everything this package raises deliberately."""


class NoViableBackend(CrucibleError):
    """The host is not one Crucible can serve from. Carries the reason verbatim."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ConfigError(CrucibleError):
    """`~/.crucible/config.toml` is missing, unreadable, or incomplete."""


class ApiError(CrucibleError):
    """An HTTP refusal. Rendered as {"error": {"code", "message", "details"?}}."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            error["details"] = self.details
        return {"error": error}


class JobError(CrucibleError):
    """Raised from inside a job's run(). Becomes the job's `error` and a `failed` event."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class JobCancelled(CrucibleError):
    """Raised by ctx.raise_if_cancelled() once a cancel has been requested."""
