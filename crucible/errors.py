from __future__ import annotations

from typing import Any


class CrucibleError(Exception):
    ...


class NoViableBackend(CrucibleError):

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ConfigError(CrucibleError):
    ...


class ApiError(CrucibleError):

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

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class JobCancelled(CrucibleError):
    ...
