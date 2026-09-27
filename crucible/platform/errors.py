from __future__ import annotations

from ..errors import CrucibleError


class HostError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class LocalError(RuntimeError):
    pass
