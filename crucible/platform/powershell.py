from __future__ import annotations

from typing import Sequence

POWERSHELL = "powershell.exe"


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def query_argv(script: str) -> list[str]:
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script]


def script_argv(script: str) -> list[str]:
    return [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]


def runas_argv(argv: Sequence[str]) -> list[str]:
    program, *rest = argv
    arguments = f" -ArgumentList {','.join(quote(word) for word in rest)}" if rest else ""
    return script_argv(f"Start-Process -Verb RunAs -Wait -FilePath {quote(program)}{arguments}")
