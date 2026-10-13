from __future__ import annotations

from typing import Sequence

POWERSHELL = "powershell.exe"


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def query_argv(script: str) -> list[str]:
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script]


def script_argv(script: str) -> list[str]:
    # No `-ExecutionPolicy Bypass`: execution policy governs script FILES, never `-Command`
    # text, so it did nothing here - and on owens-pc (2026-10-12) Windows began refusing to
    # start powershell.exe with it beside a shortcut-writing script ([WinError 5] at
    # CreateProcess), which failed every Start Menu write and with it the 2.0.11 deploy.
    return [POWERSHELL, "-NoProfile", "-Command", script]


def runas_argv(argv: Sequence[str]) -> list[str]:
    program, *rest = argv
    arguments = f" -ArgumentList {','.join(quote(word) for word in rest)}" if rest else ""
    return script_argv(f"Start-Process -Verb RunAs -Wait -FilePath {quote(program)}{arguments}")
