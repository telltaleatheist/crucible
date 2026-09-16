"""Install and remove the user's CLI entry point, with explicit ownership.

Windows uses <Crucible home>/bin and a user PATH entry. POSIX uses the standard
~/.local/bin; shells not including it get an explicit instruction, never a
silent profile rewrite. Existing unrelated launchers are refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .errors import CrucibleError

RECORD = "launcher.json"


def _user_path(value: str | None = None) -> str:
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            existing, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            existing, kind = "", winreg.REG_EXPAND_SZ
        if value is not None:
            winreg.SetValueEx(key, "Path", 0, kind, value)
            # Tell future Explorer-launched processes about the new user PATH.
            import ctypes
            result = ctypes.c_size_t()
            ctypes.windll.user32.SendMessageTimeoutW(
                0xFFFF, 0x001A, 0, "Environment", 2, 5000, ctypes.byref(result)
            )
        return existing


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def install(home: Path, executable: str, cwd: str, *, platform: str | None = None,
            user_home: Path | None = None, path_store: Any = _user_path) -> dict[str, Any]:
    platform = sys.platform if platform is None else platform
    user_home = Path.home() if user_home is None else user_home
    directory = home / "bin" if platform == "win32" else user_home / ".local" / "bin"
    path = directory / ("crucible.cmd" if platform == "win32" else "crucible")
    metadata = home / RECORD
    old = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
    adopted = False
    if path.exists() and (old is None or old.get("path") != str(path)
                         or old.get("sha256") != _digest(path.read_text(encoding="utf-8"))):
        # NOT OURS BY THE RECORD — but there are two very different files that
        # can be. One is a launcher for THIS Crucible written by something other
        # than this installer: a previous release, or the hand-written shim that
        # was on both of Owen's machines (its own comments say it exists because
        # `Scripts\crucible.exe --version` exited 1). Replacing that is what an
        # upgrade IS, and refusing it stopped the 0.6.3 upgrade twice on
        # 2026-09-16, once per machine, with "nothing changed" and no remedy.
        #
        # The other is a `crucible` of somebody else's that happens to sit on
        # the same path, and that one must still be left alone. The two are told
        # apart by what the file DOES: ours runs `crucible.cli` out of this home.
        existing = path.read_text(encoding="utf-8", errors="replace")
        if "crucible.cli" in existing and str(home) in existing:
            adopted = True
        else:
            raise CrucibleError(
                f"cli_launcher_conflict: {path} was not installed by this "
                f"Crucible and does not launch it either; nothing changed. "
                f"Move it aside and run the install again if it is not wanted"
            )
    if any(c in value for value in (str(home), executable, cwd) for c in ('\n', '\r', '"')):
        raise CrucibleError("cli_launcher_invalid_path: a path contains quotes or newlines")
    if platform == "win32":
        escaped = [value.replace("%", "%%") for value in (str(home), executable, cwd)]
        body = (f'@echo off\nsetlocal DisableDelayedExpansion\nset "CRUCIBLE_HOME={escaped[0]}"\n'
                f'cd /d "{escaped[2]}" || exit /b 1\n"{escaped[1]}" -m crucible.cli %*\n')
    else:
        body = (f"#!/bin/sh\nexport CRUCIBLE_HOME={shlex.quote(str(home))}\n"
                f"cd {shlex.quote(cwd)} || exit 1\nexec {shlex.quote(executable)} -m crucible.cli \"$@\"\n")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    added = bool(old and old.get("path_added"))
    if platform == "win32":
        current = path_store()
        if str(directory).casefold() not in [part.casefold() for part in current.split(";")]:
            path_store(current + (";" if current else "") + str(directory))
            added = True
    record = {"path": str(path), "sha256": _digest(body), "platform": platform,
              "path_added": added, "adopted": adopted}
    home.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def remove(home: Path, *, path_store: Any = _user_path) -> list[str]:
    metadata = home / RECORD
    if not metadata.exists():
        return ["no owned CLI launcher"]
    record = json.loads(metadata.read_text(encoding="utf-8"))
    path = Path(record["path"])
    if path.exists():
        if _digest(path.read_text(encoding="utf-8")) != record["sha256"]:
            raise CrucibleError(f"cli_launcher_modified: {path} changed externally and was kept")
        path.unlink()
    if record["platform"] == "win32" and record["path_added"]:
        current = path_store()
        path_store(";".join(part for part in current.split(";") if part.casefold() != str(path.parent).casefold()))
    if path.parent == home / "bin" and path.parent.is_dir() and not any(path.parent.iterdir()):
        path.parent.rmdir()
    metadata.unlink()
    return [f"removed owned CLI launcher {path}"]
