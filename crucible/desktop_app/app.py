from __future__ import annotations

import sys
import traceback
from pathlib import Path

from ..config import crucible_home
from ..errors import ConfigError
from .instance import FOCUS, Instance, signal
from .screens import install_line

LOG_NAME = "app.log"


def log_path(home: Path) -> Path:
    return home / LOG_NAME


def no_tk_message(exc: BaseException) -> str:
    return (
        f"app_no_tk: this Python cannot draw windows ({exc}). The Python that Crucible's "
        f"installer downloads can; run the installer again to get it:\n{install_line()}"
    )


def say(message: str) -> None:
    if sys.stderr is not None:
        print(f"crucible: {message}", file=sys.stderr)


def launch(home: Path) -> int:
    instance = Instance(home)
    if not instance.claim():
        signal(home, FOCUS)
        return 0
    try:
        try:
            from .window import run
        except ImportError as exc:
            say(no_tk_message(exc))
            return 1
        return run(home, instance)
    finally:
        instance.close()


def main(argv: list[str] | None = None) -> int:
    try:
        home = crucible_home()
    except ConfigError as exc:
        say(str(exc))
        return 1
    try:
        return launch(home)
    except Exception:
        log_path(home).write_text(traceback.format_exc(), encoding="utf-8")
        say(f"app_failed: the Crucible window stopped with an error. What happened is in {log_path(home)}")
        return 1
