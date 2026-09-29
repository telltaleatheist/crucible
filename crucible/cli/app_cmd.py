from __future__ import annotations

import argparse
from typing import Any


def command(_args: argparse.Namespace) -> int:
    from ..desktop_app.app import main

    return main()


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "app", help="open Crucible's window: models, voices, packages and settings on this computer"
    )
    parser.set_defaults(func=command)
