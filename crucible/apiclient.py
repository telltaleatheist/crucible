from __future__ import annotations

import importlib
import sys
import types

TARGET = "crucible.cli.api_cmd"


class MovedToApiCmd(types.ModuleType):

    def __getattr__(self, name: str):
        return getattr(importlib.import_module(TARGET), name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(importlib.import_module(TARGET), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(importlib.import_module(TARGET), name)


sys.modules[__name__].__class__ = MovedToApiCmd
