from __future__ import annotations

import importlib.util

# What the worker imports beyond the standard library, the same list
# `crucible.jobenv.HIGGS_WORKER_SMOKE_IMPORT` proves in a tts env. The `test` extra carries
# them, so CI runs these modules with no torch and no MLX; an interpreter without them does
# not collect these modules and `test_worker_env.py` says so as a skip
# (docs/internals/voices.md, "The narrator engine").
WORKER_MODULES = ("numpy", "num2words", "regex", "soundfile")

MISSING = tuple(name for name in WORKER_MODULES if importlib.util.find_spec(name) is None)

collect_ignore_glob = ["test_[!w]*.py"] if MISSING else []
