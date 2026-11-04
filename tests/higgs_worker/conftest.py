from __future__ import annotations

import importlib.util

# What the worker imports beyond the standard library, the same list
# `crucible.jobenv.HIGGS_WORKER_SMOKE_IMPORT` proves in a tts env. Crucible's own test
# env carries numpy and none of the others, so there these modules are not
# collected and `test_worker_env.py` says so as a skip; the tts env's interpreter has
# them all and runs every one (docs/internals/voices.md, "The narrator engine").
WORKER_MODULES = ("numpy", "num2words", "regex", "soundfile")

MISSING = tuple(name for name in WORKER_MODULES if importlib.util.find_spec(name) is None)

collect_ignore_glob = ["test_[!w]*.py"] if MISSING else []
