"""The directory Crucible's Higgs worker runs out of.

The worker is `crucible_higgs/` beside this file, run in the tts env's interpreter
as `python -m crucible_higgs.serve` with `PYTHONPATH` set to THIS directory, which
holds that package and nothing else - so the path adds exactly one importable
name to an env full of other people's packages. `crucible/engines/narrator.py`
(`WORKER_ROOT`, `MODULE`) is what launches it.

This file exists so setuptools ships the tree inside the `crucible` wheel; it
imports nothing, and nothing in Crucible's server imports the worker.
"""
