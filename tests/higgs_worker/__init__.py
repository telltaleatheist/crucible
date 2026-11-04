"""The Higgs worker's own tests (`crucible/jobs/tts/higgs/crucible_higgs`).

Moved from narrator@dc05e7e7 `python/narrator/tests/` with the worker, Higgs and
serve only. They import the worker as the top-level package it runs as, so the
directory that holds it goes on sys.path here, before any module of this package
is imported.
"""
import os
import sys

WORKER_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "crucible", "jobs", "tts", "higgs")
if WORKER_ROOT not in sys.path:
    sys.path.insert(0, WORKER_ROOT)
