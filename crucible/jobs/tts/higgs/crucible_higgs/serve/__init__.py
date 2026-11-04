"""crucible_higgs.serve - the resident Higgs v3 narration worker.

Which ENGINE it serves is `NARRATOR_ENGINE`, required, resolved through
`crucible_higgs.engine.registry` (`higgs-v3`); an unset or unknown value is
refused by name, never defaulted.

    python -m crucible_higgs.serve      # PYTHONPATH = the directory holding this package

The stdin/stdout JSON-lines protocol is narrator.serve's, unchanged -
`crucible/engines/narrator.py` speaks it. See worker.py's docstring for the full
message set.
"""
from .worker import StreamServer, main

__all__ = ['StreamServer', 'main']
