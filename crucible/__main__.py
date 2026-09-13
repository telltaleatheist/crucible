"""`python -m crucible ...` — the same CLI the `crucible` script runs.

The console script comes from the installed distribution, so it always points at
whichever checkout was `pip install -e`'d. Running a *different* checkout (a
worktree under test, say) means putting that checkout on `PYTHONPATH` and asking
Python for the module, which is what this file makes possible. Same parser, same
exit codes; there is no second code path.
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
