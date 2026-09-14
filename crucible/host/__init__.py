"""`crucible host` — the Windows presence, PHASE15-HOST.md section 4.

**This package is not a server.** It is the one process that runs ON Windows and
can therefore own the sentence "the engine is running" — which nothing could,
before, because WSL has no boot: nothing starts a distro at login, and a clean
stop left the engine down at 16:10 on 2026-09-14 with nobody noticing.

Four things and no fifth (4.1): which server this machine runs, starting at
login, booting the guest, and watching. The install door (4.3) is the fifth only
in the sense that it is how the FIRST of those four becomes true on a machine
that has no distro yet.

WHAT IS HERE, AND WHY EACH FILE IS SEPARATE
-------------------------------------------
Everything that can be a pure function is one, in its own module, because the
tray cannot be tested and the decisions it draws must be:

    paths.py       where the host's files are, from the environment only
    log.py         %LOCALAPPDATA%\\Crucible\\host.log, appended and rolled
    menu.py        the menu as a pure function of (distro, engine) — 4.2
    runner.py      the one door to a subprocess, injectable
    presence.py    the boot, the two recovery recipes, the 15 s watch — 4.1
    wsl_states.py  GENERATED from sdk/bootstrap/src/wsl-states.ts — 4c's table
    wslstate.py    the predicates for that table, and the detection walk
    installer.py   the 4.3 sequence: states, distro, pack, migrate, service
    door.py        POST /install on 127.0.0.1:7101 — bootstrap's other end
    startup.py     the Startup shortcut, and the two verbs that own it
    window.py      the tkinter window the install shows (no console)
    tray.py        pystray, which is the only module that cannot be tested
    app.py         the wiring, and what `crucible host` runs

**IMPORTING THIS PACKAGE MUST NOT NEED pystray OR tkinter.** pytest runs in
WSL, in an env that has neither, and a test suite that cannot import its subject
is a test suite that pins nothing. `tray` and `window` are imported inside the
functions that use them, and nothing else in here imports them at all.

**NOTHING IN HERE RUNS A MODEL.** Windows is never a backend (DESIGN.md
section 2). The host starts a server; the server it starts on a machine with no
WSL is a `backend_kind = "none"` host-mode server (3.5), which has `echo`, the
settings door and the upstream routes, and no card.
"""

from __future__ import annotations

from .errors import HostError

__all__ = ["HostError"]
