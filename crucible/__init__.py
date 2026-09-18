"""Crucible — one inference server, many client apps.

`VERSION` is the build (semver). `API_VERSION` is the contract clients check; it is
bumped only on a breaking change to the HTTP surface. The two are deliberately
separate (DESIGN.md section 8).

`API_HEADER` is the header that carries `API_VERSION`. It lives HERE and not in
`crucible/api.py`, which is where it was written, because `crucible/peer.py`'s
orchestrator half has to send it and must not import the server: that half runs
inside the Windows tray, which starts at login and must not drag FastAPI,
uvicorn or httpx in behind it. One spelling, in the one module both sides can
import for free.
"""

VERSION = "1.0.2"
API_VERSION = 1
API_HEADER = "X-Crucible-Api"

__all__ = ["VERSION", "API_VERSION", "API_HEADER"]
