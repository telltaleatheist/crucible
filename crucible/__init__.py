"""Crucible — one inference server, many client apps.

`VERSION` is the build (semver). `API_VERSION` is the contract clients check; it is
bumped only on a breaking change to the HTTP surface. The two are deliberately
separate (DESIGN.md section 8).
"""

VERSION = "0.6.0"
API_VERSION = 1

__all__ = ["VERSION", "API_VERSION"]
