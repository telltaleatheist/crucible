"""Where this server looks for the command-line tools it shells out to.

One fact, one owner (ARCHITECTURE.md R1): **`PATH`**. Three job types refuse when
ffmpeg is not on it (`asr` decodes through it, `align` decodes through it, `tts`
encodes through it), and `crucible service install` writes it into the unit or
the plist it generates. Those are the same fact read twice, and until this module
existed they were two answers nobody could compare.

Why this is not a detail
------------------------
Measured on Owen's Mac, 2026-09-13, against a real `crucible doctor` over a
**non-login** shell:

    job tts: NOT READY — there is no ffmpeg on PATH

ffmpeg was installed the whole time, at `/opt/homebrew/bin/ffmpeg`. The PATH that
shell searched was `/usr/bin:/bin:/usr/sbin:/sbin`, which is the same bare PATH a
**launchd agent and a systemd user unit are started with** — so a Crucible
installed as a service would have failed in exactly the same way, on a host where
every env was installed and every tool was present.

Two things follow, and this module is both of them:

- **A refusal names the PATH it searched.** "There is no ffmpeg on PATH" is true
  and useless; the reader's next question is always *which PATH*, and on a host
  with two shells and a service manager that is a real question with three
  answers. `searched_note()` is what answers it, at every site that says a tool
  is missing.
- **`crucible service install` records the installing shell's PATH**, because
  that shell is the one the operator proved the tools on — they just ran
  `crucible doctor` in it. Hardcoding `/opt/homebrew/bin` would fix one Mac and
  nothing else, and it would be a second owner of this fact besides the
  environment.
"""

from __future__ import annotations

import os
import shutil

PATH_ENV = "PATH"


def search_path() -> str:
    """The `PATH` this process searches. `""` when there is none.

    A module-level probe, for the reason `crucible/jobs/asr/__init__.py` gives
    about its own: a test replaces it and asserts on the refusal, instead of
    asserting on whatever the machine running the suite happens to have.
    """
    return os.environ.get(PATH_ENV, "")


def which(tool: str) -> str | None:
    """`shutil.which`, asked through the one PATH owner. None when not found."""
    return shutil.which(tool)


def searched_note() -> str:
    """`(PATH searched: …)` — the parenthetical every "tool missing" carries.

    An empty PATH is reported as empty rather than as nothing, because the two
    read identically in a sentence and only one of them is a configuration
    mistake somebody can fix.
    """
    value = search_path()
    if value == "":
        return "(PATH searched: it is empty)"
    return f"(PATH searched: {value})"
