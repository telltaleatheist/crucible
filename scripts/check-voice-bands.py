#!/usr/bin/env python3
"""Every voice manifest's safe band, against BookForge's authoritative overlay.

WHY THIS EXISTS. On 2026-09-13 an audit found `thirdreich` advertising a band of
600-1000 here while BookForge's `electron/data/higgs-safe-bands.json` said
500-700. The overlay was right: it is 620 renders at n=32/rung, pooled by the
character count the packer actually emits, and it explicitly supersedes the
n=16/rung sweep this repo's manifest was written from. Pooled correctly, 700-800
is a 12.5% failure zone and 800-900 is 18.8%.

Crucible does not pack. It ADVERTISES the band and the client packs to it
(PHASE3-TTS.md section 2) — so a wrong band here is a defect in somebody else's
audio, produced silently, with nothing failing. That is the worst shape a bug can
have and it is exactly what two copies of one fact produce.

WHY A SCRIPT AND NOT A TEST. Crucible must run on a machine that has never heard
of BookForge — that is the whole point of the server — so it cannot read the
overlay at runtime or import it in its suite. The manifest is therefore genuinely
authoritative FOR CRUCIBLE, and this is the thing that keeps it honest where both
repos happen to sit side by side. Run it after touching any band, and before a
release.

    python scripts/check-voice-bands.py [--bookforge PATH]

Exit 0 when every voice agrees or is absent from one side; exit 1 with a diff
otherwise. A voice the overlay does not mention is NOT an error — the overlay is
a sparse set of corrections, not a census.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

DEFAULT_BOOKFORGE = Path(r"C:\Users\tellt\Projects\bookforge")
OVERLAY = Path("electron/data/higgs-safe-bands.json")


def overlay_bands(bookforge: Path) -> dict[str, tuple[int, int]]:
    """`{voice: (min, max)}` from the overlay, which is the authority.

    Keys whose value is not a band (`_README` and any future note) are skipped
    rather than refused: the file is a document as much as a table.
    """
    path = bookforge / OVERLAY
    if not path.exists():
        raise SystemExit(
            f"no overlay at {path}.\n"
            f"Pass --bookforge if the checkout is elsewhere; this script needs both "
            f"repos side by side and is a no-op anywhere else."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    bands: dict[str, tuple[int, int]] = {}
    for voice, entry in raw.items():
        if isinstance(entry, dict) and "min" in entry and "max" in entry:
            bands[voice] = (int(entry["min"]), int(entry["max"]))
    return bands


def manifest_bands() -> dict[str, tuple[int | None, int | None]]:
    bands: dict[str, tuple[int | None, int | None]] = {}
    for path in sorted(Path("voices").glob("*.toml")):
        pace = tomllib.loads(path.read_text(encoding="utf-8"))["voice"].get("pace", {})
        bands[path.stem] = (pace.get("safe_min_chars"), pace.get("safe_max_chars"))
    return bands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bookforge", type=Path, default=DEFAULT_BOOKFORGE)
    args = parser.parse_args()

    authority = overlay_bands(args.bookforge)
    mine = manifest_bands()

    problems: list[str] = []
    checked = 0
    for voice, want in sorted(authority.items()):
        if voice not in mine:
            # Not an error. BookForge may carry a voice this build has no
            # manifest for; `crucible voices` simply will not offer it.
            continue
        checked += 1
        got = mine[voice]
        if got != want:
            problems.append(
                f"  {voice}: manifest says {got[0]}-{got[1]}, "
                f"overlay says {want[0]}-{want[1]}"
            )

    if problems:
        print("VOICE BANDS DISAGREE WITH BOOKFORGE'S OVERLAY:\n")
        print("\n".join(problems))
        print(
            "\nThe overlay wins. It is the measurement, it declares itself "
            "authoritative,\nand it is what promotion writes. Copy its numbers into "
            "the manifest WITH the\nreason, the sample size and the date — a band "
            "without its evidence is how\nthis drifted in the first place."
        )
        return 1

    print(f"{checked} voice band(s) agree with BookForge's overlay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
