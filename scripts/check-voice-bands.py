#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOKFORGE_ENV = "CRUCIBLE_BOOKFORGE"
SIBLING_BOOKFORGE = REPO.parent / "bookforge"
OVERLAY = Path("electron/data/higgs-safe-bands.json")


def overlay_bands(bookforge: Path) -> dict[str, tuple[int, int]]:
    path = bookforge / OVERLAY
    if not path.exists():
        raise SystemExit(
            f"no overlay at {path}.\n"
            f"Name the BookForge checkout: --bookforge <path>, or set {BOOKFORGE_ENV}=<path> "
            f"(without either it looks beside this checkout, at {SIBLING_BOOKFORGE})."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    bands: dict[str, tuple[int, int]] = {}
    for voice, entry in raw.items():
        if isinstance(entry, dict) and "min" in entry and "max" in entry:
            bands[voice] = (int(entry["min"]), int(entry["max"]))
    return bands


def manifest_bands(voices: Path) -> dict[str, tuple[int | None, int | None]]:
    bands: dict[str, tuple[int | None, int | None]] = {}
    for path in sorted(voices.glob("*.toml")):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        if "voice" not in document:
            continue
        pace = document["voice"].get("pace", {})
        bands[path.stem] = (pace.get("safe_min_chars"), pace.get("safe_max_chars"))
    if not bands:
        raise SystemExit(
            f"no voice manifest (a *.toml with a [voice] table) in {voices}, so there is "
            f"nothing to compare.\nName the directory that holds them: --voices <dir>"
        )
    return bands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check every voice manifest's safe band against BookForge's authoritative overlay.",
    )
    parser.add_argument(
        "--bookforge", type=Path,
        default=Path(os.environ.get(BOOKFORGE_ENV) or SIBLING_BOOKFORGE),
        help=f"the BookForge checkout (default: ${BOOKFORGE_ENV}, else {SIBLING_BOOKFORGE})",
    )
    parser.add_argument(
        "--voices", type=Path, default=Path("voices"),
        help="the directory of voice manifests to check (default: ./voices)",
    )
    args = parser.parse_args()

    authority = overlay_bands(args.bookforge)
    mine = manifest_bands(args.voices)

    problems: list[str] = []
    checked = 0
    for voice, want in sorted(authority.items()):
        if voice not in mine:
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
