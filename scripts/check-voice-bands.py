#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

DEFAULT_BOOKFORGE = Path(r"C:\Users\tellt\Projects\bookforge")
OVERLAY = Path("electron/data/higgs-safe-bands.json")


def overlay_bands(bookforge: Path) -> dict[str, tuple[int, int]]:
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
    parser = argparse.ArgumentParser(
        description="Check every voice manifest's safe band against BookForge's authoritative overlay.",
    )
    parser.add_argument("--bookforge", type=Path, default=DEFAULT_BOOKFORGE)
    args = parser.parse_args()

    authority = overlay_bands(args.bookforge)
    mine = manifest_bands()

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
