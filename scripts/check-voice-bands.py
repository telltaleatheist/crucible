#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from crucible.config import crucible_home
from crucible.voicerepo import _cache_path, _snapshot_path, load_pins
from crucible.voices import VoiceError

REPO_FIXTURES = REPO / "tests" / "fixtures"
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


def band_of(path: Path) -> tuple[int | None, int | None] | None:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if "voice" not in document:
        return None
    pace = document["voice"].get("pace", {})
    return (pace.get("safe_min_chars"), pace.get("safe_max_chars"))


def pinned_manifest_bands() -> dict[str, tuple[int | None, int | None]]:
    home = crucible_home()
    bands: dict[str, tuple[int | None, int | None]] = {}
    missing: list[str] = []
    for voice_id, pin in sorted(load_pins().items()):
        candidates = (
            _snapshot_path(home, pin), _cache_path(home, pin), _cache_path(REPO_FIXTURES, pin)
        )
        found = next((path for path in candidates if path is not None and path.is_file()), None)
        if found is None:
            missing.append(
                f"  {voice_id}: run `crucible voices check {pin.hf_repo}@{pin.revision}`"
            )
            continue
        band = band_of(found)
        if band is not None:
            bands[voice_id] = band
    if missing:
        print(
            "these pinned voices have no manifest pulled or cached under "
            f"{home} or {REPO_FIXTURES / 'voice-manifests'}, "
            "so they are not compared:\n" + "\n".join(missing) + "\n"
        )
    if not bands:
        raise SystemExit(
            "no pinned voice has a manifest to compare. Fetch each with the commands "
            "above, or name a directory of crucible-voice.toml files: --voices <dir>"
        )
    return bands


def manifest_bands(voices: Path) -> dict[str, tuple[int | None, int | None]]:
    bands: dict[str, tuple[int | None, int | None]] = {}
    for path in sorted(voices.glob("*.toml")):
        band = band_of(path)
        if band is not None:
            bands[path.stem] = band
    if not bands:
        raise SystemExit(
            f"no voice manifest (a *.toml with a [voice] table) in {voices}, so there is "
            "nothing to compare.\nName the directory that holds them (one <voice id>.toml "
            "each), or leave --voices off to read the pinned manifests"
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
        "--voices", type=Path, default=None,
        help="a directory of <voice id>.toml manifests to check instead of the pinned ones "
        "(default: every pinned voice's crucible-voice.toml, from the CRUCIBLE_HOME "
        "voice-manifests cache, else this checkout's tests/fixtures/voice-manifests)",
    )
    args = parser.parse_args()

    authority = overlay_bands(args.bookforge)
    try:
        mine = pinned_manifest_bands() if args.voices is None else manifest_bands(args.voices)
    except VoiceError as exc:
        raise SystemExit(str(exc)) from exc

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
