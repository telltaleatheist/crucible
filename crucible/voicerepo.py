"""A voice's facts travel with its weights — `crucible-voice.toml` and the pins.

PHASE21-VOICES-FROM-HF.md sections 2.1 to 2.3. Crucible ships no voices: it
downloads them, and the safe band, the caps, the sampling and the retake ladder
come down WITH the bytes, out of a file committed in the SAME commit as the
weights it describes.

── Why the file is in the repo and not in this package ──────────────────────

Deploying mistborn ckpt-4257 on 2026-09-19 took seven steps, and step seven was
"the manifest Crucible ships is STILL stale": `crucible/voices/mistborn.toml`
carried pace 13.33, band 400-700 and revision e5bf8017, every one of them a
retired number, while the card on the HF repo carried four of the same figures
written by a second step of the same deploy. A value and its description in two
places drift, always — that is how deathstalker's 16.64 survived onto weights
that measured 15.91, and how thirdreich's card still carried `higgs_target_chars`
ten days after that field was retired. The fix is a schema fix: one file, one
commit, one writer.

── THREE SCHEMAS, AND THEY ARE DELIBERATELY NOT ONE ─────────────────────────

    crucible-voice.toml   the REPO's, this module's, and it carries no machine
                          facts at all. `id`, `hf_repo`, `revision`,
                          `memory_bytes_estimate`, `estimate_basis`,
                          `estimate_note`, `[voice.serving]` and the word
                          `backends` are each REFUSED BY NAME here, so a file
                          converted from a packaged manifest cannot carry one by
                          accident and cannot make a claim about a box it has
                          never run on.

    pins.toml             the LOCAL side: one repo and one sha per id, packaged
                          (what this build may offer) and per-machine (what this
                          machine has chosen), the machine winning per id.

    voices/<id>.toml      `crucible/voices.py`'s, unchanged, and still the one
                          the engine reads. This module does not add a second
                          internal model; it TRANSLATES a repo document into
                          that one and hands it to the same `_parse` every
                          packaged manifest goes through. Every pace, sampling,
                          clips, takes and cap rule therefore applies verbatim,
                          by being the same code rather than a copy of it.

── The word `arms` ──────────────────────────────────────────────────────────

The repo schema says `[voice.arms.<backend>]` where the internal one says
`[voice.backends.<kind>]`, and the rename is the point: the two files look
alike, one carries machine facts and the other must never, and a reader (or a
`cp`) that confuses them would produce a file that parses as the wrong schema.
`backends` in a repo manifest is refused by name for exactly that reason.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .manifests import check_table
from .voices import (
    MANIFEST_REPO,
    MAX_CHARS_BASES,
    PACE_BASES,
    PINS_FILE,
    VoiceError,
    _parse,
    home_voices_dir,
    voices_dir,
)

#: WHAT THE FILE IS CALLED, at the ROOT of the repo. Not the card's
#: frontmatter: a machine contract is not parsed out of a human README where an
#: editorial fix breaks a loader. The card is GENERATED from this file
#: (`crucible voices card`), so there is one writer.
REPO_MANIFEST_NAME = "crucible-voice.toml"

#: The schema this build reads. A loader refuses a schema it does not read
#: rather than reading the keys it recognises out of a newer file — half a
#: manifest is a voice served on somebody else's terms.
REPO_SCHEMA = 1

#: WHERE A FETCHED MANIFEST IS KEPT, under `<CRUCIBLE_HOME>`. Content-addressed
#: by (repo, revision), which is what makes caching it honest rather than a
#: staleness hazard: a sha names one byte-state forever, so a cached file at a
#: sha cannot go out of date. It exists so `GET /v1/voices` is a directory read
#: rather than a Hub round trip per row.
MANIFEST_CACHE_DIRNAME = "voice-manifests"

#: `[voice]`'s scalars in the REPO schema. `id` is not among them, by rule: a
#: repo may be offered under any id and the PIN names it, so an id in the file
#: would be a second owner of the one fact the local side actually decides.
_REPO_VOICE_REQUIRED: dict[str, type] = {
    "display": str,
    "kind": str,
    "narrator_engine": str,
    "language": str,
    "sample_rate": int,
}

#: KEYS THIS SCHEMA REFUSES BY NAME, each with where it belongs instead. Not a
#: silent drop and not a generic "unknown key": the whole reason section 2.1
#: lists them is that a file converted from a packaged manifest would otherwise
#: carry a claim about a machine it has never run on, and `check_table`'s
#: "unknown key(s) [...]" would send its reader looking for a typo.
_REFUSED_IN_REPO: dict[str, str] = {
    "id": (
        "a repo can be offered under any id, and the PIN names it "
        "(pins.toml, section 2.2). An id here would be a second owner of the "
        "one fact the local side decides"
    ),
    "hf_repo": "the file IS the revision — the pin names the repo",
    "revision": "the file IS the revision",
    "path": (
        "a repo manifest describes published weights; the PHASE18 path arm is "
        "a local override and lives in `PUT /v1/voices/{id}`'s `voice` body"
    ),
    "identity": "a pinned revision's identity is the sha it was fetched at",
    "memory_bytes_estimate": (
        "this is a MACHINE fact and belongs in that machine's config.toml, "
        "`[tts.<engine>] memory_bytes_estimate` (section 2.3). All seven "
        "packaged manifests declared the same 19_000_000_000 on cuda-linux and "
        "12_133_000_000 on mlx-darwin, which is the proof it is a fact about a "
        "box and an engine rather than about a voice"
    ),
    "estimate_basis": "a machine fact — config.toml `[tts.<engine>]`",
    "estimate_note": "a machine fact — config.toml `[tts.<engine>]`",
    "serving": (
        "`max_num_seqs` sizes the SERVER narrator starts, which is a property of "
        "the box and the engine; it belongs in config.toml `[tts.<engine>]` "
        "(section 2.3)"
    ),
    "backends": (
        "the repo schema says `[voice.arms.<backend>]`. The two words are "
        "deliberately different so the repo schema and voices/<id>.toml cannot "
        "be mistaken for each other — one of them carries machine facts and "
        "this one must never"
    ),
}

#: `[voice.pace]`'s two extra keys in the REPO schema. `basis` is required
#: whenever the table is present; `measured_from` is required when the basis is
#: `"measured"`.
_PACE_BASIS = "basis"
_PACE_MEASURED_FROM = "measured_from"

_ARM_REQUIRED: dict[str, type] = {
    "max_chars": int,
    "max_chars_basis": str,
    "sampling": dict,
}
_ARM_OPTIONAL: dict[str, type] = {"sampling_reason": str, "clips": object}

_PIN_REQUIRED: dict[str, type] = {"hf_repo": str, "revision": str}


@dataclass(frozen=True)
class Pin:
    """One row of a `pins.toml`: which repo, at which commit, under which id."""

    id: str
    hf_repo: str
    revision: str
    #: Which `pins.toml` this row came out of, so a refusal can name the file a
    #: person has to edit.
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "path": str(self.path),
        }


# --------------------------------------------------------------------- pins


def packaged_pins_path() -> Path:
    """`crucible/voices/pins.toml` — the repos this BUILD may offer."""
    return voices_dir() / PINS_FILE


def home_pins_path() -> Path:
    """`<CRUCIBLE_HOME>/voices/pins.toml` — the repos THIS MACHINE has chosen.

    Where `PUT /v1/voices/{id}` with a `pin` body and `crucible voices pin`
    write, and the only pins file either of them touches: the packaged one is
    the install, and editing it would make the next upgrade the thing that
    "restored" a pin somebody deliberately moved.
    """
    return home_voices_dir() / PINS_FILE


def _parse_pins(text: str, path: Path) -> dict[str, Pin]:
    """One pins file, by id. Every refusal by name, as `VoiceError`."""
    from .voices import _HF_REPO, _REVISION, _VOICE_ID

    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path}: not valid TOML: {exc}") from exc
    pins: dict[str, Pin] = {}
    for voice_id in sorted(document):
        where = f"{path.name} [{voice_id}]"
        if not _VOICE_ID.match(voice_id):
            raise VoiceError(
                f"{where}: {voice_id!r} is not a voice id (lower-case letters, "
                "digits, dot, dash and underscore, starting with a letter or "
                "digit). A pins file is one table per voice id and nothing else"
            )
        block = document[voice_id]
        if not isinstance(block, dict):
            raise VoiceError(
                f"{where}: must be a table of hf_repo and revision, got "
                f"{type(block).__name__}. A pins file has no top-level keys — "
                "every row is [<voice id>]"
            )
        check_table(where, block, _PIN_REQUIRED, {}, error=VoiceError)
        if not _HF_REPO.match(block["hf_repo"]):
            raise VoiceError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise VoiceError(
                f"{where}: revision {block['revision']!r} must be a full "
                "40-character commit sha, so the manifest at that sha and the "
                "weights at that sha are the same commit; branch names are not pins"
            )
        pins[voice_id] = Pin(
            id=voice_id,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            path=path,
        )
    return pins


def load_pins() -> dict[str, Pin]:
    """Every pin this host offers, by id — packaged first, the machine's winning.

    The same precedence `voice_dirs()` has and for the same reason: the packaged
    list is what this BUILD ships, the home list is what this MACHINE decided,
    and a deploy repins one machine at a time (section 8.4). Deleting a home row
    puts the packaged pin back, which is what makes trying a new checkpoint safe.
    """
    from .voices import voices_dir_is_overridden

    # `CRUCIBLE_VOICES_DIR` REPLACES the set, this machine's own pins included:
    # `packaged_pins_path()` already follows the override, and reading the home
    # file beside it would add voices the caller did not put in that directory.
    roots = (
        (packaged_pins_path(),)
        if voices_dir_is_overridden()
        else (packaged_pins_path(), home_pins_path())
    )
    pins: dict[str, Pin] = {}
    for path in roots:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VoiceError(f"could not read {path}: {exc}") from exc
        pins.update(_parse_pins(text, path))
    return pins


def write_home_pin(voice_id: str, hf_repo: str, revision: str) -> Pin:
    """Add or replace this machine's pin for `voice_id`. Returns what was written.

    Validated by the SAME `_parse_pins` that reads one, at the path it is about
    to occupy, and written atomically — a half-written pins file is not a broken
    pin, it is a broken server, because `load_pins` reads the whole file and one
    unparseable row raises for every caller of it.
    """
    import tomli_w

    path = home_pins_path()
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise VoiceError(f"{path}: not valid TOML: {exc}") from exc
    document = {
        **existing,
        voice_id: {"hf_repo": hf_repo, "revision": revision},
    }
    document = {key: document[key] for key in sorted(document)}
    text = tomli_w.dumps(document)
    written = _parse_pins(text, path)
    if voice_id not in written:  # pragma: no cover - `_parse_pins` refuses first
        raise VoiceError(f"{path.name}: writing the pin for {voice_id!r} lost it")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VoiceError(f"could not write {path}: {exc}") from exc
    return written[voice_id]


def remove_home_pin(voice_id: str) -> bool:
    """Delete this machine's pin row for `voice_id`. True if one went.

    The PACKAGED pin is untouched and unreachable from here, exactly as the
    packaged manifests are: removing a home row that shadowed one brings the
    packaged pin back.
    """
    import tomli_w

    path = home_pins_path()
    if not path.is_file():
        return False
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VoiceError(f"{path}: not valid TOML: {exc}") from exc
    if voice_id not in document:
        return False
    del document[voice_id]
    text = tomli_w.dumps({key: document[key] for key in sorted(document)})
    temporary = path.with_name(f".{path.name}.writing")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VoiceError(f"could not write {path}: {exc}") from exc
    return True


def is_home_pin(voice_id: str) -> bool:
    """Does this machine's own pins file hold a row for this id?"""
    path = home_pins_path()
    if not path.is_file():
        return False
    try:
        return voice_id in tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False


# ------------------------------------------------------- the repo manifest


def _check_no_machine_facts(where: str, table: dict[str, Any]) -> None:
    """Refuse a machine fact in a repo manifest, by its own name and reason."""
    for key in sorted(set(table) & set(_REFUSED_IN_REPO)):
        raise VoiceError(
            f"{where}: carries {key}, which a repo manifest may not state — "
            f"{_REFUSED_IN_REPO[key]}"
        )


def _repo_pace(
    where: str, table: dict[str, Any]
) -> tuple[dict[str, Any], str, str | None]:
    """The internal `[voice.pace]` table and its `basis`, out of the repo one.

    `basis` and `measured_from` are stripped here and do not reach `_parse`:
    the internal schema has no such keys, they say how the numbers were GOT
    rather than what they are, and the one that survives to the wire does so as
    `pace_basis` on the `/v1/voices` row.
    """
    basis = table.get(_PACE_BASIS)
    if basis is None:
        raise VoiceError(
            f"{where}: states no {_PACE_BASIS}. A pace table says how its "
            f"numbers were got — {sorted(PACE_BASES)} — and an inherited pace is "
            "indistinguishable from a measured one at the point of use, which is "
            "exactly how deathstalker's 16.64 survived onto weights that measured "
            "15.91. Omit the whole table for a voice whose pace is not known yet"
        )
    if basis not in PACE_BASES:
        raise VoiceError(
            f"{where}: {_PACE_BASIS} {basis!r} is not one of {sorted(PACE_BASES)}"
        )
    measured_from = table.get(_PACE_MEASURED_FROM)
    if basis == "measured" and (
        not isinstance(measured_from, str) or measured_from.strip() == ""
    ):
        raise VoiceError(
            f"{where}: {_PACE_BASIS} is 'measured' and there is no "
            f"{_PACE_MEASURED_FROM}. A measured pace was measured on something — "
            "a checkpoint, a ladder, a sample count — and the number is only "
            "worth what the reader can find out about where it came from"
        )
    if measured_from is not None and not isinstance(measured_from, str):
        raise VoiceError(
            f"{where}: {_PACE_MEASURED_FROM} must be prose, got "
            f"{type(measured_from).__name__}"
        )
    return (
        {k: v for k, v in table.items()
         if k not in (_PACE_BASIS, _PACE_MEASURED_FROM)},
        basis,
        measured_from,
    )


@dataclass(frozen=True)
class RepoManifest:
    """A parsed `crucible-voice.toml`, before a pin and a machine are applied.

    Its own type because two of the three things that make a servable voice are
    NOT in it: which id it is offered under, and what it costs on this box. A
    function returning a `VoiceManifest` straight out of a repo file would have
    had to invent both.
    """

    schema: int
    voice: dict[str, Any]
    pace: dict[str, Any] | None
    pace_basis: str | None
    #: The prose behind a measured pace, kept off the internal table (which has
    #: no such key) and on this record, because `crucible voices card` prints it
    #: into the card's `## Measured limits` section verbatim. That is the one
    #: piece of prose the card no longer has to be typed with per deploy.
    measured_from: str | None
    arms: dict[str, dict[str, Any]]
    #: `max_chars_basis` per arm, stripped out of the arm tables for the reason
    #: `_repo_pace` strips `basis`: the internal schema has no such key.
    max_chars_basis: dict[str, str]
    takes: list[dict[str, Any]] | None
    path: Path


def parse_repo_manifest(text: str, path: Path) -> RepoManifest:
    """Read and check a `crucible-voice.toml`. Raises `VoiceError` by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path.name}: not valid TOML: {exc}") from exc

    unknown = sorted(set(document) - {"schema", "voice"})
    if unknown:
        raise VoiceError(
            f"{path.name}: unknown top-level key(s) {unknown}; a repo manifest is "
            "`schema` and [voice], and everything else hangs off [voice]"
        )
    if "schema" not in document:
        raise VoiceError(
            f"{path.name}: voice_manifest_schema — states no `schema`. The first "
            f"line of the file says which schema it is written in; this build "
            f"reads schema {REPO_SCHEMA}"
        )
    schema = document["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise VoiceError(
            f"{path.name}: voice_manifest_schema — `schema` must be an integer, "
            f"got {type(schema).__name__}"
        )
    if schema != REPO_SCHEMA:
        raise VoiceError(
            f"{path.name}: voice_manifest_schema {schema} — this build reads "
            f"schema {REPO_SCHEMA} and will not read half of a newer one. Upgrade "
            "Crucible on this machine, or pin the revision that was current for it"
        )

    if "voice" not in document:
        raise VoiceError(f"{path.name}: missing the [voice] table")
    voice = document["voice"]
    if not isinstance(voice, dict):
        raise VoiceError(f"{path.name}: [voice] must be a table")

    scalars = {
        key: value for key, value in voice.items()
        if key not in ("pace", "arms", "takes")
    }
    _check_no_machine_facts(f"{path.name} [voice]", scalars)
    check_table(
        f"{path.name} [voice]", scalars, _REPO_VOICE_REQUIRED, {}, error=VoiceError
    )

    pace: dict[str, Any] | None = None
    pace_basis: str | None = None
    measured_from: str | None = None
    if "pace" in voice:
        if not isinstance(voice["pace"], dict):
            raise VoiceError(f"{path.name}: [voice.pace] must be a table")
        pace, pace_basis, measured_from = _repo_pace(
            f"{path.name} [voice.pace]", voice["pace"]
        )

    if "arms" not in voice:
        raise VoiceError(
            f"{path.name}: missing every [voice.arms.<backend>] table; a voice "
            "nothing can serve is not a voice"
        )
    arms_table = voice["arms"]
    if not isinstance(arms_table, dict) or not arms_table:
        raise VoiceError(
            f"{path.name}: [voice.arms] must hold one table per arm, and at "
            "least one"
        )
    arms: dict[str, dict[str, Any]] = {}
    bases: dict[str, str] = {}
    for arm in sorted(arms_table):
        where = f"{path.name} [voice.arms.{arm}]"
        block = arms_table[arm]
        if not isinstance(block, dict):
            raise VoiceError(f"{where}: must be a table")
        _check_no_machine_facts(where, block)
        check_table(where, block, _ARM_REQUIRED, _ARM_OPTIONAL, error=VoiceError)
        basis = block["max_chars_basis"]
        if basis not in MAX_CHARS_BASES:
            raise VoiceError(
                f"{where}: max_chars_basis {basis!r} is not one of "
                f"{sorted(MAX_CHARS_BASES)}. A cap is either a number a sweep "
                "produced on these weights on this arm or a number somebody wrote "
                "down so the arm could be served — thirdreich shipped a "
                "`higgs_max_chars_mlx: 900` that was never measured, and a schema "
                "that cannot say so ships it as a measured fact"
            )
        bases[arm] = basis
        arms[arm] = {k: v for k, v in block.items() if k != "max_chars_basis"}

    takes = voice.get("takes")
    if takes is not None and not isinstance(takes, list):
        raise VoiceError(
            f"{path.name}: [[voice.takes]] must be a non-empty list of tables; a "
            "voice with no ladder simply omits it and gets take 0"
        )

    return RepoManifest(
        schema=schema,
        voice=scalars,
        pace=pace,
        pace_basis=pace_basis,
        measured_from=measured_from,
        arms=arms,
        max_chars_basis=bases,
        takes=takes,
        path=path,
    )


# ------------------------------------------------------------------- merge


def merge(repo: RepoManifest, pin: Pin, footprint: Any):
    """Repo manifest + pin + this box's `[tts.<engine>]` -> one `VoiceManifest`.

    THE TRANSLATION IS INTO THE INTERNAL DOCUMENT, NOT INTO THE INTERNAL MODEL,
    and that is the whole reason this phase does not double the schema's surface:
    what comes out of here goes through the SAME `crucible/voices.py::_parse`
    every packaged manifest goes through, so the pace ordering rule, the band
    symmetry rule, the `safe_max_chars <= max_chars` rule, the boson-default
    sampling rule, the clips rules and the takes ladder rules apply to a repo
    manifest verbatim — by being the same code, rather than by a second copy of
    them that would agree until the first time one was edited.

    `footprint` is a `config.EngineFootprint`; it is typed loosely here only to
    keep `crucible/config.py` from having to import this module back.
    """
    engine = repo.voice["narrator_engine"]
    document: dict[str, Any] = {
        "voice": {
            "id": pin.id,
            **repo.voice,
            # ABSENT MEANS ABSENT, and it reaches `_parse` as an EMPTY table
            # rather than as a missing one: `_check_pace` reads an empty table as
            # "nothing was measured" and builds a `Pace` of Nones, which is
            # PHASE18 section 4.1's uncertified voice. A missing table is refused
            # by `_parse`, and rightly — inside voices/<id>.toml it means somebody
            # deleted a block.
            "pace": dict(repo.pace) if repo.pace is not None else {},
            "serving": {
                "max_num_seqs": footprint.max_num_seqs,
                "max_num_seqs_note": footprint.max_num_seqs_note,
            },
            "backends": {
                arm: {
                    "hf_repo": pin.hf_repo,
                    "revision": pin.revision,
                    "memory_bytes_estimate": footprint.memory_bytes_estimate,
                    "estimate_basis": footprint.estimate_basis,
                    **(
                        {}
                        if footprint.estimate_note is None
                        else {"estimate_note": footprint.estimate_note}
                    ),
                    **block,
                }
                for arm, block in repo.arms.items()
            },
        }
    }
    if repo.takes is not None:
        document["voice"]["takes"] = repo.takes
    # `[voice.serving]` is refused on an engine that reads no HIGGS_* variable
    # (`_check_serving`), and a second engine will arrive that way. The footprint
    # is still REQUIRED of it — `memory_bytes_estimate` is — so the table is
    # dropped here rather than the footprint being made optional.
    if engine != "higgs-v3":
        del document["voice"]["serving"]

    manifest = _parse(document, repo.path, pin.id)
    backends = {
        arm: replace(spec, max_chars_basis=repo.max_chars_basis[arm])
        for arm, spec in manifest.backends.items()
    }
    return replace(
        manifest,
        backends=backends,
        manifest_source=MANIFEST_REPO,
        pace_basis=repo.pace_basis,
    )


# ---------------------------------------------------------------- fetching


def _cache_path(home: Path, pin: Pin) -> Path:
    """Where a fetched manifest is kept — content-addressed by repo and sha."""
    return (
        home
        / MANIFEST_CACHE_DIRNAME
        / pin.hf_repo.replace("/", "--")
        / pin.revision
        / REPO_MANIFEST_NAME
    )


def _snapshot_path(home: Path, pin: Pin) -> Path | None:
    """The manifest inside the PULLED weights, when they are at this pin.

    The pull fetches the whole repo at the pin, so the manifest rides in with
    the bytes (section 5). Read from there when it is there: it is the copy that
    is provably beside the weights being served, and no Hub call can disagree
    with it.

    The STAMP is what says which revision those bytes are, and it is checked —
    a directory left over from the previous pin holds the previous manifest, and
    serving new bytes under an old manifest is the substitution this whole phase
    exists to prevent.
    """
    import json

    root = home / "voices" / pin.id
    if not root.is_dir():
        return None
    for backend_dir in sorted(root.iterdir()):
        stamp = backend_dir / "crucible-pull.json"
        if not stamp.is_file():
            continue
        try:
            record = json.loads(stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("revision") != pin.revision or record.get("hf_repo") != pin.hf_repo:
            continue
        found = backend_dir / REPO_MANIFEST_NAME
        if found.is_file():
            return found
        # THE WEIGHTS ARE HERE AND THE MANIFEST IS NOT, which is a revision that
        # carries no manifest and is said as such rather than quietly falling
        # through to the Hub to be told the same thing more slowly.
        raise VoiceError(
            f"voice_manifest_missing: {pin.hf_repo}@{pin.revision[:12]} is pulled "
            f"into {backend_dir} and carries no {REPO_MANIFEST_NAME}. A pinned "
            "repo whose manifest is missing is not served, and its facts are not "
            f"read from anywhere else — see {pin.path}"
        )
    return None


def fetch_repo_manifest(home: Path, pin: Pin) -> tuple[str, Path]:
    """The `crucible-voice.toml` at this pin, and where it was read from.

    Three places, in order: the pulled weights, this home's manifest cache, and
    the Hub. Only the last one costs anything, and only once per (repo, sha).

    `hf_hub_download` FETCHES THE FILE ALONE, which is what lets
    `GET /v1/voices` list an uninstalled voice with its real facts instead of
    asking a person to download 8.5 GB before the row can say what the voice is.
    """
    found = _snapshot_path(home, pin)
    if found is not None:
        return _read(found), found
    cached = _cache_path(home, pin)
    if cached.is_file():
        return _read(cached), cached

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise VoiceError(f"huggingface_hub is not importable: {exc}") from exc

    from .config import config_path
    from .weights import hf_token_at

    cached.parent.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=pin.hf_repo,
            filename=REPO_MANIFEST_NAME,
            revision=pin.revision,
            local_dir=str(cached.parent),
            token=hf_token_at(config_path(home)),
        )
    except EntryNotFoundError as exc:
        raise VoiceError(
            f"voice_manifest_missing: {pin.hf_repo}@{pin.revision[:12]} carries no "
            f"{REPO_MANIFEST_NAME}. A voice's facts travel with its weights in the "
            "same commit (PHASE21 section 2.1); a pinned repo whose manifest is "
            "missing is not served, and its band, caps and sampling are not read "
            f"from any other source. Pin a revision that carries one, or run "
            f"`crucible voices card` and commit the manifest — see {pin.path}: {exc}"
        ) from exc
    except RevisionNotFoundError as exc:
        raise VoiceError(
            f"voice_manifest_missing: {pin.hf_repo} has no revision "
            f"{pin.revision}; {pin.path} pins a commit that repo does not "
            f"have: {exc}"
        ) from exc
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise VoiceError(
            f"voice_manifest_unreadable: {pin.hf_repo} is private, gated or does "
            f"not exist, and this server has no HF token that opens it (set "
            f"$HF_TOKEN or [hf] token in {config_path(home)}): {exc}"
        ) from exc
    except Exception as exc:
        # NOT `voice_manifest_missing`. "This revision has no manifest" is a fact
        # about a commit and is permanent; "the Hub did not answer" is a fact
        # about a minute. Telling a reader the first about the second would send
        # them to re-commit a manifest that is already there.
        raise VoiceError(
            f"voice_manifest_unreadable: could not fetch {REPO_MANIFEST_NAME} from "
            f"{pin.hf_repo}@{pin.revision[:12]}: {type(exc).__name__}: {exc}"
        ) from exc
    if not cached.is_file():  # pragma: no cover - the hub just wrote it
        raise VoiceError(
            f"voice_manifest_unreadable: {pin.hf_repo}@{pin.revision[:12]} was "
            f"fetched but there is no {cached}"
        )
    return _read(cached), cached


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoiceError(f"could not read {path}: {exc}") from exc


# ------------------------------------------------------------ the whole set


def voice_for_pin(pin: Pin) -> Any:
    """One pin -> one `VoiceManifest`, or the named refusal that stops it.

    The whole of "a pinned repo whose manifest is missing is NOT served and is
    NOT read from any other source" lives here, and so does
    `engine_footprint_unset`. It is a function rather than a loop body because
    `PUT /v1/voices/{id}` calls it BEFORE it writes a pin: a door that wrote the
    row first and discovered the refusal on the next listing would leave a
    server holding a pin nothing can load, and the operator would be told about
    it by a catalog that had stopped working.
    """
    from .config import crucible_home, tts_engine_footprints

    home = crucible_home()
    text, path = fetch_repo_manifest(home, pin)
    repo = parse_repo_manifest(text, path)
    engine = repo.voice["narrator_engine"]
    footprint = tts_engine_footprints(home).get(engine)
    if footprint is None:
        raise VoiceError(
            f"engine_footprint_unset: voice {pin.id!r} is served by {engine!r} "
            f"and this server's config states no [tts.{engine}] table, so there "
            "is no memory estimate and no serving width for it. A voice's facts "
            "travel with its weights and a BOX's facts stay with the box "
            "(PHASE21 section 2.3) — nothing here is defaulted. Run `crucible "
            "init` on this machine, or write the table into config.toml by hand"
        )
    return merge(repo, pin, footprint)


def pinned_voices() -> dict[str, Any]:
    """Every pinned voice this host offers, by id, as a `VoiceManifest`.

    `crucible/voices.py::load_all_voices` calls this FIRST — the lowest
    precedence — so a packaged manifest still wins on a shared id while section
    8.1 is true, and adding a pin regresses nothing.
    """
    return {voice_id: voice_for_pin(pin) for voice_id, pin in load_pins().items()}
