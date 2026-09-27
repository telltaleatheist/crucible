from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict

from .narratorengines import HIGGS_V3, EngineFootprint, declared_tts_footprints
from .tomltable import (
    HF_REPO_PATTERN,
    REVISION_PATTERN,
    VOICE_ID_PATTERN,
    check_table,
)
from .voices import (
    MANIFEST_REPO,
    MAX_CHARS_BASES,
    PACE_BASES,
    PINS_FILE,
    SERVING_KEYS,
    VoiceError,
    VoiceManifest,
    home_voices_dir,
    parse_document,
    voices_dir,
    voices_dir_is_overridden,
)

REPO_MANIFEST_NAME = "crucible-voice.toml"

REPO_SCHEMA = 1

MANIFEST_CACHE_DIRNAME = "voice-manifests"

_REPO_VOICE_REQUIRED: dict[str, type] = {
    "display": str,
    "kind": str,
    "narrator_engine": str,
    "language": str,
    "sample_rate": int,
}

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

_PACE_BASIS = "basis"
_PACE_MEASURED_FROM = "measured_from"
_PACE_INHERITED_FROM = "inherited_from"

_PACE_BASIS_PROSE: dict[str, str] = {
    "measured": _PACE_MEASURED_FROM,
    "inherited": _PACE_INHERITED_FROM,
}

_ARM_REQUIRED: dict[str, type] = {
    "sampling": dict,
}
_ARM_OPTIONAL: dict[str, type] = {
    "max_chars": int,
    "max_chars_basis": str,
    "sampling_reason": str,
    "clips": object,
}

_PIN_REQUIRED: dict[str, type] = {"hf_repo": str, "revision": str}


@dataclass(frozen=True)
class Pin:
    id: str
    hf_repo: str
    revision: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "path": str(self.path),
        }


def packaged_pins_path() -> Path:
    return voices_dir() / PINS_FILE


def home_pins_path() -> Path:
    return home_voices_dir() / PINS_FILE


def parse_pins(text: str, path: Path) -> dict[str, Pin]:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path}: not valid TOML: {exc}") from exc
    pins: dict[str, Pin] = {}
    for voice_id in sorted(document):
        where = f"{path.name} [{voice_id}]"
        if not VOICE_ID_PATTERN.match(voice_id):
            raise VoiceError(
                f"{where}: {voice_id!r} is not a voice id (lower-case letters, "
                "digits, dot, dash and underscore, starting with a letter or "
                "digit, at most 64 characters). A pins file is one table per voice id and nothing else"
            )
        block = document[voice_id]
        if not isinstance(block, dict):
            raise VoiceError(
                f"{where}: must be a table of hf_repo and revision, got "
                f"{type(block).__name__}. A pins file has no top-level keys — "
                "every row is [<voice id>]"
            )
        check_table(where, block, _PIN_REQUIRED, {}, error=VoiceError)
        if not HF_REPO_PATTERN.match(block["hf_repo"]):
            raise VoiceError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not REVISION_PATTERN.match(block["revision"]):
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


_parse_pins = parse_pins


def packaged_pins() -> dict[str, Pin]:
    path = packaged_pins_path()
    if not path.is_file():
        return {}
    return parse_pins(_read(path), path)


def load_pins() -> dict[str, Pin]:
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
        pins.update(parse_pins(text, path))
    return pins


def write_home_pin(voice_id: str, hf_repo: str, revision: str) -> Pin:
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
    written = parse_pins(text, path)
    if voice_id not in written:
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


def _check_no_machine_facts(where: str, table: dict[str, Any]) -> None:
    for key in sorted(set(table) & set(_REFUSED_IN_REPO)):
        raise VoiceError(
            f"{where}: carries {key}, which a repo manifest may not state — "
            f"{_REFUSED_IN_REPO[key]}"
        )


def _check_arm_cap(where: str, block: dict[str, Any]) -> str | None:
    cap = block.get("max_chars")
    basis = block.get("max_chars_basis")
    if cap is None:
        if basis is not None:
            raise VoiceError(
                f"{where}: states max_chars_basis {basis!r} and no max_chars. "
                "The basis says how a cap was got and there is no cap; drop it, "
                "or state the number it describes"
            )
        return None
    if basis is None:
        raise VoiceError(
            f"{where}: states max_chars {cap} and no max_chars_basis. A cap is "
            "either a number a sweep produced on these weights on this arm or a "
            "number somebody wrote down so the arm could be served, and a file "
            "that cannot say which ships the second as the first"
        )
    if basis not in MAX_CHARS_BASES:
        raise VoiceError(
            f"{where}: max_chars_basis {basis!r} is not one of "
            f"{sorted(MAX_CHARS_BASES)}. A cap is either a number a sweep "
            "produced on these weights on this arm or a number somebody wrote "
            "down so the arm could be served — thirdreich shipped a "
            "`higgs_max_chars_mlx: 900` that was never measured, and a schema "
            "that cannot say so ships it as a measured fact"
        )
    return basis


def _repo_pace(
    where: str, table: dict[str, Any]
) -> tuple[dict[str, Any], str, str | None, str | None]:
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
    owed = _PACE_BASIS_PROSE[basis]
    refused = _PACE_BASIS_PROSE["inherited" if basis == "measured" else "measured"]

    prose = table.get(owed)
    if prose is not None and not isinstance(prose, str):
        raise VoiceError(
            f"{where}: {owed} must be prose, got {type(prose).__name__}"
        )
    if prose is None or prose.strip() == "":
        raise VoiceError(
            f"{where}: {_PACE_BASIS} is {basis!r} and there is no {owed}. "
            + (
                "A measured pace was measured on something — a checkpoint, a "
                "ladder, a sample count — and the number is only worth what the "
                "reader can find out about where it came from"
                if basis == "measured"
                else "An inherited pace was measured on OTHER weights, and WHICH "
                "ones decides whether it is near enough: a sibling checkpoint of "
                "the same corpus is (mistborn 13.29/13.33/13.76 across three "
                "retrains), a different corpus two versions back is the "
                "deathstalker defect (16.64 onto weights that measured 15.91). "
                "Name the run and checkpoint it came from, and say why these "
                "weights have no ladder yet"
            )
        )
    if table.get(refused) is not None:
        raise VoiceError(
            f"{where}: {_PACE_BASIS} is {basis!r} and it also carries {refused}. "
            f"Each basis owes exactly its own sentence — {owed} — and a "
            f"{refused} beside a {basis!r} pace is prose about a measurement this "
            "voice did not make. It is the rule estimate_basis has, for the same "
            "reason: a reason attached to the wrong basis is a reason a reader "
            "will trust the next time it means something"
        )
    return (
        {
            k: v
            for k, v in table.items()
            if k not in (_PACE_BASIS, _PACE_MEASURED_FROM, _PACE_INHERITED_FROM)
        },
        basis,
        prose if basis == "measured" else None,
        prose if basis == "inherited" else None,
    )


class VoiceDocument(TypedDict):
    display: str
    kind: str
    narrator_engine: str
    language: str
    sample_rate: int


@dataclass(frozen=True)
class RepoManifest:
    schema: int
    voice: VoiceDocument
    pace: dict[str, Any] | None
    pace_basis: str | None
    measured_from: str | None
    inherited_from: str | None
    arms: dict[str, dict[str, Any]]
    max_chars_basis: dict[str, str | None]
    takes: list[dict[str, Any]] | None
    path: Path


def parse_repo_manifest(text: str, path: Path) -> RepoManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path.name}: not valid TOML: {exc}") from exc
    schema = _check_top_level(path, document)
    voice = document["voice"]
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
    inherited_from: str | None = None
    if "pace" in voice:
        if not isinstance(voice["pace"], dict):
            raise VoiceError(f"{path.name}: [voice.pace] must be a table")
        pace, pace_basis, measured_from, inherited_from = _repo_pace(
            f"{path.name} [voice.pace]", voice["pace"]
        )
    arms, bases = _repo_arms(path, voice)

    takes = voice.get("takes")
    if takes is not None and not isinstance(takes, list):
        raise VoiceError(
            f"{path.name}: [[voice.takes]] must be a non-empty list of tables; a "
            "voice with no ladder simply omits it and gets take 0"
        )

    return RepoManifest(
        schema=schema,
        voice=VoiceDocument(
            display=scalars["display"],
            kind=scalars["kind"],
            narrator_engine=scalars["narrator_engine"],
            language=scalars["language"],
            sample_rate=scalars["sample_rate"],
        ),
        pace=pace,
        pace_basis=pace_basis,
        measured_from=measured_from,
        inherited_from=inherited_from,
        arms=arms,
        max_chars_basis=bases,
        takes=takes,
        path=path,
    )


def _check_top_level(path: Path, document: dict[str, Any]) -> int:
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
    if not isinstance(document["voice"], dict):
        raise VoiceError(f"{path.name}: [voice] must be a table")
    return schema


def _repo_arms(
    path: Path, voice: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
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
    bases: dict[str, str | None] = {}
    for arm in sorted(arms_table):
        where = f"{path.name} [voice.arms.{arm}]"
        block = arms_table[arm]
        if not isinstance(block, dict):
            raise VoiceError(f"{where}: must be a table")
        _check_no_machine_facts(where, block)
        check_table(where, block, _ARM_REQUIRED, _ARM_OPTIONAL, error=VoiceError)
        bases[arm] = _check_arm_cap(where, block)
        arms[arm] = {k: v for k, v in block.items() if k != "max_chars_basis"}
    return arms, bases


def merge(repo: RepoManifest, pin: Pin, footprint: EngineFootprint) -> VoiceManifest:
    facts = footprint.to_dict()
    serving = {key: value for key, value in facts.items() if key in SERVING_KEYS}
    machine = {key: value for key, value in facts.items() if key not in SERVING_KEYS}
    voice: dict[str, Any] = {
        "id": pin.id,
        **repo.voice,
        "pace": dict(repo.pace) if repo.pace is not None else {},
        "backends": {
            arm: {
                "hf_repo": pin.hf_repo,
                "revision": pin.revision,
                **machine,
                **block,
            }
            for arm, block in repo.arms.items()
        },
    }
    if repo.voice["narrator_engine"] == HIGGS_V3:
        voice["serving"] = serving
    if repo.takes is not None:
        voice["takes"] = repo.takes
    manifest = parse_document({"voice": voice}, repo.path, pin.id)
    backends = {
        arm: replace(spec, max_chars_basis=repo.max_chars_basis[arm])
        for arm, spec in manifest.backends.items()
    }
    return replace(
        manifest,
        backends=backends,
        manifest_source=MANIFEST_REPO,
        pace_basis=repo.pace_basis,
        inherited_from=repo.inherited_from,
    )


def _cache_path(home: Path, pin: Pin) -> Path:
    return (
        home
        / MANIFEST_CACHE_DIRNAME
        / pin.hf_repo.replace("/", "--")
        / pin.revision
        / REPO_MANIFEST_NAME
    )


def _snapshot_path(home: Path, pin: Pin) -> Path | None:
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
        raise VoiceError(
            f"voice_manifest_missing: {pin.hf_repo}@{pin.revision[:12]} is pulled "
            f"into {backend_dir} and carries no {REPO_MANIFEST_NAME}. A pinned "
            "repo whose manifest is missing is not served, and its facts are not "
            f"read from anywhere else — see {pin.path}"
        )
    return None


def fetch_repo_manifest(home: Path, pin: Pin) -> tuple[str, Path]:
    found = _snapshot_path(home, pin)
    if found is not None:
        return _read(found), found
    cached = _cache_path(home, pin)
    if cached.is_file():
        return _read(cached), cached
    _download(home, pin, cached)
    if not cached.is_file():
        raise VoiceError(
            f"voice_manifest_unreadable: {pin.hf_repo}@{pin.revision[:12]} was "
            f"fetched but there is no {cached}"
        )
    return _read(cached), cached


def _download(home: Path, pin: Pin, cached: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
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
        raise VoiceError(
            f"voice_manifest_unreadable: could not fetch {REPO_MANIFEST_NAME} from "
            f"{pin.hf_repo}@{pin.revision[:12]}: {type(exc).__name__}: {exc}"
        ) from exc


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoiceError(f"could not read {path}: {exc}") from exc


def footprint_unset(engine: str) -> str:
    from .backend import CUDA_LINUX, MLX_DARWIN
    from .config import config_path, crucible_home

    written_by_init = {
        entry.engine
        for arm in (CUDA_LINUX, MLX_DARWIN)
        for entry in declared_tts_footprints(arm)
    }
    if engine in written_by_init:
        return (
            f"`crucible init --force` on this machine writes the [tts.{engine}] "
            "table for its backend (pass `--token <token>` from `crucible token "
            "--show` to keep the token apps already hold)"
        )
    return (
        f"no Crucible command writes a [tts.{engine}] table yet: measure the "
        f"engine here, then add [tts.{engine}] to {config_path(crucible_home())} "
        f"with the keys [tts.{HIGGS_V3}] carries (memory_bytes_estimate, "
        "estimate_basis, max_num_seqs, max_num_seqs_note)"
    )


def voice_for_pin(pin: Pin) -> VoiceManifest:
    from .config import crucible_home, tts_engine_footprints
    from .errors import ConfigError

    home = crucible_home()
    text, path = fetch_repo_manifest(home, pin)
    repo = parse_repo_manifest(text, path)
    engine = repo.voice["narrator_engine"]
    try:
        footprint = tts_engine_footprints(home).get(engine)
    except ConfigError as exc:
        raise VoiceError(
            f"config_unreadable: voice {pin.id!r} cannot be sized because this "
            f"machine's config did not read: {exc}"
        ) from exc
    if footprint is None:
        raise VoiceError(
            f"engine_footprint_unset: voice {pin.id!r} is served by {engine!r} "
            f"and this server's config states no [tts.{engine}] table, so there "
            "is no memory estimate and no serving width for it. A voice's facts "
            "travel with its weights and a BOX's facts stay with the box "
            f"(PHASE21 section 2.3) — nothing here is defaulted. "
            f"{footprint_unset(engine)}"
        )
    return merge(repo, pin, footprint)


_REFUSALS_SAID: set[tuple[str, str]] = set()

_log = logging.getLogger(__name__)


def _say_once(voice_id: str, pin: Pin, why: str) -> None:
    key = (voice_id, why)
    if key in _REFUSALS_SAID:
        return
    _REFUSALS_SAID.add(key)
    _log.warning(
        "voice %r is pinned to %s@%s in %s but not served: %s",
        voice_id,
        pin.hf_repo,
        pin.revision[:12],
        pin.path,
        " ".join(why.split()),
    )


def load_pinned() -> tuple[dict[str, VoiceManifest], dict[str, tuple[Pin, str]]]:
    voices: dict[str, VoiceManifest] = {}
    refused: dict[str, tuple[Pin, str]] = {}
    for voice_id, pin in load_pins().items():
        try:
            voices[voice_id] = voice_for_pin(pin)
        except VoiceError as exc:
            refused[voice_id] = (pin, str(exc))
            _say_once(voice_id, pin, str(exc))
    return voices, refused


def pinned_voices() -> dict[str, VoiceManifest]:
    return load_pinned()[0]
