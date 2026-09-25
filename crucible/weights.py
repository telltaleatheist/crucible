"""Weights on disk — `~/.crucible/<family>/<id>/<backend>/`.

Pulled by `crucible models pull <id>` (or `crucible voices pull <id>`) with
`huggingface_hub` at the manifest's pinned revision, never from GitHub Releases
(PHASE2-LLM.md section 1). A pull that finishes writes `crucible-pull.json`
beside the weights; nothing downstream treats a directory without that stamp as
installed, so an interrupted 19 GB download can never be handed to an engine as
if it were a model.

`family` is `models`, `voices` or `rvc`, and it comes off the manifest
(`weights_family`) rather than being passed around. Those are separate namespaces
— nothing stops a voice being called `qwen3.5-9b` — and one directory holding two
kinds would let a `crucible voices pull` overwrite a 19 GB model with an 8.5 GB
checkpoint and leave a stamp that reads as installed to either.

Three shapes of pull, and each of the second two exists because of how a real
repo is laid out rather than because anybody wanted another shape:

* `pull` snapshot-downloads a whole repo, which is what a model or a voice is.
* `pull_archive` fetches ONE file, verifies its SHA-256 against the manifest and
  unpacks it, which is what an RVC model is (see `crucible/rvcmodels.py`).
* `pull_files` fetches NAMED files and places each one exactly where an engine
  looks for it, which is what ultimate-rvc's shared base assets are: four files
  scattered through a repo that also holds six pretrained GAN checkpoints, read
  back from a tree with different names (see `crucible/rvcbase.py`). It is also
  what a separator checkpoint is — two files out of a mirror of every UVR model
  there is, landing under the two names audio-separator resolves by (see
  `crucible/denoisemodels.py`), which is why it takes a `stamp_name`: that
  target root holds one set per model rather than one set.

All three write the same stamp, so nothing downstream has to know which ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .config import Config
from .errors import CrucibleError

HF_TOKEN_ENV = "HF_TOKEN"
STAMP_NAME = "crucible-pull.json"

#: WHERE A BACKEND BLOCK'S BYTES COME FROM (PHASE18-UNCERTIFIED.md section 3).
#:
#:   PINNED  bytes Crucible FETCHED at something it can name — an HF repo at a
#:           commit for a model, a voice or an RVC archive, and the llama.cpp
#:           release for the engine row (`crucible/llamacpp.py`). This module
#:           fetches it, stamps it, and the catalog owns it.
#:   LOCAL   a directory somebody else put on the serving machine. This module
#:           NEVER fetches, stamps or deletes it, and it may vanish between
#:           jobs without that being an error.
#:
#: The vocabulary lives here rather than in `crucible/voices.py` because this is
#: the module that ACTS on the difference — everything downstream only reports
#: it — and because `crucible/llamacpp.py` needs the word too and has no
#: business importing a voice schema.
PINNED = "pinned"
LOCAL = "local"

#: What to call the thing, and which command pulls it, per weights family. A
#: refusal that says "run `crucible models pull deathstalker`" for a voice sends
#: its reader to a command that will tell them there is no such model.
_FAMILY_WORDS: dict[str, tuple[str, str]] = {
    "models": ("model", "crucible models pull"),
    "voices": ("voice", "crucible voices pull"),
    "rvc": ("RVC model", "crucible rvc pull"),
}


@runtime_checkable
class WeightsSubject(Protocol):
    """What this module needs from a manifest, model or voice alike.

    Structural rather than a base class: `ModelManifest` and `VoiceManifest`
    describe different things and share no fields beyond these, and inventing a
    parent for them would put the id and the path somewhere neither schema's
    reader would look for them.
    """

    id: str
    path: Path
    weights_family: str


@runtime_checkable
class WeightsSource(Protocol):
    """What this module needs from a backend block.

    `files` is the one place "which files does this backend fetch" is
    answered, and every spec answers it. An EMPTY tuple means *the repository
    is the weights* — `snapshot_download` of the whole thing, which is what
    every safetensors backend does and what this module did unconditionally
    until PHASE15. A NON-EMPTY tuple means *these files and no others*, which
    is what `llama-windows` needs: `unsloth/Qwen3.8-27B-GGUF` holds every
    quantization, hundreds of gigabytes, and a row there IS one of them.

    Two things read it and they must not be able to disagree: `pull` passes it
    as `allow_patterns`, and `installed` requires every one of them present.
    That second half is what makes `dots-ocr` report `installed: false` when
    the text tower arrived and the vision projector did not (PHASE15-HOST.md
    3.10, fact 2 — the mmproj is not optional), and what makes 3.5's *"the
    catalog's `installed` list is the input to the host's weights migration"*
    exact rather than approximately right.
    """

    backend: str
    #: BOTH None ON A LOCAL VOICE BLOCK, which names a `path` instead
    #: (PHASE18-UNCERTIFIED.md section 3). Declared nullable here rather than
    #: left saying `str`, because this protocol is what a reader consults
    #: before writing `spec.revision[:12]` — and every such line is a
    #: TypeError on the shape that declares no pin. `local_source` is the
    #: question to ask first; it answers None for the three spec types that
    #: can only ever be pinned.
    hf_repo: str | None
    revision: str | None
    files: tuple[str, ...]


class WeightsError(CrucibleError):
    """Weights are missing, half-pulled, or could not be fetched."""


class PullCancelled(CrucibleError):
    """A pull stopped because its caller's progress hook said to stop.

    Its own type and NOT a `WeightsError`, because the two must never be caught
    together: a failed pull is news and a cancelled one is what was asked for.
    Every `except Exception` in this module re-raises it untouched for that
    reason — wrapped in a `WeightsError` it would reach an operator as "pulling
    … failed", which is a lie about their own cancel.

    Whatever was on disk is REMOVED before this leaves the module. R6 does not
    apply and PHASE13-OPERATOR.md section 3.3 says why: half a safetensors file
    is not partial work anybody can resume or use, and leaving it would make the
    next `installed()` read a directory with no stamp — which is honest, but
    also several gigabytes of nothing.
    """


#: What a caller is told while bytes are arriving: `(done, total|None, file)`.
#:
#: `total` is None where the server did not say how big the file is, which
#: happens and must not be reported as zero. `file` is the name the hub is
#: fetching, so a progress line says which of a repo's forty shards is in
#: flight.
#:
#: THE HOOK IS ALSO THE CANCEL POINT, and that is not a second job bolted onto
#: it — it is the only place in a `snapshot_download` where this process's own
#: code runs. A hook that raises `PullCancelled` stops the pull; nothing else
#: can, because a thread cannot be killed and the hub takes no cancellation
#: token. See `crucible/tasks.py`, whose `DELETE /v1/tasks/{id}` is the caller.
ProgressHook = Callable[[int, "int | None", str], None]


@dataclass(frozen=True)
class InstalledWeights:
    path: Path
    hf_repo: str | None
    revision: str | None
    bytes: int
    #: When the pull finished. None for a LOCAL entry: no download ever
    #: happened, and a timestamp there would make a directory that has sat on
    #: the machine for a month read as a recent install.
    pulled: str | None
    #: `"pinned"` or `"local"` (`crucible/voices.py`). A LOCAL entry is not an
    #: install: nothing was fetched, there is no stamp, `hf_repo` and `pulled`
    #: are None because no download ever happened, and the bytes may be gone
    #: by the next job. It is reported through this type anyway because every
    #: caller is asking the same question — "can this be served, and from
    #: where" — and a second return type would make each of them branch before
    #: it could read a path.
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "bytes": self.bytes,
            "pulled": self.pulled,
            "source": self.source,
        }


def weights_root(config: Config, family: str) -> Path:
    return config.home / family


def weights_dir(
    config: Config, family: str, subject_id: str, backend_kind: str
) -> Path:
    return weights_root(config, family) / subject_id / backend_kind


def stamp_path(
    config: Config, family: str, subject_id: str, backend_kind: str
) -> Path:
    return weights_dir(config, family, subject_id, backend_kind) / STAMP_NAME


# ------------------------------------------------------ one copy, two rows
#
# PHASE22-DECIDE.md section 2.9, Owen 2026-09-23: *"One copy on disk, two fit
# rows in the catalog."* A model manifest may declare `[model] weights_of =
# "<base id>"` (crucible/manifests.py, `resolve_weights_of`): an ALIAS. Its
# weights are the base's download, in the base's folder, and what it owns on
# disk is only what its block names beyond the base's (`extra_files` — the
# llama-windows projector). Every function below that touches the store asks
# `_store_id` where that is, so an alias cannot be downloaded twice by any door.


def _store_id(subject: Any) -> str:
    """The id whose folder holds this subject's weights.

    `getattr` because only a MODEL or an ASR manifest can be an alias: a voice
    and an RVC model have no `weights_of` in their schemas, and this is asking
    which schema the subject is, exactly as `local_source` does.
    """
    weights_of = getattr(subject, "weights_of", None)
    return subject.id if weights_of is None else weights_of


def _extra_files(subject: Any, spec: WeightsSource) -> tuple[str, ...]:
    """What an alias owns on this backend; () for anything that is not one."""
    if getattr(subject, "weights_of", None) is None:
        return ()
    return subject.extra_files(spec.backend)


def subject_dir(config: Config, subject: WeightsSubject, backend_kind: str) -> Path:
    """Where THIS subject's weights are for `backend_kind` — the base's folder
    for an alias. The one question every reader of the layout should ask."""
    return weights_dir(config, subject.weights_family, _store_id(subject), backend_kind)


#: Beside the base's stamp, one per alias PULLED into that folder. It is the
#: alias's statement that it was asked for here, and it is what makes "an alias
#: exists on this machine" a fact on backends where the alias owns no file of
#: its own (cuda-linux: the whole repo is shared). Without it, removing the
#: base could never be refused there, and removing the alias could never end
#: the refusal. It does NOT decide `installed` — the ruling is "the base's
#: stamp AND its own extra files present" — it decides only whether removing
#: the base would take a pulled alias away.
ALIAS_RECORD_PREFIX = "crucible-alias-"


def alias_record_path(config: Config, alias: Any, backend_kind: str) -> Path:
    return subject_dir(config, alias, backend_kind) / f"{ALIAS_RECORD_PREFIX}{alias.id}.json"


class WeightsShared(WeightsError):
    """`weights_shared`: a base's folder is also an alias's, and it was asked
    to go. Carries the aliases by id so a refusal can name each of them."""

    code = "weights_shared"

    def __init__(self, base_id: str, backend_kind: str, aliases: Sequence[str]) -> None:
        self.base_id = base_id
        self.backend = backend_kind
        self.aliases = tuple(aliases)
        named = ", ".join(repr(a) for a in self.aliases)
        super().__init__(
            f"weights_shared — {base_id!r}'s {backend_kind} weights are also the "
            f"weights of {named}, pulled on this machine. Removing them would take "
            f"{'that model' if len(self.aliases) == 1 else 'those models'} away "
            f"too. Remove {named} first (`crucible remove model <id>` removes only "
            f"what an alias owns), then {base_id!r}"
        )


def aliases_holding(
    config: Config, manifest: WeightsSubject, backend_kind: str
) -> tuple[str, ...]:
    """The aliases, by id, that were pulled into this base's folder and are
    installed there now. Empty for anything that is not a model base.

    Installed AND recorded, both: a record beside an alias whose projector was
    deleted by hand names an alias that is not there any more, and refusing the
    base for it would leave nothing any door could remove to lift the refusal.
    """
    from .asrmodels import AsrManifest, asr_aliases_of
    from .manifests import ModelManifest, aliases_of

    if getattr(manifest, "weights_of", None) is not None:
        return ()
    # Two catalogs can alias (models/ and, since 2026-09-24, asr/); each asks
    # its own directory who shares its folder.
    if isinstance(manifest, ModelManifest):
        aliases = aliases_of(manifest)
    elif isinstance(manifest, AsrManifest):
        aliases = asr_aliases_of(manifest)
    else:
        return ()
    holding: list[str] = []
    for alias in aliases:
        if not alias.supports(backend_kind):
            continue
        if not alias_record_path(config, alias, backend_kind).is_file():
            continue
        if installed(config, alias, alias.spec(backend_kind)) is None:
            continue
        holding.append(alias.id)
    return tuple(holding)


def refuse_if_shared(
    config: Config, manifest: WeightsSubject, backend_kind: str
) -> None:
    """`WeightsShared` if deleting this subject's folder would take an alias."""
    holding = aliases_holding(config, manifest, backend_kind)
    if holding:
        raise WeightsShared(manifest.id, backend_kind, holding)


def missing_files(directory: Path, spec: WeightsSource) -> tuple[str, ...]:
    """The files this spec NAMES that are not on disk, in the spec's order.

    Empty for a spec that names none, which is every safetensors backend: the
    repository is the weights and `snapshot_download` either wrote it or did
    not. For `llama-windows` it is the whole of "is this subject complete" —
    a `dots-ocr` directory holding the text tower and no `mmproj` is a
    directory `llama-server` will start against, answer `/v1/models` from,
    and then refuse every page (3.10, fact 2).
    """
    return tuple(name for name in spec.files if not (directory / name).is_file())


def local_source(spec: Any) -> Path | None:
    """The directory a LOCAL spec names, or None for a pinned one.

    `getattr` and not an attribute on `WeightsSource`, because only a VOICE can
    be local today (PHASE18-UNCERTIFIED.md section 3). A model is a catalog
    subject with a download, an installer, a host migration and a `crucible
    models pull` behind it, and not one of those has been designed for bytes
    this server does not own; adding the member to the protocol would advertise
    a capability three spec types do not have.
    """
    found = getattr(spec, "local_path", None)
    return None if found is None else Path(found)


def _local_installed(directory: Path, spec: Any) -> InstalledWeights | None:
    """A local directory reported as servable, or None when it is not there.

    NOTHING IS VERIFIED HERE beyond existence and the files the spec names. The
    bytes were not fetched by this server, there is no stamp to compare a pin
    against, and the `identity` on the row is the registrant's word (section
    3.1). `pulled` is None because no download ever happened — reporting a
    timestamp would make a directory that has sat there for a month look like a
    recent install.

    `bytes` is a stat walk rather than a read: it is the same
    `directory_bytes` every other caller uses and it costs nothing next to an
    8 GiB checkpoint load.
    """
    if not directory.is_dir():
        return None
    if missing_files(directory, spec):
        return None
    return InstalledWeights(
        path=directory,
        hf_repo=None,
        # `spec.identity` and NOT `getattr(spec, "identity", None)`. The
        # `getattr` in `local_source` asks a question every spec type may
        # answer no to; this one is reached only after it answered yes, so a
        # spec that names a path and no identity is a schema the loader let
        # through — and a None here would be reported as this voice's
        # revision. Loud by name is the answer to that, not a default.
        revision=spec.identity,
        bytes=directory_bytes(directory),
        pulled=None,
        source=LOCAL,
    )


def installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights | None:
    """The installed weights for this (model or voice, backend), or None.

    A stamp naming a different revision than the manifest pins is *not* installed:
    the manifest moved, and serving the old bytes under the new id would be a
    silent substitution.

    A stamp beside a MISSING NAMED FILE is not installed either, and that is
    not the same check wearing a second hat: the stamp says a pull finished,
    and `spec.files` says what finishing means for this backend. A subject
    whose mmproj was deleted by hand has a perfectly good stamp.

    A LOCAL SPEC IS NEVER STAMPED and is answered off the directory itself —
    see `_local_installed`.
    """
    local = local_source(spec)
    if local is not None:
        return _local_installed(local, spec)

    # AN ALIAS READS ITS BASE'S FOLDER AND ITS BASE'S STAMP (section 2.9): the
    # pins are equal by `resolve_weights_of`, so the base's stamp answers for
    # the alias's pin as well. `missing_files` below then asks for every file
    # the ALIAS's block names — the base's GGUF and its own projector — so a
    # base-only pull reads as not installed here, honestly.
    directory = subject_dir(config, manifest, spec.backend)
    stamp = directory / STAMP_NAME
    if not stamp.is_file():
        return None
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["revision"] != spec.revision or record["hf_repo"] != spec.hf_repo:
        return None
    if missing_files(directory, spec):
        return None
    extras = _extra_files(manifest, spec)
    return InstalledWeights(
        path=directory,
        hf_repo=record["hf_repo"],
        revision=record["revision"],
        source=PINNED,
        # THE BYTES THIS SUBJECT OWNS. An alias's download is its base's and is
        # counted once, on the base (section 2.9); what the alias adds is its
        # extra files, which is also exactly what removing it frees — 0 where
        # it adds none.
        bytes=(
            record["bytes"]
            if getattr(manifest, "weights_of", None) is None
            else sum((directory / name).stat().st_size for name in extras)
        ),
        pulled=record["pulled"],
    )


def require_installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights:
    """Installed weights, or `model_not_installed` / `voice_not_installed`."""
    found = installed(config, manifest, spec)
    if found is not None:
        return found

    local = local_source(spec)
    if local is not None:
        # NOT "not installed", and not an instruction to pull. Nothing here was
        # ever installed, and there is no command that would fetch it: the
        # directory belongs to whoever put it there (section 3), so a refusal
        # that said `crucible voices pull` would send its reader to a command
        # that cannot help. The bytes going away mid-run is EXPECTED of a
        # screening merge — 8 GiB deleted the moment its renders land — so this
        # is a plain statement of what is not there.
        absent = missing_files(local, spec)
        if local.is_dir() and absent:
            raise WeightsError(
                f"voice {manifest.id!r} names {local} for {spec.backend} and the "
                f"directory is there, but {len(absent)} of the file(s) it needs are "
                f"not: {', '.join(absent)}"
            )
        raise WeightsError(
            f"voice {manifest.id!r} names {local} for {spec.backend} and there is no "
            "such directory on this server. Crucible does not fetch a local voice's "
            "weights and cannot replace them — whatever put them there has to put "
            "them back, or the voice's manifest should be removed"
        )

    family = manifest.weights_family
    noun, command = _FAMILY_WORDS[family]
    directory = subject_dir(config, manifest, spec.backend)
    stamp = directory / STAMP_NAME
    base = getattr(manifest, "weights_base", None)
    if base is not None:
        # AN ALIAS SAYS WHICH HALF IS MISSING: the shared download, or its own
        # files beside it. "No weights at <the base's folder>" would send its
        # reader to look at a directory that may be full.
        extras = _extra_files(manifest, spec)
        if installed(config, base, base.spec(spec.backend)) is None:
            raise WeightsError(
                f"{noun} {manifest.id!r} shares the weights of {base.id!r}, and "
                f"{base.id!r} is not installed for {spec.backend} — run "
                f"`{command} {manifest.id}`, which pulls {base.id!r}'s download"
                + (f" plus {', '.join(extras)}" if extras else "")
                + " into one folder"
            )
        absent = missing_files(directory, spec)
        raise WeightsError(
            f"{noun} {manifest.id!r} shares the weights of {base.id!r}, which is "
            f"installed at {directory}, and {len(absent)} of its own file(s) are "
            f"not there: {', '.join(absent)} — run `{command} {manifest.id}`"
        )
    if stamp.is_file():
        record = json.loads(stamp.read_text(encoding="utf-8"))
        if (
            record["revision"] == spec.revision
            and record["hf_repo"] == spec.hf_repo
        ):
            # The pin matches and the stamp is there, so what is wrong is a
            # NAMED FILE that is not. Said as itself rather than as a pin
            # mismatch, which is what the sentence below would claim.
            absent = missing_files(directory, spec)
            raise WeightsError(
                f"{directory} is stamped for {spec.hf_repo}@{spec.revision[:12]} "
                f"but {len(absent)} of the {len(spec.files)} file(s) it names "
                f"are not there: {', '.join(absent)} — run `{command} "
                f"{manifest.id} --force`"
            )
        raise WeightsError(
            f"{directory} holds {record['hf_repo']}@{record['revision'][:12]}, but "
            f"{manifest.path.name} now pins {spec.hf_repo}@{spec.revision[:12]} — "
            f"run `{command} {manifest.id}`"
        )
    raise WeightsError(
        f"{noun} {manifest.id!r} is not installed for {spec.backend}; there are no "
        f"weights at {directory} — run `{command} {manifest.id}`"
    )


class RemoveFailed(WeightsError):
    """A subject's files would not go. Carries the path that refused.

    Its own type because PHASE15-HOST.md 3.5a gives it its own name and its
    own `details.path`: "this subject is not installed" and "this file is
    locked by something" are different things to do about, and a caller told
    one about the other deletes the wrong problem.
    """

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


def _remove(path: Path) -> None:
    """One file or one tree, with the failure named by its own path."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        raise RemoveFailed(
            path, f"{path} would not be removed: {type(exc).__name__}: {exc}"
        ) from None


def _prune_empty(directory: Path, stop: Path) -> None:
    """Remove `directory` and its empty parents, up to but not past `stop`.

    3.5a: *"and the subject's directory if it is then empty"*. A directory
    left behind is not a failure — it is a directory — so this never raises
    for one that is not empty; what it refuses to do is climb past the tree
    this module owns.
    """
    current = directory
    while current != stop and stop in current.parents:
        try:
            next(current.iterdir())
        except StopIteration:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent
            continue
        except OSError:
            return
        return


def _refuse_local(spec: Any, manifest: WeightsSubject, verb: str, why: str) -> None:
    """Refuse a door that would fetch or delete bytes this server does not own.

    Both refusals in one place because they are one rule — Crucible manages a
    PIN and does not manage a PATH (PHASE18-UNCERTIFIED.md section 3) — and two
    hand-written copies would be two answers about one voice the first time
    either was edited.
    """
    local = local_source(spec)
    if local is None:
        return
    raise WeightsError(
        f"{manifest.id!r} cannot be {verb}: {why}. Its {spec.backend} block names "
        f"{local}"
    )


def remove(config: Config, manifest: WeightsSubject, spec: WeightsSource) -> Path:
    """Delete this subject's weights for this backend. Returns what went.

    PHASE15-HOST.md 3.5a, and it is the door the host's weights migration
    needs so that it never reaches into this module's layout from outside
    (`crucible/host/catalog.py` says why at length).

    A LOCAL SPEC IS REFUSED (section 3): the bytes are not Crucible's and
    deleting somebody else's 8 GiB directory because a manifest mentioned it is
    the one thing this door must never do.

    **The whole backend directory**, not a file list, and the difference is
    only visible on `llama-windows`: that backend's directory holds exactly
    the files its spec names plus the stamp, so removing the directory and
    removing the named files are the same act with one fewer way to leave a
    stamp behind. Every other backend's directory IS the snapshot.

    What it does NOT touch is another backend's copy of the same subject. A
    machine that ran `cuda-linux` yesterday and `llama-windows` today has
    two, and 3.5's migration deletes one of them.
    """
    _refuse_local(
        spec,
        manifest,
        "removed",
        "its weights are a directory on this server that something else owns, "
        "and deleting them because a manifest mentioned them is the one thing "
        "this door must never do",
    )
    directory = subject_dir(config, manifest, spec.backend)
    if getattr(manifest, "weights_of", None) is not None:
        # AN ALIAS TAKES ONLY WHAT IS ITS OWN (section 2.9): its extra files
        # and its record. Never the folder, never the base's stamp, never a
        # byte of the shared download.
        for name in _extra_files(manifest, spec):
            _remove(directory / name)
        _remove(alias_record_path(config, manifest, spec.backend))
        return directory
    # A BASE'S FOLDER IS REFUSED WHILE AN ALIAS HOLDS IT, by name, here in the
    # store rather than in either door, so the CLI, the API and the host's
    # migration cannot come to disagree about it.
    refuse_if_shared(config, manifest, spec.backend)
    _remove(directory)
    _prune_empty(
        directory.parent, weights_root(config, manifest.weights_family)
    )
    return directory


def remove_files(
    target_root: Path,
    targets: Sequence[str],
    *,
    stamp_name: str = STAMP_NAME,
) -> Path:
    """Delete a NAMED FILE SET and its stamp, leaving the directory alone.

    The counterpart of `pull_files`, and the reason it cannot be `remove`:
    `~/.crucible/denoise-models/` holds every separator in one flat directory
    (`crucible/denoisemodels.py` says why), so removing the directory would
    remove somebody else's model. One stamp per set, one removal per set.
    """
    for name in targets:
        _remove(_safe_target(target_root, name))
    _remove(target_root / stamp_name)
    return target_root


# --------------------------------------------------- an id renamed, and after
#
# 2026-09-24, Owen's asr lineup ruling: `faster-whisper-large-v3-turbo` and
# `mlx-whisper-large-v3-turbo` became `whisper-large-v3-turbo`, and the two
# tinies became `whisper-tiny`. The store is laid out by id
# (`<family>/<id>/<backend>`), so a rename strands every byte already pulled
# under the old one: 1.6 GB of turbo on a machine that has it, invisible to
# every door, because every inventory is walked from the manifests. The two
# functions below are the store's half of a rename — it moves what it can
# PROVE is the new id's, and it names what nothing owns.


def adopt_renamed(
    config: Config,
    old_id: str,
    manifest: WeightsSubject,
    specs: Mapping[str, WeightsSource],
) -> list[str]:
    """Move `old_id`'s pulled weights into `manifest`'s folder, per backend.

    `specs` is the new manifest's blocks by backend kind.

    One sentence per backend directory found, saying what happened — moved, or
    left and why. Nothing is moved on a guess: a directory is adopted only when
    its stamp names EXACTLY the repo and revision the new manifest pins for
    that backend, because moving other bytes under the new id would be the
    silent substitution `installed()` exists to refuse. Everything left is
    what `stranded()` reports.

    Weather, not misconfiguration (CLAUDE.md): a rename that fails for an
    OSError is reported and left for the next start, never raised — the server
    is the thing starting, and bytes that could not be moved are still on disk
    for `crucible doctor` to name.
    """
    root = weights_root(config, manifest.weights_family)
    old_root = root / old_id
    if not old_root.is_dir():
        return []
    lines: list[str] = []
    for source in sorted(p for p in old_root.iterdir() if p.is_dir()):
        backend_kind = source.name
        spec = specs.get(backend_kind)
        if spec is None:
            lines.append(
                f"left {source}: {manifest.id!r} has no {backend_kind} block, so "
                f"these are not its weights"
            )
            continue
        stamp = source / STAMP_NAME
        if not stamp.is_file():
            lines.append(
                f"left {source}: no {STAMP_NAME}, so no finished pull says what "
                "these bytes are"
            )
            continue
        try:
            record = json.loads(stamp.read_text(encoding="utf-8"))
            pinned = (record["hf_repo"], record["revision"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            lines.append(
                f"left {source}: its stamp would not read "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if pinned != (spec.hf_repo, spec.revision):
            lines.append(
                f"left {source}: stamped {pinned[0]}@{str(pinned[1])[:12]}, and "
                f"{manifest.id!r} pins {spec.hf_repo}@{spec.revision[:12]} on "
                f"{backend_kind}"
            )
            continue
        target = subject_dir(config, manifest, backend_kind)
        if target.exists():
            lines.append(
                f"left {source}: {target} already exists, and one of the two "
                "copies is surplus"
            )
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # A RENAME, not a copy: one filesystem (`~/.crucible/models/`), so
            # it is one metadata operation and never a half-copied 1.6 GB.
            os.replace(source, target)
        except OSError as exc:
            lines.append(
                f"left {source}: the move to {target} failed "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        # The stamp now describes its new home. `installed()` reads only the
        # repo and revision, so a stamp that failed to rewrite would still
        # answer truly; the id is for a human reading the file.
        record["id"] = manifest.id
        record["renamed_from"] = old_id
        try:
            (target / STAMP_NAME).write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            lines.append(
                f"moved {source} -> {target}; its stamp still says {old_id!r} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        lines.append(f"moved {source} -> {target} (renamed {old_id} -> {manifest.id})")
    _prune_empty(old_root, root)
    return lines


@dataclass(frozen=True)
class StrandedWeights:
    """A directory under the store that no manifest in this build owns."""

    family: str
    subject_id: str
    backend: str
    path: Path
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "id": self.subject_id,
            "backend": self.backend,
            "path": str(self.path),
            "bytes": self.bytes,
        }


def stranded(
    config: Config,
    family: str,
    backends: Sequence[str],
    declared_backends: Callable[[str], Sequence[str]],
) -> list[StrandedWeights]:
    """Every `<family>/<id>/<backend>` directory nothing in this build declares.

    `backends` is every backend kind a directory may be named for, and
    `declared_backends(id)` which of them this build has a manifest block for;
    the caller supplies both because the store does not load manifests. A
    directory named for anything else is not a weights directory and is not
    this function's business.

    THE RECONCILER FOR THE STORE (CLAUDE.md: every hold has an owner, and a
    reconciler finds orphans). It REPORTS and never deletes: the bytes may be
    an operator's only copy of something, and the operator decides.
    """
    root = weights_root(config, family)
    if not root.is_dir():
        return []
    found: list[StrandedWeights] = []
    for subject in sorted(p for p in root.iterdir() if p.is_dir()):
        declared = set(declared_backends(subject.name))
        for directory in sorted(p for p in subject.iterdir() if p.is_dir()):
            if directory.name not in backends or directory.name in declared:
                continue
            found.append(
                StrandedWeights(
                    family=family,
                    subject_id=subject.name,
                    backend=directory.name,
                    path=directory,
                    bytes=directory_bytes(directory),
                )
            )
    return found


def hf_token(config: Config) -> str | None:
    """`$HF_TOKEN`, else `[hf] token` in config.toml, else None."""
    return hf_token_at(config.path)


def hf_token_at(config_path: Path) -> str | None:
    """The same credential, asked of a PATH rather than of a loaded Config.

    `crucible/voicerepo.py` fetches a voice's manifest out of a private repo
    while merging the catalog, and it does that without a `Config` — the merge
    happens inside `load_all_voices()`, which the CLI, the tests and three job
    types call with no server around. One reader either way: `hf_token` is this
    function with a Config's path, so there is no second answer to "which token
    does this machine use".
    """
    from_env = os.environ.get(HF_TOKEN_ENV)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError:
        return None
    section = document.get("hf")
    if not isinstance(section, dict):
        return None
    token = section.get("token")
    if isinstance(token, str) and token.strip() != "":
        return token.strip()
    return None


def resolve_revision(config: Config, hf_repo: str) -> str:
    """The repo's current head sha, so a caller can pin what it just looked at.

    ── Why the ENGINE resolves this and not the app ───────────────────────────

    A voice manifest requires a full 40-character commit sha, never a branch
    name, so that a pull is reproducible (`crucible/voices.py`). That makes
    "add the voice at this repo" impossible to ask for without first turning a
    repo id into a sha — and the thing that should do the turning is the thing
    that will do the fetching. This process already holds the HuggingFace
    credential (`hf_token`) and already talks to the Hub; an app resolving the
    sha would need its own copy of the token to read a private repo, which is
    the credential sprawl PHASE15 section 0 exists to prevent.

    ── It pins the head, and that is a MOMENT rather than a promise ───────────

    Between this call and the pull the repo may move. That is not a race worth
    locking: the point of the pin is that whatever is fetched is *recorded*, so
    two machines asked for the same voice get the same bytes. A caller that
    wants a specific older revision passes one and never reaches here.

    Refuses by name. `revision_unresolved` carries the Hub's own words, because
    "no such repo", "you are not authorised" and "the Hub is down" are three
    different things to do about it and only the Hub can tell them apart.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - import guard
        raise WeightsError(
            f"huggingface_hub is not importable: {exc}"
        ) from exc
    try:
        info = HfApi(token=hf_token(config)).model_info(hf_repo)
    except Exception as exc:
        raise WeightsError(
            f"could not read {hf_repo!r} on HuggingFace: {exc}. A voice pins a "
            "full commit sha, so the repo has to be readable from this machine "
            "before it can be added — check the id, and check [hf] token in "
            "config.toml if the repo is private"
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) != 40:
        raise WeightsError(
            f"HuggingFace answered for {hf_repo!r} without a commit sha "
            f"({sha!r}), so there is nothing to pin"
        )
    return sha


def reporting_tqdm(on_progress: ProgressHook) -> Any:
    """A `tqdm_class` for `huggingface_hub` that reports bytes to `on_progress`.

    R4, applied to the one thing in this module that was only ever a log line: a
    pull's progress existed exclusively as the hub's own progress BAR on
    somebody's terminal, so a client that asked a server to pull 19 GB could be
    told "pulling" and then nothing for twenty minutes. The bar is untouched —
    `on_line` still prints what it always printed — and the fact is promoted to
    an event beside it.

    **Only byte bars are reported.** `snapshot_download` also raises a bar
    counting FILES (`unit="it"`), and forwarding both would interleave two
    different meanings of the same three numbers into one event stream. The
    file count is recoverable from the sequence of `file` names; a byte count
    mistaken for a file count is not recoverable from anything.
    """
    try:
        from huggingface_hub.utils import tqdm as hub_tqdm
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise WeightsError(
            f"huggingface_hub is not importable: {exc}"
        ) from exc

    class _Reporting(hub_tqdm):  # type: ignore[misc, valid-type]
        """A hub progress bar that also reports, and reports even when hidden.

        THE COUNTING IS OUR OWN, and that is not redundancy. `tqdm.update`
        returns immediately when the bar is `disable`d, and `tqdm.__init__`
        does not even set `self.unit` or `self.desc` in that case — so a server
        whose environment carries `HF_HUB_DISABLE_PROGRESS_BARS` would have
        emitted no `progress` events at all, and, far worse, would have had
        **no cancel point**: the hook is the only place a
        `snapshot_download` can be interrupted (see `PullCancelled`). A
        download nobody can stop because somebody turned off a progress bar is
        exactly the kind of coupling ARCHITECTURE.md R4 is about.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._crucible_unit = kwargs.get("unit")
            self._crucible_desc = kwargs.get("desc") or ""
            self._crucible_done = int(kwargs.get("initial", 0) or 0)
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            displayed = super().update(n)
            if self._crucible_unit == "B":
                self._crucible_done += int(n or 0)
                # `self.total` IS set on a disabled bar, and the hub revises it
                # once it has read the content length, so it is asked rather
                # than remembered.
                on_progress(
                    self._crucible_done, self.total, self._crucible_desc
                )
            return displayed

    return _Reporting


def directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def pull(
    config: Config,
    manifest: WeightsSubject,
    spec: WeightsSource,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> InstalledWeights:
    """Fetch this model's or voice's weights for this backend at its pin.

    AN ALIAS (`[model] weights_of`) is `_pull_alias`: its base's download, then
    only the files its own block adds, into the one folder.
    """
    _refuse_local(
        spec,
        manifest,
        "pulled",
        "its weights are a directory on this server that something else put "
        "there, and fetching would overwrite them from a repo the block does "
        "not name",
    )
    if getattr(manifest, "weights_of", None) is not None:
        return _pull_alias(
            config, manifest, spec, force=force, on_line=on_line,
            on_progress=on_progress,
        )

    target = subject_dir(config, manifest, spec.backend)
    existing = installed(config, manifest, spec)
    if existing is not None and not force:
        return existing
    if force and target.exists():
        # A forced pull empties the folder, and an alias's projector and record
        # are in it: the same act as a removal, refused by the same rule.
        refuse_if_shared(config, manifest, spec.backend)
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    stamp = target / STAMP_NAME
    if stamp.exists():
        stamp.unlink()

    token = hf_token(config)
    if on_line is not None:
        on_line(
            f"pulling {spec.hf_repo}@{spec.revision[:12]} -> {target} "
            f"({'with' if token else 'without'} an HF token)"
        )
    started = time.monotonic()
    if spec.files and on_line is not None:
        on_line(f"only {len(spec.files)} file(s) of that repo: " + ", ".join(spec.files))
    try:
        _snapshot(
            config, manifest.path.name, spec, target,
            patterns=spec.files, on_progress=on_progress,
        )
    except PullCancelled:
        # The caller asked for this. Everything written so far goes, and the
        # cancellation travels untouched — see `PullCancelled`.
        shutil.rmtree(target, ignore_errors=True)
        raise

    elapsed = time.monotonic() - started
    # BEFORE THE STAMP. A stamp is this module's statement that the subject is
    # complete, and writing one over a repo that answered for the text tower
    # and not for the projector would make `installed` say yes about a server
    # that will refuse every page. `allow_patterns` silently matches nothing
    # when a name is wrong, so this is the only thing that catches a manifest
    # with a typo in it.
    absent = missing_files(target, spec)
    if absent:
        raise WeightsError(
            f"{spec.hf_repo}@{spec.revision[:12]} was fetched but "
            f"{len(absent)} of the file(s) {manifest.path.name} names for "
            f"{spec.backend} are not in {target}: {', '.join(absent)}. Either "
            "the manifest names a file this revision does not have, or the "
            "download was incomplete; nothing is stamped either way"
        )
    size = directory_bytes(target)
    record = {
        "family": manifest.weights_family,
        "id": manifest.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        # What the pin MEANT on this backend, so a reader of the stamp alone
        # can see that a repo was fetched in part and which part.
        "files": list(spec.files),
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(
            f"pulled {size / 1e9:.2f} GB in {elapsed:.0f}s "
            f"({size / 1e6 / max(elapsed, 1e-6):.0f} MB/s)"
        )
    result = installed(config, manifest, spec)
    if result is None:  # pragma: no cover - the stamp was just written
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


def _snapshot(
    config: Config,
    manifest_name: str,
    spec: WeightsSource,
    target: Path,
    *,
    patterns: Sequence[str],
    on_progress: ProgressHook | None,
) -> None:
    """`snapshot_download` of `spec`'s pin into `target`, refusals by name.

    `patterns` empty = the whole repo. Non-empty = ONLY those files: without
    it a `llama-windows` row on `unsloth/Qwen3.8-27B-GGUF` fetches every
    quantization in the repo — hundreds of gigabytes for one 16 GB file.
    `allow_patterns` takes literal names as well as globs, and these are
    literal: the manifest names the file, so a pattern that matched two would
    be this module deciding which.

    A `PullCancelled` travels out untouched; what to delete on a cancel is the
    caller's, because a base owns its whole folder and an alias only its files.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise WeightsError(
            f"huggingface_hub is not importable in {config.name}'s interpreter: {exc}"
        ) from exc
    extra: dict[str, Any] = {}
    if on_progress is not None:
        extra["tqdm_class"] = reporting_tqdm(on_progress)
    if patterns:
        extra["allow_patterns"] = list(patterns)
    try:
        snapshot_download(
            repo_id=spec.hf_repo,
            revision=spec.revision,
            local_dir=str(target),
            token=hf_token(config),
            max_workers=8,
            **extra,
        )
    except PullCancelled:
        raise
    except GatedRepoError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is gated and this server has no HF token that opens it "
            f"(set ${HF_TOKEN_ENV} or [hf] token in {config.path}): {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is private or does not exist; if it is private set "
            f"${HF_TOKEN_ENV} or [hf] token in {config.path}: {exc}"
        ) from exc
    except RevisionNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} has no revision {spec.revision}; "
            f"{manifest_name} pins a commit that repo does not have: {exc}"
        ) from exc
    except Exception as exc:
        raise WeightsError(
            f"pulling {spec.hf_repo}@{spec.revision[:12]} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _pull_alias(
    config: Config,
    alias: Any,
    spec: WeightsSource,
    *,
    force: bool,
    on_line: Callable[[str], None] | None,
    on_progress: ProgressHook | None,
) -> InstalledWeights:
    """An alias's pull: the base's download, then the alias's own files.

    PHASE22-DECIDE.md section 2.9. **The base is pulled as the base** — the
    same `pull`, the same stamp, the same folder — so a machine that pulls
    `qwen3.5-9b-vl` first and `qwen3.5-9b` second downloads once. Then only
    what the alias's block names beyond the base's (`extra_files`: the
    llama-windows projector; nothing on a whole-repo backend) is fetched into
    that folder, and the alias's record is written beside the base's stamp.

    `force` re-fetches the ALIAS's files only. It never forces the base: that
    is a pull of the base, asked for by name, and it is refused while an alias
    holds the folder (`refuse_if_shared`).

    A cancel removes only the files this pull was fetching. The folder is the
    base's, and a cancelled projector download must not take a 19 GB model
    with it.
    """
    base = alias.weights_base
    base_spec = base.spec(spec.backend)
    target = subject_dir(config, alias, spec.backend)
    record_path = alias_record_path(config, alias, spec.backend)
    existing = installed(config, alias, spec)
    if existing is not None and record_path.is_file() and not force:
        return existing

    if installed(config, base, base_spec) is None:
        if on_line is not None:
            on_line(
                f"{alias.id} shares the weights of {base.id}; pulling {base.id} "
                "first, once, into its own folder"
            )
        pull(config, base, base_spec, on_line=on_line, on_progress=on_progress)

    extras = alias.extra_files(spec.backend)
    if force:
        for name in extras:
            _remove(target / name)
    wanted = [name for name in extras if not (target / name).is_file()]
    started = time.monotonic()
    if wanted:
        if on_line is not None:
            on_line(
                f"pulling {alias.id}'s own file(s) from "
                f"{spec.hf_repo}@{spec.revision[:12]} into {target}: "
                + ", ".join(wanted)
            )
        try:
            _snapshot(
                config, alias.path.name, spec, target,
                patterns=wanted, on_progress=on_progress,
            )
        except PullCancelled:
            for name in wanted:
                (target / name).unlink(missing_ok=True)
            raise
    absent = missing_files(target, spec)
    if absent:
        raise WeightsError(
            f"{spec.hf_repo}@{spec.revision[:12]} was fetched but {len(absent)} of "
            f"the file(s) {alias.path.name} names for {spec.backend} are not in "
            f"{target}: {', '.join(absent)}. Either the manifest names a file this "
            "revision does not have, or the download was incomplete; no alias "
            "record is written either way"
        )
    own = sum((target / name).stat().st_size for name in extras)
    record = {
        "family": alias.weights_family,
        "id": alias.id,
        "weights_of": base.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        # What this alias OWNS in the folder, which is what removing it frees.
        "files": list(extras),
        "bytes": own,
        "seconds": round(time.monotonic() - started, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(
            f"{alias.id}: {own / 1e9:.2f} GB of its own beside {base.id}'s weights "
            f"at {target}"
        )
    result = installed(config, alias, spec)
    if result is None:  # pragma: no cover - checked just above
        raise WeightsError(f"wrote {record_path} but {alias.id} does not read as installed")
    return result


# ------------------------------------------------------------ one file, unpacked


@runtime_checkable
class ArchiveSource(Protocol):
    """A backend block whose weights are ONE file in a shared repo.

    `crucible/rvcmodels.py` is the only user and explains why it exists: every
    RVC model Owen has published is a `.tar.gz` under `rvc/` in one repo
    alongside six others and the XTTS weights, so `snapshot_download` would fetch
    about 800 MB to get at 80.
    """

    backend: str
    hf_repo: str
    revision: str
    archive: str
    archive_sha256: str


def pull_archive(
    config: Config,
    manifest: WeightsSubject,
    spec: ArchiveSource,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> InstalledWeights:
    """Fetch one archive from a repo, verify it, and unpack it into the weights dir.

    The stamp it writes is byte-identical in shape to `pull`'s, so `installed`
    and `require_installed` read either without knowing which one ran — which is
    the point of putting this here rather than in `rvcmodels.py`.

    **The digest is checked before anything is unpacked**, and a mismatch is a
    refusal rather than a warning. `snapshot_download` verifies what it fetches
    against the revision; a single file fetched by path gets the same assurance
    from the manifest, because the failure it prevents is a truncated or
    substituted checkpoint that converts a whole book into something subtly
    wrong and says nothing.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise WeightsError(
            f"huggingface_hub is not importable in {config.name}'s interpreter: {exc}"
        ) from exc

    target = weights_dir(config, manifest.weights_family, manifest.id, spec.backend)
    existing = installed(config, manifest, spec)
    if existing is not None and not force:
        return existing
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    stamp = target / STAMP_NAME
    if stamp.exists():
        stamp.unlink()

    token = hf_token(config)
    if on_line is not None:
        on_line(
            f"pulling {spec.hf_repo}@{spec.revision[:12]}:{spec.archive} -> {target} "
            f"({'with' if token else 'without'} an HF token)"
        )
    started = time.monotonic()
    # Into a staging directory beside the target, never into the hub's shared
    # cache-by-default: an interrupted download must not leave bytes somewhere a
    # later run would treat as complete, and the target is the one place this
    # module cleans up.
    staging = target / ".crucible-archive"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    extra: dict[str, Any] = {}
    if on_progress is not None:
        extra["tqdm_class"] = reporting_tqdm(on_progress)
    try:
        downloaded = hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.archive,
            revision=spec.revision,
            local_dir=str(staging),
            token=token,
            **extra,
        )
    except PullCancelled:
        shutil.rmtree(target, ignore_errors=True)
        raise
    except GatedRepoError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is gated and this server has no HF token that opens it "
            f"(set ${HF_TOKEN_ENV} or [hf] token in {config.path}): {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is private or does not exist; if it is private set "
            f"${HF_TOKEN_ENV} or [hf] token in {config.path}: {exc}"
        ) from exc
    except RevisionNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} has no revision {spec.revision}; "
            f"{manifest.path.name} pins a commit that repo does not have: {exc}"
        ) from exc
    except EntryNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo}@{spec.revision[:12]} has no file {spec.archive!r}; "
            f"{manifest.path.name} names an archive that revision does not hold: "
            f"{exc}"
        ) from exc
    except Exception as exc:
        raise WeightsError(
            f"pulling {spec.hf_repo}:{spec.archive} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    digest = sha256_of(Path(downloaded))
    if digest != spec.archive_sha256:
        shutil.rmtree(staging, ignore_errors=True)
        raise WeightsError(
            f"{spec.archive} from {spec.hf_repo}@{spec.revision[:12]} hashes to "
            f"{digest}, but {manifest.path.name} pins {spec.archive_sha256}. Nothing "
            "was unpacked. Either the manifest is wrong or these are not the bytes "
            "it names, and both are worse than no weights at all."
        )

    try:
        _unpack(Path(downloaded), target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = time.monotonic() - started
    size = directory_bytes(target)
    record = {
        "family": manifest.weights_family,
        "id": manifest.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "archive": spec.archive,
        "archive_sha256": digest,
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(f"unpacked {size / 1e9:.2f} GB in {elapsed:.0f}s at {target}")
    result = installed(config, manifest, spec)
    if result is None:  # pragma: no cover - the stamp was just written
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


# ------------------------------------------------- named files, placed exactly


@runtime_checkable
class FileSource(Protocol):
    """One file to fetch by path, verify, and put somewhere specific.

    The third shape of pull, and the reason it exists is a layout nobody chose:
    ultimate-rvc's shared base assets are four files scattered through one
    HuggingFace repo that also holds six pretrained GAN checkpoints, and the
    engine reads them from a tree of its own with different names and different
    directories (`crucible/rvcbase.py`). `pull` would fetch the whole repo;
    `pull_archive` has nothing to unpack. This fetches exactly what is named and
    puts each file exactly where the engine looks.
    """

    source: str
    target: str
    sha256: str
    bytes: int


def files_installed(
    target_root: Path,
    hf_repo: str,
    revision: str,
    *,
    stamp_name: str = STAMP_NAME,
) -> InstalledWeights | None:
    """The stamped file set at `target_root`, or None.

    A stamp naming a different repo or revision is *not* installed, for
    `installed`'s reason: the declaration moved, and serving the old bytes under
    the new pin would be a silent substitution. Every target is checked for
    presence too — a stamp beside a file somebody deleted is a stamp that lies.

    `stamp_name` is the default when one directory holds exactly one set, which
    is ultimate-rvc's base assets. It is NOT the default for
    `~/.crucible/denoise-models`, where audio-separator reads every separator
    model by filename out of one flat directory: one stamp there would be
    overwritten by the second model's pull and would then report the first as
    never installed. One stamp per set, named after the set.
    """
    stamp = target_root / stamp_name
    if not stamp.is_file():
        return None
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record.get("hf_repo") != hf_repo or record.get("revision") != revision:
        return None
    for entry in record.get("files", []):
        if not (target_root / entry["target"]).is_file():
            return None
    return InstalledWeights(
        path=target_root,
        hf_repo=record["hf_repo"],
        revision=record["revision"],
        bytes=record["bytes"],
        pulled=record["pulled"],
        source=PINNED,
    )


def _safe_target(target_root: Path, target: str) -> Path:
    """`target_root/target`, or a refusal if it would land outside it."""
    root = target_root.resolve()
    destination = (root / target).resolve()
    if destination != root and root not in destination.parents:
        raise WeightsError(
            f"{target!r} would be written outside {target_root}. A declared "
            "target is a path inside the tree the engine reads, never a way to "
            "write somewhere else"
        )
    return target_root / target


def pull_files(
    config: Config,
    *,
    hf_repo: str,
    revision: str,
    files: Sequence[FileSource],
    target_root: Path,
    label: str,
    stamp_name: str = STAMP_NAME,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> InstalledWeights:
    """Fetch each named file at one revision, verify it, and place it.

    **Every digest is checked before ANY file is placed.** The same rule
    `pull_archive` follows and for the same reason: a half-placed set is a tree
    an engine will happily start against, and the failure then arrives inside
    somebody's book rather than here. Files land in a staging directory under
    the target root, are hashed there, and are moved into place only once all of
    them have passed.

    `label` is what the progress lines call this set, because a caller pulling
    "ultimate-rvc's base assets" should not read lines about a model id.

    `stamp_name` is `files_installed`'s: a target root that holds more than one
    set — `~/.crucible/denoise-models`, where audio-separator reads every
    separator by filename out of one flat directory — needs one stamp per set,
    or the second pull's stamp says the first was never made. Note that force
    does NOT empty the target root, unlike `pull` and `pull_archive`: it
    replaces this set's files and leaves anybody else's alone.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise WeightsError(
            f"huggingface_hub is not importable in {config.name}'s interpreter: {exc}"
        ) from exc

    if not files:
        raise WeightsError(
            f"{label} declares no files; a set with nothing in it is not a set"
        )

    existing = files_installed(
        target_root, hf_repo, revision, stamp_name=stamp_name
    )
    if existing is not None and not force:
        return existing

    target_root.mkdir(parents=True, exist_ok=True)
    stamp = target_root / stamp_name
    if stamp.exists():
        # Removed FIRST: from here until the new stamp is written this tree is
        # honestly "not installed", so a run interrupted half way cannot be read
        # as a complete set by anything downstream.
        stamp.unlink()
    staging = target_root / ".crucible-files"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    token = hf_token(config)
    if on_line is not None:
        on_line(
            f"pulling {len(files)} file(s) of {label} from "
            f"{hf_repo}@{revision[:12]} -> {target_root} "
            f"({'with' if token else 'without'} an HF token)"
        )
    started = time.monotonic()
    fetched: list[tuple[FileSource, Path, Path]] = []
    extra: dict[str, Any] = {}
    if on_progress is not None:
        extra["tqdm_class"] = reporting_tqdm(on_progress)
    try:
        for entry in files:
            destination = _safe_target(target_root, entry.target)
            try:
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=entry.source,
                    revision=revision,
                    local_dir=str(staging),
                    token=token,
                    **extra,
                )
            except PullCancelled:
                # Nothing has been MOVED yet — every file is placed only after
                # all of them verify — so the staging tree the `finally` below
                # removes is the whole of what this pull wrote.
                raise
            except GatedRepoError as exc:
                raise WeightsError(
                    f"{hf_repo} is gated and this server has no HF token that "
                    f"opens it (set ${HF_TOKEN_ENV} or [hf] token in "
                    f"{config.path}): {exc}"
                ) from exc
            except RepositoryNotFoundError as exc:
                raise WeightsError(
                    f"{hf_repo} is private or does not exist; if it is private "
                    f"set ${HF_TOKEN_ENV} or [hf] token in {config.path}: {exc}"
                ) from exc
            except RevisionNotFoundError as exc:
                raise WeightsError(
                    f"{hf_repo} has no revision {revision}; {label} pins a commit "
                    f"that repo does not have: {exc}"
                ) from exc
            except EntryNotFoundError as exc:
                raise WeightsError(
                    f"{hf_repo}@{revision[:12]} has no file {entry.source!r}; "
                    f"{label} names a path that revision does not hold: {exc}"
                ) from exc
            except Exception as exc:
                raise WeightsError(
                    f"pulling {hf_repo}:{entry.source} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            digest = sha256_of(Path(downloaded))
            if digest != entry.sha256:
                raise WeightsError(
                    f"{entry.source} from {hf_repo}@{revision[:12]} hashes to "
                    f"{digest}, but {label} pins {entry.sha256}. NOTHING was "
                    "placed. Either the declaration is wrong or these are not "
                    "the bytes it names, and both are worse than no weights"
                )
            if on_line is not None:
                on_line(f"  verified {entry.target} ({digest[:12]})")
            fetched.append((entry, Path(downloaded), destination))

        total = 0
        for entry, downloaded, destination in fetched:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            shutil.move(str(downloaded), str(destination))
            total += destination.stat().st_size
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = time.monotonic() - started
    record = {
        "label": label,
        "hf_repo": hf_repo,
        "revision": revision,
        "files": [
            {"source": entry.source, "target": entry.target, "sha256": entry.sha256}
            for entry in files
        ],
        "bytes": total,
        "seconds": round(elapsed, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(f"placed {total / 1e9:.2f} GB in {elapsed:.0f}s at {target_root}")
    result = files_installed(
        target_root, hf_repo, revision, stamp_name=stamp_name
    )
    if result is None:  # pragma: no cover - the stamp was just written
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """The file's SHA-256, read in chunks so a 180 MB archive is not held twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack(archive: Path, target: Path) -> None:
    """Extract a `.tar.gz` into `target`, refusing any member that escapes it.

    `filter="data"` is python 3.12+'s extraction filter and is exactly this rule
    — no absolute paths, no `..`, no devices, no links out — but this server
    supports 3.11, where the default is the permissive one and the argument is
    absent. So the check is written out: a member whose resolved destination is
    not under `target` is a refusal naming it, never a skip.
    """
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            root = target.resolve()
            for member in members:
                destination = (root / member.name).resolve()
                if destination != root and root not in destination.parents:
                    raise WeightsError(
                        f"{archive.name} contains {member.name!r}, which would be "
                        f"written outside {target}. Refusing to unpack any of it."
                    )
                if member.issym() or member.islnk():
                    raise WeightsError(
                        f"{archive.name} contains a link, {member.name!r}. A weights "
                        "archive is files; a link is a way to write somewhere else."
                    )
            handle.extractall(path=target, members=members)
    except tarfile.TarError as exc:
        raise WeightsError(f"could not unpack {archive.name}: {exc}") from exc
