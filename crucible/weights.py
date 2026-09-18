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
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .config import Config
from .errors import CrucibleError

HF_TOKEN_ENV = "HF_TOKEN"
STAMP_NAME = "crucible-pull.json"

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
    hf_repo: str
    revision: str
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
    hf_repo: str
    revision: str
    bytes: int
    pulled: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "bytes": self.bytes,
            "pulled": self.pulled,
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
    """
    family = manifest.weights_family
    directory = weights_dir(config, family, manifest.id, spec.backend)
    stamp = stamp_path(config, family, manifest.id, spec.backend)
    if not stamp.is_file():
        return None
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["revision"] != spec.revision or record["hf_repo"] != spec.hf_repo:
        return None
    if missing_files(directory, spec):
        return None
    return InstalledWeights(
        path=weights_dir(config, family, manifest.id, spec.backend),
        hf_repo=record["hf_repo"],
        revision=record["revision"],
        bytes=record["bytes"],
        pulled=record["pulled"],
    )


def require_installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights:
    """Installed weights, or `model_not_installed` / `voice_not_installed`."""
    found = installed(config, manifest, spec)
    if found is not None:
        return found
    family = manifest.weights_family
    noun, command = _FAMILY_WORDS[family]
    directory = weights_dir(config, family, manifest.id, spec.backend)
    stamp = stamp_path(config, family, manifest.id, spec.backend)
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


def remove(config: Config, manifest: WeightsSubject, spec: WeightsSource) -> Path:
    """Delete this subject's weights for this backend. Returns what went.

    PHASE15-HOST.md 3.5a, and it is the door the host's weights migration
    needs so that it never reaches into this module's layout from outside
    (`crucible/host/catalog.py` says why at length).

    **The whole backend directory**, not a file list, and the difference is
    only visible on `llama-windows`: that backend's directory holds exactly
    the files its spec names plus the stamp, so removing the directory and
    removing the named files are the same act with one fewer way to leave a
    stamp behind. Every other backend's directory IS the snapshot.

    What it does NOT touch is another backend's copy of the same subject. A
    machine that ran `cuda-linux` yesterday and `llama-windows` today has
    two, and 3.5's migration deletes one of them.
    """
    directory = weights_dir(config, manifest.weights_family, manifest.id, spec.backend)
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


def hf_token(config: Config) -> str | None:
    """`$HF_TOKEN`, else `[hf] token` in config.toml, else None."""
    from_env = os.environ.get(HF_TOKEN_ENV)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    try:
        with config.path.open("rb") as handle:
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
    """Fetch this model's or voice's weights for this backend at its pin."""
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
            f"pulling {spec.hf_repo}@{spec.revision[:12]} -> {target} "
            f"({'with' if token else 'without'} an HF token)"
        )
    started = time.monotonic()
    extra: dict[str, Any] = {}
    if on_progress is not None:
        extra["tqdm_class"] = reporting_tqdm(on_progress)
    if spec.files:
        # ONLY THE FILES THIS BACKEND NAMES. Without this a `llama-windows`
        # row on `unsloth/Qwen3.8-27B-GGUF` fetches every quantization in the
        # repo — hundreds of gigabytes for one 16 GB file. `allow_patterns`
        # takes literal names as well as globs, and these are literal: the
        # manifest names the file, so a pattern that matched two would be this
        # module deciding which.
        extra["allow_patterns"] = list(spec.files)
        if on_line is not None:
            on_line(
                f"only {len(spec.files)} file(s) of that repo: "
                + ", ".join(spec.files)
            )
    try:
        snapshot_download(
            repo_id=spec.hf_repo,
            revision=spec.revision,
            local_dir=str(target),
            token=token,
            max_workers=8,
            **extra,
        )
    except PullCancelled:
        # The caller asked for this. Everything written so far goes, and the
        # cancellation travels untouched — see `PullCancelled`.
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
    except Exception as exc:
        raise WeightsError(
            f"pulling {spec.hf_repo}@{spec.revision[:12]} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

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
