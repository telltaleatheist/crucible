from __future__ import annotations

import hashlib
import json
import logging
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


class _QuietUnauthenticated(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()


logging.getLogger("huggingface_hub.utils._http").addFilter(_QuietUnauthenticated())

PINNED = "pinned"
LOCAL = "local"

_FAMILY_WORDS: dict[str, tuple[str, str]] = {
    "models": ("model", "crucible models pull"),
    "voices": ("voice", "crucible voices pull"),
    "rvc": ("RVC model", "crucible rvc pull"),
}


@runtime_checkable
class WeightsSubject(Protocol):

    id: str
    path: Path
    weights_family: str


@runtime_checkable
class WeightsSource(Protocol):

    backend: str
    hf_repo: str | None
    revision: str | None
    files: tuple[str, ...]


class WeightsError(CrucibleError):
    ...


class PullCancelled(CrucibleError):
    ...


ProgressHook = Callable[[int, "int | None", str], None]


@dataclass(frozen=True)
class InstalledWeights:
    path: Path
    hf_repo: str | None
    revision: str | None
    bytes: int
    pulled: str | None
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


def _store_id(subject: Any) -> str:
    weights_of = getattr(subject, "weights_of", None)
    return subject.id if weights_of is None else weights_of


def _extra_files(subject: Any, spec: WeightsSource) -> tuple[str, ...]:
    if getattr(subject, "weights_of", None) is None:
        return ()
    return subject.extra_files(spec.backend)


def subject_dir(config: Config, subject: WeightsSubject, backend_kind: str) -> Path:
    return weights_dir(config, subject.weights_family, _store_id(subject), backend_kind)


ALIAS_RECORD_PREFIX = "crucible-alias-"


def alias_record_path(config: Config, alias: Any, backend_kind: str) -> Path:
    return subject_dir(config, alias, backend_kind) / f"{ALIAS_RECORD_PREFIX}{alias.id}.json"


class WeightsShared(WeightsError):

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
    from .asrmodels import AsrManifest, asr_aliases_of
    from .manifests import ModelManifest, aliases_of
    from .voices import VoiceManifest, voice_aliases_of

    if getattr(manifest, "weights_of", None) is not None:
        return ()
    if isinstance(manifest, ModelManifest):
        aliases = aliases_of(manifest)
    elif isinstance(manifest, AsrManifest):
        aliases = asr_aliases_of(manifest)
    elif isinstance(manifest, VoiceManifest):
        aliases = voice_aliases_of(manifest)
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
    holding = aliases_holding(config, manifest, backend_kind)
    if holding:
        raise WeightsShared(manifest.id, backend_kind, holding)


def missing_files(directory: Path, spec: WeightsSource) -> tuple[str, ...]:
    return tuple(name for name in spec.files if not (directory / name).is_file())


def local_source(spec: Any) -> Path | None:
    found = getattr(spec, "local_path", None)
    return None if found is None else Path(found)


def _local_installed(directory: Path, spec: Any) -> InstalledWeights | None:
    if not directory.is_dir():
        return None
    if missing_files(directory, spec):
        return None
    return InstalledWeights(
        path=directory,
        hf_repo=None,
        revision=spec.identity,
        bytes=directory_bytes(directory),
        pulled=None,
        source=LOCAL,
    )


def installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights | None:
    local = local_source(spec)
    if local is not None:
        return _local_installed(local, spec)

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
    found = installed(config, manifest, spec)
    if found is not None:
        return found

    local = local_source(spec)
    if local is not None:
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

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


def _remove(path: Path) -> None:
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
    local = local_source(spec)
    if local is None:
        return
    raise WeightsError(
        f"{manifest.id!r} cannot be {verb}: {why}. Its {spec.backend} block names "
        f"{local}"
    )


def remove(config: Config, manifest: WeightsSubject, spec: WeightsSource) -> Path:
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
        for name in _extra_files(manifest, spec):
            _remove(directory / name)
        _remove(alias_record_path(config, manifest, spec.backend))
        return directory
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
    for name in targets:
        _remove(_safe_target(target_root, name))
    _remove(target_root / stamp_name)
    return target_root


@dataclass(frozen=True)
class StrandedWeights:

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
    return hf_token_at(config.path)


def hf_token_at(config_path: Path) -> str | None:
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
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
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
    try:
        from huggingface_hub.utils import tqdm as hub_tqdm
    except ImportError as exc:
        raise WeightsError(
            f"huggingface_hub is not importable: {exc}"
        ) from exc

    class _Reporting(hub_tqdm):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._crucible_unit = kwargs.get("unit")
            self._crucible_desc = kwargs.get("desc") or ""
            self._crucible_done = int(kwargs.get("initial", 0) or 0)
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            displayed = super().update(n)
            if self._crucible_unit == "B":
                self._crucible_done += int(n or 0)
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
        shutil.rmtree(target, ignore_errors=True)
        raise

    elapsed = time.monotonic() - started
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
    if result is None:
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
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
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
    if result is None:
        raise WeightsError(f"wrote {record_path} but {alias.id} does not read as installed")
    return result


@runtime_checkable
class ArchiveSource(Protocol):

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
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
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
    if result is None:
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


@runtime_checkable
class FileSource(Protocol):

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
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
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
    if result is None:
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack(archive: Path, target: Path) -> None:
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
