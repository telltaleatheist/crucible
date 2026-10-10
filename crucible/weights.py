from __future__ import annotations

import fnmatch
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
from .tomltable import SHA256_PATTERN

HF_TOKEN_ENV = "HF_TOKEN"
STAMP_NAME = "crucible-pull.json"
HUB_CACHE_ENVS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")


class _QuietUnauthenticated(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()


logging.getLogger("huggingface_hub.utils._http").addFilter(_QuietUnauthenticated())

PINNED = "pinned"
LOCAL = "local"

FAMILY_NOUNS: dict[str, str] = {
    "models": "model",
    "voices": "voice",
    "rvc": "RVC model",
    "denoise": "denoise model",
}


@runtime_checkable
class WeightsSubject(Protocol):

    id: str
    path: Path
    weights_family: str

    @property
    def pull_command(self) -> str:
        ...

    def aliases(self) -> "tuple[WeightsSubject, ...]":
        ...


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
    if getattr(manifest, "weights_of", None) is not None:
        return ()
    holding: list[str] = []
    for alias in manifest.aliases():
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
        _refuse_missing_local(manifest, spec, local)
    directory = subject_dir(config, manifest, spec.backend)
    if getattr(manifest, "weights_base", None) is not None:
        _refuse_missing_alias(config, manifest, spec, directory)
    stamp = directory / STAMP_NAME
    if stamp.is_file():
        _refuse_stamped(manifest, spec, directory, stamp)
    raise WeightsError(
        f"{FAMILY_NOUNS[manifest.weights_family]} {manifest.id!r} is not installed "
        f"for {spec.backend}; there are no weights at {directory} — run "
        f"`{manifest.pull_command}`"
    )


def _refuse_missing_local(manifest: WeightsSubject, spec: WeightsSource, local: Path) -> None:
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


def _refuse_missing_alias(
    config: Config, manifest: Any, spec: WeightsSource, directory: Path
) -> None:
    noun = FAMILY_NOUNS[manifest.weights_family]
    command = manifest.pull_command
    base = manifest.weights_base
    extras = _extra_files(manifest, spec)
    if installed(config, base, base.spec(spec.backend)) is None:
        raise WeightsError(
            f"{noun} {manifest.id!r} shares the weights of {base.id!r}, and "
            f"{base.id!r} is not installed for {spec.backend} — run "
            f"`{command}`, which pulls {base.id!r}'s download"
            + (f" plus {', '.join(extras)}" if extras else "")
            + " into one folder"
        )
    absent = missing_files(directory, spec)
    raise WeightsError(
        f"{noun} {manifest.id!r} shares the weights of {base.id!r}, which is "
        f"installed at {directory}, and {len(absent)} of its own file(s) are "
        f"not there: {', '.join(absent)} — run `{command}`"
    )


def _refuse_stamped(
    manifest: WeightsSubject, spec: WeightsSource, directory: Path, stamp: Path
) -> None:
    command = manifest.pull_command
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["revision"] == spec.revision and record["hf_repo"] == spec.hf_repo:
        absent = missing_files(directory, spec)
        raise WeightsError(
            f"{directory} is stamped for {spec.hf_repo}@{spec.revision[:12]} "
            f"but {len(absent)} of the {len(spec.files)} file(s) it names "
            f"are not there: {', '.join(absent)} — run `{command} "
            "--force`"
        )
    raise WeightsError(
        f"{directory} holds {record['hf_repo']}@{record['revision'][:12]}, but "
        f"{manifest.path.name} now pins {spec.hf_repo}@{spec.revision[:12]} — "
        f"run `{command}`"
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


def remove_stranded(config: Config, entry: StrandedWeights) -> Path:
    # A stranded directory has no manifest left to ask about local sources or
    # aliases: `stranded` only ever names a folder under the store's own root,
    # which Crucible wrote, so the folder itself is what is removed.
    _remove(entry.path)
    _prune_empty(entry.path.parent, weights_root(config, entry.family))
    return entry.path


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


def _say(on_line: Callable[[str], None] | None, text: str) -> None:
    if on_line is not None:
        on_line(text)


def _clear_stamp(directory: Path, stamp_name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / stamp_name
    if stamp.exists():
        stamp.unlink()
    return stamp


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read_back(result: InstalledWeights | None, stamp: Path) -> InstalledWeights:
    if result is None:
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result


def _pulled_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _token_phrase(token: Any) -> str:
    return "with" if token else "without"


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
    stamp = _clear_stamp(target, STAMP_NAME)
    linked = adopt_hub_cache(spec, target, on_line)
    started = _fetch_snapshot(config, manifest, spec, target, on_line, on_progress)
    elapsed = time.monotonic() - started
    _require_snapshot_complete(manifest, spec, target)
    size = directory_bytes(target)
    _write_record(stamp, {**_snapshot_record(manifest, spec, size, elapsed), "linked_bytes": linked})
    _say(
        on_line,
        f"pulled {size / 1e9:.2f} GB in {elapsed:.0f}s "
        f"({size / 1e6 / max(elapsed, 1e-6):.0f} MB/s)",
    )
    return _read_back(installed(config, manifest, spec), stamp)


def hub_cache_root() -> Path:
    for name in HUB_CACHE_ENVS:
        stated = os.environ.get(name)
        if stated:
            return Path(stated).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def cached_snapshot(spec: WeightsSource) -> Path | None:
    if not spec.hf_repo or not spec.revision:
        return None
    owner, _, name = spec.hf_repo.partition("/")
    snapshot = hub_cache_root() / f"models--{owner}--{name}" / "snapshots" / spec.revision
    return snapshot if snapshot.is_dir() else None


def _wanted(relative: str, spec: WeightsSource) -> bool:
    return not spec.files or any(fnmatch.fnmatch(relative, pattern) for pattern in spec.files)


def _hashed_blob(source: Path) -> Path | None:
    blob = source.resolve()
    return blob if SHA256_PATTERN.match(blob.name) else None


def _link_into(source: Path, destination: Path) -> int:
    blob = _hashed_blob(source)
    if blob is None or destination.exists():
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(blob, destination)
    except OSError:
        return 0
    return destination.stat().st_size


def adopt_hub_cache(
    spec: WeightsSource, target: Path, on_line: Callable[[str], None] | None = None
) -> int:
    snapshot = cached_snapshot(spec)
    if snapshot is None:
        return 0
    linked = sum(
        _link_into(source, target / source.relative_to(snapshot))
        for source in sorted(snapshot.rglob("*"))
        if source.is_file() and _wanted(source.relative_to(snapshot).as_posix(), spec)
    )
    if linked:
        _say(
            on_line,
            f"linked {linked / 1e9:.2f} GB from the Hugging Face cache at {snapshot} "
            "(the same revision, no second copy on disk); each file is hashed "
            "against the hub before it counts, and anything missing or different "
            "is downloaded",
        )
    return linked


def _fetch_snapshot(
    config: Config,
    manifest: WeightsSubject,
    spec: WeightsSource,
    target: Path,
    on_line: Callable[[str], None] | None,
    on_progress: ProgressHook | None,
) -> float:
    token = hf_token(config)
    _say(
        on_line,
        f"pulling {spec.hf_repo}@{spec.revision[:12]} -> {target} "
        f"({_token_phrase(token)} an HF token)",
    )
    started = time.monotonic()
    if spec.files:
        _say(on_line, f"only {len(spec.files)} file(s) of that repo: " + ", ".join(spec.files))
    try:
        _snapshot(
            config, manifest.path.name, spec, target,
            patterns=spec.files, on_progress=on_progress,
            retry=manifest.pull_command,
        )
    except PullCancelled:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return started


def _require_snapshot_complete(
    manifest: WeightsSubject, spec: WeightsSource, target: Path
) -> None:
    absent = missing_files(target, spec)
    if absent:
        raise WeightsError(
            f"{spec.hf_repo}@{spec.revision[:12]} was fetched but "
            f"{len(absent)} of the file(s) {manifest.path.name} names for "
            f"{spec.backend} are not in {target}: {', '.join(absent)}. Either "
            "the manifest names a file this revision does not have, or the "
            "download was incomplete; nothing is stamped either way"
        )


def _snapshot_record(
    manifest: WeightsSubject, spec: WeightsSource, size: int, elapsed: float
) -> dict[str, Any]:
    return {
        "family": manifest.weights_family,
        "id": manifest.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "files": list(spec.files),
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": _pulled_now(),
    }


HF_ACCEPT_URL = "https://huggingface.co/{repo}"

HF_TOKENS_URL = "https://huggingface.co/settings/tokens"


def gated_message(
    repo: str, config: Config, retry: str | None, cause: Exception | None = None
) -> str:
    again = f"`{retry}`" if retry else "the same pull"
    return (
        f"{repo} is gated: Hugging Face serves it only to an account that has "
        "accepted its licence, and this server has no HF token that opens it. "
        "Nothing was downloaded. 1) Signed in to Hugging Face, open "
        f"{HF_ACCEPT_URL.format(repo=repo)} and accept the licence there. "
        f"2) Make a read token at {HF_TOKENS_URL} and give it to this server: "
        f"set ${HF_TOKEN_ENV} in the environment Crucible runs in, or put it "
        f"under [hf] token in {config.path}. 3) Run {again} again"
        + ("" if cause is None else f" (Hugging Face said: {cause})")
    )


def _snapshot(
    config: Config,
    manifest_name: str,
    spec: WeightsSource,
    target: Path,
    *,
    patterns: Sequence[str],
    on_progress: ProgressHook | None,
    retry: str | None = None,
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
        raise WeightsError(gated_message(spec.hf_repo, config, retry, exc)) from exc
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
    target = subject_dir(config, alias, spec.backend)
    record_path = alias_record_path(config, alias, spec.backend)
    existing = installed(config, alias, spec)
    if existing is not None and record_path.is_file() and not force:
        return existing
    _ensure_base_pulled(config, alias, spec, on_line, on_progress)
    extras = alias.extra_files(spec.backend)
    if force:
        for name in extras:
            _remove(target / name)
    wanted = [name for name in extras if not (target / name).is_file()]
    started = time.monotonic()
    _fetch_alias_files(config, alias, spec, target, wanted, on_line, on_progress)
    _require_alias_complete(alias, spec, target)
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
        "pulled": _pulled_now(),
    }
    _write_record(record_path, record)
    _say(
        on_line,
        f"{alias.id}: {own / 1e9:.2f} GB of its own beside {base.id}'s weights "
        f"at {target}",
    )
    result = installed(config, alias, spec)
    if result is None:
        raise WeightsError(f"wrote {record_path} but {alias.id} does not read as installed")
    return result


def _ensure_base_pulled(
    config: Config,
    alias: Any,
    spec: WeightsSource,
    on_line: Callable[[str], None] | None,
    on_progress: ProgressHook | None,
) -> None:
    base = alias.weights_base
    base_spec = base.spec(spec.backend)
    if installed(config, base, base_spec) is not None:
        return
    _say(
        on_line,
        f"{alias.id} shares the weights of {base.id}; pulling {base.id} "
        "first, once, into its own folder",
    )
    pull(config, base, base_spec, on_line=on_line, on_progress=on_progress)


def _fetch_alias_files(
    config: Config,
    alias: Any,
    spec: WeightsSource,
    target: Path,
    wanted: list[str],
    on_line: Callable[[str], None] | None,
    on_progress: ProgressHook | None,
) -> None:
    if not wanted:
        return
    _say(
        on_line,
        f"pulling {alias.id}'s own file(s) from "
        f"{spec.hf_repo}@{spec.revision[:12]} into {target}: "
        + ", ".join(wanted),
    )
    try:
        _snapshot(
            config, alias.path.name, spec, target,
            patterns=wanted, on_progress=on_progress,
            retry=alias.pull_command,
        )
    except PullCancelled:
        for name in wanted:
            (target / name).unlink(missing_ok=True)
        raise


def _require_alias_complete(alias: Any, spec: WeightsSource, target: Path) -> None:
    absent = missing_files(target, spec)
    if absent:
        raise WeightsError(
            f"{spec.hf_repo}@{spec.revision[:12]} was fetched but {len(absent)} of "
            f"the file(s) {alias.path.name} names for {spec.backend} are not in "
            f"{target}: {', '.join(absent)}. Either the manifest names a file this "
            "revision does not have, or the download was incomplete; no alias "
            "record is written either way"
        )


@runtime_checkable
class ArchiveSource(Protocol):

    backend: str
    hf_repo: str
    revision: str
    archive: str
    archive_sha256: str


def _import_hub(config: Config) -> Any:
    try:
        import huggingface_hub
        import huggingface_hub.errors
    except ImportError as exc:
        raise WeightsError(
            f"huggingface_hub is not importable in {config.name}'s interpreter: {exc}"
        ) from exc
    return huggingface_hub


def _progress_extra(on_progress: ProgressHook | None) -> dict[str, Any]:
    if on_progress is None:
        return {}
    return {"tqdm_class": reporting_tqdm(on_progress)}


@dataclass(frozen=True)
class HubFile:

    repo: str
    revision: str
    filename: str
    pinned_by: str
    names: str


def download_error(exc: Exception, errors: Any, config: Config, wanted: HubFile) -> WeightsError:
    repo, revision = wanted.repo, wanted.revision
    messages = (
        (errors.GatedRepoError, lambda: gated_message(repo, config, None, exc)),
        (errors.RepositoryNotFoundError, lambda: (
            f"{repo} is private or does not exist; if it is private set "
            f"${HF_TOKEN_ENV} or [hf] token in {config.path}: {exc}"
        )),
        (errors.RevisionNotFoundError, lambda: (
            f"{repo} has no revision {revision}; "
            f"{wanted.pinned_by} pins a commit that repo does not have: {exc}"
        )),
        (errors.EntryNotFoundError, lambda: (
            f"{repo}@{revision[:12]} has no file {wanted.filename!r}; "
            f"{wanted.pinned_by} names {wanted.names} that revision does not hold: "
            f"{exc}"
        )),
    )
    for kind, message in messages:
        if isinstance(exc, kind):
            return WeightsError(message())
    return WeightsError(
        f"pulling {repo}:{wanted.filename} failed: {type(exc).__name__}: {exc}"
    )


def _hub_download(
    hub: Any,
    config: Config,
    wanted: HubFile,
    *,
    local_dir: Path,
    token: Any,
    extra: dict[str, Any],
) -> str:
    try:
        return hub.hf_hub_download(
            repo_id=wanted.repo,
            filename=wanted.filename,
            revision=wanted.revision,
            local_dir=str(local_dir),
            token=token,
            **extra,
        )
    except PullCancelled:
        raise
    except Exception as exc:
        raise download_error(exc, hub.errors, config, wanted) from exc


def pull_archive(
    config: Config,
    manifest: WeightsSubject,
    spec: ArchiveSource,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> InstalledWeights:
    hub = _import_hub(config)
    target = weights_dir(config, manifest.weights_family, manifest.id, spec.backend)
    existing = installed(config, manifest, spec)
    if existing is not None and not force:
        return existing
    if force and target.exists():
        shutil.rmtree(target)
    stamp = _clear_stamp(target, STAMP_NAME)
    token = hf_token(config)
    _say(
        on_line,
        f"pulling {spec.hf_repo}@{spec.revision[:12]}:{spec.archive} -> {target} "
        f"({_token_phrase(token)} an HF token)",
    )
    started = time.monotonic()
    staging = _fresh_dir(target / ".crucible-archive")
    downloaded = Path(
        _download_archive(hub, config, manifest, spec, target, staging, token, on_progress)
    )
    digest = _verify_archive(manifest, spec, downloaded, staging)
    try:
        _unpack(downloaded, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    elapsed = time.monotonic() - started
    size = directory_bytes(target)
    _write_record(stamp, _archive_record(manifest, spec, digest, size, elapsed))
    _say(on_line, f"unpacked {size / 1e9:.2f} GB in {elapsed:.0f}s at {target}")
    return _read_back(installed(config, manifest, spec), stamp)


def _download_archive(
    hub: Any,
    config: Config,
    manifest: WeightsSubject,
    spec: ArchiveSource,
    target: Path,
    staging: Path,
    token: Any,
    on_progress: ProgressHook | None,
) -> str:
    wanted = HubFile(
        repo=spec.hf_repo,
        revision=spec.revision,
        filename=spec.archive,
        pinned_by=manifest.path.name,
        names="an archive",
    )
    extra = _progress_extra(on_progress)
    try:
        return _hub_download(
            hub, config, wanted, local_dir=staging, token=token, extra=extra
        )
    except PullCancelled:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _verify_archive(
    manifest: WeightsSubject, spec: ArchiveSource, downloaded: Path, staging: Path
) -> str:
    digest = sha256_of(downloaded)
    if digest != spec.archive_sha256:
        shutil.rmtree(staging, ignore_errors=True)
        raise WeightsError(
            f"{spec.archive} from {spec.hf_repo}@{spec.revision[:12]} hashes to "
            f"{digest}, but {manifest.path.name} pins {spec.archive_sha256}. Nothing "
            "was unpacked. Either the manifest is wrong or these are not the bytes "
            "it names, and both are worse than no weights at all."
        )
    return digest


def _archive_record(
    manifest: WeightsSubject, spec: ArchiveSource, digest: str, size: int, elapsed: float
) -> dict[str, Any]:
    return {
        "family": manifest.weights_family,
        "id": manifest.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "archive": spec.archive,
        "archive_sha256": digest,
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": _pulled_now(),
    }


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
    hub = _import_hub(config)
    if not files:
        raise WeightsError(
            f"{label} declares no files; a set with nothing in it is not a set"
        )
    existing = files_installed(target_root, hf_repo, revision, stamp_name=stamp_name)
    if existing is not None and not force:
        return existing
    stamp = _clear_stamp(target_root, stamp_name)
    staging = _fresh_dir(target_root / ".crucible-files")
    token = hf_token(config)
    _say(
        on_line,
        f"pulling {len(files)} file(s) of {label} from "
        f"{hf_repo}@{revision[:12]} -> {target_root} "
        f"({_token_phrase(token)} an HF token)",
    )
    started = time.monotonic()
    extra = _progress_extra(on_progress)
    try:
        fetched = [
            _fetch_verified(
                hub, config, entry, hf_repo=hf_repo, revision=revision, label=label,
                target_root=target_root, staging=staging, token=token, extra=extra,
                on_line=on_line,
            )
            for entry in files
        ]
        total = _place_fetched(fetched)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    elapsed = time.monotonic() - started
    _write_record(stamp, _files_record(label, hf_repo, revision, files, total, elapsed))
    _say(on_line, f"placed {total / 1e9:.2f} GB in {elapsed:.0f}s at {target_root}")
    return _read_back(
        files_installed(target_root, hf_repo, revision, stamp_name=stamp_name), stamp
    )


def _fetch_verified(
    hub: Any,
    config: Config,
    entry: FileSource,
    *,
    hf_repo: str,
    revision: str,
    label: str,
    target_root: Path,
    staging: Path,
    token: Any,
    extra: dict[str, Any],
    on_line: Callable[[str], None] | None,
) -> tuple[FileSource, Path, Path]:
    destination = _safe_target(target_root, entry.target)
    wanted = HubFile(
        repo=hf_repo, revision=revision, filename=entry.source,
        pinned_by=label, names="a path",
    )
    downloaded = _hub_download(
        hub, config, wanted, local_dir=staging, token=token, extra=extra
    )
    digest = sha256_of(Path(downloaded))
    if digest != entry.sha256:
        raise WeightsError(
            f"{entry.source} from {hf_repo}@{revision[:12]} hashes to "
            f"{digest}, but {label} pins {entry.sha256}. NOTHING was "
            "placed. Either the declaration is wrong or these are not "
            "the bytes it names, and both are worse than no weights"
        )
    _say(on_line, f"  verified {entry.target} ({digest[:12]})")
    return entry, Path(downloaded), destination


def _place_fetched(fetched: Sequence[tuple[FileSource, Path, Path]]) -> int:
    total = 0
    for _entry, downloaded, destination in fetched:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(downloaded), str(destination))
        total += destination.stat().st_size
    return total


def _files_record(
    label: str,
    hf_repo: str,
    revision: str,
    files: Sequence[FileSource],
    total: int,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "label": label,
        "hf_repo": hf_repo,
        "revision": revision,
        "files": [
            {"source": entry.source, "target": entry.target, "sha256": entry.sha256}
            for entry in files
        ],
        "bytes": total,
        "seconds": round(elapsed, 1),
        "pulled": _pulled_now(),
    }


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_member(archive: Path, target: Path, root: Path, member: tarfile.TarInfo) -> None:
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


def _unpack(archive: Path, target: Path) -> None:
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            root = target.resolve()
            for member in members:
                _check_member(archive, target, root, member)
            handle.extractall(path=target, members=members)
    except tarfile.TarError as exc:
        raise WeightsError(f"could not unpack {archive.name}: {exc}") from exc
