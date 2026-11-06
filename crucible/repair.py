"""Verify what a server has installed, and put back what is broken.

Owen, 2026-11-04: "a doctor that repairs broken environments by redownloading them if they
arent right". One repair, two doors: the `repair` task and `crucible doctor --repair` both run
`run()` here, and answer the same result.

What "broken" means is deliberately narrow, because the cure for a broken env is a rebuild of
many gigabytes:

- An env is broken when its stamp no longer matches its recipe (`jobenv.env_status`), when its
  smoke import fails, or when a file a package installed is missing or hashes to nothing any
  package's RECORD says of it. Two packages can own one path, and the last installed wins
  (flashinfer and torch_c_dlpack_ext both ship `build_backend.py`), so matching ANY claim is
  healthy. A file an env patch rewrites on purpose is not held to its RECORD; `envpatches.check`
  judges those. A wheel whose RECORD is wrong about its own files is named in `RECORD_WRONG`,
  because a rebuild would install the same bytes. `pip check` is NOT a test: several recipes
  install packages with `--no-deps` on purpose, and it reports those healthy envs as conflicts.
- A subject that finished a pull of its pin (`Subject.was_pulled`) and no longer reads as
  installed has lost a file, and is broken: nobody pulling it and a pull that lost a file both read
  "not installed", and only the second is a repair's to mend.
- A subject's weights are broken when a file the subject needs is missing, or when a file does
  not hash to what the hub says that file is at the pinned revision, or, for a companion from
  another repo, to the sha256 its manifest pins. If the hub cannot be asked (the network is
  weather), or none of a subject's files can be matched to anything, the subject is `unverified`,
  never broken: nothing is deleted on a guess. What could not be hashed against anything is listed
  in `not_hash_checked`, so a clean result never claims more than was checked. An RVC voice is one
  of those: its manifest pins the archive it came in and not the files unpacked from it.

The cure for weights is the bad files alone, and when a subject's pull cannot fetch just those it
is the whole subject again. If that does not cure it, or the subject has no source, it fails by
name.

Nothing here touches the card, and nothing here takes files out from under a running engine: an
env or a subject that is in use is still READ and judged, and its repair waits until it is free,
for `WAIT_FOR_FREE_S` at most, then fails by name so another repair can finish the job.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import catalog, envpatches, jobenv, pagecache, weights
from .atomicjson import write_json
from .backend import Backend
from .config import Config

# How long a repair waits for what it must replace to come free, and how often it looks.
WAIT_FOR_FREE_S = 30 * 60.0
WAIT_POLL_S = 15.0

VERIFIED_CACHE = "repair-verified.json"
HASH_CHUNK = 1 << 20
PROGRESS_EVERY_BYTES = 256 << 20

ENV = "env"

# Published wheels whose RECORD does not describe their own files: {dist-info directory: paths}.
# pynvvideocodec 2.0.4's cp311 manylinux wheel says 2,402,945 bytes for a .so it holds at
# 2,439,136 (read out of the wheel itself, 2026-11-04). Named by version: a new release is
# judged again.
RECORD_WRONG: dict[str, frozenset[str]] = {
    "pynvvideocodec-2.0.4.dist-info": frozenset({
        "PyNvVideoCodec/PyNvVideoCodec_121.cpython-311-x86_64-linux-gnu.so",
        "PyNvVideoCodec/PyNvVideoCodec_130.cpython-311-x86_64-linux-gnu.so",
    }),
}


class RepairCancelled(Exception):
    ...


@dataclass(frozen=True)
class Scope:
    """What to look at. `envs` is None for every installed env, or the job types named; a
    `weights` of False leaves weights alone; `verify_only` reports and changes nothing."""

    envs: tuple[str, ...] | None = None
    weights: bool = True
    verify_only: bool = False

    @staticmethod
    def from_request(raw: dict[str, Any] | None) -> "Scope":
        raw = raw or {}
        envs = raw.get("envs", "all")
        return Scope(
            envs=None if envs == "all" else tuple(envs),
            weights=raw.get("weights", "all") != "none",
            verify_only=bool(raw.get("verify_only", False)),
        )


def scope_problems(raw: Any) -> list[str]:
    """Why a request's `scope` cannot be run, in words; empty when it can."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["scope must be an object"]
    problems = [f"scope has unknown key {key!r}" for key in sorted(set(raw) - {"envs", "weights", "verify_only"})]
    envs = raw.get("envs", "all")
    if envs != "all" and not (isinstance(envs, list) and all(isinstance(e, str) for e in envs)):
        problems.append("scope.envs must be 'all' or a list of job types")
    if raw.get("weights", "all") not in ("all", "none"):
        problems.append("scope.weights must be 'all' or 'none'")
    if not isinstance(raw.get("verify_only", False), bool):
        problems.append("scope.verify_only must be true or false")
    return problems


@dataclass
class Report:
    """The result both doors answer. `healthy` is judged after repair: a subject counts as
    repaired only once it has been verified again."""

    checked: list[dict[str, str]] = field(default_factory=list)
    broken: list[dict[str, Any]] = field(default_factory=list)
    repaired: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    unverified: list[dict[str, Any]] = field(default_factory=list)
    not_hash_checked: list[dict[str, Any]] = field(default_factory=list)
    bytes_hashed: int = 0
    bytes_skipped_verified: int = 0
    verify_only: bool = False

    @property
    def healthy(self) -> bool:
        return not self.failed and not (self.verify_only and self.broken)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "verify_only": self.verify_only,
            "checked": self.checked,
            "broken": self.broken,
            "repaired": self.repaired,
            "failed": self.failed,
            "unverified": self.unverified,
            "not_hash_checked": self.not_hash_checked,
            "bytes_hashed": self.bytes_hashed,
            "bytes_skipped_verified": self.bytes_skipped_verified,
        }


Emit = Callable[[str, dict[str, Any]], None]


def _subject(kind: str, subject_id: str) -> dict[str, str]:
    return {"kind": kind, "id": subject_id}


class VerifiedCache:
    """Files already hashed, by path, size and modified time. Any change to a file changes one
    of those, and an entry only answers for the digest it recorded, so a file that was replaced
    or a pin that moved is hashed again."""

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        self._rows: dict[str, list[Any]] = loaded if isinstance(loaded, dict) else {}

    def knows(self, path: Path, algorithm: str, digest: str) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False
        return self._rows.get(str(path)) == [stat.st_size, stat.st_mtime_ns, algorithm, digest]

    def remember(self, path: Path, algorithm: str, digest: str) -> None:
        stat = path.stat()
        self._rows[str(path)] = [stat.st_size, stat.st_mtime_ns, algorithm, digest]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self._path, self._rows, indent=None)


class Hasher:
    """Hashes files, skipping what the cache already vouches for, and says how far it has got."""

    def __init__(
        self, report: Report, cache: VerifiedCache, emit: Emit, cancelled: Callable[[], bool]
    ) -> None:
        self._report = report
        self._cache = cache
        self._emit = emit
        self._cancelled = cancelled
        self._since_said = 0
        self.total = 0

    def _progress(self, name: str) -> None:
        self._emit(
            "progress",
            {
                "bytes_done": self._report.bytes_hashed + self._report.bytes_skipped_verified,
                "bytes_total": self.total,
                "file": name,
            },
        )

    def matches(
        self, path: Path, algorithm: str, wants: tuple[str, ...], digest: Callable[[Path], str]
    ) -> bool:
        """Whether the file hashes to one of `wants` (more than one when packages share a path)."""
        size = path.stat().st_size
        if any(self._cache.knows(path, algorithm, want) for want in wants):
            self._report.bytes_skipped_verified += size
            return True
        got = digest(path)
        self._report.bytes_hashed += size
        self._since_said += size
        if self._since_said >= PROGRESS_EVERY_BYTES:
            self._since_said = 0
            self._progress(path.name)
        if got not in wants:
            return False
        self._cache.remember(path, algorithm, got)
        return True

    def read(self, path: Path, into: Any) -> None:
        with path.open("rb") as handle:
            done = 0
            while block := handle.read(HASH_CHUNK):
                if self._cancelled():
                    raise RepairCancelled()
                into.update(block)
                done += len(block)
                pagecache.forget(handle.fileno(), done)

    def sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        self.read(path, digest)
        return digest.hexdigest()

    def record_digest(self, path: Path) -> str:
        """`sha256=<urlsafe base64, no padding>`, RECORD's own spelling."""
        digest = hashlib.sha256()
        self.read(path, digest)
        encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
        return f"sha256={encoded}"

    def git_blob_sha1(self, path: Path) -> str:
        """What the hub calls a file that is not in LFS: git's blob id."""
        digest = hashlib.sha1()  # noqa: S324 (git's own identifier, not a security check)
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
        self.read(path, digest)
        return digest.hexdigest()


# ── envs ─────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecordRow:
    path: Path
    digests: tuple[str, ...]
    size: int


def _rewritten_on_purpose(spec: jobenv.EnvSpec, packages: Path) -> frozenset[Path]:
    patches = envpatches.patches_for(spec.job_type)
    if patches is None:
        return frozenset()
    return frozenset((packages / patch.rel_path) for patch in patches.patches)


def record_wrong_here(directory: Path) -> list[str]:
    """The files of this env left out because their own wheel's RECORD is wrong about them."""
    packages = envpatches.site_packages(directory)
    if packages is None:
        return []
    return sorted(
        name
        for dist, names in RECORD_WRONG.items()
        if (packages / dist).is_dir()
        for name in names
        if (packages / name).exists()
    )


def record_rows(spec: jobenv.EnvSpec, directory: Path) -> tuple[list[RecordRow], list[str]]:
    """Every file with a hash in a RECORD of this env, and the rows that cannot be read. A
    compiled .pyc is left out (its RECORD hash is of a file Python may write again), and so is
    what an env patch rewrites on purpose."""
    packages = envpatches.site_packages(directory)
    if packages is None:
        return [], [f"{directory} has no site-packages"]
    skip = _rewritten_on_purpose(spec, packages)
    claims: dict[Path, tuple[tuple[str, ...], int]] = {}
    unreadable: list[str] = []
    for record in sorted(packages.glob("*.dist-info/RECORD")):
        try:
            text = record.read_text(encoding="utf-8")
        except OSError as exc:
            unreadable.append(f"{record}: {exc}")
            continue
        wrong = RECORD_WRONG.get(record.parent.name, frozenset())
        for line in csv.reader(text.splitlines()):
            if len(line) < 3 or not line[1].startswith("sha256=") or line[0].endswith(".pyc"):
                continue
            if line[0] in wrong:
                continue
            path = (packages / line[0]).resolve() if ".." in Path(line[0]).parts else packages / line[0]
            if path in skip:
                continue
            digests, _size = claims.get(path, ((), 0))
            claims[path] = ((*digests, line[1]), int(line[2]) if line[2].isdigit() else 0)
    return [RecordRow(path, *claim) for path, claim in claims.items()], unreadable


def _smoke_failure(python: Path, spec: jobenv.EnvSpec, backend_kind: str) -> str | None:
    module = jobenv.SMOKE_IMPORT.get(spec.key, {}).get(backend_kind)
    if module is None:
        return None
    done = subprocess.run(
        [str(python), "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if done.returncode == 0:
        return None
    said = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return f"`import {module}` exited {done.returncode}: " + " / ".join(said[-3:] or ["no output"])


def env_problem(
    config: Config, backend: Backend, spec: jobenv.EnvSpec, rows: Iterable[RecordRow], hasher: Hasher
) -> str | None:
    """What is wrong with an installed env, in words, or None when nothing is."""
    status = jobenv.env_status(config.home, spec, backend.kind)
    if not status.installed:
        return status.detail
    python = jobenv.env_python(config.home, spec)
    missing: list[str] = []
    different: list[str] = []
    for row in rows:
        if not row.path.is_file():
            missing.append(str(row.path))
        elif not hasher.matches(row.path, "record", row.digests, hasher.record_digest):
            different.append(str(row.path))
    if missing or different:
        named = (different + missing)[:5]
        return (
            f"{len(missing)} file(s) missing and {len(different)} changed against what their "
            f"packages installed, among them {', '.join(named)}"
        )
    return _smoke_failure(python, spec, backend.kind)


def installed_envs(config: Config, backend: Backend, scope: Scope) -> list[jobenv.EnvSpec]:
    return [
        spec
        for spec in jobenv.every_env(backend.kind)
        if jobenv.env_python(config.home, spec).is_file()
        and (scope.envs is None or spec.job_type in scope.envs)
    ]


# ── weights ──────────────────────────────────────────────────────────────────────────────────


class HubUnreachable(Exception):
    ...


@dataclass(frozen=True)
class HubFile:
    path: str
    sha256: str | None
    blob_id: str | None


def hub_files(config: Config, hf_repo: str, revision: str) -> dict[str, HubFile]:
    """What the hub says each file of this revision is. Raises HubUnreachable for anything that
    is not an answer: a repair never decides weights are broken because the network was down."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=weights.hf_token(config)).model_info(
            hf_repo, revision=revision, files_metadata=True, timeout=60
        )
    except Exception as exc:  # noqa: BLE001 (every way of not getting an answer is the same answer)
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            # With a good login this is an answer, and a serious one: this machine may hold the
            # only copy. Nothing here can check it or fetch it again, so nothing here deletes it.
            raise HubUnreachable(
                f"the hub has no {hf_repo}@{revision[:12]} any more (or this login cannot see "
                "it), so this copy can be neither checked nor fetched again; repin it to what "
                "the hub holds, or upload it"
            ) from exc
        raise HubUnreachable(f"could not ask the hub about {hf_repo}@{revision[:12]}: {exc}") from exc
    found: dict[str, HubFile] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        found[sibling.rfilename] = HubFile(sibling.rfilename, sha, getattr(sibling, "blob_id", None))
    return found


def _match_hub_file(relative: str, hub: dict[str, HubFile]) -> HubFile | None:
    found = hub.get(relative)
    if found is not None:
        return found
    # A subject stored flat from a path deep in its repo (a denoise checkpoint): one file in the
    # repo with that name is that file; two are not an answer.
    named = [entry for entry in hub.values() if Path(entry.path).name == Path(relative).name]
    return named[0] if len(named) == 1 else None


def _on_disk(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and ".cache" not in path.relative_to(directory).parts
        and path.name != weights.STAMP_NAME
    ]


def _pins_of(subject: catalog.Subject) -> dict[str, str]:
    pinned = getattr(subject, "pinned_files", None)
    return {} if pinned is None else pinned()


def weights_problem(
    subject: catalog.Subject, found: weights.InstalledWeights, hub: dict[str, HubFile], hasher: Hasher
) -> tuple[list[str], list[str], list[str]]:
    """(the files that do not hash to what they should, the files the subject lacks, the files
    nothing could be compared with)."""
    missing = list(subject.missing_files()) if subject.missing_files is not None else []
    pins = _pins_of(subject)
    missing += [relative for relative in sorted(pins) if not (found.path / relative).is_file()]
    wants = getattr(subject, "wants_file", None)
    if wants is not None:
        missing += [
            relative
            for relative in sorted(hub)
            if relative not in pins and wants(relative) and not (found.path / relative).is_file()
        ]
    different: list[str] = []
    unjudged: list[str] = []
    only_pinned = getattr(subject, "pinned_only", False)
    for path in _on_disk(found.path):
        relative = path.relative_to(found.path).as_posix()
        pin = pins.get(relative)
        if pin is None and only_pinned:
            continue
        if pin is not None:
            if not hasher.matches(path, "sha256", (pin,), hasher.sha256):
                different.append(relative)
            continue
        entry = _match_hub_file(relative, hub)
        if entry is None:
            unjudged.append(relative)
            continue
        if entry.sha256 is not None:
            good = hasher.matches(path, "sha256", (entry.sha256,), hasher.sha256)
        elif entry.blob_id is not None:
            good = hasher.matches(path, "git", (entry.blob_id,), hasher.git_blob_sha1)
        else:
            unjudged.append(relative)
            continue
        if not good:
            different.append(relative)
    return different, missing, unjudged


def _drop_hub_copy(hf_repo: str, revision: str, relative: str) -> None:
    """The hub cache's copy of a file, which `adopt_hub_cache` would otherwise link straight back
    (the two are often one inode, so a damaged file is damaged in both)."""
    root = weights.hub_cache_root() / f"models--{hf_repo.replace('/', '--')}" / "snapshots" / revision
    linked = root / relative
    if not linked.exists() and not linked.is_symlink():
        return
    blob = linked.resolve()
    linked.unlink(missing_ok=True)
    if blob != linked:
        blob.unlink(missing_ok=True)


def _put_back(
    subject: catalog.Subject,
    found: weights.InstalledWeights,
    different: list[str],
    on_line: Callable[[str], None],
    on_progress: Callable[[int, int | None, str], None],
) -> None:
    """Delete only the bad files and fetch again: what is good stays, and is not downloaded."""
    pins = _pins_of(subject)
    for relative in different:
        (found.path / relative).unlink(missing_ok=True)
        if relative not in pins and found.hf_repo is not None and found.revision is not None:
            _drop_hub_copy(found.hf_repo, found.revision, relative)
    stamp = found.path / weights.STAMP_NAME
    kept = stamp.read_bytes() if stamp.is_file() else None
    stamp.unlink(missing_ok=True)
    try:
        # A pull_files subject (a separator, the RVC base) has its own stamp and answers from it,
        # so this can come back having fetched nothing; `_repair_subject` then asks for all of it.
        subject.pull(force=False, on_line=on_line, on_progress=on_progress)
    except BaseException:
        # Cancelled or failed part way: the stamp is what lets the next repair see this subject
        # lost a file, instead of reading it as one nobody pulled.
        if kept is not None and not stamp.exists() and found.path.is_dir():
            stamp.write_bytes(kept)
        raise


# ── the repair ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Doors:
    """How a repair learns what is in use. Both answer None when the thing is free, or who
    holds it in words. The CLI runs with no server up, so its doors always answer None."""

    env_in_use: Callable[[jobenv.EnvSpec], str | None]
    subject_in_use: Callable[[catalog.Subject], str | None]


def _wait_until_free(
    what: dict[str, str],
    holder: Callable[[], str | None],
    emit: Emit,
    cancelled: Callable[[], bool],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    budget_s: float,
) -> str | None:
    """None once it is free, or who still holds it when the budget is spent."""
    deadline = clock() + budget_s
    said = False
    while True:
        who = holder()
        if who is None:
            return None
        if clock() >= deadline:
            return who
        if not said:
            emit("repair", {"step": "wait", "subject": what, "state": "running", "detail": who})
            said = True
        if cancelled():
            raise RepairCancelled()
        sleep(WAIT_POLL_S)


def _env_bytes(rows: Iterable[RecordRow]) -> int:
    return sum(row.size for row in rows)


def run(
    config: Config,
    backend: Backend,
    scope: Scope,
    doors: Doors,
    *,
    emit: Emit,
    cancelled: Callable[[], bool] = lambda: False,
    on_line: Callable[[str], None] = lambda _line: None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wait_budget_s: float = WAIT_FOR_FREE_S,
) -> Report:
    report = Report(verify_only=scope.verify_only)
    cache = VerifiedCache(config.home / VERIFIED_CACHE)
    hasher = Hasher(report, cache, emit, cancelled)
    envs = installed_envs(config, backend, scope)
    env_rows = {spec.key: record_rows(spec, jobenv.env_dir(config.home, spec)) for spec in envs}
    subjects: list[tuple[catalog.Subject, weights.InstalledWeights, bool]] = []
    if scope.weights:
        for subject in catalog.subjects(config, backend):
            if subject.shares_weights_of is not None:
                continue
            found = subject.installed()
            lost = False
            if found is None:
                was_pulled = getattr(subject, "was_pulled", None)
                found = None if was_pulled is None else was_pulled()
                lost = found is not None
            if found is not None:
                subjects.append((subject, found, lost))
    hasher.total = sum(_env_bytes(env_rows[spec.key][0]) for spec in envs) + sum(
        found.bytes for _subject_row, found, _lost in subjects
    )
    emit(
        "repair",
        {
            "step": "plan",
            "state": "running",
            "envs": [spec.key for spec in envs],
            "subjects": [_subject(s.kind, s.id) for s, _found, _lost in subjects],
            "bytes_to_check": hasher.total,
        },
    )
    try:
        for spec in envs:
            _repair_env(config, backend, spec, env_rows[spec.key], scope, doors, report, hasher,
                        emit, cancelled, on_line, sleep, clock, wait_budget_s)
        for subject, found, lost in subjects:
            _repair_subject(config, subject, found, lost, scope, doors, report, hasher,
                            emit, cancelled, on_line, sleep, clock, wait_budget_s)
    finally:
        cache.save()
    return report


def _repair_env(
    config: Config,
    backend: Backend,
    spec: jobenv.EnvSpec,
    found_rows: tuple[list[RecordRow], list[str]],
    scope: Scope,
    doors: Doors,
    report: Report,
    hasher: Hasher,
    emit: Emit,
    cancelled: Callable[[], bool],
    on_line: Callable[[str], None],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    wait_budget_s: float,
) -> None:
    if cancelled():
        raise RepairCancelled()
    subject = _subject(ENV, spec.key)
    report.checked.append(subject)
    emit("repair", {"step": "verify", "subject": subject, "state": "running"})
    rows, unreadable = found_rows
    wrong = record_wrong_here(jobenv.env_dir(config.home, spec))
    if wrong:
        report.not_hash_checked.append({
            "subject": subject,
            "files": wrong,
            "why": "the published wheel's RECORD is wrong about these files",
        })
    was = env_problem(config, backend, spec, rows, hasher)
    if was is None and unreadable:
        was = f"{len(unreadable)} RECORD file(s) could not be read: {unreadable[0]}"
    if was is None:
        emit("repair", {"step": "verify", "subject": subject, "state": "ok"})
        return
    emit("repair", {"step": "verify", "subject": subject, "state": "broken", "detail": was})
    report.broken.append({"subject": subject, "was": was})
    if scope.verify_only:
        return
    held = _wait_until_free(
        subject, lambda: doors.env_in_use(spec), emit, cancelled, sleep, clock, wait_budget_s
    )
    if held is not None:
        report.failed.append({
            "subject": subject,
            "code": "subject_in_use",
            "message": f"{was}. It was not rebuilt: {held}. Run repair again once it is free",
        })
        return
    emit("repair", {"step": "rebuild", "subject": subject, "state": "running"})
    try:
        jobenv.install_env(config.home, spec, backend.kind, force=True, on_line=on_line)
    except Exception as exc:  # noqa: BLE001 (any way a rebuild fails is one failed row, by name)
        report.failed.append({"subject": subject, "code": "env_build_failed", "message": str(exc)})
        emit("repair", {"step": "rebuild", "subject": subject, "state": "failed", "detail": str(exc)})
        return
    after = env_problem(
        config, backend, spec, record_rows(spec, jobenv.env_dir(config.home, spec))[0], hasher
    )
    if after is not None:
        report.failed.append({"subject": subject, "code": "env_build_failed", "message": after})
        emit("repair", {"step": "rebuild", "subject": subject, "state": "failed", "detail": after})
        return
    report.repaired.append({"subject": subject, "was": was})
    emit("repair", {"step": "rebuild", "subject": subject, "state": "repaired", "detail": was})


def _repair_subject(
    config: Config,
    subject: catalog.Subject,
    found: weights.InstalledWeights,
    lost: bool,
    scope: Scope,
    doors: Doors,
    report: Report,
    hasher: Hasher,
    emit: Emit,
    cancelled: Callable[[], bool],
    on_line: Callable[[str], None],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    wait_budget_s: float,
) -> None:
    if cancelled():
        raise RepairCancelled()
    row = _subject(subject.kind, subject.id)
    report.checked.append(row)
    if found.hf_repo is None or found.revision is None:
        report.unverified.append({"subject": row, "why": f"its source is {found.source}, not a hub pin"})
        emit("repair", {"step": "verify", "subject": row, "state": "unverified", "detail": found.source})
        return
    emit("repair", {"step": "verify", "subject": row, "state": "running"})
    try:
        hub = hub_files(config, found.hf_repo, found.revision)
    except HubUnreachable as exc:
        report.unverified.append({"subject": row, "why": str(exc)})
        emit("repair", {"step": "verify", "subject": row, "state": "unverified", "detail": str(exc)})
        return
    different, missing, unjudged = weights_problem(subject, found, hub, hasher)
    if lost and not missing:
        missing = ["a file its pull placed"]
    judged = len(_on_disk(found.path)) - len(unjudged)
    if not judged and not different and not missing:
        why = f"none of its files are named in {found.hf_repo}@{found.revision[:12]}"
        report.unverified.append({"subject": row, "why": why})
        emit("repair", {"step": "verify", "subject": row, "state": "unverified", "detail": why})
        return
    if unjudged:
        report.not_hash_checked.append({
            "subject": row,
            "files": unjudged,
            "why": f"not named in {found.hf_repo}@{found.revision[:12]} or pinned by the manifest",
        })
    if not different and not missing:
        emit("repair", {"step": "verify", "subject": row, "state": "ok"})
        return
    was = f"{len(different)} file(s) changed and {len(missing)} missing, among them " + ", ".join(
        (different + missing)[:5]
    )
    emit("repair", {"step": "verify", "subject": row, "state": "broken", "detail": was})
    report.broken.append({"subject": row, "was": was})
    if scope.verify_only:
        return
    held = _wait_until_free(
        row, lambda: doors.subject_in_use(subject), emit, cancelled, sleep, clock, wait_budget_s
    )
    if held is not None:
        report.failed.append({
            "subject": row,
            "code": "subject_in_use",
            "message": f"{was}. It was not fetched again: {held}. Run repair again once it is free",
        })
        return
    emit("repair", {"step": "repull", "subject": row, "state": "running"})

    def on_progress(done: int, total: int | None, name: str) -> None:
        if cancelled():
            raise weights.PullCancelled("the repair was cancelled")
        emit("progress", {"bytes_done": done, "bytes_total": total, "file": name})

    def cured() -> tuple[bool, list[str]]:
        now = subject.installed()
        if now is None:
            return False, ["the subject still does not read as installed"]
        still, gone, _unjudged = weights_problem(subject, now, hub, hasher)
        # What this repair deleted has to be back: a file that is absent is not one that differs.
        gone += [name for name in different if not (now.path / name).is_file()]
        return not still and not gone, still + gone

    try:
        _put_back(subject, found, different, on_line, on_progress)
        fixed, left = cured()
        if not fixed:
            on_line(f"{subject.id}: {', '.join(left[:3])} after fetching the bad files; fetching all of it")
            subject.pull(force=True, on_line=on_line, on_progress=on_progress)
            fixed, left = cured()
    except weights.PullCancelled:
        raise RepairCancelled() from None
    except Exception as exc:  # noqa: BLE001 (any way a fetch fails is one failed row, by name)
        report.failed.append({"subject": row, "code": "pull_failed", "message": str(exc)})
        emit("repair", {"step": "repull", "subject": row, "state": "failed", "detail": str(exc)})
        return
    if not fixed:
        still = left
        message = f"after fetching all of it again, {', '.join(still[:5])} still do(es) not match"
        report.failed.append({"subject": row, "code": "weights_sha_mismatch", "message": message})
        emit("repair", {"step": "repull", "subject": row, "state": "failed", "detail": message})
        return
    report.repaired.append({"subject": row, "was": was})
    emit("repair", {"step": "repull", "subject": row, "state": "repaired", "detail": was})


def in_use_by_nothing(_thing: Any) -> str | None:
    return None


NOBODY = Doors(env_in_use=in_use_by_nothing, subject_in_use=in_use_by_nothing)


__all__ = [
    "Doors",
    "HubUnreachable",
    "NOBODY",
    "RepairCancelled",
    "Report",
    "Scope",
    "run",
    "scope_problems",
]

