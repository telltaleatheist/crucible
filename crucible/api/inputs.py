from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ApiError
from ..jobs.base import Job, validate_member_name
from ..jobs.queue import JobStore
from ..journal import InputDigest, sha256_file
from .schemas import ArtifactRef, JobInput


def _refuse_resume_without_a_journal(
    plugin: Any, job_type: str, params: dict[str, Any]
) -> None:
    """`resume_unsupported`, by name, for a type that keeps no journal.

    Accepted by every type's params IN PRINCIPLE (2026-09-27): the key means
    the same thing everywhere, so it is refused here, once, for the types that
    have not been brought onto the journal yet, rather than as an unknown
    field by each type's own model, which would read as a typo. A type that
    journals only some of its models (`asr`: Qwen3-ASR yes, whisper not yet)
    refuses the rest in its own preflight, with the same code.
    """
    if "resume" not in params:
        return
    if getattr(plugin, "journal_identity", None) is None:
        raise ApiError(
            400,
            "resume_unsupported",
            f"{job_type} jobs keep no resume journal yet, so there is nothing to "
            "resume; send the job without resume to run it from the start",
            {"type": job_type, "resume": params.get("resume")},
        )


def _journal_identity(plugin: Any, model: str | None, params: dict[str, Any]) -> Any:
    """What this job's journal is the work OF, or None when it keeps none."""
    identify = getattr(plugin, "journal_identity", None)
    if identify is None:
        return None
    return identify(model, params)


def _referenced_artifact(store: JobStore, name: str, ref: ArtifactRef) -> Path:
    """The file an artifact input names, or `409 artifact_expired` by name.

    EXPIRED IS THE ONE ANSWER for every way the bytes are not here: the job was
    reaped (released, fetched, or past the seven-day collector), this server
    never ran it (a different server, or ids from before a wipe), or the job
    exists and published no such file. In every case the client's correct next
    move is the same: upload the bytes. It is refused before the job exists.
    """
    try:
        validate_member_name(ref.name)
    except ValueError as exc:
        raise ApiError(400, "invalid_artifact_name", str(exc)) from None
    try:
        source_job = store.get(ref.job_id)
    except ApiError as exc:
        raise ApiError(
            409,
            "artifact_expired",
            f"input {name!r} names artifact {ref.name!r} of job {ref.job_id!r}, and "
            f"this server no longer holds that job ({exc.code}: {exc.message}). "
            "Upload the bytes instead",
            {"input": name, "job_id": ref.job_id, "artifact": ref.name, "why": exc.code},
        ) from None
    source = source_job.artifacts_dir / ref.name
    if ref.name not in source_job.artifacts or not source.is_file():
        raise ApiError(
            409,
            "artifact_expired",
            f"input {name!r} names artifact {ref.name!r} of job {ref.job_id!r}, "
            f"which that job does not hold (it holds {len(source_job.artifacts)} "
            "artifact(s)). Upload the bytes instead",
            {"input": name, "job_id": ref.job_id, "artifact": ref.name, "why": "no_such_artifact"},
        )
    return source


def _input_digests(
    config: Config, store: JobStore, inputs: dict[str, JobInput]
) -> list[InputDigest]:
    """Each input's name, sha256 and size, for a journaled job (2026-09-27).

    ASKED BEFORE ANYTHING MOVES. A `resume` is checked against these, and a
    refused resume must not have consumed the client's upload: an upload is
    moved into the job that names it, and a discarded job takes it with it.
    An upload's sha256 is the one `POST /v1/uploads` computed and wrote beside
    it, so a four-hour book is not read twice; inline bytes are hashed in
    memory; an artifact input is hashed from disk, the one case that reads the
    file (seconds a GB). Anything unresolvable is left to `_materialise_inputs`
    to refuse by its own name.
    """
    digests: list[InputDigest] = []
    for name, declared in inputs.items():
        if declared.blob_id is not None:
            try:
                validate_member_name(declared.blob_id)
            except ValueError:
                continue
            source = Path(config.uploads_dir) / declared.blob_id
            if not source.is_file():
                continue
            meta_path = Path(config.uploads_dir) / f"{declared.blob_id}.json"
            sha, size = None, source.stat().st_size
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta.get("bytes", -1)) == size:
                    sha = str(meta["sha256"])
            except (OSError, ValueError, KeyError, TypeError):
                sha = None
            digests.append(
                InputDigest(name, sha if sha is not None else sha256_file(source), size)
            )
        elif declared.inline_base64 is not None:
            try:
                payload = base64.b64decode(declared.inline_base64, validate=True)
            except (binascii.Error, ValueError):
                continue
            digests.append(
                InputDigest(name, hashlib.sha256(payload).hexdigest(), len(payload))
            )
        elif declared.artifact is not None:
            source = _referenced_artifact(store, name, declared.artifact)
            digests.append(InputDigest(name, sha256_file(source), source.stat().st_size))
    return digests


def _materialise_inputs(
    config: Config, store: JobStore, job: Job, inputs: dict[str, JobInput]
) -> None:
    """Write every declared input into the job's scratch dir before it is queued.

    **AN UPLOAD IS MOVED, NOT COPIED** (Owen's ruling, 2026-09-18). It used to
    be `copyfile`d with the original left in `uploads/`, so every byte a client
    sent was on this disk twice for ever: nothing deleted an upload, and the PC
    was measured at 2.4 GB in 2,935 of them. A blob exists to become a job's
    input, and `os.replace` is the same operation with one copy — and, on one
    filesystem, without reading the bytes at all.

    **EVERYTHING IS CHECKED BEFORE ANYTHING MOVES**, which the copy did not
    have to care about. A refusal on the third input used to leave the first
    two copied and the uploads untouched; now it would leave the first two
    MOVED, and a job that is then discarded takes them with it. So the two
    passes below: nothing is written until every name, every blob and every
    inline payload has been read and accepted.
    """
    planned: list[tuple[str, Path, str | None, bytes | None]] = []
    linked: list[tuple[Path, Path]] = []
    for name, declared in inputs.items():
        try:
            validate_member_name(name)
        except ValueError as exc:
            raise ApiError(400, "invalid_input_name", str(exc)) from None
        target = job.inputs_dir / name
        if declared.blob_id is not None:
            try:
                validate_member_name(declared.blob_id)
            except ValueError as exc:
                raise ApiError(400, "invalid_blob_id", str(exc)) from None
            # Asked BEFORE the file check, because the two are different
            # answers about the same missing file: a blob some job already
            # took is not one this server never had, and telling a client the
            # second when the first is true sends it looking for a bug in its
            # own upload.
            store.refuse_if_blob_consumed(declared.blob_id)
            source = Path(config.uploads_dir) / declared.blob_id
            if not source.is_file():
                raise ApiError(
                    400,
                    "unknown_blob",
                    f"input {name!r} names blob {declared.blob_id!r}, which this "
                    "server does not hold",
                )
            planned.append((name, target, declared.blob_id, None))
        elif declared.inline_base64 is not None:
            try:
                payload = base64.b64decode(declared.inline_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(
                    400,
                    "invalid_inline_input",
                    f"input {name!r} is not valid base64: {exc}",
                ) from None
            planned.append((name, target, None, payload))
        elif declared.artifact is not None:
            linked.append((target, _referenced_artifact(store, name, declared.artifact)))
        else:  # unreachable: JobInput's validator requires exactly one source
            raise ApiError(
                400,
                "invalid_input",
                f"input {name!r} names neither a blob nor inline bytes",
            )

    # A REFERENCE IS A HARD LINK, not a copy: the same bytes under the new
    # job's name, on the one filesystem both directories live in. Reaping the
    # referenced job afterwards unlinks its name and leaves this one intact.
    for target, source in linked:
        try:
            os.link(source, target)
        except OSError as exc:
            raise ApiError(
                500,
                "artifact_link_failed",
                f"could not link {source} as input {target.name}: "
                f"{type(exc).__name__}: {exc}",
            ) from None
    for name, target, blob_id, payload in planned:
        if blob_id is None:
            assert payload is not None  # one of the two, by the loop above
            target.write_bytes(payload)
            continue
        # The record first, so that the one line that makes the bytes
        # unreachable from `uploads/` is the one line that says who has them.
        # Two inputs of one request naming the same blob land here, and are
        # refused naming this job — which is what happened: the first of them
        # took it.
        store.consume_blob(blob_id, job)
        source = Path(config.uploads_dir) / blob_id
        os.replace(source, target)
        # The upload's metadata sidecar describes a blob that is no longer in
        # `uploads/`. It is written by `POST /v1/uploads` and read by nothing,
        # so it goes with the blob rather than being moved beside it. A
        # CLEANUP FAILURE IS NOT AN OPERATION FAILURE: the input is in place
        # and the job is going to run, so a sidecar that will not unlink is
        # said out loud and left.
        meta = Path(config.uploads_dir) / f"{blob_id}.json"
        try:
            meta.unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"crucible: blob {blob_id} moved into job {job.id} but its "
                f"metadata {meta} would not delete: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
