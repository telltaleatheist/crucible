from __future__ import annotations

import base64
import binascii
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol

from .config import Config
from .errors import ApiError
from .jobs.base import Job, validate_member_name
from .jobs.queue import JobStore
from .journal import InputDigest, sha256_file
from .uploads import blob_path, recorded_sha256, sidecar_path


class ArtifactSource(Protocol):
    @property
    def job_id(self) -> str: ...

    @property
    def name(self) -> str: ...


class InputSource(Protocol):
    @property
    def blob_id(self) -> str | None: ...

    @property
    def inline_base64(self) -> str | None: ...

    @property
    def artifact(self) -> ArtifactSource | None: ...


def refuse_resume_without_a_journal(
    plugin: Any, job_type: str, params: dict[str, Any]
) -> None:
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


def journal_identity(plugin: Any, model: str | None, params: dict[str, Any]) -> Any:
    identify = getattr(plugin, "journal_identity", None)
    if identify is None:
        return None
    return identify(model, params)


def referenced_artifact(store: JobStore, name: str, ref: ArtifactSource) -> Path:
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


def input_digests(
    config: Config, store: JobStore, inputs: Mapping[str, InputSource]
) -> list[InputDigest]:
    digests: list[InputDigest] = []
    for name, declared in inputs.items():
        if declared.blob_id is not None:
            try:
                validate_member_name(declared.blob_id)
            except ValueError:
                continue
            source = blob_path(config.uploads_dir, declared.blob_id)
            if not source.is_file():
                continue
            size = source.stat().st_size
            sha = recorded_sha256(config.uploads_dir, declared.blob_id, size)
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
            source = referenced_artifact(store, name, declared.artifact)
            digests.append(InputDigest(name, sha256_file(source), source.stat().st_size))
    return digests


def materialise_inputs(
    config: Config, store: JobStore, job: Job, inputs: Mapping[str, InputSource]
) -> None:
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
            store.refuse_if_blob_consumed(declared.blob_id)
            if not blob_path(config.uploads_dir, declared.blob_id).is_file():
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
            linked.append((target, referenced_artifact(store, name, declared.artifact)))
        else:
            raise ApiError(
                400,
                "invalid_input",
                f"input {name!r} names neither a blob nor inline bytes",
            )

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
            assert payload is not None
            target.write_bytes(payload)
            continue
        store.consume_blob(blob_id, job)
        os.replace(blob_path(config.uploads_dir, blob_id), target)
        meta = sidecar_path(config.uploads_dir, blob_id)
        try:
            meta.unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"crucible: blob {blob_id} moved into job {job.id} but its "
                f"metadata {meta} would not delete: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
