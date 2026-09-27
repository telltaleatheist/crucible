from __future__ import annotations

import argparse
import json

from .. import catalog, denoisemodels, rvcbase, weights
from ..alignmodels import AlignManifest, AlignManifestError, load_all_align_manifests
from ..asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from ..errors import ConfigError, CrucibleError, NoViableBackend
from ..manifests import ManifestError, ModelManifest, load_all_manifests
from ..rvcmodels import RvcManifestError, load_all_rvc_manifests, load_rvc_manifest
from . import common
from .common import EXIT_OK, _backend_mismatch, _fail, _models_config


def _all_manifests() -> dict[str, "ModelManifest | AsrManifest | AlignManifest"]:
    """Every model this build ships, from all three manifest directories, by id.

    `models/`, `asr/` and `align/` are three directories with three loaders
    (crucible/asrmodels.py explains why they are not one yet), but from the
    command line there is a single namespace of model ids, because `crucible
    models pull <id>` is a single question. A collision between any two would
    make that question ambiguous, so it is refused rather than settled by which
    directory was read first.

    `rvc/` is deliberately NOT in here. Its weights are a single archive fetched
    by name and unpacked rather than a repo snapshot, so `crucible models pull`
    could not serve one; they live under their own `crucible rvc` command and
    their own subtree of `~/.crucible`, which is also what keeps an RVC model
    named `sigma` from colliding with a narrator voice of the same name.
    """
    merged: dict[str, "ModelManifest | AsrManifest | AlignManifest"] = dict(
        load_all_manifests()
    )
    for extra in (load_all_asr_manifests(), load_all_align_manifests()):
        for model_id, manifest in extra.items():
            if model_id in merged:
                raise ManifestError(
                    f"{model_id!r} is declared by both {merged[model_id].path} and "
                    f"{manifest.path}; a model id names one model"
                )
            merged[model_id] = manifest
    return merged


def cmd_remove(args: argparse.Namespace) -> int:
    """`crucible remove <kind> <id>` — PHASE15-HOST.md 3.5a, from a terminal.

    REFUSES IDENTICALLY TO THE DOOR, and it does so by asking the same
    questions in the same order: an unknown kind or id first (true whatever
    this server is doing), then not-installed, then in-use. What it CANNOT
    ask is whether a running server holds the subject — that is a fact about
    a process this command is not inside, and the names it would need
    (`Residency`, `Leases`, the task store) live in one. So it asks the two
    it can and says so: on a machine with a server running, the door is the
    one to use, and `DELETE /v1/catalog/{kind}/{id}` is what the host calls.
    """
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force`"
        )
    subject = catalog.find(config, backend, args.kind, args.id)
    if subject is None:
        return _fail(
            f"subject_unknown: this server has no {args.kind} called "
            f"{args.id!r} for {backend.kind}. `crucible catalog` lists every "
            "subject it can hold"
        )
    found = subject.installed()
    if found is None:
        return _fail(
            f"subject_not_installed: {args.kind} {args.id!r} is not installed "
            "on this server, so there is nothing to remove"
        )
    try:
        gone = subject.remove()
    except weights.WeightsShared as exc:
        # The door's code verbatim (PHASE22 section 2.9); the message already
        # begins with it and names every alias holding the folder.
        return _fail(str(exc))
    except weights.RemoveFailed as exc:
        return _fail(f"subject_remove_failed: {exc}")
    except CrucibleError as exc:
        return _fail(f"subject_remove_failed: {type(exc).__name__}: {exc}")
    if args.json:
        print(json.dumps(
            {
                "kind": args.kind,
                "id": args.id,
                "path": str(gone),
                "bytes_freed": found.bytes,
            },
            indent=2,
        ))
    else:
        print(f"removed:  {args.kind} {args.id}")
        print(f"path:     {gone}")
        print(f"freed:    {found.bytes / 1e9:.2f} GB")
    return EXIT_OK


def cmd_models_list(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = _all_manifests()
    except (ManifestError, AsrManifestError, AlignManifestError) as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                # An ASR manifest carries no context. Whisper's window is 30
                # seconds of audio and is not a number anybody sets, so null
                # here means "this model has no such knob", not "unknown".
                "context_default": (
                    manifest.context_for(backend.kind)
                    if isinstance(manifest, ModelManifest)
                    else None
                ),
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"not pulled — `crucible models pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<16} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_models_pull(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = _all_manifests()
    except (ManifestError, AsrManifestError, AlignManifestError) as exc:
        return _fail(str(exc))
    manifest = manifests.get(args.model)
    if manifest is None:
        return _fail(
            f"no manifest for model {args.model!r}; this build ships "
            f"{sorted(manifests)}"
        )
    if not manifest.supports(backend.kind):
        return _fail(
            f"model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    try:
        result = weights.pull(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def cmd_rvc_list(args: argparse.Namespace) -> int:
    """Every RVC manifest this build ships and where it stands on this host.

    Its own command rather than a row in `crucible models list`, for the reason
    `_all_manifests` gives: an RVC model's weights are one archive fetched by
    name, not a repo snapshot, so `models pull` could not fetch one — and the ids
    are a separate namespace, which is what stops an RVC model called `sigma`
    from colliding with the narrator voice of the same name.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = load_all_rvc_manifests()
    except RvcManifestError as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "model_name": manifest.model_name,
                "has_index": manifest.has_index,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "archive": spec.archive,
                "revision": spec.revision,
                "archive_bytes": spec.archive_bytes,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"not pulled — `crucible rvc pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_rvc_pull(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = load_rvc_manifest(args.model)
    except RvcManifestError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"RVC model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(
        f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]}:{spec.archive} "
        f"for {backend.kind}"
    )
    try:
        result = weights.pull_archive(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def cmd_rvc_pull_base(args: argparse.Namespace) -> int:
    """`crucible rvc pull-base` — the engine's shared assets, at a pinned sha.

    Its own verb rather than a step inside `crucible install rvc`, for the
    reason every other weights pull is its own verb: installing an env and
    fetching 600 MB of weights are different acts with different failure modes,
    and `crucible install llm` does not pull a 19 GB model either. One set, one
    command, one owner (PHASE4-AUDIO.md section 4.1).
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, _backend = resolved
    try:
        assets = rvcbase.load_rvc_base()
    except rvcbase.RvcBaseError as exc:
        return _fail(str(exc))
    print(
        f"{assets.id}: {assets.hf_repo}@{assets.revision[:12]}, "
        f"{len(assets.files)} file(s), {assets.total_bytes / 1e9:.2f} GB"
    )
    for entry in assets.files:
        print(f"  {entry.target} — {entry.why}")
    try:
        result = rvcbase.pull(
            config, assets, force=args.force, on_line=lambda line: print(f"  {line}")
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    absent = rvcbase.missing(config, assets)
    if absent:
        # Unreachable unless something removed a file between the place and
        # this read; said out loud rather than reported as success, because the
        # next thing to look at this tree is a job that will fail inside urvc.
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{rvcbase.base_root(config)}"
        )
    print(f"{assets.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


# ------------------------------------------------------------------ denoise


def cmd_denoise_list(args: argparse.Namespace) -> int:
    """Every denoise manifest this build ships and where it stands here.

    Its own command rather than a row in `crucible models list`, for `crucible
    rvc list`'s reason one job type along: a separator's weights are two named
    files placed under names an engine resolves by, not a repo snapshot, so
    `models pull` could not fetch one — and the ids are their own namespace.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = denoisemodels.load_all_denoise_manifests()
    except denoisemodels.DenoiseManifestError as exc:
        return _fail(str(exc))
    root = denoisemodels.denoise_models_root(config.home)
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = denoisemodels.installed(config.home, manifest, spec)
        absent = denoisemodels.missing(config.home, manifest)
        # Stamped and present are different facts and this prints both. A
        # stamp with a file missing beside it is not installed; two files
        # somebody placed by hand are usable and unstamped, which is the state
        # every host was in before this command existed.
        if found is not None:
            detail = f"{found.bytes / 1e9:.2f} GB at {found.path}"
        elif not absent:
            detail = (
                f"both files are in {root} but Crucible did not place them — "
                f"`{denoisemodels.PULL_COMMAND} {manifest.id} --force` to pin them"
            )
        else:
            detail = f"not pulled — `{denoisemodels.PULL_COMMAND} {manifest.id}`"
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "model_filename": manifest.model_filename,
                "config_filename": manifest.config_filename,
                "primary_stem": manifest.primary_stem,
                "sample_rate": manifest.sample_rate,
                "backend_supported": True,
                "installed": found is not None,
                "present": not absent,
                "missing": absent,
                "root": str(root),
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "model_path": spec.model_path,
                "config_path": spec.config_path,
                "total_bytes": spec.total_bytes,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "detail": detail,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else (
                "unstamped" if row["present"] else "not pulled"
            )
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_denoise_pull(args: argparse.Namespace) -> int:
    """`crucible denoise pull <id>` — the checkpoint and its config, at the pin.

    Both files, both digests, one revision, into the flat directory
    audio-separator reads by name. The layout is `crucible/denoisemodels.py`'s
    and the job reads the same function, so what this places is what a job
    looks for (ARCHITECTURE.md R1).
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = denoisemodels.load_denoise_manifest(args.model)
    except denoisemodels.DenoiseManifestError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"denoise model {args.model!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    print(
        f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]}, 2 file(s), "
        f"{spec.total_bytes / 1e9:.2f} GB for {backend.kind}"
    )
    for entry in denoisemodels.model_files(manifest, spec):
        print(f"  {entry.target} — {entry.why}")
    try:
        result = denoisemodels.pull(
            config,
            manifest,
            spec,
            force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    absent = denoisemodels.missing(config.home, manifest)
    if absent:
        # Unreachable unless something removed a file between the place and
        # this read; said out loud rather than reported as success, because the
        # next thing to look at this tree is a job that will fail inside
        # audio-separator. `rvc pull-base`'s rule, and its reason.
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{denoisemodels.denoise_models_root(config.home)}"
        )
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def add_model_parsers(subparsers: argparse._SubParsersAction) -> None:
    remove = subparsers.add_parser(
        "remove",
        help="delete an installed subject's files (PHASE15-HOST.md 3.5a)",
    )
    remove.add_argument(
        "kind",
        choices=list(catalog.KINDS),
        help="the subject kind, as `crucible catalog` and GET /v1/catalog spell it",
    )
    remove.add_argument("id", help="the subject id, e.g. qwen3.5-9b")
    remove.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    remove.set_defaults(func=cmd_remove)

    models = subparsers.add_parser("models", help="list and pull model weights")
    model_commands = models.add_subparsers(dest="models_command", required=True)

    models_list = model_commands.add_parser(
        "list", help="every manifest this build ships and where it stands here"
    )
    models_list.add_argument("--json", action="store_true", help="machine-readable")
    models_list.set_defaults(func=cmd_models_list)

    models_pull = model_commands.add_parser(
        "pull", help="fetch a model's weights at the manifest's pinned revision"
    )
    models_pull.add_argument("model", help="the Crucible model id, e.g. qwen3.5-9b")
    models_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    models_pull.set_defaults(func=cmd_models_pull)


def add_rvc_denoise_parsers(subparsers: argparse._SubParsersAction) -> None:
    rvc = subparsers.add_parser("rvc", help="list and pull RVC voice-conversion models")
    rvc_commands = rvc.add_subparsers(dest="rvc_command", required=True)

    rvc_list = rvc_commands.add_parser(
        "list", help="every RVC manifest this build ships and where it stands here"
    )
    rvc_list.add_argument("--json", action="store_true", help="machine-readable")
    rvc_list.set_defaults(func=cmd_rvc_list)

    rvc_pull = rvc_commands.add_parser(
        "pull", help="fetch and unpack an RVC model at the manifest's pinned revision"
    )
    rvc_pull.add_argument("model", help="the Crucible RVC id, e.g. deathstalker-rvc-v1")
    rvc_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    rvc_pull.set_defaults(func=cmd_rvc_pull)

    rvc_pull_base = rvc_commands.add_parser(
        "pull-base",
        help="fetch ultimate-rvc's shared base assets (the embedder and the "
        "pitch predictors) — the engine's, not any model's",
    )
    rvc_pull_base.add_argument(
        "--force", action="store_true", help="re-pull even if they are already there"
    )
    rvc_pull_base.set_defaults(func=cmd_rvc_pull_base)

    denoise = subparsers.add_parser(
        "denoise", help="list and pull separator checkpoints for the denoise job"
    )
    denoise_commands = denoise.add_subparsers(dest="denoise_command", required=True)

    denoise_list = denoise_commands.add_parser(
        "list", help="every denoise manifest this build ships and where it stands here"
    )
    denoise_list.add_argument("--json", action="store_true", help="machine-readable")
    denoise_list.set_defaults(func=cmd_denoise_list)

    denoise_pull = denoise_commands.add_parser(
        "pull",
        help="fetch a separator's checkpoint and its config at the manifest's "
        "pinned revision, into the directory audio-separator reads by name",
    )
    denoise_pull.add_argument(
        "model", help="the Crucible denoise id, e.g. denoise-roformer"
    )
    denoise_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    denoise_pull.set_defaults(func=cmd_denoise_pull)
