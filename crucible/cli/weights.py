from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path
from typing import Any

from .. import audioweights, catalog, denoisemodels, rvcbase, videoweights, weights
from ..alignmodels import AlignManifest, AlignManifestError, load_all_align_manifests
from ..asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from ..audiomodels import AudioManifest, AudioManifestError, load_all_audio_manifests
from ..client import transport
from ..client.connection import Connection
from ..errors import CrucibleError
from ..imagemodels import ImageManifest, ImageManifestError, load_all_image_manifests
from ..manifests import (
    ManifestError,
    ModelManifest,
    UnknownForm,
    host_fit_of,
    load_all_manifests,
)
from ..rvcmodels import RvcManifestError, load_all_rvc_manifests, load_rvc_manifest
from ..segmentmodels import SegmentManifest, SegmentManifestError, load_all_segment_manifests
from ..videomodels import VideoManifest, VideoManifestError, load_all_video_manifests
from . import common
from .api_cmd import report_http_error
from .common import EXIT_OK, _fail

AnyManifest = (
    ModelManifest
    | AsrManifest
    | AlignManifest
    | ImageManifest
    | AudioManifest
    | SegmentManifest
    | VideoManifest
)

MANIFEST_ERRORS = (
    ManifestError,
    AsrManifestError,
    AlignManifestError,
    ImageManifestError,
    AudioManifestError,
    SegmentManifestError,
    VideoManifestError,
)


def _all_manifests() -> dict[str, AnyManifest]:
    merged: dict[str, AnyManifest] = dict(load_all_manifests())
    for extra in (
        load_all_asr_manifests(),
        load_all_align_manifests(),
        load_all_image_manifests(),
        load_all_audio_manifests(),
        load_all_segment_manifests(),
        load_all_video_manifests(),
    ):
        for model_id, manifest in extra.items():
            if model_id in merged:
                raise ManifestError(
                    f"{model_id!r} is declared by both {merged[model_id].path} and "
                    f"{manifest.path}; a model id names one model"
                )
            merged[model_id] = manifest
    return merged


def _installed(config: Any, manifest: AnyManifest, spec: Any) -> weights.InstalledWeights | None:
    if isinstance(manifest, AudioManifest):
        return audioweights.installed(config, manifest, spec)
    if isinstance(manifest, VideoManifest):
        return videoweights.installed(config, manifest, spec)
    return weights.installed(config, manifest, spec)


def _nobody_holds(_subject: catalog.Subject) -> None:
    return None


def _remove_through_the_server(
    server: Connection,
    found: weights.InstalledWeights,
    args: argparse.Namespace,
) -> tuple[Path, int] | int:
    try:
        transport.call(server, "DELETE", f"/v1/catalog/{args.kind}/{args.id}")
    except urllib.error.HTTPError as exc:
        return report_http_error(exc, server)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(
            f"server_unreachable: the server at {server.url} answered a moment "
            f"ago and not now ({exc}); nothing was removed. Run `crucible remove "
            f"{args.kind} {args.id}` again"
        )
    return found.path, found.bytes


def cmd_remove(args: argparse.Namespace) -> int:
    config, backend = common.here()
    server = common.server_here(config, backend)
    try:
        if server is not None:
            _subject, found = catalog.locate_installed(
                config, backend, args.kind, args.id
            )
            outcome = _remove_through_the_server(server, found, args)
            if isinstance(outcome, int):
                return outcome
            gone, bytes_freed = outcome
        else:
            removed = catalog.remove_subject(
                config, backend, args.kind, args.id, holder=_nobody_holds
            )
            gone, bytes_freed = removed.path, removed.found.bytes
    except catalog.RemoveRefused as exc:
        return _fail(f"{exc.code}: {exc}")
    except weights.WeightsShared as exc:
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
                "bytes_freed": bytes_freed,
                "through": "server" if server is not None else "files",
            },
            indent=2,
        ))
    else:
        print(f"removed:  {args.kind} {args.id}")
        print(f"path:     {gone}")
        print(f"freed:    {bytes_freed / 1e9:.2f} GB")
        if server is not None:
            print(f"through:  the server at {server.url}, which checked nothing holds it")
    return EXIT_OK


def cmd_models_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
    try:
        manifests = _all_manifests()
    except MANIFEST_ERRORS as exc:
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
        forms = (
            _form_facts(config, backend, manifest)
            if isinstance(manifest, ModelManifest)
            else None
        )
        spec = (
            manifest.spec(backend.kind)
            if forms is None
            else manifest.spec(backend.kind, forms["form"])
        )
        found = _installed(config, manifest, spec)
        detail = (
            f"{found.bytes / 1e9:.2f} GB at {found.path}"
            if found is not None
            else f"not pulled — `crucible models pull {manifest.id}`"
        )
        if forms is not None:
            detail = _form_detail(manifest, forms, detail, found is not None)
        rows.append(
            {
                "id": manifest.id,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "context_default": (
                    manifest.context_for(backend.kind)
                    if isinstance(manifest, ModelManifest)
                    else None
                ),
                **({} if forms is None else forms),
                "detail": detail,
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


def _form_facts(config: Any, backend: Any, manifest: ModelManifest) -> dict[str, Any] | None:
    """A model's forms here: the one this card takes, why, and which are installed."""
    pick = manifest.form_pick(backend.kind, host_fit_of(config, backend))
    if pick is None:
        return None
    block = manifest.block(backend.kind)
    installed = [
        name
        for name in block.form_names
        if weights.installed(config, manifest, block.with_form(name, manifest.id)) is not None
    ]
    return {
        "form": pick.form.name,
        "form_reason": pick.reason,
        "forms": list(block.form_names),
        "installed_forms": installed,
    }


def _form_detail(
    manifest: ModelManifest, forms: dict[str, Any], detail: str, picked_installed: bool
) -> str:
    """The listing's line for a model with forms: which form, and, when the form this card
    takes is not the one installed, that it needs a pull."""
    form = forms["form"]
    if picked_installed:
        return f"{form} form; {detail}"
    others = forms["installed_forms"]
    if not others:
        return f"{form} form (the one this card takes) {detail}"
    verb = "is" if len(others) == 1 else "are"
    return (
        f"{form} form needed — {forms['form_reason']}; {' and '.join(others)} {verb} "
        f"installed instead — run `crucible models pull {manifest.id}`"
    )


def cmd_models_pull(args: argparse.Namespace) -> int:
    config, backend = common.here()
    try:
        manifests = _all_manifests()
    except MANIFEST_ERRORS as exc:
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
    if args.form is not None and not isinstance(manifest, ModelManifest):
        return _fail(f"{manifest.id!r} is not a chat model; only a chat model has forms")
    host = host_fit_of(config, backend)
    try:
        spec = (
            manifest.spec(backend.kind, args.form, host=host)
            if isinstance(manifest, ModelManifest)
            else manifest.spec(backend.kind)
        )
    except UnknownForm as exc:
        return _fail(f"{exc.code}: {exc}")
    print(f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    if spec.form is not None:
        assert isinstance(manifest, ModelManifest)
        pick = manifest.form_pick(backend.kind, host)
        assert pick is not None
        if args.form is None:
            print(f"  the {spec.form} form ({spec.file}): {pick.reason}")
        else:
            taken = "" if args.form == pick.form.name else f"; this card takes {pick.form.name}"
            print(f"  the {spec.form} form ({spec.file}), as named{taken}")
    try:
        pull = (
            audioweights.pull if isinstance(manifest, AudioManifest)
            else videoweights.pull if isinstance(manifest, VideoManifest)
            else weights.pull
        )
        result = pull(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def cmd_rvc_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
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
    config, backend = common.here()
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
    config, _backend = common.here()
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
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{rvcbase.base_root(config)}"
        )
    print(f"{assets.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def cmd_denoise_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
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
    config, backend = common.here()
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
        return _fail(
            f"the pull finished but {sorted(absent)} are not under "
            f"{denoisemodels.denoise_models_root(config.home)}"
        )
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def add_model_parsers(subparsers: argparse._SubParsersAction) -> None:
    remove = subparsers.add_parser(
        "remove",
        help="delete an installed subject's files (docs/internals/cli.md, \"Weights verbs\")",
    )
    remove.add_argument(
        "kind",
        choices=list(catalog.KINDS),
        help="the subject kind, as `crucible api catalog` and GET /v1/catalog spell it",
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
    models_pull.add_argument(
        "--form",
        default=None,
        help=(
            "for a model that comes in more than one form (`crucible models list` names "
            "them), pull this one; omitted, the form this card takes"
        ),
    )
    models_pull.set_defaults(func=cmd_models_pull)

    models_concurrency = model_commands.add_parser(
        "concurrency",
        help=(
            "[llm.concurrency]: how many requests a chat model runs at once here, "
            "lower than its manifest's (fewer leaves the GPU room for the displays). "
            "No arguments lists them; `default` returns a model to its manifest's"
        ),
    )
    models_concurrency.add_argument("model", nargs="?", default=None)
    models_concurrency.add_argument("width", nargs="?", default=None)
    models_concurrency.set_defaults(func=cmd_models_concurrency)


def cmd_models_concurrency(args: argparse.Namespace) -> int:
    from .. import llmconcurrency
    from ..errors import ConfigError

    config, _backend = common.here()
    try:
        rows = llmconcurrency.rows(config)
    except ConfigError as exc:
        return _fail(str(exc))
    if args.model is None:
        for row in rows:
            now = row["set"] if row["set"] is not None else row["manifest"]
            whose = "set in config" if row["set"] is not None else "the manifest's"
            print(f"{row['model']}: {now} at once ({whose}; manifest {row['manifest']})")
        return EXIT_OK
    row = next((r for r in rows if r["model"] == args.model), None)
    if args.width is None:
        if row is None:
            return _fail(f"concurrency_not_settable: {args.model!r} has no concurrency here")
        now = row["set"] if row["set"] is not None else row["manifest"]
        print(f"{args.model}: {now} at once (manifest {row['manifest']})")
        return EXIT_OK
    if args.width == "default":
        width = None
    else:
        try:
            width = int(args.width)
        except ValueError:
            return _fail(f"{args.width!r} is not a whole number or `default`")
    try:
        path = llmconcurrency.set_concurrency(config, args.model, width)
    except ConfigError as exc:
        return _fail(str(exc))
    print(
        f"[llm.concurrency] {args.model} = "
        f"{'the manifest' + chr(39) + 's' if width is None else width} — {path}"
    )
    print(
        "read when the model loads: if it is on the card now, it keeps its old width "
        "until it is unloaded and loaded again"
    )
    return EXIT_OK


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
