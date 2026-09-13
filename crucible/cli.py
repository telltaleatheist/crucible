"""The `crucible` command line.

    crucible init      mint the token, write the config, record the backend
    crucible serve     run the API in the foreground
    crucible doctor    probe the host and every job type; exit 0 only when healthy
    crucible token     print the bearer token (needs --show)

Exit codes: 0 success, 1 refused (named reason on stderr), 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from . import API_VERSION, VERSION, llmenv, weights, workerenv
from .asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from .backend import WINDOWS_REFUSAL, Backend, detect_backend
from .config import (
    CRUCIBLE_HOME_ENV,
    DEFAULT_DESKTOP_ALLOWANCE_BYTES,
    DEFAULT_HOST,
    DEFAULT_PORT,
    Config,
    config_mode,
    config_path,
    crucible_home,
    default_server_name,
    load_config,
    mint_token,
    write_config,
)
from .errors import ConfigError, NoViableBackend
from .jobs import ALL_JOB_TYPES, build_registry
from .manifests import ManifestError, ModelManifest, load_all_manifests

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


def _fail(message: str) -> int:
    print(f"crucible: {message}", file=sys.stderr)
    return EXIT_REFUSED


# --------------------------------------------------------------------- init


def cmd_init(args: argparse.Namespace) -> int:
    home = crucible_home()
    path = config_path(home)
    if path.exists() and not args.force:
        return _fail(
            f"{path} already exists; pass --force to replace it (this mints a new "
            "token and every client will need the new one)"
        )

    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")

    token = mint_token()
    written = write_config(
        home,
        name=args.name if args.name is not None else default_server_name(),
        host=args.host,
        port=args.port,
        token=token,
        backend_kind=backend.kind,
        enable_echo=args.enable_echo,
        enable_llm=args.enable_llm,
        enable_asr=args.enable_asr,
        desktop_allowance_bytes=args.desktop_allowance_bytes,
    )
    print(f"backend:  {backend.kind} ({backend.gpu.name}, {backend.detail})")
    print(f"config:   {written} (mode {config_mode(written)})")
    print(f"serving:  http://{args.host}:{args.port}/v1")
    print(f"echo job: {'enabled' if args.enable_echo else 'disabled'}")
    print(f"llm job:  {'enabled' if args.enable_llm else 'disabled'}")
    print(f"asr job:  {'enabled' if args.enable_asr else 'disabled'}")
    print(
        f"desktop:  {args.desktop_allowance_bytes / 1024 ** 3:.1f} GiB of VRAM "
        "treated as this host's own desktop, not somebody's job"
    )
    print("token:    minted; print it with `crucible token --show`")
    return EXIT_OK


# -------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force` "
            "on this host"
        )

    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port

    from .api import create_app  # imported here so `init`/`token` stay light

    app = create_app(config, backend)

    print(f"crucible {VERSION} (api {API_VERSION}) — {config.name}")
    print(f"backend: {backend.kind} ({backend.gpu.name})")
    print(f"listening on http://{host}:{port}/v1")
    if host in ("127.0.0.1", "localhost", "::1"):
        print(
            "bound to loopback: only this host can reach it. To serve the tailnet, "
            "pass --host 0.0.0.0 (or the tailnet IP); the bearer token is the lock."
        )
    else:
        print("bound beyond loopback: the bearer token is the only lock.")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return EXIT_OK


# ------------------------------------------------------------------ install


INSTALLABLE_JOB_TYPES = ("llm", *workerenv.WORKER_JOB_TYPES)


def cmd_install(args: argparse.Namespace) -> int:
    if args.job_type not in INSTALLABLE_JOB_TYPES:
        return _fail(
            f"there is no installer for job type {args.job_type!r}; this build "
            f"installs {sorted(INSTALLABLE_JOB_TYPES)}"
        )
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            f"this host detects backend {backend.kind}, but {config.path} was "
            f"initialised for {config.backend_kind}; re-run `crucible init --force`"
        )
    if args.job_type != "llm":
        return _install_worker_env(config, backend, args)

    try:
        recipe = llmenv.recipe_for(backend.kind)
    except llmenv.EnvError as exc:
        return _fail(str(exc))
    print(f"backend: {backend.kind}")
    print(f"recipe:  {recipe}")
    print(f"target:  {llmenv.llm_env_dir(config.home)}")
    started = time.monotonic()
    try:
        status = llmenv.install_llm_env(
            config.home,
            backend.kind,
            force=args.force,
            on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
        )
    except llmenv.EnvError as exc:
        return _fail(str(exc))
    elapsed = time.monotonic() - started
    if not status.installed:
        return _fail(f"the env did not come out installed: {status.detail}")
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    headline = llmenv.BACKEND_HEADLINE_PACKAGE[backend.kind]
    for name in sorted(status.packages):
        if name in (headline, "torch", "numpy", "transformers", "mlx"):
            print(f"  {name}=={status.packages[name]}")
    return EXIT_OK


def _install_worker_env(
    config: Config, backend: Backend, args: argparse.Namespace
) -> int:
    """`crucible install <type>` for a type whose work runs in its own venv.

    PHASE4-AUDIO.md section 0: the phase 4 types are libraries rather than
    servers, so each gets an env of its own and a worker script run with that
    env's python. The `llm` branch above does the same job through `llmenv`; the
    two modules are one module's worth of code twice over, and merging them is a
    follow-up (crucible/workerenv.py says so at the top).
    """
    try:
        recipe = workerenv.recipe_for(args.job_type, backend.kind)
    except workerenv.WorkerEnvError as exc:
        return _fail(str(exc))
    print(f"backend: {backend.kind}")
    print(f"recipe:  {recipe}")
    print(f"target:  {workerenv.worker_env_dir(config.home, args.job_type)}")
    started = time.monotonic()
    try:
        status = workerenv.install_worker_env(
            config.home,
            args.job_type,
            backend.kind,
            force=args.force,
            on_line=(lambda line: print(f"  {line}")) if args.verbose else None,
        )
    except workerenv.WorkerEnvError as exc:
        return _fail(str(exc))
    elapsed = time.monotonic() - started
    if not status.installed:
        return _fail(f"the env did not come out installed: {status.detail}")
    print(f"installed in {elapsed:.0f}s: {status.detail}")
    headline = workerenv.HEADLINE_PACKAGE[args.job_type]
    for name in sorted(status.packages):
        if name in (headline, "ctranslate2", "numpy", "onnxruntime"):
            print(f"  {name}=={status.packages[name]}")
    return EXIT_OK


# ------------------------------------------------------------------- models


def _all_manifests() -> dict[str, "ModelManifest | AsrManifest"]:
    """Every model this build ships, from both manifest directories, by id.

    `models/` and `asr/` are two directories with two loaders (crucible/
    asrmodels.py explains why they are not one yet), but from the command line
    there is a single namespace of model ids, because `crucible models pull <id>`
    is a single question. A collision between the two would make that question
    ambiguous, so it is refused rather than settled by which directory was read
    first.
    """
    merged: dict[str, "ModelManifest | AsrManifest"] = dict(load_all_manifests())
    for model_id, manifest in load_all_asr_manifests().items():
        if model_id in merged:
            raise ManifestError(
                f"{model_id!r} is declared by both {merged[model_id].path} and "
                f"{manifest.path}; a model id names one model"
            )
        merged[model_id] = manifest
    return merged


def _models_config() -> tuple[Config, Backend] | int:
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    return config, backend


def cmd_models_list(args: argparse.Namespace) -> int:
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifests = _all_manifests()
    except (ManifestError, AsrManifestError) as exc:
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
    except (ManifestError, AsrManifestError) as exc:
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
            config, manifest, spec, force=args.force, on_line=lambda line: print(f"  {line}")
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


# ------------------------------------------------------------------- doctor


def _job_type_reports(config: Config, backend: Backend) -> list[dict[str, Any]]:
    registry = build_registry(config, backend)
    reports: list[dict[str, Any]] = []
    for name in sorted(ALL_JOB_TYPES):
        plugin = registry.get(name)
        if plugin is None:
            reports.append(
                {
                    "name": name,
                    "enabled": False,
                    "ready": False,
                    "detail": "not enabled in config.toml",
                    "models": [],
                }
            )
            continue
        status = plugin.check(backend)
        reports.append(
            {
                "name": name,
                "enabled": True,
                "ready": status.ready,
                "detail": status.detail,
                "models": [m.to_dict() for m in plugin.describe_models()],
            }
        )
    return reports


def _doctor_report() -> dict[str, Any]:
    home = crucible_home()
    report: dict[str, Any] = {
        "healthy": False,
        "crucible": {"version": VERSION, "api_version": API_VERSION},
        "home": str(home),
        "config": None,
        "backend": None,
        "job_types": [],
        "llm_env": None,
        "worker_envs": [],
        "problems": [],
    }

    try:
        backend = detect_backend()
        report["backend"] = backend.to_dict()
    except NoViableBackend as exc:
        report["problems"].append(f"no_viable_backend: {exc.reason}")
        backend = None

    try:
        config = load_config(home)
    except ConfigError as exc:
        report["problems"].append(f"config: {exc}")
        config = None

    if config is not None:
        mode = config_mode(config.path)
        report["config"] = {
            "path": str(config.path),
            "mode": mode,
            "name": config.name,
            "host": config.host,
            "port": config.port,
            "enable_echo": config.enable_echo,
            "enable_llm": config.enable_llm,
            "enable_asr": config.enable_asr,
            "desktop_allowance_bytes": config.desktop_allowance_bytes,
            "backend_kind": config.backend_kind,
        }
        if mode != "0o600":
            report["problems"].append(
                f"config_permissions: {config.path} is mode {mode}; the token should "
                "be readable only by its owner (chmod 600)"
            )
        if backend is not None and backend.kind != config.backend_kind:
            report["problems"].append(
                f"backend_changed: config says {config.backend_kind}, this host is "
                f"{backend.kind}"
            )

    if config is not None and backend is not None:
        if config.enable_llm:
            try:
                env = llmenv.env_status(config.home, backend.kind)
                report["llm_env"] = env.to_dict()
                if not env.installed:
                    report["problems"].append(f"llm_env: {env.detail}")
            except llmenv.EnvError as exc:
                report["llm_env"] = {"installed": False, "detail": str(exc)}
                report["problems"].append(f"llm_env: {exc}")
        for job_type in workerenv.WORKER_JOB_TYPES:
            if not getattr(config, f"enable_{job_type}"):
                continue
            try:
                worker_env = workerenv.env_status(config.home, job_type, backend.kind)
                report["worker_envs"].append(worker_env.to_dict())
                if not worker_env.installed:
                    report["problems"].append(f"{job_type}_env: {worker_env.detail}")
            except workerenv.WorkerEnvError as exc:
                report["worker_envs"].append(
                    {"job_type": job_type, "installed": False, "detail": str(exc)}
                )
                report["problems"].append(f"{job_type}_env: {exc}")
        report["job_types"] = _job_type_reports(config, backend)
        for entry in report["job_types"]:
            if entry["enabled"] and not entry["ready"]:
                report["problems"].append(
                    f"job_type_not_ready: {entry['name']}: {entry['detail']}"
                )

    report["healthy"] = not report["problems"]
    return report


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _doctor_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"crucible {VERSION} (api {API_VERSION})")
        print(f"home:    {report['home']}")
        backend = report["backend"]
        if backend is None:
            print("backend: NONE")
        else:
            gpu = backend["gpu"]
            print(
                f"backend: {backend['kind']} on {backend['platform']}/{backend['arch']}"
            )
            print(
                f"gpu:     {gpu['vendor']} {gpu['name']} "
                f"({gpu['vram_bytes'] / 1024 ** 3:.1f} GiB) — {backend['detail']}"
            )
        config = report["config"]
        if config is None:
            print("config:  MISSING")
        else:
            print(f"config:  {config['path']} (mode {config['mode']})")
            print(f"serves:  {config['name']} on {config['host']}:{config['port']}")
        env = report["llm_env"]
        if env is not None:
            mark = "ready" if env["installed"] else "NOT READY"
            print(f"llm env: {mark} — {env['detail']}")
        for worker_env in report["worker_envs"]:
            mark = "ready" if worker_env["installed"] else "NOT READY"
            print(
                f"{worker_env['job_type']} env: {mark} — {worker_env['detail']}"
            )
        for entry in report["job_types"]:
            mark = "ready" if entry["ready"] else ("off" if not entry["enabled"] else "NOT READY")
            print(f"job {entry['name']}: {mark} — {entry['detail']}")
        for problem in report["problems"]:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("healthy" if report["healthy"] else "unhealthy")
    return EXIT_OK if report["healthy"] else EXIT_REFUSED


# -------------------------------------------------------------------- token


def cmd_token(args: argparse.Namespace) -> int:
    if not args.show:
        return _fail("pass --show to print the bearer token")
    try:
        config = load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    print(config.token)
    return EXIT_OK


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="One inference server, many client apps.",
        epilog=(
            f"State lives under ${CRUCIBLE_HOME_ENV} (default ~/.crucible). "
            "Crucible runs on Linux with an NVIDIA card and on Apple Silicon macOS."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crucible {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init", help="detect the backend, mint a token, write config.toml"
    )
    init.add_argument("--force", action="store_true", help="replace an existing config")
    init.add_argument("--name", default=None, help="server name (default crucible@<hostname>)")
    init.add_argument("--host", default=DEFAULT_HOST, help=f"default bind host ({DEFAULT_HOST})")
    init.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default port ({DEFAULT_PORT})")
    init.add_argument(
        "--enable-echo",
        action="store_true",
        help="register the echo test job type ([jobs] enable_echo)",
    )
    init.add_argument(
        "--enable-llm",
        action="store_true",
        help="register the load-model / unload-model job types and the OpenAI "
        "proxy ([jobs] enable_llm)",
    )
    init.add_argument(
        "--enable-asr",
        action="store_true",
        help="register the asr (faster-whisper transcription) job type "
        "([jobs] enable_asr)",
    )
    init.add_argument(
        "--desktop-allowance-bytes",
        type=int,
        default=DEFAULT_DESKTOP_ALLOWANCE_BYTES,
        help=(
            "VRAM this host's own desktop holds, which the accelerator guard does "
            f"not count as somebody's job (default {DEFAULT_DESKTOP_ALLOWANCE_BYTES}"
            " = 3 GiB; use 0 on a headless box)"
        ),
    )
    init.set_defaults(func=cmd_init)

    install = subparsers.add_parser(
        "install", help="create a job type's env and install its recipe"
    )
    install.add_argument(
        "job_type",
        choices=sorted(INSTALLABLE_JOB_TYPES),
        help="the job type to install",
    )
    install.add_argument(
        "--force", action="store_true", help="rebuild the env from scratch"
    )
    install.add_argument(
        "--verbose", action="store_true", help="echo pip's output line by line"
    )
    install.set_defaults(func=cmd_install)

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

    serve = subparsers.add_parser("serve", help="run the API in the foreground")
    serve.add_argument("--host", default=None, help="bind host (default from config)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    serve.add_argument("--log-level", default="info", help="uvicorn log level")
    serve.set_defaults(func=cmd_serve)

    doctor = subparsers.add_parser("doctor", help="probe the host and the job types")
    doctor.add_argument("--json", action="store_true", help="machine-readable report")
    doctor.set_defaults(func=cmd_doctor)

    token = subparsers.add_parser("token", help="print the bearer token")
    token.add_argument("--show", action="store_true", help="required; prints the secret")
    token.set_defaults(func=cmd_token)

    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        print(f"crucible: {WINDOWS_REFUSAL}", file=sys.stderr)
        return EXIT_REFUSED
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
