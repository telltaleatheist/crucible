from __future__ import annotations

from typing import Any

from ... import VERSION, features, pairing, weights
from ... import pages as pages_module
from ... import peer as peer_module
from ...errors import ApiError
from ...interfaces import InterfaceError
from ...jobs import ALL_JOB_TYPES, model_rows, voice_rows
from ...jobs.base import TERMINAL_STATES as JOB_TERMINAL_STATES
from ...manifests import ManifestError, load_manifest
from ...protocol import API_VERSION
from ...tasks.states import TERMINAL_STATES as TASK_TERMINAL_STATES
from ...voices import MANIFEST_ENGINE, MANIFEST_OVERRIDE, MANIFEST_REPO
from ..context import AppContext, Routers
from ..responses import Info, ServiceCommand, TerminalStates, VoiceSourceLabel

TERMINAL = TerminalStates(
    jobs=sorted(JOB_TERMINAL_STATES), tasks=sorted(TASK_TERMINAL_STATES)
)
VOICE_SOURCES = {
    MANIFEST_REPO: VoiceSourceLabel(label="from its repo", tone="ok"),
    MANIFEST_OVERRIDE: VoiceSourceLabel(label="set on this machine", tone="warn"),
    MANIFEST_ENGINE: VoiceSourceLabel(label="the engine's own", tone="floor"),
}
SERVICE_COMMANDS = [
    ServiceCommand(command="crucible service status", does="is it installed, and is it up"),
    ServiceCommand(command="crucible service start", does="start it now"),
    ServiceCommand(command="crucible service stop", does="stop it"),
    ServiceCommand(command="crucible service install", does="have the machine run it at boot"),
    ServiceCommand(command="crucible service uninstall", does="stop having it do that"),
    ServiceCommand(command="crucible token --url", does="print the pairing lines again"),
]


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    @private.get("/info", response_model=Info, response_model_exclude_unset=True)
    async def info() -> dict[str, Any]:
        """Who this server is, what it runs on, what it serves, and the few fixed
        tables (terminal states, voice sources, service commands) a console shows."""
        store = ctx.store
        rows_for: dict[str, list[dict[str, Any]]] = {}
        for name, plugin in sorted(store.registry.items()):
            capability = ALL_JOB_TYPES[name]
            if capability in rows_for:
                continue
            rows_for[capability] = [m.to_dict() for m in plugin.describe_models()]
        if config.enable_llm:
            rows_for["llm"] = model_rows(config, backend, residency)
        if config.enable_tts:
            rows_for["tts"] = voice_rows(config, backend, residency, store=store)
        capabilities = [
            {"job_type": capability, "models": rows}
            for capability, rows in sorted(rows_for.items())
        ]
        return {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
            },
            "role": peer_module.ROLE_ENGINE,
            "managed_by": ctx.peer.managed_by(),
            "host": {
                "platform": backend.platform,
                "arch": backend.arch,
                "backend": backend.kind,
                "gpu": {
                    "vendor": backend.gpu.vendor,
                    "name": backend.gpu.name,
                    "vram_bytes": backend.gpu.vram_bytes,
                },
            },
            "features": features.names(),
            "job_types": sorted(store.registry),
            "capabilities": capabilities,
            "pages_engine": _pages_engine(),
            "terminal_states": TERMINAL,
            "voice_sources": VOICE_SOURCES,
            "service_commands": SERVICE_COMMANDS,
        }

    def _pages_engine() -> dict[str, Any]:
        try:
            manifest = load_manifest(pages_module.MODEL_ID)
        except ManifestError as exc:
            return pages_module.engine_block(
                engine=None,
                installed=False,
                detail=f"{pages_module.MODEL_ID}'s manifest will not read: {exc}",
            )
        if not manifest.supports(backend.kind):
            return pages_module.engine_block(
                engine=None,
                installed=False,
                detail=(
                    f"{manifest.path.name} has no {backend.kind} block, so this "
                    f"host reads no pages. It declares {sorted(manifest.backends)}"
                ),
            )
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        if found is None:
            return pages_module.engine_block(
                engine=spec.engine,
                installed=False,
                detail=(
                    f"{spec.engine} would serve {manifest.id} here, and its "
                    f"weights are not pulled — `crucible models pull {manifest.id}`"
                ),
            )
        return pages_module.engine_block(
            engine=spec.engine,
            installed=True,
            detail=(
                f"{spec.engine} serves {manifest.id} from {found.path} "
                f"({found.bytes / 1e9:.2f} GB, {spec.hf_repo}@{spec.revision[:12]})"
            ),
        )

    @private.get("/health")
    async def health() -> dict[str, Any]:
        """Is this process alive, in one cheap read: `status` (`ok`, `busy` running a job,
        `warming` loading a model), the queue depth and what is resident."""
        store = ctx.store
        if residency.warming is not None:
            status = "warming"
        elif store.running_id is not None:
            status = "busy"
        else:
            status = "ok"
        return {
            "status": status,
            "queue_depth": store.queue_depth,
            "resident_models": residency.ids(),
            "resident_kind": residency.resident_kind,
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
        }

    @private.get("/setup")
    async def setup() -> dict[str, Any]:
        """Everything an app needs to be pointed at this server in one read, including
        its token and pairing lines.
        """
        host, port = ctx.bind_host, ctx.bind_port
        try:
            urls = pairing.reachable_urls(
                host, port, config.advertise + config.tailscale_advertise + config.lan_advertise
            )
        except InterfaceError as exc:
            raise ApiError(
                503,
                "interfaces_unreadable",
                f"this server is bound to {host!r} and cannot list its own "
                f"interfaces, so it cannot say where an app should reach it: "
                f"{exc}",
            ) from None
        return {
            "name": config.name,
            "version": VERSION,
            "backend": backend.kind,
            "bind": f"http://{host}:{port}",
            "urls": urls,
            "token": config.token,
            "pairing": pairing.pairing_lines(config.name, urls, config.token),
            "job_types": sorted(ctx.store.registry),
            "config_path": str(config.path),
        }
