from __future__ import annotations

from typing import Any

from fastapi import Request

from ... import API_VERSION, VERSION, pairing, weights
from ... import pages as pages_module
from ... import peer as peer_module
from ...config import Config
from ...errors import ApiError
from ...interfaces import InterfaceError
from ...jobs import ALL_JOB_TYPES, model_rows, voice_rows
from ...jobs.queue import JobStore
from ...manifests import ManifestError, load_manifest
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    app, config, backend, residency = ctx.app, ctx.config, ctx.backend, ctx.residency

    @private.get("/info")
    async def info(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store
        rows_for: dict[str, list[dict[str, Any]]] = {}
        for name, plugin in sorted(store.registry.items()):
            capability = ALL_JOB_TYPES[name]
            if capability in rows_for:
                continue
            rows_for[capability] = [m.to_dict() for m in plugin.describe_models()]
        if config.enable_llm:
            rows_for["llm"] = model_rows(config, backend, residency)
        if config.enable_tts:
            rows_for["tts"] = voice_rows(
                config, backend, residency,
                leases=app.state.leases, store=app.state.store,
            )
        capabilities = [
            {"job_type": capability, "models": rows}
            for capability, rows in sorted(rows_for.items())
        ]
        peer_state: peer_module.PeerState = request.app.state.peer
        return {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
            },
            "role": peer_module.ROLE_ENGINE,
            "managed_by": peer_state.managed_by(),
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
            "job_types": sorted(store.registry),
            "capabilities": capabilities,
            "pages_engine": _pages_engine(),
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
    async def health(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store
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
    async def setup(request: Request) -> dict[str, Any]:
        """Everything an app needs to be pointed at this server in one read, including
        its token and pairing lines.
        """
        live: Config = request.app.state.config
        store: JobStore = request.app.state.store
        host = request.app.state.bind_host
        port = request.app.state.bind_port
        try:
            urls = pairing.reachable_urls(
                host, port, live.advertise + live.tailscale_advertise + live.lan_advertise
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
            "name": live.name,
            "version": VERSION,
            "backend": backend.kind,
            "bind": f"http://{host}:{port}",
            "urls": urls,
            "token": live.token,
            "pairing": pairing.pairing_lines(live.name, urls, live.token),
            "job_types": sorted(store.registry),
            "config_path": str(live.path),
        }
