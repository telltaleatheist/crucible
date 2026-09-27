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

    # ------------------------------------------------------------------ info

    @private.get("/info")
    async def info(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store

        # ONE CAPABILITY PER CAPABILITY, not one per POSTable job type.
        #
        # This listed every registered job type as its own capability until
        # 2026-09-13, when running the merged server and reading its answer
        # showed `qwen3.5-9b` appearing three times — under `load-model`, under
        # `unload-model`, and under `llm` — in TWO different shapes, because the
        # lifecycle types describe a model with `ModelDescriptor.to_dict()` and
        # the `llm` capability uses the far richer `/v1/models` row. That is
        # exactly the thing PHASE2-LLM.md section 5 was written to forbid: "One
        # model, one description: a client reads a model's standing in one shape
        # wherever it finds it, and never reconciles two."
        #
        # `ALL_JOB_TYPES` already maps a job type to the capability it operates
        # (`load-model` and `unload-model` both to `llm`), so the grouping is not
        # a new table anybody has to keep in step — it is the one the registry is
        # already built from. What you can POST is answered by `job_types`, which
        # is what the lifecycle rows were really there to tell anyone.
        rows_for: dict[str, list[dict[str, Any]]] = {}
        for name, plugin in sorted(store.registry.items()):
            capability = ALL_JOB_TYPES[name]
            if capability in rows_for:
                continue
            rows_for[capability] = [m.to_dict() for m in plugin.describe_models()]
        if config.enable_llm:
            # PHASE2-LLM.md section 5: the `llm` rows are `/v1/models`' rows,
            # from the same producer, revision included.
            rows_for["llm"] = model_rows(config, backend, residency)
        if config.enable_tts:
            # PHASE3-TTS.md section 8: the `tts` rows are `/v1/voices`' rows
            # VERBATIM, for the same reason `llm`'s are.
            #
            # This assignment REPLACES whatever the loop above produced, and for
            # `tts` that is not a no-op the way it is for `llm`. `llm` is a
            # capability name and nothing else — the types you POST are
            # `load-model` and `unload-model` — while `tts` is both: the render
            # door's job type is literally `tts`, so the loop has already filled
            # this key from its `describe_models()`. The richer row wins, which
            # is the same rule applied one level down.
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
            # WHICH HALF OF THE RELATION THIS PROCESS IS
            # (PHASE17-ORCHESTRATOR.md 3.1). A property of a PROCESS, never of
            # an install: on a Windows machine with no WSL, one install runs
            # an orchestrator and an engine as two processes, and only one of
            # them answers this document.
            #
            # A client reads it all-or-nothing, exactly as it reads `route`
            # (PHASE15 3.3): a document with NO `role` comes from a server
            # that predates this phase, and such a server IS an engine with
            # `managed_by: null` — a fact the document states by its vintage,
            # not a default the client fills.
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
            # What this server will accept in `POST /v1/jobs`. A capability
            # above says what it can serve; this says what to ask it with, and
            # the two are not the same list: `llm` is served through
            # `load-model` and `unload-model`, neither of which is a capability.
            "job_types": sorted(store.registry),
            "capabilities": capabilities,
            # WHICH ENGINE READS A PAGE HERE, AND WHAT A PAGE REQUEST IS
            # (PHASE15-HOST.md 3.10). Two halves, and the second is the
            # load-bearing one: `engine` is for an operator looking at a
            # machine, and `request` is the prompt, the dpi, the pixel
            # budget, the ceiling and the dialect — read from
            # `crucible/pages.py` on EVERY backend, so a client builds the
            # same bytes whether vLLM, llama.cpp or mlx-vlm is behind them.
            #
            # It is on the wire because page reading has no job type of its
            # own (PHASE3-VLM.md section 1): the CLIENT builds the chat
            # completion, and it was building it out of constants pinned in
            # its own source — a prompt and a pixel budget are facts about
            # the weights, and the division-of-knowledge ruling puts those
            # here.
            "pages_engine": _pages_engine(),
        }

    def _pages_engine() -> dict[str, Any]:
        """`pages_engine` for THIS host. Null engine where none is served."""
        try:
            manifest = load_manifest(pages_module.MODEL_ID)
        except ManifestError as exc:
            # A build whose page manifest will not read has no page engine,
            # and says which file rather than answering an empty block.
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
            # Keeps its name and its shape — a list of ids — and gains the kind
            # beside it, because since PHASE3-TTS.md section 5 the one thing on
            # the card may be a voice, and a client has to know which door to
            # knock on. `resident_models` is not renamed: every phase-2 client
            # reads it, and one id is one id whatever kind of thing it names.
            "resident_models": residency.ids(),
            "resident_kind": residency.resident_kind,
            # WHAT WAS TOLD TO GO AND HAS NOT (ledger R13, Owen's ruling
            # 2026-09-18). Null when nothing is. It belongs on the smallest
            # read this server has because it is the reason every load, the
            # claim and the streaming door are refusing. `status` is untouched
            # and still reports the LANE — `ok` there has always meant "no job
            # is running", never "the card is free" — so this is the field
            # that makes the difference readable instead of a redefinition of
            # one every phase-2 client already reads. The object is
            # `Residency.stopping`'s own (`DyingResident.to_dict`), so this
            # and `/v1/activity` cannot tell two stories.
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
        }

    # ----------------------------------------------------------------- setup

    @private.get("/setup")
    async def setup(request: Request) -> dict[str, Any]:
        """Everything an app needs to be pointed at this server, in one read.

        PHASE13-OPERATOR.md section 3.1. Owen, 2026-09-14: *"crucible has its own
        ui. and it provides the token or whatever else we need to set it up on
        foundry or bookforge."* This is "whatever else we need".

        **It returns the token, and that reveals nothing.** Every `/v1/*` route
        is behind the bearer token, so the only caller who can read this is one
        who already has it. What it buys is that nobody types a secret twice:
        the operator page fetches this and draws a copyable pairing line, and
        the person pasting that line into BookForge has not seen a token at all.

        `urls` and `pairing` are the same list read two ways, and both are
        derived rather than stored — the bind address this process actually
        holds, made dialable (`crucible/pairing.py`). A wildcard bind becomes
        one entry per non-loopback IPv4 interface; a concrete bind becomes
        exactly one. Never a hostname lookup: an interface is a fact about this
        host, a name is a fact about somebody else's resolver.

        `job_types` repeats `/v1/info`'s list rather than making the page read
        twice, and it repeats it from the same producer — `store.registry` — so
        the two cannot disagree. After an install task's reload (3.4) both
        answer the new list in the same tick.
        """
        live: Config = request.app.state.config
        store: JobStore = request.app.state.store
        host = request.app.state.bind_host
        port = request.app.state.bind_port
        try:
            # `live.advertise` is the operator's statement that something
            # forwards here from elsewhere. The bind is what this host can see;
            # that is what it cannot.
            urls = pairing.reachable_urls(
                host, port, live.advertise + live.tailscale_advertise + live.lan_advertise
            )
        except InterfaceError as exc:
            # 503 and not an empty `urls`: an empty list reads as "reachable
            # from nowhere", which is a claim about this host rather than a
            # report that the question could not be asked (R3). The page shows
            # the reason and the operator can still bind a concrete address.
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
