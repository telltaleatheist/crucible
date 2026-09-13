"""Which model is on the accelerator right now.

PHASE2-LLM.md section 3: **one resident model at a time**. Loading a second
unloads the first. LRU across several comes later.

Only the exclusive job lane mutates this (the `load-model` and `unload-model`
jobs), so the proxy and `/v1/health` read a value that is never half-written: an
engine is published as resident only once it has answered its own `/v1/models`,
and it is unpublished before it is signalled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ...config import Config
from ...engines import (
    EngineError,
    SubprocessEngine,
    build_engine,
    engine_log_path,
    engine_model_name,
    find_free_port,
)
from ...manifests import BackendSpec, ModelManifest, fingerprint

#: How long a load waits for the engine to answer `/v1/models`. vLLM on a 19 GB
#: model spends most of it reading weights and capturing CUDA graphs.
DEFAULT_READY_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class ResidentModel:
    model_id: str
    backend: str
    engine: str
    engine_model_name: str
    base_url: str
    port: int
    revision: str
    #: The context this engine was actually started with — vLLM's
    #: `--max-model-len`, mlx-lm's own config. It comes from
    #: `ModelManifest.context_for(backend)` at load time, which is why it is not
    #: simply read back off the manifest: a manifest edited while this engine is
    #: up would then describe a context nothing is serving.
    max_model_len: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

    @property
    def fingerprint(self) -> str:
        """`<id>@<revision>` for the weights this engine actually read."""
        return fingerprint(self.model_id, self.revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "engine": self.engine,
            "engine_model_name": self.engine_model_name,
            "base_url": self.base_url,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "max_model_len": self.max_model_len,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Residency:
    """The one-resident-model holder for a server instance."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._resident: ResidentModel | None = None
        self._engine: SubprocessEngine | None = None
        self._warming: str | None = None

    # -------------------------------------------------------------- reading

    @property
    def resident(self) -> ResidentModel | None:
        return self._resident

    @property
    def resident_id(self) -> str | None:
        return None if self._resident is None else self._resident.model_id

    @property
    def warming(self) -> str | None:
        """The model id a load job is currently warming, or None."""
        return self._warming

    def ids(self) -> list[str]:
        return [] if self._resident is None else [self._resident.model_id]

    def begin_warming(self, model_id: str) -> None:
        """Mark a load as in progress, so `/v1/health` says `warming`.

        Set for the whole load job, not just the engine's readiness poll: from
        the client's side, "this server is warming a model" is true from the
        moment the lane picks the job up.
        """
        self._warming = model_id

    def end_warming(self) -> None:
        self._warming = None

    def owned_pids(self) -> frozenset[int]:
        return frozenset() if self._engine is None else self._engine.pids

    def reclaimable_bytes(self, excluding: str | None = None) -> int:
        """What unloading the current resident would give back.

        Zero when the resident *is* `excluding` — reloading a model does not free
        its own memory before it needs it again.
        """
        if self._resident is None or self._resident.model_id == excluding:
            return 0
        return self._resident.memory_bytes_estimate

    # -------------------------------------------------------------- writing

    def load(
        self,
        manifest: ModelManifest,
        spec: BackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentModel:
        """Make this model the resident one, unloading whatever was there."""

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        if self._resident is not None:
            previous = self._resident.model_id
            say(f"unloading {previous} — one resident model at a time")
            self.unload(previous)

        log_path = engine_log_path(self._config.home, manifest.id)
        engine = build_engine(spec.engine, python, log_path)
        served = engine_model_name(spec.engine, weights_dir, manifest.id)
        port = find_free_port()

        context = manifest.context_for(spec.backend)
        self.begin_warming(manifest.id)
        say(
            f"starting {spec.engine} for {manifest.id} on 127.0.0.1:{port} "
            f"(context {context}); log {log_path}"
        )
        try:
            engine.start(
                weights_dir,
                served,
                port,
                self._engine_args(manifest, spec),
            )
            engine.ready(timeout, on_progress=say)
        except EngineError as start_failure:
            # Tidy up the half-started engine, but report the *start* failure —
            # that is the one that explains the load. A stop failure on top of it
            # is appended, never substituted.
            try:
                engine.stop()
            except EngineError as stop_failure:
                raise EngineError(
                    f"{start_failure}\n...and stopping it also failed: {stop_failure}"
                ) from start_failure
            raise
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentModel(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            engine_model_name=served,
            base_url=engine.base_url,
            port=port,
            revision=spec.revision,
            max_model_len=context,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident at {engine.base_url}")
        return self._resident

    @staticmethod
    def _engine_args(manifest: ModelManifest, spec: BackendSpec) -> list[str]:
        """The manifest's args plus what Crucible always sets.

        `--max-model-len` only goes to vLLM; mlx-lm takes the context from the
        model's own config and has no such flag (see engines/mlx_lm.py).
        """
        args = list(spec.engine_args)
        if spec.engine == "vllm":
            args += ["--max-model-len", str(manifest.context_for(spec.backend))]
        return args

    def unload(self, model_id: str) -> ResidentModel:
        """Stop the engine serving `model_id`. Raises KeyError if it is not resident."""
        resident = self._resident
        if resident is None or resident.model_id != model_id:
            raise KeyError(model_id)
        engine = self._engine
        # Unpublish first: from here on nothing new is proxied to a dying engine.
        self._resident = None
        self._engine = None
        if engine is not None:
            engine.stop()
        return resident

    def shutdown(self) -> None:
        """Stop whatever is resident. Called when the server exits."""
        if self._resident is not None:
            self.unload(self._resident.model_id)
