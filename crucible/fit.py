from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from . import asrplan, ttsplan
from .audiomodels import held_need
from .backend import CardFacts
from .enginespec import bf16_fallback, card_needs, declared_dtype, dtype_on
from .manifests import MemoryTerms
from .precision import below_floor, weight_bits
from .precision import label as precision_label
from .servingplan import ServingVariant


@dataclass(frozen=True)
class WorkingContext:
    tokens: int
    concurrency: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "concurrency": self.concurrency,
            "source": self.source,
        }


@dataclass(frozen=True)
class Candidate:
    id: str
    memory_bytes_estimate: int
    memory: "MemoryTerms | None" = None
    served_context: int | None = None
    needs: tuple[str, ...] = ()
    bits: int | None = None
    dtype: str | None = None
    bf16_fallback: str | None = None
    serving: "tuple[ServingVariant, ...] | None" = None
    # An audio model's declared need under `[audio] low_vram` (None: it cannot be split).
    low_vram_bytes: int | None = None
    # Set by on_host when this host holds the model under `[audio] low_vram`: the whole
    # figure memory_bytes_estimate stood at before it became the held one.
    whole_bytes: int | None = None
    # The manifest's `[model] params_b`, where its catalog states one: the first key of a
    # goal class's pick (docs/VERB-SIZING.md rule 3).
    params_b: float | None = None
    # The base whose weights this alias serves another way (`[model] weights_of`), or None
    # for a model's own form.
    weights_of: str | None = None
    # This backend's block serves images (`serves` names "image"): what a request that
    # carries images needs.
    serves_images: bool = False

    @classmethod
    def of(cls, manifest: Any, backend_kind: str, form: str | None = None) -> "Candidate":
        """The candidate a manifest's block is here: for a block with forms, the form this
        host takes, or `form` where one is named."""
        spec = (
            manifest.spec(backend_kind)
            if form is None
            else manifest.spec(backend_kind, form)
        )
        return cls(
            id=manifest.id,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            memory=getattr(spec, "memory", None),
            served_context=(
                manifest.max_context_for(backend_kind)
                if hasattr(manifest, "max_context_for")
                else None
            ),
            needs=card_needs(spec),
            bits=weight_bits(spec),
            dtype=declared_dtype(spec),
            bf16_fallback=bf16_fallback(spec),
            serving=(
                ttsplan.ladder_for(manifest, spec, backend_kind)
                or asrplan.ladder_for(manifest, spec, backend_kind)
            ),
            low_vram_bytes=getattr(spec, "low_vram_memory_bytes_estimate", None),
            params_b=getattr(manifest, "params_b", None),
            weights_of=getattr(manifest, "weights_of", None),
            serves_images="image" in getattr(spec, "serves", ()),
        )

    @property
    def alias(self) -> bool:
        """A `weights_of` alias: the same weights as its base, served another way."""
        return self.weights_of is not None

    def form_of(self, model_id: str) -> bool:
        """This candidate is `model_id` itself or an alias of its weights."""
        return self.id == model_id or self.weights_of == model_id

    @property
    def held_low_vram(self) -> bool:
        return self.whole_bytes is not None

    def on_host(self, audio_low_vram: bool) -> "Candidate":
        """This candidate as a host with this `[audio] low_vram` holds it: the need is
        audiomodels.held_need's, the rule the audio job admits against."""
        if self.held_low_vram:
            raise ValueError(f"{self.id} is already weighed for a host; weigh the catalog's")
        need = held_need(self.memory_bytes_estimate, self.low_vram_bytes, audio_low_vram)
        if not need.low_vram:
            return self
        return replace(self, memory_bytes_estimate=need.bytes, whole_bytes=self.memory_bytes_estimate)

    def would_fit_low_vram(self, budget: int) -> bool:
        """Not held low here, and its declared low-VRAM need fits: `[audio] low_vram`
        is what this host lacks for it."""
        if self.held_low_vram:
            return False
        would = held_need(self.memory_bytes_estimate, self.low_vram_bytes, True)
        return would.low_vram and would.bytes <= budget

    def serving_on(self, budget: int) -> "ServingVariant | None":
        if self.serving is None:
            return None
        return ttsplan.choose(self.serving, budget)

    def holds(self, work: "WorkingContext | None", budget: int) -> bool:
        if self.serving is not None:
            return self.serving_on(budget) is not None
        return self.need_bytes(work) <= budget

    def floor_bytes(self, work: "WorkingContext | None") -> int:
        if self.serving is not None:
            return min(v.need_bytes for v in self.serving if v.available)
        return self.need_bytes(work)

    def lacks(self, card: "CardFacts | None") -> tuple[str, ...]:
        if card is None:
            return ()
        return tuple(need for need in self.needs if card.has(need) is False)

    def run_dtype(self, card: "CardFacts | None") -> str | None:
        return dtype_on(self.dtype, self.bf16_fallback, card)

    def precision_on(self, card: "CardFacts | None") -> str:
        return precision_label(self.bits, self.run_dtype(card))

    def degraded_on(self, card: "CardFacts | None") -> bool:
        return self.run_dtype(card) != self.dtype

    def context_ceiling(
        self, available_bytes: int, concurrency: int
    ) -> "ContextCeiling | None":
        if self.served_context is None:
            return None
        memory = (
            None
            if self.memory is None
            else self.memory.max_context(
                available_bytes=available_bytes, concurrency=concurrency
            )
        )
        if memory is None or self.served_context <= memory:
            ceiling, bound_by = self.served_context, "served"
        else:
            ceiling, bound_by = memory, "memory"
        return ContextCeiling(
            model=self.id,
            tokens=ceiling,
            bound_by=bound_by,
            served_context=self.served_context,
            memory_context=memory,
            concurrency=concurrency,
        )

    def need_bytes(self, work: "WorkingContext | None") -> int:
        if work is None or self.memory is None:
            return self.memory_bytes_estimate
        return self.memory.bytes_for(
            context=work.tokens, concurrency=work.concurrency
        )

    def max_context(self, available_bytes: int, work: "WorkingContext | None") -> int | None:
        if self.memory is None:
            return None
        concurrency = 1 if work is None else work.concurrency
        return self.memory.max_context(
            available_bytes=available_bytes, concurrency=concurrency
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory": None if self.memory is None else self.memory.to_dict(),
            "served_context": self.served_context,
            "needs": list(self.needs),
            "bits": self.bits,
            "dtype": self.dtype,
            "bf16_fallback": self.bf16_fallback,
            "low_vram_bytes": self.low_vram_bytes,
            "low_vram": self.held_low_vram,
            "whole_bytes": self.whole_bytes,
            "params_b": self.params_b,
            "alias": self.alias,
            "weights_of": self.weights_of,
            "serves_images": self.serves_images,
        }


def by_need(candidate: Candidate) -> tuple[int, str]:
    return -candidate.memory_bytes_estimate, candidate.id


def on_host(found: tuple[Candidate, ...], audio_low_vram: bool) -> tuple[Candidate, ...]:
    """A class's candidates as this host holds them, largest need first."""
    return tuple(sorted((c.on_host(audio_low_vram) for c in found), key=by_need))


@dataclass(frozen=True)
class ContextCeiling:
    model: str
    tokens: int
    bound_by: str
    served_context: int
    memory_context: int | None
    concurrency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokens": self.tokens,
            "bound_by": self.bound_by,
            "served_context": self.served_context,
            "served_context_source": (
                "the most this backend ever starts an engine with: the model "
                "manifest's max_context for this backend, or its "
                "context_default where it states none "
                "(manifest.max_context_for; a load-model's params.context may "
                "raise --max-model-len / -c / max_model_len up to it)"
            ),
            "memory_context": self.memory_context,
            "memory_context_source": (
                None
                if self.memory_context is None
                else (
                    "this host's available bytes less the model's weights and "
                    f"overhead, over its KV bytes per token x {self.concurrency} "
                    "in flight (MemoryTerms.max_context)"
                )
            ),
            "concurrency": self.concurrency,
        }


Fingerprint = tuple[str, int, tuple[tuple[str, int, int], ...]]


def catalog_fingerprint(root: Path) -> Fingerprint:
    files = []
    for path in sorted(root.glob("*.toml")):
        stat = path.stat()
        files.append((path.name, stat.st_mtime_ns, stat.st_size))
    return str(root.resolve()), root.stat().st_mtime_ns, tuple(files)


_catalogs: dict[tuple[Any, Fingerprint], dict[str, Any]] = {}
_candidates: dict[tuple["CatalogCandidates", str], tuple[Fingerprint, tuple[Candidate, ...]]] = {}


def cached_catalog(
    load: Callable[..., dict[str, Any]], directory: Callable[[], Path]
) -> dict[str, Any]:
    root = directory()
    key = (load, catalog_fingerprint(root))
    found = _catalogs.get(key)
    if found is None:
        found = load(root)
        for stale in [k for k in list(_catalogs) if k[0] is load]:
            _catalogs.pop(stale, None)
        _catalogs[key] = found
    return dict(found)


def forget_cached_catalogs() -> None:
    _catalogs.clear()
    _candidates.clear()


@dataclass(frozen=True)
class CatalogCandidates:
    load: Callable[..., dict[str, Any]]
    families: tuple[str, ...] | None = None
    aliases: bool = False
    directory: Callable[[], Path] | None = None

    def __call__(self, backend_kind: str) -> tuple[Candidate, ...]:
        if self.directory is None:
            return self._select(self.load(), backend_kind)
        fingerprint = catalog_fingerprint(self.directory())
        held = _candidates.get((self, backend_kind))
        if held is not None and held[0] == fingerprint:
            return held[1]
        found = self._select(cached_catalog(self.load, self.directory), backend_kind)
        _candidates[(self, backend_kind)] = (fingerprint, found)
        return found

    def _select(self, catalog: dict[str, Any], backend_kind: str) -> tuple[Candidate, ...]:
        found: list[Candidate] = []
        for manifest in catalog.values():
            if self.families is not None and manifest.family not in self.families:
                continue
            if not self.aliases and getattr(manifest, "weights_of", None) is not None:
                continue
            if not manifest.supports(backend_kind):
                continue
            candidate = Candidate.of(manifest, backend_kind)
            if below_floor(candidate.bits):
                continue
            found.append(candidate)
        found.sort(key=by_need)
        return tuple(found)


__all__ = [
    "Candidate",
    "CatalogCandidates",
    "ContextCeiling",
    "WorkingContext",
    "by_need",
    "cached_catalog",
    "catalog_fingerprint",
    "forget_cached_catalogs",
    "on_host",
]
