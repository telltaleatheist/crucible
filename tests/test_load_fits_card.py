"""A load that names no context fits the card: the model's default, or the most this card
affords under it, never under MIN_LOAD_CONTEXT.

Briefcase on the Mac against owens-pc (RTX 3090 Ti, 4.5 GiB desktop allowance,
Crucible 2.0.17), 2026-10-20: a session for `generate` with max_params_b 27 was given
qwen3.8-27b-4bit, which fits the verb's working context, and its load was refused
`insufficient_kv_cache` because it was started at the manifest's 16384 where the card
afforded 14_247. A ceiling request on an idle card reached a refusal. A context the request
STATES is still held to and refused when it does not fit."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import accelerator
from crucible.contextceiling import MIN_LOAD_CONTEXT
from crucible.errors import ApiError
from crucible.jobs.llm import LoadParams
from crucible.manifests import load_manifest
from crucible.memorybudget import MIB
from crucible.queuesessions import QueueSession
from crucible.sessionqueue import _resident
from crucible.vram import VLLM_CUDA_CONTEXT_BYTES, VLLM_SEQUENCE_SLACK_TOKENS

from .test_vllm_budget_after_eviction import (  # noqa: F401  (the fixture)
    FAKE_BACKEND,
    MODEL,
    TOTAL,
    Card,
    loader,
)

TERMS = load_manifest(MODEL).spec(FAKE_BACKEND.kind).memory
DEFAULT = load_manifest(MODEL).context_for(FAKE_BACKEND.kind)


def _free_for(tokens: int) -> int:
    """The card's free memory that leaves a KV pool of exactly `tokens` (plus vLLM's
    per-sequence slack) once the weights and vLLM's CUDA context are taken out."""
    assert TERMS is not None
    pool = TERMS.kv_bytes_per_token * (tokens + VLLM_SEQUENCE_SLACK_TOKENS)
    return TERMS.fixed_bytes + pool + VLLM_CUDA_CONTEXT_BYTES


class Done:
    def __init__(self, ctx: Any) -> None:
        self.extra: dict[str, Any] = {}
        ctx.done_extra = self.extra.update


def _idle(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free, TOTAL))


def test_the_default_does_not_fit_so_it_loads_at_what_the_card_affords(
    loader, monkeypatch
) -> None:
    job, built, ctx = loader(None)
    _idle(monkeypatch, _free_for(DEFAULT // 2))
    done = Done(ctx)

    needs = job.requirements(MODEL, LoadParams())
    assert needs.fits_to_card and needs.forecast.fits
    assert MIN_LOAD_CONTEXT <= needs.context < DEFAULT
    assert abs(needs.context - DEFAULT // 2) <= 1

    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]

    assert len(built) == 1
    resident = job.residency.resident_model
    assert resident is not None and resident.max_model_len == needs.context
    assert done.extra["context"] == needs.context
    assert done.extra["context_default"] == DEFAULT
    assert any(f"(context {needs.context})" in line for line in ctx.said)


def test_a_card_that_holds_the_default_loads_at_the_default(loader, monkeypatch) -> None:
    job, built, ctx = loader(None)
    _idle(monkeypatch, _free_for(DEFAULT) + 64 * MIB)
    done = Done(ctx)
    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]
    assert done.extra["context"] == DEFAULT


def test_a_stated_context_is_held_to_and_refused_when_it_does_not_fit(
    loader, monkeypatch
) -> None:
    job, built, ctx = loader(None)
    _idle(monkeypatch, _free_for(DEFAULT // 2))
    with pytest.raises(ApiError) as refused:
        job.requirements(MODEL, LoadParams(context=DEFAULT))
    assert refused.value.code == "insufficient_kv_cache"
    assert built == []


def test_under_the_minimum_context_the_load_is_refused_by_name(loader, monkeypatch) -> None:
    job, built, ctx = loader(None)
    _idle(monkeypatch, _free_for(MIN_LOAD_CONTEXT - 256))
    with pytest.raises(ApiError) as refused:
        job.requirements(MODEL, LoadParams())
    assert refused.value.code == "insufficient_kv_cache"
    assert f"at {DEFAULT} tokens" in refused.value.message


def test_a_card_with_less_after_the_eviction_is_fitted_again(loader) -> None:
    """The forecast (free plus the resident's figure) holds the default; the card read once
    the resident has gone holds less, and the engine starts at what that card affords,
    saying so, rather than being refused."""
    after = _free_for(DEFAULT // 2)
    card = Card(held_bytes=4608 * MIB, after_stop=[after, after])
    job, built, ctx = loader(card)
    done = Done(ctx)

    assert job.requirements(MODEL, LoadParams()).context == DEFAULT
    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]

    assert len(built) == 1
    assert MIN_LOAD_CONTEXT <= done.extra["context"] < DEFAULT
    assert any("the most this card affords now" in line for line in ctx.said)


def test_a_session_opening_on_its_resident_model_takes_that_models_context() -> None:
    @asynccontextmanager
    async def settled_for(what: str):  # type: ignore[no-untyped-def]
        yield

    resident = SimpleNamespace(model_id=MODEL, max_model_len=14_247)
    ctx = SimpleNamespace(
        residency=SimpleNamespace(settled_for=settled_for, resident_model=resident)
    )
    session = QueueSession(
        act="generate", client="briefcase", model=MODEL, idle_s=60, max_wait_s=60,
        created=None,  # type: ignore[arg-type]
    )
    assert asyncio.run(_resident(ctx, session)) is True  # type: ignore[arg-type]
    assert session.context == 14_247

    other = QueueSession(
        act="generate", client="briefcase", model="qwen3.5-9b", idle_s=60, max_wait_s=60,
        created=None,  # type: ignore[arg-type]
    )
    assert asyncio.run(_resident(ctx, other)) is False  # type: ignore[arg-type]
    assert other.context is None

