"""`GET /v1/resolve`: which model a request would be served by, before it is sent (Owen,
2026-10-11: "we should have some way to tell which model will be used before crucible
picks it").

A preview, read-only: nothing is loaded, reserved, pulled or queued, and no GPU is
touched. It calls the very functions the doors call to choose (the chat door's
`verbmodel.chat_model`, embed's and rerank's `verbmodel.resolve`, decide's
`decide_model`, the image job's `jobs.resolve_model`), never a copy of them, so the
preview and the real call cannot disagree: what the door would refuse while choosing,
the preview answers under `refused` with the same status, code and message.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from fastapi import Query

from ... import jobs as jobs_module
from ... import playground, rerank, upstreamrecord, verbmodel
from ...capabilityclasses import BY_NAME
from ...classnames import CHAT_VERBS, DEFAULT_CHAT_VERB
from ...decide import check_image_count
from ...engines import likelihood_reading
from ...errors import ApiError
from ...jobs import image as image_job
from ...jobs.llm import model_rows
from ...manifests import ManifestError, load_manifest
from ...memorybudget import available_bytes
from ..context import AppContext, Routers
from .decide import decide_model
from .openai import refuse_unservable

IMAGE = image_job.JOB_TYPE

PREVIEWED: tuple[str, ...] = (*CHAT_VERBS, "decide", "embed", "rerank", IMAGE)
"""The verbs the preview answers for: every one whose door chooses a model."""

DOORS: dict[str, str] = {
    **{verb: "POST /v1/openai/chat/completions" for verb in CHAT_VERBS},
    "decide": "POST /v1/decide",
    "embed": "POST /v1/embed",
    "rerank": "POST /v1/rerank",
    IMAGE: "POST /v1/jobs (type image)",
}


def _ceiling(raw: str | None) -> float | None:
    """The query's `max_params_b`, read as the doors read the body's: through
    `verbmodel.take_ceiling`, so a value they refuse is refused here the same way."""
    if raw is None:
        return None
    try:
        number: Any = float(raw)
    except ValueError:
        number = raw
    if isinstance(number, float) and not math.isfinite(number):
        number = raw
    return verbmodel.take_ceiling({verbmodel.CEILING: number})


def _images(raw: str | None, verb: str) -> int:
    if raw is None:
        return 0
    if verb != "decide":
        raise ApiError(
            400,
            "invalid_request",
            f"`images` is the count of images a decision carries; {verb} takes none",
            {"verb": verb},
        )
    try:
        count = int(raw)
    except ValueError:
        count = -1
    if count < 0:
        raise ApiError(
            400, "invalid_request",
            f"`images` is how many images the decision carries, a whole number; got {raw!r}",
            {"images": raw},
        )
    return check_image_count(["image"] * count)


def _no_ceiling(verb: str, ceiling: float | None) -> None:
    """The image job takes no ceiling: a body that sent one would be refused as a field
    it does not have. (decide takes one, as chat does.)"""
    if ceiling is None:
        return
    if verb == IMAGE:
        verbmodel.refuse_unsized_verb(verb, "this request's `max_params_b`")
    raise ApiError(
        400,
        "invalid_request",
        f"the {verb} door takes no `max_params_b`: name a `model`, or none for the one "
        "this server registered",
        {"verb": verb},
    )


def _choose(
    ctx: AppContext, verb: str, model: str | None, form: str | None,
    ceiling: float | None, images: int, verb_stated: bool,
) -> tuple[str, str]:
    """(model, source): the door's own choice, every refusal it makes while choosing
    raised as it raises it."""
    config, backend = ctx.config, ctx.backend
    resident = ctx.residency.resident_model
    resident_id = None if resident is None else resident.model_id
    if verb in CHAT_VERBS:
        chosen = verbmodel.chat_model(
            config, backend, model=model, verb=verb if verb_stated else None, form=form,
            max_params_b=ceiling, resident=resident_id,
        )
        if upstreamrecord.split_model(chosen.model) is None:
            refuse_unservable(chosen.model, form, backend.kind)
        return chosen.model, verbmodel.source_of(chosen, verb, config)
    if verb in ("embed", "rerank"):
        chosen = verbmodel.resolve(
            verb, config, backend, model=model, form=form, max_params_b=ceiling,
            resident=resident_id,
        )
        if verb == "rerank":
            manifest = load_manifest(chosen.model)
            engine = manifest.spec(backend.kind).engine
            rerank.refuse_unscorable(chosen.model, manifest, engine, likelihood_reading(engine))
        return chosen.model, verbmodel.source_of(chosen, verb, config)
    if verb == "decide":
        picked = decide_model(config, backend, model, form, images, ceiling)
        if ceiling is not None:
            chosen = verbmodel.resolve(
                verb, config, backend, model=None, form=form, max_params_b=ceiling,
                resident=resident_id,
            )
            return picked, verbmodel.source_of(chosen, verb, config)
    else:
        _no_ceiling(verb, ceiling)
        plugin = jobs_module.resolve(ctx.store.registry, IMAGE, config)
        picked = jobs_module.resolve_model(plugin, model, {})
        if picked is None:
            raise ApiError(
                409, "capability_disabled",
                "this server's image job offers no model on this backend",
                {"capability": IMAGE},
            )
    if model is not None:
        return picked, verbmodel.MODEL
    return picked, verbmodel.SETTINGS if config.local_model(verb) == picked else verbmodel.AUTO


def _text_facts(ctx: AppContext, verb: str, model: str, form: str | None) -> dict[str, Any]:
    """A text model's facts: its GET /v1/models row (form, forms, context, estimate,
    installed, resident, loadable) and its fit for the verb's working context."""
    try:
        manifest = load_manifest(model)
    except ManifestError:
        return {"installed": None, "resident": None, "fit": None, "facts": None}
    row = None
    if ctx.config.enable_llm:
        row = next(
            (r for r in model_rows(ctx.config, ctx.backend, ctx.residency) if r["id"] == model),
            None,
        )
    spec = manifest.spec(ctx.backend.kind) if manifest.supports(ctx.backend.kind) else None
    return {
        "params_b": manifest.params_b,
        "bits": None if spec is None else spec.bits,
        "engine": None if spec is None else spec.engine,
        "form": form if form is not None else (None if row is None else row.get("form")),
        "installed": None if row is None else row["installed"],
        "resident": None if row is None else row["resident"],
        "fit": verbmodel.fit_of(verb, ctx.config, ctx.backend, model, form),
        "facts": row,
    }


def _image_facts(ctx: AppContext, model: str) -> dict[str, Any]:
    """An image model's facts: the playground page's (native size, quality floor,
    defaults, what a negative prompt costs) and where it stands here."""
    manifest = image_job.MANIFESTS.all().get(model)
    if manifest is None or not manifest.supports(ctx.backend.kind):
        return {"installed": None, "resident": None, "fit": None, "facts": None}
    plugin = ctx.store.registry.get(IMAGE)
    row = None if plugin is None else next(
        (d for d in plugin.describe_models() if d.id == model), None
    )
    budget = available_bytes(ctx.backend.gpu.vram_bytes, ctx.config.desktop_allowance_bytes)
    need = None if row is None else row.vram_bytes
    return {
        "params_b": None,
        "bits": None,
        "engine": None,
        "form": None,
        "installed": None if row is None else row.installed,
        "resident": None if row is None else row.resident,
        "fit": None if need is None else {
            "fits": need <= budget, "need_bytes": need, "available_bytes": budget,
            "context_tokens": None, "concurrency": None,
        },
        "facts": playground.image_facts(manifest, manifest.spec(ctx.backend.kind)),
    }


def preview(ctx: AppContext, query: dict[str, str | None]) -> dict[str, Any]:
    raw_verb = query.get("verb")
    verb_stated = raw_verb is not None
    verb = DEFAULT_CHAT_VERB if raw_verb is None else raw_verb
    model = query.get("model")
    form = query.get("form")
    answer: dict[str, Any] = {
        "verb": verb,
        "door": DOORS.get(verb),
        "request": {
            "model": model, "verb": raw_verb, "max_params_b": query.get("max_params_b"),
            "form": form, "images": query.get("images"),
        },
        "model": None,
        "source": None,
        "params_b": None,
        "bits": None,
        "engine": None,
        "form": None,
        "installed": None,
        "resident": None,
        "fit": None,
        "facts": None,
        "refused": None,
    }
    try:
        if verb not in PREVIEWED:
            if verb in BY_NAME:
                raise ApiError(
                    400, "verb_not_previewed",
                    f"{verb!r} chooses no model at a door this preview reads; the verbs it "
                    f"answers for are {list(PREVIEWED)}",
                    {"verb": verb, "verbs": list(PREVIEWED)},
                )
            verbmodel.refuse_unchattable_verb(verb)
        if model == "":
            raise ApiError(400, "model_required", "`model` is a model id, or left out")
        ceiling = _ceiling(query.get("max_params_b"))
        images = _images(query.get("images"), verb)
        chosen, source = _choose(ctx, verb, model, form, ceiling, images, verb_stated)
    except ApiError as refusal:
        answer["refused"] = {
            "status": refusal.status_code,
            "code": refusal.code,
            "message": refusal.message,
            "details": refusal.details,
        }
        return answer
    answer["model"], answer["source"] = chosen, source
    if upstreamrecord.split_model(chosen) is not None:
        return answer
    answer.update(
        _image_facts(ctx, chosen) if verb == IMAGE else _text_facts(ctx, verb, chosen, form)
    )
    return answer


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/resolve")
    async def resolve_preview(
        verb: str | None = Query(None, description=(
            "The verb whose door is previewed: `generate` (the default: a chat), `clean`, "
            "`translate`, `simplify`, `analysis` (a chat naming that verb), `decide`, "
            "`embed`, `rerank`, `image`.")),
        model: str | None = Query(None, description="As the request's `model`."),
        max_params_b: str | None = Query(None, description=(
            "As the request's `max_params_b`, read the same way (a number above 0); "
            "`decide` and `image` take none.")),
        form: str | None = Query(None, description="As the request's `form`."),
        images: str | None = Query(None, description=(
            "For `decide`: how many images the decision carries (with images it may be "
            "served by a vision form).")),
    ) -> dict[str, Any]:
        """Which model a request would be served by, without sending it: nothing is
        loaded, reserved or queued. The query is the request's own choosing fields:
        `verb` (default `generate`, a chat; also `clean`, `translate`, `simplify`,
        `analysis`, `decide`, `embed`, `rerank`, `image`), `model`, `max_params_b`,
        `form`, and for `decide` `images` (how many the decision carries). The answer
        names the `model`, its `source` (`model`, `max_params_b`, `settings`, `auto`), its
        `params_b`, `bits`, `engine` and `form`, whether it is `installed` and `resident`,
        its `fit` for the verb's working context (`need_bytes`, `available_bytes`), and its
        `facts`: a text model's GET /v1/models row, an image model's playground facts
        (native size, `quality_min_side`, defaults, negative prompt). A request the door
        would refuse while choosing answers 200 with `refused` holding the same `status`,
        `code`, `message` and `details` the door would send, and `model` null. It reads
        what the door reads before anything waits; what a door refuses later, about the
        body itself (a schema, a prefill), it does not."""
        query = {"verb": verb, "model": model, "max_params_b": max_params_b, "form": form,
                 "images": images}
        return await asyncio.to_thread(preview, ctx, query)


__all__ = ["PREVIEWED", "preview", "register"]
