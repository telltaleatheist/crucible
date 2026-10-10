from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from ... import embed as embed_core
from ... import verbmodel
from ...embed import EmbedRequest, EmbedResponse
from ...engines import embed_reading
from ...errors import ApiError
from ...formrequest import refuse_unknown_form
from ...jobs.llm import MANIFESTS
from ...manifests import load_manifest
from ...queuerequest import QueueChoice, queue_field
from ..context import AppContext, Routers
from ..verbcall import serve_verb
from .decide import engine_call

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class OpenAIEmbeddingRequest(BaseModel):
    """`POST /v1/openai/embeddings` (and `/openai/v1/embeddings`): OpenAI's embeddings
    request, so an OpenAI client works unchanged, with Crucible's members beside it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    input: _NonEmpty | Annotated[list[_NonEmpty], Field(min_length=1, max_length=embed_core.MAX_EMBED_INPUTS)]
    """One text or a list of texts. Token arrays are refused (`400 invalid_request`):
    the model's format is written around text, by Crucible."""
    model: _NonEmpty | None = None
    """A Crucible model id; absent, the model this server registered for embed. An
    OpenAI model name (`text-embedding-3-small`) is refused `model_not_for_verb`."""
    encoding_format: Literal["float", "base64", "base64_float16"] = "float"
    """`float` or `base64` (float32 little-endian), as OpenAI's; `base64_float16` is
    Crucible's."""
    dimensions: Annotated[int, Field(ge=1)] | None = None
    """As on `POST /v1/embed`."""
    user: str | None = None
    """OpenAI's end-user tag. Read and not used."""
    input_type: Literal["query", "document"] = "document"
    """Crucible's: `query` writes the model's instruction prefix. Absent: `document`,
    the text as it is, which is what an OpenAI client means."""
    instruction: _NonEmpty | None = None
    """Crucible's: as on `POST /v1/embed`."""
    fingerprint: _NonEmpty | None = None
    """Crucible's: as on `POST /v1/embed`."""
    form: _NonEmpty | None = None
    """Crucible's: which form of `model`, for a model whose block states several."""
    max_params_b: Annotated[float, Field(gt=0)] | None = None
    """Crucible's: a ceiling in billions of parameters, when no model is named."""
    queue: QueueChoice = queue_field()
    """Crucible's: absent, the request waits in the server's line; `false` refuses at
    once."""

    def native(self) -> EmbedRequest:
        try:
            return EmbedRequest(
                inputs=[self.input] if isinstance(self.input, str) else list(self.input),
                input_type=self.input_type,
                instruction=self.instruction,
                model=self.model,
                form=self.form,
                fingerprint=self.fingerprint,
                max_params_b=self.max_params_b,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format,
                # Left out when it was left out: an explicit null is refused.
                **({} if self.queue is None else {"queue": self.queue}),
            )
        except ValidationError as exc:
            raise ApiError(
                400,
                "invalid_request",
                f"the request cannot be read as an embedding: {exc}",
            ) from None


def openai_shape(answer: EmbedResponse) -> dict[str, Any]:
    """OpenAI's list of embeddings, with what wrote them under `crucible`."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(answer.embeddings)
        ],
        "model": answer.model.id,
        "usage": {"prompt_tokens": answer.tokens.total, "total_tokens": answer.tokens.total},
        "crucible": {
            "model": answer.model.model_dump(mode="json"),
            "dimensions": answer.dimensions,
            "input_type": answer.input_type,
            "instruction": answer.instruction,
            "encoding_format": answer.encoding_format,
            "timing_ms": answer.timing_ms.model_dump(mode="json"),
        },
    }


async def _resolve(ctx: AppContext, body: EmbedRequest) -> verbmodel.VerbModel:
    manifests = await asyncio.to_thread(MANIFESTS.all)
    named = embed_core.named_by_fingerprint(body, manifests, ctx.backend.kind)
    if named.model is not None:
        refuse_unknown_form(named.model, named.form, ctx.backend.kind)
    resident = ctx.residency.resident_model
    return verbmodel.resolve(
        "embed", ctx.config, ctx.backend,
        model=named.model, form=named.form, max_params_b=body.max_params_b,
        resident=None if resident is None else resident.model_id,
    )


async def embed_call(
    request: Request, ctx: AppContext, body: EmbedRequest, render: Any
) -> Response:
    chosen = await _resolve(ctx, body)
    manifest = load_manifest(chosen.model)
    spec = manifest.embed
    assert spec is not None, f"{chosen.model} serves embed and has no [embed]"
    embed_core.refuse_unfit_request(body, spec, chosen.model)
    backend_kind = ctx.backend.kind

    def prepare(resident: Any, concurrency: int) -> Any:
        reading = embed_reading(resident.engine)
        if reading.route is None:
            raise ApiError(
                400,
                "embed_unsupported_on_engine",
                f"{resident.model_id!r} runs on {resident.engine}, which writes no "
                f"vectors here: {reading.basis}",
                {"model": resident.model_id, "engine": resident.engine},
            )
        model = embed_core.identity(resident, manifest, backend_kind)
        embed_core.refuse_other_fingerprint(body.fingerprint, model)
        return embed_core.embed_on_engine(
            engine_call(ctx.http, resident), resident, reading.route, body, spec, model
        )

    return await serve_verb(
        request, ctx, kind="embed", what="an embedding", model=chosen.model,
        form=chosen.form, queue=body.queue, prepare=prepare, render=render,
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai

    @private.post(
        "/embed",
        response_model=None,
        responses={200: {"model": EmbedResponse}},
    )
    async def embed(request: Request, body: EmbedRequest) -> Response:
        """Unit-length vectors for a list of texts, each answer naming exactly what
        wrote them (`model.fingerprint`). Vectors are comparable only with vectors of
        the same fingerprint: store it with them and send it on later calls, and a
        server that would write anything else refuses `fingerprint_mismatch` instead
        of answering. `input_type` says whether the texts are queries (written with the
        model's instruction prefix) or documents (written as they are). The model's
        format, its pooling and the normalisation are Crucible's (the model manifest's
        [embed]). The model is the request's `model` (or its fingerprint's), else the
        biggest under `max_params_b` that fits, else what this server registered for
        embed; its models are the optional retrieval package (`crucible install
        retrieval`), refused `package_not_installed` where it is not installed. A
        request whose model is not resident waits in the server's line and the model is
        loaded for it; with `"queue": false` it is refused at once instead."""
        return await embed_call(request, ctx, body, lambda answer: answer.model_dump(mode="json"))

    @private.post("/openai/embeddings")
    @openai.post("/embeddings")
    async def openai_embeddings(request: Request, body: OpenAIEmbeddingRequest) -> Response:
        """OpenAI's embeddings: `{object: "list", data: [{embedding, index}], model,
        usage}`, unit-length vectors, with what wrote them under `crucible` (its
        `model.fingerprint` is what to store beside them; docs/internals/api.md
        "Embed"). `input_type` defaults to `document`; a search query is sent with
        `"input_type": "query"`, which OpenAI's API does not have."""
        return await embed_call(request, ctx, body.native(), openai_shape)


__all__ = ["OpenAIEmbeddingRequest", "embed_call", "openai_shape", "register"]
