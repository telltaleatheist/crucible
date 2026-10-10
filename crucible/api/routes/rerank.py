from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from ... import rerank as rerank_core
from ... import verbmodel
from ...engines import likelihood_reading
from ...errors import ApiError
from ...formrequest import refuse_unknown_form
from ...manifests import load_manifest
from ...queuerequest import QueueChoice, queue_field
from ...rerank import RerankRequest, RerankResponse
from ..context import AppContext, Routers
from ..verbcall import serve_verb
from .decide import engine_call

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class TextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: _NonEmpty


class CompatRerankRequest(BaseModel):
    """`POST /v1/openai/rerank` (and `/openai/v1/rerank`): the Cohere and Jina rerank
    request, so their clients work unchanged, with Crucible's members beside it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    query: _NonEmpty
    """What the documents are judged against."""
    documents: Annotated[
        list[_NonEmpty | TextDocument], Field(min_length=1, max_length=rerank_core.MAX_DOCUMENTS)
    ]
    """Strings, or `{"text": ...}` objects (Cohere's and Jina's)."""
    model: _NonEmpty | None = None
    """A Crucible model id: a reranker or any decide model; absent, the model this server
    registered for rerank."""
    top_n: Annotated[int, Field(ge=1)] | None = None
    """How many results, most relevant first. Absent: every document."""
    return_documents: bool = False
    """Each result carries its document's text under `document.text`."""
    instruction: _NonEmpty | None = None
    """Crucible's: as on `POST /v1/rerank`."""
    form: _NonEmpty | None = None
    """Crucible's: which form of `model`, for a model whose block states several."""
    max_params_b: Annotated[float, Field(gt=0)] | None = None
    """Crucible's: a ceiling in billions of parameters, when no model is named."""
    queue: QueueChoice = queue_field()
    """Crucible's: absent, the request waits in the server's line; `false` refuses at
    once."""

    def texts(self) -> list[str]:
        return [doc if isinstance(doc, str) else doc.text for doc in self.documents]

    def native(self) -> RerankRequest:
        try:
            return RerankRequest(
                query=self.query,
                documents=self.texts(),
                instruction=self.instruction,
                model=self.model,
                form=self.form,
                max_params_b=self.max_params_b,
                # Left out when it was left out: an explicit null is refused.
                **({} if self.queue is None else {"queue": self.queue}),
            )
        except ValidationError as exc:
            raise ApiError(
                400, "invalid_request", f"the request cannot be read as a rerank: {exc}"
            ) from None


def compat_shape(body: CompatRerankRequest) -> Any:
    texts = body.texts()

    def render(answer: RerankResponse) -> dict[str, Any]:
        results = answer.results[: body.top_n] if body.top_n is not None else answer.results
        return {
            "id": f"rerank-{uuid.uuid4().hex}",
            "model": answer.model.id,
            "results": [
                {
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    **({"document": {"text": texts[result.index]}} if body.return_documents else {}),
                }
                for result in results
            ],
            "usage": {"prompt_tokens": answer.tokens.total, "total_tokens": answer.tokens.total},
            "crucible": {
                "model": answer.model.model_dump(mode="json"),
                "instruction": answer.instruction,
                "scores": answer.scores,
                "timing_ms": answer.timing_ms.model_dump(mode="json"),
            },
        }

    return render


async def rerank_call(
    request: Request, ctx: AppContext, body: RerankRequest, render: Any
) -> Response:
    if body.model is not None:
        refuse_unknown_form(body.model, body.form, ctx.backend.kind)
    resident = ctx.residency.resident_model
    chosen = verbmodel.resolve(
        "rerank", ctx.config, ctx.backend,
        model=body.model, form=body.form, max_params_b=body.max_params_b,
        resident=None if resident is None else resident.model_id,
    )
    manifest = load_manifest(chosen.model)
    backend_kind = ctx.backend.kind
    engine = manifest.spec(backend_kind).engine
    rerank_core.refuse_unscorable(chosen.model, manifest, engine, likelihood_reading(engine))

    def prepare(resident: Any, concurrency: int) -> Any:
        reading = likelihood_reading(resident.engine)
        rerank_core.refuse_unscorable(resident.model_id, manifest, resident.engine, reading)
        assert reading.route is not None
        model = rerank_core.identity(resident, manifest, backend_kind)
        return rerank_core.rerank_on_engine(
            engine_call(ctx.http, resident), resident, reading.route, body, manifest, model,
            concurrency=concurrency,
        )

    return await serve_verb(
        request, ctx, kind="rerank", what="a rerank", model=chosen.model,
        form=chosen.form, queue=body.queue, prepare=prepare, render=render,
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai

    @private.post(
        "/rerank",
        response_model=None,
        responses={200: {"model": RerankResponse}},
    )
    async def rerank(request: Request, body: RerankRequest) -> Response:
        """A relevance probability per document for one query: P(yes) normalised
        against P(no) under the model's prompt, so the same model's scores compare
        across calls and a fixed cutoff means the same thing every time. `scores` are
        in document order; `results` the same, most relevant first. The prompt is
        Crucible's: a dedicated reranker's own (its manifest's [rerank]), or for any
        decide model Crucible's general template; the instruction and the query are read
        once and shared by every document where the engine keeps a cache (llama-server,
        mlx-lm). The model is the request's `model`, else the biggest under
        `max_params_b` that fits, else what this server registered for rerank; the
        dedicated rerankers are the optional retrieval package (`crucible install
        retrieval`, refused `package_not_installed` where it is not installed), and a
        decide model named in the request reranks wherever it decides. A request whose
        model is not resident waits in the server's line and the model is loaded for
        it; with `"queue": false` it is refused at once instead."""
        return await rerank_call(request, ctx, body, lambda answer: answer.model_dump(mode="json"))

    @private.post("/openai/rerank")
    @openai.post("/rerank")
    async def compat_rerank(request: Request, body: CompatRerankRequest) -> Response:
        """Cohere's and Jina's rerank: `{results: [{index, relevance_score, document?}],
        model, usage}`, most relevant first, `top_n` of them, with what judged them and
        every score in document order under `crucible`."""
        return await rerank_call(request, ctx, body.native(), compat_shape(body))


__all__ = ["CompatRerankRequest", "compat_shape", "register", "rerank_call"]
