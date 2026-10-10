"""The rerank verb: one query and a list of documents to a relevance probability per
document, read off the decision door's likelihood machinery (crucible/decide_likelihood.py)
on every engine that scores candidates.

Each document is a group of two candidate replies, the model's "yes" and "no", after a
prompt that asks whether the document meets the query. Its score is P(yes) normalised
against P(no): the softmax of the two replies' log-probabilities, which is exactly what
Qwen3-Reranker's model card computes (log_softmax over the last position's logits for
"no" and "yes", the exp of yes's), since the log-probabilities differ from the logits by
one shared constant. A probability, so the same model's scores compare across calls and a
fixed cutoff means the same thing every time.

The prompt is Crucible's, never the app's: a dedicated reranker's own prompt from its
manifest ([rerank]; written as text and tokenized as it is, the prompt form), or, for any
decide model, Crucible's general template (chat turns rendered by the model's own chat
template, the chat form). Either way the instruction and the query come first and are
shared by every document, so an engine that keeps a cache reads them once: the Mac's
items route keeps them as its state, llama-server's slot reuses its cached prefix for each
next document. vLLM reads prompt log-probabilities, which never read its prefix cache, so
there every document's request reads the query again (engines/vllm.py)."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .decide_items import EngineCall
from .decide_likelihood import GroupScore, Scoring, score_groups, softmax
from .embed import engine_build, fingerprint_of
from .errors import ApiError
from .manifests import ModelManifest
from .queuerequest import QueueChoice, queue_field
from .verbspec import fill

MAX_DOCUMENTS = 256
"""Documents one request may carry."""

RERANK_SCHEME = 1
"""Crucible's own scoring: the two replies, the softmax over them. A change is a new
fingerprint."""

GENERAL_TEMPLATE = "crucible-general-1"
"""Crucible's template for a model with no rerank prompt of its own: the dedicated
reranker's wording as chat turns, the instruction and the query in the system turn (so
they are the shared state), the document as the user turn."""

GENERAL_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "\n<Instruct>: {instruction}\n<Query>: {query}"
)

GENERAL_USER = "<Document>: {document}"

GENERAL_YES = "yes"
GENERAL_NO = "no"

GENERAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

MODEL_TEMPLATE = "model"

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class RerankRequest(_Strict):
    """`POST /v1/rerank`: how relevant each document is to one query."""

    query: _NonEmpty
    """What the documents are judged against."""
    documents: Annotated[list[_NonEmpty], Field(min_length=1, max_length=MAX_DOCUMENTS)]
    """The documents, in order; at most 256. A document and the query together may be
    at most the served context in tokens; a longer one is refused
    `item_prompt_too_long` naming it."""
    instruction: _NonEmpty | None = None
    """What relevant means here, in one sentence ("Given a question about a podcast,
    find the transcript passages that answer it"). Absent: the model's default (the
    reply says which was used)."""
    model: _NonEmpty | None = None
    """The Crucible model id: a dedicated reranker, or any decide model (qwen3.5-9b and
    the like), which is judged with Crucible's general template. Absent: the model this
    server registered for rerank (`GET /v1/capability`)."""
    form: _NonEmpty | None = None
    """Which form of `model`, for a model whose block here states more than one."""
    max_params_b: Annotated[float, Field(gt=0)] | None = None
    """A ceiling in billions of parameters: the biggest model that fits this server at
    or below it. Ignored when `model` names one."""
    queue: QueueChoice = queue_field()
    """Absent: while the model is not resident, or its engine is busy, the request waits
    in the server's line up to an hour and the model is loaded for it. `{"max_wait_s":
    N}` changes the wait; `false` refuses at once."""


class RerankModel(_Strict):
    """What judged the documents."""

    id: str
    revision: str
    file: str | None
    form: str | None
    engine: str
    engine_build: str
    template: str
    """`model`: the reranker's own prompt. `crucible-general-1`: Crucible's general
    template, for a decide model."""
    fingerprint: str
    """Scores are comparable across calls with the same fingerprint."""


class RerankResult(_Strict):
    index: int
    """The document's position in the request."""
    relevance_score: float
    """P(yes) / (P(yes) + P(no)), from 0 to 1."""


class RerankTiming(_Strict):
    total: float
    """The run, ms, from leaving the server's line to the answer."""
    queued: float | None = None
    """Ms the request waited in the server's line."""


class RerankTokens(_Strict):
    per_document: list[int]
    """Each document's context: the prompt up to where the reply opens."""
    total: int
    """Every document's prompt with its query, once per candidate (yes and no), as
    llama-server is sent it: cached ones included."""
    cached: int | None
    """Of those, what no pass read again (the query shared by every document, a held
    cache), so `total - cached` is what the engine read; null when it did not say."""


class RerankResponse(_Strict):
    """A relevance probability per document, in document order, and the same sorted."""

    object: Literal["crucible.rerank"] = "crucible.rerank"
    model: RerankModel
    instruction: str
    """The instruction the documents were judged under."""
    scores: list[float]
    """Each document's relevance, in the request's order."""
    results: list[RerankResult]
    """Every document, most relevant first (a tie keeps the request's order)."""
    tokens: RerankTokens
    timing_ms: RerankTiming


def instruction_of(body: RerankRequest, manifest: ModelManifest) -> str:
    if body.instruction is not None:
        return body.instruction
    if manifest.rerank is not None:
        return manifest.rerank.default_instruction
    return GENERAL_INSTRUCTION


def scoring(body: RerankRequest, manifest: ModelManifest) -> Scoring:
    """The documents as likelihood groups: the reranker's own prompt (the prompt form),
    or Crucible's general template (the chat form)."""
    instruction = instruction_of(body, manifest)
    spec = manifest.rerank
    if spec is not None:
        return Scoring(
            messages=None,
            prompt=fill(spec.prefix, {"instruction": instruction, "query": body.query}),
            questions=[fill(spec.document, {"document": text}) for text in body.documents],
            candidates=[[spec.yes, spec.no] for _ in body.documents],
        )
    system = fill(GENERAL_SYSTEM, {"instruction": instruction, "query": body.query})
    return Scoring(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": ""}],
        prompt=None,
        questions=[fill(GENERAL_USER, {"document": text}) for text in body.documents],
        candidates=[[GENERAL_YES, GENERAL_NO] for _ in body.documents],
    )


def refuse_unscorable(model: str, manifest: ModelManifest, engine: str, reading: Any) -> None:
    """An engine that scores no candidates, or a reranker whose own prompt the engine's
    route cannot take as text, is refused by name: the model and the engine do not
    serve rerank here."""
    if reading.route is None:
        raise ApiError(
            400,
            "rerank_unsupported_on_engine",
            f"{model!r} runs on {engine}, which cannot score a reply: {reading.basis}",
            {"model": model, "engine": engine},
        )
    if manifest.rerank is not None and not reading.prompt:
        raise ApiError(
            400,
            "rerank_unsupported_on_engine",
            f"{model!r} is a reranker with its own prompt, and {engine} scores chat turns "
            f"only: {reading.prompt_basis}",
            {"model": model, "engine": engine},
        )


def identity(resident: Any, manifest: ModelManifest, backend_kind: str) -> RerankModel:
    spec = manifest.spec(backend_kind, resident.form)
    build = engine_build(resident.engine, backend_kind)
    template = MODEL_TEMPLATE if manifest.rerank is not None else GENERAL_TEMPLATE
    return RerankModel(
        id=resident.model_id,
        revision=resident.revision,
        file=spec.file,
        form=resident.form,
        engine=resident.engine,
        engine_build=build,
        template=template,
        fingerprint=fingerprint_of(
            resident.model_id, resident.revision, spec.file, build,
            f"r{RERANK_SCHEME}-{template}",
        ),
    )


def by_document(error: ApiError) -> ApiError:
    """A refusal the engine made about a group, said with the document's index."""
    details = dict(error.details or {})
    group = details.get("group")
    if not isinstance(group, int):
        return error
    details["document"] = group
    return ApiError(error.status_code, error.code, f"document {group}: {error.message}", details)


def relevance(group: GroupScore) -> float:
    """P(yes) normalised against P(no): the softmax of the two replies' totals."""
    yes, no = (sum(row) for row in group.rows)
    return softmax([yes, no])[0]


def answer(
    groups: list[GroupScore], model: RerankModel, instruction: str, started: float
) -> RerankResponse:
    scores = [relevance(group) for group in groups]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    cached = [group.timing.cached_tokens for group in groups]
    return RerankResponse(
        model=model,
        instruction=instruction,
        scores=scores,
        results=[RerankResult(index=index, relevance_score=scores[index]) for index in order],
        tokens=RerankTokens(
            per_document=[group.context_tokens for group in groups],
            total=sum(group.prompt_tokens for group in groups),
            cached=None if any(c is None for c in cached) else sum(c for c in cached if c is not None),
        ),
        timing_ms=RerankTiming(total=round((time.perf_counter() - started) * 1000.0, 1)),
    )


async def rerank_on_engine(
    call: EngineCall,
    resident: Any,
    route: str,
    body: RerankRequest,
    manifest: ModelManifest,
    model: RerankModel,
    *,
    concurrency: int,
) -> RerankResponse:
    started = time.perf_counter()
    groups = await score_groups(
        call, resident, scoring(body, manifest), route,
        concurrency=concurrency, refusal=by_document,
    )
    return answer(groups, model, instruction_of(body, manifest), started)


__all__ = [
    "GENERAL_TEMPLATE",
    "MAX_DOCUMENTS",
    "RERANK_SCHEME",
    "RerankModel",
    "RerankRequest",
    "RerankResponse",
    "RerankResult",
    "answer",
    "by_document",
    "identity",
    "refuse_unscorable",
    "relevance",
    "rerank_on_engine",
    "scoring",
]
