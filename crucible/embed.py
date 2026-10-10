"""The embed verb: texts to unit-length vectors, with the model that wrote them named.

Vectors from different models are not comparable, nor are vectors from the same model in
another precision, another engine build or another version of Crucible's own reading of
it (two runtime versions of one quantised model gave cosines of 0.97 to 0.99 against each
other, Briefcase 2026-10). So every answer carries a FINGERPRINT that changes whenever
anything that changes the floats does: the model id, the revision of the weights, the
file read, the engine and its build, and Crucible's embedding scheme. An app that stores
vectors stores the fingerprint beside them and sends it on every later call; this server
then serves exactly that or refuses by name (`fingerprint_mismatch`), and never
substitutes another model.

The input format is the model's (its manifest's [embed]; crucible/verbspec.py): a query
takes the model's instruction prefix, a document none, and the app says which it sends
(`input_type`) and, optionally, the instruction. The vector is the model's pooling,
truncated to `dimensions` where the model is trained for that (Matryoshka), and
normalised to unit length here, by Crucible, after the truncation, whatever the engine
returned.
"""

from __future__ import annotations

import base64
import math
import struct
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from . import hosttools, jobenv
from .backend import CUDA_LINUX
from .decide_items import EngineCall
from .engines.items_forward import EMBED_INPUT_TOO_LONG, ITEMS_PATH
from .errors import ApiError
from .llamacpp import LLAMA_CPP_RELEASE
from .manifests import ManifestError, ModelManifest, load_manifest
from .queuerequest import QueueChoice, queue_field
from .verbspec import EmbedSpec

MAX_EMBED_INPUTS = 256
"""Inputs one request may carry. A bigger corpus is many requests, in one queue session so
the model stays on the card between them (docs/internals/api.md "Embed")."""

EMBED_SCHEME = 1
"""Crucible's own reading of a vector: how the input is rendered and tokenized, which
hidden state is pooled, and how it is truncated and normalised. A change to any of them
is a new number, and so a new fingerprint."""

ENCODINGS = ("float", "base64", "base64_float16")

TOKENIZE_PATH = "/tokenize"

EMBEDDINGS_PATH = "/v1/embeddings"

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class EmbedRequest(_Strict):
    """`POST /v1/embed`: texts to unit-length vectors, each named by the model that wrote it."""

    inputs: Annotated[list[_NonEmpty], Field(min_length=1, max_length=MAX_EMBED_INPUTS)]
    """The texts, in order; at most 256 a request. Each may be at most the served
    context in tokens (`GET /v1/models`, the row's `embed.max_input_tokens`), once the
    model's format is around it; a longer one is refused `embed_input_too_long` naming
    it, before anything is embedded."""
    input_type: Literal["query", "document"]
    """`query`: written with the model's instruction prefix (Qwen3-Embedding: "Instruct:
    <instruction>\\nQuery:<text>"). `document`: written as it is. Retrieval embeds the
    corpus as documents and the searches as queries; the two are meant to be compared."""
    instruction: _NonEmpty | None = None
    """What the query is for, in one sentence ("Given a podcast transcript passage,
    find the passages that discuss the same topic"). Absent: the model's default
    instruction (the reply says which was used). Refused `instruction_not_taken` on a
    `document`, or by a model whose queries take none."""
    model: _NonEmpty | None = None
    """The Crucible model id. An app that stores vectors names the model that wrote them
    (better: sends `fingerprint`). One that does not fit this server is refused
    `model_does_not_fit`, never swapped for another. Absent: the model this server
    registered for embed (`GET /v1/capability`)."""
    form: _NonEmpty | None = None
    """Which form of `model`, for a model whose block here states more than one."""
    fingerprint: _NonEmpty | None = None
    """An answer's `model.fingerprint`, sent back: this server serves exactly that
    identity or refuses `fingerprint_mismatch` (409) naming what it would serve
    instead, so stored vectors are never mixed with incomparable ones. Implies `model`
    (and `form`); sending them as well, they must agree."""
    max_params_b: Annotated[float, Field(gt=0)] | None = None
    """A ceiling in billions of parameters: the biggest embedding model that fits this
    server at or below it (docs/VERB-SIZING.md rule 8). Ignored when `model` or
    `fingerprint` names one."""
    dimensions: Annotated[int, Field(ge=1)] | None = None
    """The vector's length: a prefix of the model's own, normalised again (Matryoshka).
    Any size the model is trained for (`GET /v1/models`, `embed.dimensions_range`; 32
    to 4096 on Qwen3-Embedding-8B); refused `dimensions_not_supported` otherwise.
    Absent: the model's own length."""
    encoding_format: Literal["float", "base64", "base64_float16"] = "float"
    """`float`: JSON arrays of numbers. `base64`: each vector's float32 values,
    little-endian, base64 (OpenAI's `base64`). `base64_float16`: IEEE half precision,
    little-endian, base64: half the bytes, about three decimal digits."""
    queue: QueueChoice = queue_field()
    """Absent: while the model is not resident, or its engine is busy, the request waits
    in the server's line up to an hour and the model is loaded for it. `{"max_wait_s":
    N}` changes the wait; `false` refuses at once (`409 model_not_resident`, `503
    chat_queue_full`)."""

    @field_validator("inputs")
    @classmethod
    def _no_blank_input(cls, value: list[str]) -> list[str]:
        blank = [index for index, text in enumerate(value) if not text.strip()]
        if blank:
            raise ValueError(f"inputs {blank[:10]} are blank; a vector of nothing is not one")
        return value


class EmbedModel(_Strict):
    """What wrote the vectors: everything that changes a float."""

    id: str
    """The Crucible model id."""
    revision: str
    """The weights' revision (the repo commit pinned in the manifest)."""
    file: str | None
    """The weights file read, for a block that is one file (a GGUF); null for a repo."""
    form: str | None
    """The form served, for a model whose block states several; null otherwise."""
    bits: int | None
    """The weights' precision."""
    engine: str
    """`llama-server` or `mlx-lm`."""
    engine_build: str
    """The engine's build: `llama-server-b10970-llg1.7.6-cuda13.0`, `mlx-lm-0.31.3+mlx-0.32.2`."""
    scheme: int
    """Crucible's own reading of a vector (rendering, pooling, truncation, normalisation)."""
    fingerprint: str
    """All of the above in one string. Vectors are comparable only with vectors of the
    same fingerprint: store it beside them, send it on later calls."""
    dimensions: int
    """The model's own vector length (the answer's `dimensions` may be a prefix of it)."""


class EmbedTiming(_Strict):
    total: float
    """The run, ms, from leaving the server's line to the answer."""
    queued: float | None = None
    """Ms the request waited in the server's line (a model loading, the engine busy)."""


class EmbedTokens(_Strict):
    per_input: list[int]
    """Tokens each input was, the model's format around it included."""
    total: int


class EmbedResponse(_Strict):
    """Vectors, in input order, and what wrote them."""

    object: Literal["crucible.embeddings"] = "crucible.embeddings"
    model: EmbedModel
    dimensions: int
    """Each vector's length."""
    input_type: Literal["query", "document"]
    instruction: str | None
    """The instruction the queries were written with; null for documents."""
    encoding_format: Literal["float", "base64", "base64_float16"]
    embeddings: list[list[float]] | list[str]
    """Unit-length vectors (float arrays, or base64 strings by `encoding_format`)."""
    tokens: EmbedTokens
    timing_ms: EmbedTiming


# --- the identity of a vector -------------------------------------------------------


def engine_build(engine: str, backend_kind: str) -> str:
    """The build of the engine that runs here, as this Crucible installs it: the
    llama-server binary Crucible pins on cuda-linux (hosttools.LLAMA_SERVER_BUILDS), the
    ggml-org release on llama-windows, the llm env's pins for mlx-lm and vLLM."""
    if engine == "llama-server":
        if backend_kind == CUDA_LINUX:
            build = hosttools.llama_server_build()
            if build is None:
                raise ApiError(
                    409,
                    "engine_not_pinned",
                    f"Crucible pins no llama-server build for {hosttools.host_platform()}, "
                    "so nothing here writes a vector it can name",
                    {"engine": engine},
                )
            return f"llama-server-{build.version}"
        return f"llama-server-{LLAMA_CPP_RELEASE}"
    pins = jobenv.recipe_pins(jobenv.recipe_for(jobenv.llm_env(backend_kind)))
    if engine == "mlx-lm":
        return f"mlx-lm-{pins['mlx-lm']}+mlx-{pins['mlx']}"
    if engine == "vllm":
        return f"vllm-{pins['vllm']}+torch-{pins['torch']}"
    raise ApiError(
        409,
        "engine_not_pinned",
        f"{engine} writes no vector Crucible names",
        {"engine": engine},
    )


def fingerprint_of(
    model_id: str, revision: str, file: str | None, build: str, scheme: str
) -> str:
    """`<id>@<revision, 12>:<file, or repo>:<engine build>:<Crucible's scheme>`."""
    return f"{model_id}@{revision[:12]}:{file or 'repo'}:{build}:{scheme}"


def identity(resident: Any, manifest: ModelManifest, backend_kind: str) -> EmbedModel:
    spec = manifest.spec(backend_kind, resident.form)
    assert manifest.embed is not None
    build = engine_build(resident.engine, backend_kind)
    return EmbedModel(
        id=resident.model_id,
        revision=resident.revision,
        file=spec.file,
        form=resident.form,
        bits=spec.bits,
        engine=resident.engine,
        engine_build=build,
        scheme=EMBED_SCHEME,
        fingerprint=fingerprint_of(
            resident.model_id, resident.revision, spec.file, build, f"e{EMBED_SCHEME}"
        ),
        dimensions=manifest.embed.dimensions,
    )


@dataclass(frozen=True)
class Named:
    """The model and form a request names, its fingerprint read."""

    model: str | None
    form: str | None


def named_by_fingerprint(body: EmbedRequest, manifests: dict[str, ModelManifest], backend_kind: str) -> Named:
    """`model` and `form` as the request names them, read off its `fingerprint` where it
    sends one: the id before `@`, and the form whose file is the fingerprint's file."""
    if body.fingerprint is None:
        return Named(body.model, body.form)
    model = body.fingerprint.split("@", 1)[0]
    manifest = manifests.get(model)
    if manifest is None or "@" not in body.fingerprint:
        raise ApiError(
            400,
            "fingerprint_unknown",
            f"fingerprint {body.fingerprint!r} names no model this build ships; a "
            "fingerprint is an earlier answer's model.fingerprint, sent back as it was",
            {"fingerprint": body.fingerprint},
        )
    parts = body.fingerprint.split(":")
    file = parts[1] if len(parts) > 1 else None
    form = None
    if manifest.supports(backend_kind):
        block = manifest.block(backend_kind)
        form = next((f.name for f in block.forms if f.file == file), None)
    if body.model is not None and body.model != model or (
        body.form is not None and body.form != form
    ):
        raise ApiError(
            400,
            "fingerprint_conflict",
            f"the request names model {body.model!r} form {body.form!r} and its "
            f"fingerprint {body.fingerprint!r} names {model!r} form {form!r}; send one",
            {"fingerprint": body.fingerprint, "model": body.model, "form": body.form},
        )
    return Named(model, form)


def refuse_other_fingerprint(asked: str | None, served: EmbedModel) -> None:
    if asked is None or asked == served.fingerprint:
        return
    raise ApiError(
        409,
        "fingerprint_mismatch",
        f"the request's vectors are {asked!r} and this server writes "
        f"{served.fingerprint!r}: another weights file, engine build or Crucible scheme, "
        "whose vectors are not comparable with the stored ones. Nothing was embedded. "
        "Embed the corpus again with this server (and store its fingerprint), or send "
        "the request to a server whose answer names the stored fingerprint",
        {"asked": asked, "served": served.fingerprint, "model": served.model_dump()},
    )


def refuse_vectors_only(model: str, door: str) -> None:
    """An embedding model at a door that generates or decides: it is started to write
    vectors (llama-server with --embedding) and answers nothing else."""
    try:
        manifest = load_manifest(model)
    except ManifestError:
        return
    if manifest.embed is not None:
        raise ApiError(
            400,
            "model_embeds_only",
            f"{model!r} is an embedding model: it writes vectors (POST /v1/embed) and "
            f"does not {door}",
            {"model": model},
        )


# --- the request, rendered and checked -----------------------------------------------


def refuse_unfit_request(body: EmbedRequest, spec: EmbedSpec, model: str) -> None:
    """What the model's format says no to, before anything waits."""
    if body.instruction is not None and not spec.takes_instruction(body.input_type):
        why = (
            "a document is written as it is"
            if body.input_type == "document"
            else f"{model}'s queries take no instruction"
        )
        raise ApiError(
            400,
            "instruction_not_taken",
            f"`instruction` is sent with input_type {body.input_type!r}, and {why}",
            {"model": model, "input_type": body.input_type},
        )
    if body.dimensions is None:
        return
    low, high = spec.dimensions_allowed()
    if not low <= body.dimensions <= high:
        raise ApiError(
            400,
            "dimensions_not_supported",
            f"{model} writes {spec.dimensions}-float vectors"
            + (
                f" and is trained to be read at any prefix from {low} to {high}"
                if spec.min_dimensions is not None
                else " and is not trained to be read at a shorter length"
            )
            + f"; {body.dimensions} is not one",
            {"model": model, "dimensions": body.dimensions, "range": [low, high]},
        )


def instruction_used(body: EmbedRequest, spec: EmbedSpec) -> str | None:
    if not spec.takes_instruction(body.input_type):
        return None
    return body.instruction if body.instruction is not None else spec.default_instruction


def rendered(body: EmbedRequest, spec: EmbedSpec) -> list[str]:
    return [spec.render(body.input_type, text, body.instruction) for text in body.inputs]


# --- the vector -----------------------------------------------------------------------


def unit(vector: list[float], dimensions: int, index: int, engine: str) -> list[float]:
    """The first `dimensions` values, scaled to unit length. A vector with no length
    (llama-server writes zeros for a state it could not read) is not a vector."""
    kept = vector[:dimensions]
    norm = math.sqrt(math.fsum(value * value for value in kept))
    if not math.isfinite(norm) or norm == 0.0:
        raise ApiError(
            502,
            "engine_error",
            f"the {engine} engine's vector for input {index} has no length ({norm}); it "
            "wrote no embedding for it",
            {"engine": engine, "input": index},
        )
    return [value / norm for value in kept]


def encoded(vectors: list[list[float]], encoding: str) -> list[list[float]] | list[str]:
    if encoding == "float":
        return vectors
    letter = "f" if encoding == "base64" else "e"
    return [
        base64.b64encode(struct.pack(f"<{len(vector)}{letter}", *vector)).decode("ascii")
        for vector in vectors
    ]


def _engine_error(engine: str, detail: str) -> ApiError:
    return ApiError(
        502,
        "engine_error",
        f"the {engine} engine's reply cannot be read as embeddings: {detail}",
        {"engine": engine},
    )


def _vector(value: Any, engine: str, where: str) -> list[float]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        raise _engine_error(engine, f"{where} is not a list of numbers")
    return [float(v) for v in value]


def too_long(index: int, tokens: int, cap: int) -> ApiError:
    return ApiError(
        400,
        EMBED_INPUT_TOO_LONG,
        f"input {index} is {tokens} tokens with the model's format around it, and this "
        f"server embeds at most {cap} in one input (the context the model is loaded "
        "at). Split it, or load the model with a longer context (POST /v1/jobs "
        '{"type": "load-model", "model": ..., "params": {"context": ...}})',
        {"input": index, "tokens": tokens, "max_tokens": cap},
    )


@dataclass(frozen=True)
class Read:
    vectors: list[list[float]]
    tokens: list[int]


async def on_openai_embeddings(call: EngineCall, resident: Any, texts: list[str]) -> Read:
    """llama-server: every input tokenized first as the model reads it (no special token
    added, special-token text read as those tokens), each checked against the context,
    then all of them as token ids in one /v1/embeddings request, unnormalised."""
    engine = resident.engine
    ids: list[list[int]] = []
    for index, text in enumerate(texts):
        data = await call(TOKENIZE_PATH, {"content": text, "add_special": False, "parse_special": True})
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in tokens
        ):
            raise _engine_error(engine, "the /tokenize reply carries no list of token ids")
        if len(tokens) > resident.max_model_len:
            raise too_long(index, len(tokens), resident.max_model_len)
        ids.append(tokens)
    data = await call(
        EMBEDDINGS_PATH,
        {"model": resident.engine_model_name, "input": ids, "embd_normalize": -1},
    )
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise _engine_error(engine, f"reply.data is not a list of {len(ids)} embeddings")
    vectors = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != position:
            raise _engine_error(engine, f"reply.data[{position}] is not embedding {position}")
        vectors.append(_vector(row.get("embedding"), engine, f"reply.data[{position}].embedding"))
    return Read(vectors=vectors, tokens=[len(row) for row in ids])


async def on_items(call: EngineCall, resident: Any, texts: list[str]) -> Read:
    """mlx-lm: the inputs in one items request; the engine tokenizes and refuses an input
    past the context by name."""
    engine = resident.engine
    data = await call(
        ITEMS_PATH,
        {"model": resident.engine_model_name, "inputs": texts,
         "max_input_tokens": resident.max_model_len},
    )
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise _engine_error(engine, f"reply.data is not a list of {len(texts)} embeddings")
    vectors, tokens = [], []
    for position, row in enumerate(rows):
        where = f"reply.data[{position}]"
        vectors.append(_vector(row.get("embedding") if isinstance(row, dict) else None, engine,
                               f"{where}.embedding"))
        count = row.get("tokens")
        if not isinstance(count, int) or isinstance(count, bool):
            raise _engine_error(engine, f"{where}.tokens is not an integer")
        tokens.append(count)
    return Read(vectors=vectors, tokens=tokens)


async def embed_on_engine(
    call: EngineCall, resident: Any, route: str, body: EmbedRequest, spec: EmbedSpec,
    model: EmbedModel,
) -> EmbedResponse:
    started = time.perf_counter()
    texts = rendered(body, spec)
    read = await (on_items if route == "items" else on_openai_embeddings)(call, resident, texts)
    widths = {len(vector) for vector in read.vectors}
    if widths != {spec.dimensions}:
        raise _engine_error(
            resident.engine,
            f"vectors of {sorted(widths)} floats from a model that writes {spec.dimensions}",
        )
    dimensions = body.dimensions if body.dimensions is not None else spec.dimensions
    vectors = [
        unit(vector, dimensions, index, resident.engine)
        for index, vector in enumerate(read.vectors)
    ]
    return EmbedResponse(
        model=model,
        dimensions=dimensions,
        input_type=body.input_type,
        instruction=instruction_used(body, spec),
        encoding_format=body.encoding_format,
        embeddings=encoded(vectors, body.encoding_format),
        tokens=EmbedTokens(per_input=read.tokens, total=sum(read.tokens)),
        timing_ms=EmbedTiming(total=round((time.perf_counter() - started) * 1000.0, 1)),
    )


__all__ = [
    "EMBED_SCHEME",
    "ENCODINGS",
    "EmbedModel",
    "EmbedRequest",
    "EmbedResponse",
    "MAX_EMBED_INPUTS",
    "embed_on_engine",
    "encoded",
    "engine_build",
    "fingerprint_of",
    "identity",
    "named_by_fingerprint",
    "refuse_other_fingerprint",
    "refuse_unfit_request",
    "unit",
]
