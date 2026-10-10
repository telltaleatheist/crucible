"""Structured output for mlx-lm's server: `response_format` and `structured_outputs`
enforced with llguidance, the grammar engine vLLM runs on the PC.

Stock mlx-lm 0.31.3 reads no `response_format` at all (mlx_lm/server.py builds its
logits processors from the repetition penalties and logit_bias only,
`_make_logits_processors`), so a chat that asked for a JSON schema was answered
unconstrained and nobody was told. This module is placed into the llm env as
`mlx_lm/_crucible_grammar.py` (envs/llm/patches/patch_mlx_lm_structured_output.py)
and the patched server calls it:

- `constraint_of_body` reads the request on the HTTP thread and turns what it asks
  for into an llguidance grammar exactly as vLLM 0.29.0's guidance backend does
  (vllm/v1/structured_output/backend_guidance.py serialize_guidance_grammar:
  `grammar_from_json_schema(schema, defaults={"whitespace_flexible": True})`, a
  json_object as the schema `{"type": "object"}`, regex / choice / grammar through
  `grammar_from`). What it cannot enforce it refuses by name (`GrammarRefusal`),
  before anything is generated.
- `GrammarProcessor` is one sequence's logits processor: its own `LLMatcher`, the
  allowed-token bitmask computed every step and applied to the logits, last in the
  list so no bias or penalty can lift a token the grammar forbids. EOS is in the mask
  only while the grammar accepts, and once the grammar is complete EOS is the only
  token left, so the answer stops at its closing brace.
- If the matcher ever fails (a resource limit inside llguidance, never a token it
  allowed), the sequence is stopped at once and the request is answered with the
  error (`answer_failure`), never with the partial text.

It imports nothing from Crucible: it runs inside the llm env's python.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
from dataclasses import dataclass
from typing import Any

GRAMMAR_VERSION = 1

# vLLM 0.29.0's guidance backend compiles every JSON schema with whitespace_flexible
# on unless the server was started with disable_any_whitespace, which Crucible does not
# (engines/vllm.py STRUCTURED_OUTPUTS_ARGS names the backend only).
JSON_DEFAULTS: dict[str, Any] = {"whitespace_flexible": True}

ANY_OBJECT = '{"type": "object"}'

RESPONSE_FORMATS: tuple[str, ...] = ("json_object", "json_schema")

STRUCTURED_KINDS: tuple[str, ...] = ("json", "json_object", "regex", "choice", "grammar")

# structured_outputs members this server does not act on, with the value that means
# "not asked for". Anything else set is refused rather than dropped.
STRUCTURED_OPTIONS: dict[str, Any] = {
    "disable_any_whitespace": False,
    "disable_additional_properties": False,
    "whitespace_pattern": None,
    "structural_tag": None,
}

# Grammar fields other engines read and this one does not: refused by name.
UNSERVED_FIELDS: tuple[str, ...] = (
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "grammar",
    "json_schema",
)

SERVED = (
    "this server enforces response_format json_object and json_schema, and "
    "structured_outputs json, json_object, regex, choice and grammar"
)


class GrammarRefusal(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Constraint:
    """One request's grammar. `error` is written by the generation thread when the
    matcher fails, and read by the HTTP thread when the answer is done."""

    grammar: str
    source: str
    error: str | None = None


def _refuse(code: str, message: str, status: int = 400) -> GrammarRefusal:
    return GrammarRefusal(status, code, message)


def _from_response_format(value: Any) -> tuple[str, Any, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _refuse(
            "invalid_response_format",
            f"response_format must be an object, not {type(value).__name__}",
        )
    kind = value.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return ("json_object", None, "response_format json_object")
    if kind == "json_schema":
        wrapper = value.get("json_schema")
        if not isinstance(wrapper, dict):
            raise _refuse(
                "invalid_response_format",
                "response_format json_schema needs a json_schema object holding the "
                "schema: {\"type\": \"json_schema\", \"json_schema\": {\"name\": ..., "
                "\"schema\": {...}}}",
            )
        schema = wrapper.get("schema")
        if not isinstance(schema, dict):
            raise _refuse(
                "invalid_response_format",
                "response_format.json_schema.schema must be an object (the JSON "
                f"schema itself), not {type(schema).__name__}",
            )
        return ("json", schema, "response_format json_schema")
    raise _refuse(
        "structured_output_not_served",
        f"response_format type {kind!r} is not enforced here; {SERVED}",
    )


def _from_structured_outputs(value: Any) -> tuple[str, Any, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _refuse(
            "invalid_structured_outputs",
            f"structured_outputs must be an object, not {type(value).__name__}",
        )
    unknown = sorted(set(value) - set(STRUCTURED_KINDS) - set(STRUCTURED_OPTIONS))
    stated = sorted(
        key
        for key, default in STRUCTURED_OPTIONS.items()
        if key in value and value[key] != default
    )
    if unknown or stated:
        raise _refuse(
            "structured_output_not_served",
            f"structured_outputs carries {unknown + stated}, which this server does "
            f"not act on; {SERVED}",
        )
    kinds = [key for key in STRUCTURED_KINDS if value.get(key) not in (None, False)]
    if len(kinds) != 1:
        raise _refuse(
            "invalid_structured_outputs",
            "structured_outputs must set exactly one of "
            f"{list(STRUCTURED_KINDS)}; it sets {kinds or 'none'}",
        )
    kind = kinds[0]
    spec = value[kind]
    if kind == "json" and isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError as exc:
            raise _refuse(
                "invalid_structured_outputs", f"structured_outputs.json is not JSON: {exc}"
            ) from None
    if kind == "json" and not isinstance(spec, dict):
        raise _refuse(
            "invalid_structured_outputs",
            "structured_outputs.json must be a JSON schema object",
        )
    if kind == "choice" and not (
        isinstance(spec, list) and spec and all(isinstance(item, str) for item in spec)
    ):
        raise _refuse(
            "invalid_structured_outputs",
            "structured_outputs.choice must be a non-empty list of strings",
        )
    if kind in ("regex", "grammar") and not (isinstance(spec, str) and spec):
        raise _refuse(
            "invalid_structured_outputs",
            f"structured_outputs.{kind} must be a non-empty string",
        )
    return (kind, spec, f"structured_outputs {kind}")


def asked_for(body: dict[str, Any]) -> tuple[str, Any, str] | None:
    """What the request constrains its answer with, as (kind, spec, source); None when
    it asks for nothing. Pure: no llguidance, so Crucible's own tests read it."""
    unserved = [field for field in UNSERVED_FIELDS if body.get(field) is not None]
    if unserved:
        raise _refuse(
            "structured_output_not_served",
            f"this request constrains its answer with {unserved}, which this server "
            f"does not read; {SERVED}",
        )
    from_format = _from_response_format(body.get("response_format"))
    from_outputs = _from_structured_outputs(body.get("structured_outputs"))
    if from_format is not None and from_outputs is not None:
        raise _refuse(
            "invalid_request",
            "this request constrains its answer with both response_format and "
            "structured_outputs; send one",
        )
    return from_format if from_format is not None else from_outputs


def compile_grammar(kind: str, spec: Any) -> str:
    """The llguidance grammar for one constraint, as vLLM 0.29.0's guidance backend
    serializes it."""
    import llguidance

    matcher = llguidance.LLMatcher
    try:
        if kind == "json":
            return matcher.grammar_from_json_schema(spec, defaults=dict(JSON_DEFAULTS))
        if kind == "json_object":
            return matcher.grammar_from_json_schema(ANY_OBJECT, defaults=dict(JSON_DEFAULTS))
        if kind == "choice":
            return llguidance.grammar_from("choice", json.dumps(spec))
        return llguidance.grammar_from(kind, spec)
    except ValueError as exc:
        raise _refuse("invalid_grammar", f"the {kind} constraint will not compile: {exc}") from None


def constraint_of_body(body: dict[str, Any]) -> Constraint | None:
    found = asked_for(body)
    if found is None:
        return None
    kind, spec, source = found
    grammar = compile_grammar(kind, spec)
    import llguidance

    error = llguidance.LLMatcher.validate_grammar(grammar)
    if error:
        raise _refuse("invalid_grammar", f"the {source} will not compile: {error}")
    return Constraint(grammar=grammar, source=source)


_TOKENIZERS: dict[tuple[int, int], tuple[Any, Any]] = {}
_TOKENIZERS_LOCK = threading.Lock()


def llg_tokenizer(tokenizer: Any, width: int) -> Any:
    """The llguidance tokenizer for mlx-lm's TokenizerWrapper, built once per tokenizer
    and logits width (~1 s on a 248k vocabulary). Its vocabulary is the wider of the
    logits and the tokenizer, as vLLM sizes it (backend_guidance.py
    `max(self.vocab_size, len(self.tokenizer))`), and its EOS set is every id mlx-lm's
    state machine stops on (`eos_token_ids`)."""
    key = (id(tokenizer), width)
    with _TOKENIZERS_LOCK:
        hit = _TOKENIZERS.get(key)
        if hit is not None and hit[0] is tokenizer:
            return hit[1]
        import llguidance

        hf = getattr(tokenizer, "_tokenizer", tokenizer)
        backend = getattr(hf, "backend_tokenizer", None)
        if backend is None:
            raise RuntimeError(
                f"{type(hf).__name__} has no backend_tokenizer: llguidance reads a "
                "fast (Rust) tokenizer's JSON, and this model's tokenizer is not one"
            )
        backend = copy.copy(backend)
        backend.no_padding()
        backend.no_truncation()
        eos = sorted(getattr(tokenizer, "eos_token_ids", None) or [hf.eos_token_id])
        built = llguidance.LLTokenizer(
            backend.to_str(), n_vocab=max(width, len(hf)), eos_token=list(eos)
        )
        _TOKENIZERS[key] = (tokenizer, built)
        return built


_UNPACK: dict[int, tuple[Any, Any]] = {}


def _unpack(width: int) -> tuple[Any, Any]:
    found = _UNPACK.get(width)
    if found is None:
        import mlx.core as mx

        index = mx.arange(width, dtype=mx.uint32)
        found = (index >> 5, index & 31)
        _UNPACK[width] = found
    return found


def apply_bitmask(logits: Any, row: Any) -> Any:
    """`logits` ([1, V], any float dtype) with every token whose bit is clear in `row`
    (llguidance's packed int32 mask, one bit per token) set to -inf. Plain mlx ops, so
    it runs on Metal and on mlx's CPU backend alike."""
    import mlx.core as mx
    import numpy as np

    width = logits.shape[-1]
    words, shifts = _unpack(width)
    packed = mx.array(np.ascontiguousarray(row).view(np.uint32))
    allowed = ((mx.take(packed, words) >> shifts) & 1).astype(mx.bool_)
    return mx.where(allowed, logits, mx.array(-float("inf"), dtype=logits.dtype))


class GrammarProcessor:
    """One sequence's grammar, as mlx-lm calls a logits processor: `(tokens, logits)`
    once per generated token, `tokens` the sequence so far. The first call comes with
    the prompt's last token (generate_step's first `_step`, GenerationBatch's first
    `_step` with the prompt's last input), so it starts the matcher; every later call's
    last token is the one sampled after the previous call, which the matcher consumes."""

    def __init__(self, constraint: Constraint, tokenizer: Any) -> None:
        if tokenizer is None:
            raise RuntimeError("a grammar processor needs the model's tokenizer")
        self._constraint = constraint
        self._tokenizer = tokenizer
        self._matcher: Any = None
        self._mask: Any = None
        self._eos: list[int] = []

    def __call__(self, tokens: Any, logits: Any) -> Any:
        import llguidance
        import llguidance.numpy as llgnp

        if self._matcher is None:
            llt = llg_tokenizer(self._tokenizer, logits.shape[-1])
            self._eos = list(llt.eos_tokens)
            self._matcher = llguidance.LLMatcher(llt, self._constraint.grammar, log_level=0)
            self._mask = llgnp.allocate_token_bitmask(1, llt.vocab_size)
        else:
            self._matcher.consume_token(int(tokens[-1].item()))
        if self._matcher.is_error():
            return self._stop(logits, self._matcher.get_error())
        llgnp.fill_next_token_bitmask(self._matcher, self._mask, 0)
        return apply_bitmask(logits, self._mask[0])

    def _stop(self, logits: Any, error: str) -> Any:
        import mlx.core as mx

        if self._constraint.error is None:
            # llguidance follows its one-line reason with the parser state and the
            # whole grammar; the client is told the reason, the log keeps the rest.
            self._constraint.error = error.split("\n", 1)[0]
            print(
                f"crucible structured output: the {self._constraint.source} matcher "
                f"failed and the answer is stopped: {error[:4000]}",
                file=sys.stderr,
                flush=True,
            )
        eos = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
        eos[mx.array(self._eos)] = True
        return mx.where(eos, logits, mx.array(-float("inf"), dtype=logits.dtype))


def refuse(handler: Any, refusal: GrammarRefusal) -> None:
    """Answer a request whose constraint cannot be enforced, before generation."""
    body = json.dumps(
        {
            "error": {
                "message": refusal.message,
                "type": "invalid_request_error",
                "code": refusal.code,
            }
        }
    ).encode("utf-8")
    handler._set_completion_headers(refusal.status)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


def answer_failure(handler: Any, constraint: Constraint) -> None:
    """Answer a request whose matcher failed during generation with the error, never
    the partial text. A whole answer has sent no byte yet: http.server buffers the
    status line and headers until end_headers, so the buffered 200 is dropped and a 500
    sent in its place. A stream has sent its 200 and its chunks; it ends with an error
    event before [DONE]."""
    error = {
        "message": (
            f"the {constraint.source} could not be kept to the end of the answer: "
            f"{constraint.error}. The answer was stopped and is not returned"
        ),
        "type": "server_error",
        "code": "structured_output_failed",
    }
    if handler.stream:
        handler.wfile.write(f"data: {json.dumps({'error': error})}\n\n".encode("utf-8"))
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
        return
    body = json.dumps({"error": error}).encode("utf-8")
    handler._headers_buffer = []
    handler._set_completion_headers(500)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


__all__ = [
    "Constraint",
    "GRAMMAR_VERSION",
    "GrammarProcessor",
    "GrammarRefusal",
    "answer_failure",
    "apply_bitmask",
    "asked_for",
    "compile_grammar",
    "constraint_of_body",
    "llg_tokenizer",
    "refuse",
]
