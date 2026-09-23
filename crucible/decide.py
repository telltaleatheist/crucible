"""The decision door's reading: `POST /v1/decide` (PHASE22-DECIDE.md).

A state and some questions with fixed answer sets go in; a probability
distribution over each answer set comes out, read off ONE forward pass of an
ordinary instruct model. No decoding: the prompt ends where the answer begins,
the engine reports the next-token distribution at that position, and the letters
`A`..`Z` that tag the options are read off it by token string, renormalised and
returned.

WHY THIS IS CRUCIBLE'S AND NOT THE APP'S
----------------------------------------
Everything here was snap's (`C:\\Users\\tellt\\Projects\\snap`: `prompt.py`,
`labels.py`, the pure half of `decide.py`, and `chat_engine.py`'s reader), and
it moved because the frame that makes an instruct model report a distribution
is a fact about the WEIGHTS it is read from — `pages.py` is the precedent: the
page prompt lives here because the model it addresses does (PHASE22 section
2.3). The client sends the ORDER: the state, the questions, the options. It
never sends the system prompt, the legend or the letters.

WHAT THIS MODULE IS NOT
-----------------------
It holds no HTTP. The door in `crucible/api.py` owns the wire, the admission,
the in-flight record and the engine; this module owns the request and answer
shapes, the messages, the letters, the reply parser and the arithmetic, all of
them pure so the semantics snap measured are tested without a socket
(`tests/test_decide_core.py`).
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import string
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .errors import ApiError
from .jobs.base import validate_member_name

# ------------------------------------------------------------------- numbers

#: The label alphabet. Bare capital letters, because each is ONE token on every
#: tokenizer snap measured and because an engine reports decoded token strings
#: (vLLM, llama-server) or raw BPE pieces (mlx-lm's `convert_ids_to_tokens`), and
#: a bare capital letter is the same string in both: `A` is `A`, where a leading
#: space would be `" A"` on one and `"ĠA"` on the other.
LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)

#: A choice question's ceiling, and the reason `too_many_options` exists: past
#: `Z` there is no single-token label left to read.
MAX_OPTIONS = len(LETTERS)  # 26

#: A score's ceiling (PHASE22 section 2.2: "2–10 unique ordered levels"). A
#: score is an expected value over an ordered scale; past ten the scale is a
#: choice question pretending to be a number.
MAX_LEVELS = 10

#: `yesno` is A = Yes, B = No, always, in that order (snap `labels.py`).
YESNO_OPTIONS: tuple[str, str] = ("Yes", "No")

#: Images per request. snap's number and snap's reason: each image costs up to
#: 4096 tokens on a Qwen-VL projector (llama.cpp `clip.cpp`,
#: `set_limit_image_tokens(8, 4096)`), so 8 keeps a request inside a modest
#: context and makes a runaway client fail by name.
MAX_IMAGES = 8

#: How many tokens past the labels the engine is asked for. A margin for the
#: tokens that outrank a letter when the model wanted to say something else —
#: which `label_mass` then reports — so a letter that came fifth behind four
#: fillers is still read rather than refused (PHASE22 section 2.4).
LABEL_MARGIN = 4

#: How many questions of one decision are in flight at once when the engine
#: states no admission of its own (vLLM, which batches). snap's
#: `DEFAULT_CONCURRENCY` for its openai-chat engine; a number of Crucible's and
#: never the wire's. An engine that DOES state one is held to that instead
#: (`crucible.engines.chat_admission`), because a serial engine given sixteen
#: at once is the starvation 1.0.10 fixed on the Mac.
UNSTATED_ENGINE_CONCURRENCY = 16

#: The frame. snap's words, unchanged: its accuracy checks were measured with
#: exactly this sentence, and a rewording is a re-measurement.
SYSTEM_PROMPT = (
    "You are a precise classifier. You are shown a state and one question about it, with "
    "lettered options. Reply with the single letter of the best option and nothing else."
)

#: What a request's image bytes are recognised as, by their own magic numbers.
#: The data URI names a media type and the engine decodes by it, so a type
#: guessed wrong is an image read wrong; one that matches nothing is refused.
IMAGE_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "png"),
    (b"\xff\xd8\xff", 0, "jpeg"),
    (b"GIF87a", 0, "gif"),
    (b"GIF89a", 0, "gif"),
    (b"WEBP", 8, "webp"),
)


# ------------------------------------------------------------------ request


_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class _Strict(BaseModel):
    # `use_attribute_docstrings`: the string under each field becomes its
    # OpenAPI description, which `scripts/gen-api-docs.py` carries into
    # `docs/API.md`. A field with no docstring is a field the reference cannot
    # explain.
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class ChoiceQuestion(_Strict):
    """Pick one of named options. Labelled A, B, C… in the order given."""

    type: Literal["choice"]
    """`choice`."""
    instructions: _NonEmpty
    """The question, as a person would ask it: "Which team should handle this?"."""
    options: dict[_NonEmpty, _NonEmpty] = Field(min_length=2)
    """Option name to a one-line description, in the order the letters are
    assigned: the first option is `A`. At least 2; more than 26 is refused as
    `too_many_options`, because past `Z` there is no one-token label to read."""


class ScoreQuestion(_Strict):
    """Place the state on an ordered scale. `score` is the expected level."""

    type: Literal["score"]
    """`score`."""
    instructions: _NonEmpty
    """The question: "How frustrated is the customer?"."""
    levels: list[_NonEmpty] = Field(min_length=2, max_length=MAX_LEVELS)
    """The scale, lowest first, 2 to 10 unique levels. Level i (1-based) is the
    value the expected `score` is computed with."""

    @field_validator("levels")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("levels must be unique")
        return value


class YesNoQuestion(_Strict):
    """Is a statement true of the state? `p` is P(Yes)."""

    type: Literal["yesno"]
    """`yesno`."""
    instructions: _NonEmpty
    """The statement to judge: "The message conveys urgency"."""


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, YesNoQuestion], Field(discriminator="type")
]


class DecideRequest(_Strict):
    """`POST /v1/decide` — PHASE22-DECIDE.md section 2.2.

    One forward pass per question at the RESIDENT model; nothing is decoded and
    nothing is loaded to answer it.
    """

    model: _NonEmpty
    """The Crucible model id, which must already be resident (`409
    model_not_resident` otherwise). An upstream id (`<upstream>/<id>`) is
    refused `400 decide_needs_logprobs`: no upstream returns a distribution."""
    state: Any
    """What the questions are about: a string, used verbatim, or any other JSON
    value, serialised as compact JSON. Required and never null; may be `""` only
    when `images` carry the state."""
    questions: dict[_NonEmpty, Question] = Field(min_length=1)
    """Question name to question. Names are single path members (no `/`, `\\`,
    leading dot) and key the answers. Answers come back in this order."""
    images: list[str] | None = None
    """Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no
    whitespace, no `data:` prefix), read as part of the state, before its text.
    At most 8 (`too_many_images`), and only on a model whose manifest declares
    `image` (`400 model_text_only` otherwise). `[]` is the same as none."""

    @field_validator("state")
    @classmethod
    def _state_not_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("state is required and may not be null")
        return value

    @field_validator("questions")
    @classmethod
    def _names_are_path_members(cls, value: dict[str, Any]) -> dict[str, Any]:
        for name in value:
            validate_member_name(name)
        return value

    @field_validator("images")
    @classmethod
    def _images_are_strict_base64(cls, value: list[str] | None) -> list[str] | None:
        # Strict, for snap's measured reason: llama-server's `base64_decode`
        # (tools/server/server-common.cpp) stops silently at the first character
        # outside the alphabet, so a line-wrapped string or a pasted data URI
        # would reach the model as a truncated file. Refused here, by index.
        for index, text in enumerate(value or []):
            if not text:
                raise ValueError(f"images[{index}] is empty")
            try:
                raw = base64.b64decode(text, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(
                    f"images[{index}] is not base64 (standard alphabet, padded, no "
                    f"whitespace, no data: prefix): {exc}"
                ) from None
            if not raw:
                raise ValueError(f"images[{index}] decodes to zero bytes")
            if image_format(raw) is None:
                raise ValueError(
                    f"images[{index}] is not a PNG, JPEG, GIF or WebP file by its "
                    "own first bytes; the engine is told the media type, so one "
                    "that cannot be named is not sent"
                )
        return value

    @model_validator(mode="after")
    def _state_or_images(self) -> "DecideRequest":
        if isinstance(self.state, str) and not self.state.strip() and not self.images:
            raise ValueError("state may not be empty unless images carry the state")
        return self


# ------------------------------------------------------------------- answer


class ChoiceAnswer(_Strict):
    """A choice question's distribution."""

    type: Literal["choice"] = "choice"
    """`choice`."""
    choice: str
    """The most probable option."""
    probabilities: dict[str, float]
    """Option name to probability, renormalised over the letters so they sum
    to 1 (a softmax over the label logits)."""
    confidence: float
    """The largest renormalised probability."""
    label_mass: float
    """The raw probability the letters held together before renormalising. Low
    means the model wanted to say something that is not an option."""


class ScoreAnswer(_Strict):
    """A score question's distribution and its expected level."""

    type: Literal["score"] = "score"
    """`score`."""
    score: float
    """Σ (1-based level index × p): 1.0 is certainly the lowest level."""
    level: str
    """The most probable level."""
    probabilities: dict[str, float]
    """Level to renormalised probability."""
    confidence: float
    """The largest renormalised probability."""
    label_mass: float
    """The raw probability the letters held together before renormalising."""


class YesNoAnswer(_Strict):
    """A yesno question's probability."""

    type: Literal["yesno"] = "yesno"
    """`yesno`."""
    p: float
    """Renormalised P(Yes)."""
    label_mass: float
    """The raw probability `A` and `B` held together before renormalising."""


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, YesNoAnswer], Field(discriminator="type")
]


class ForwardTiming(_Strict):
    """One request to the engine, timed by Crucible."""

    wall_ms: float
    """Crucible's wall clock around the request, queueing and the one decoded
    token included. The OpenAI reply carries no prefill time, so this is the
    only duration there is."""
    prompt_tokens: int
    """`usage.prompt_tokens`: the whole prompt, cached part included."""
    cached_tokens: int | None
    """`usage.prompt_tokens_details.cached_tokens`, or null when the engine did
    not say. Never 0 for "unknown": a number nobody measured is not a
    measurement."""


class DecideTiming(_Strict):
    """Where the time went."""

    total: float
    """The whole decision, ms, Crucible's clock."""
    per_question: dict[str, ForwardTiming]
    """Each question's own request."""
    prime: ForwardTiming | None
    """The shared prefix sent alone first — present when the decision had more
    than one question, null when it had one."""


class DecideTokens(_Strict):
    """How big the prompts were."""

    per_question: dict[str, int]
    """`usage.prompt_tokens` for each question's prompt."""
    images: int
    """How many images every prompt of this decision carried."""


class ModelProvenance(_Strict):
    """The weights that made the decision, as an artifact sidecar names them."""

    id: str
    """The Crucible model id."""
    revision: str
    """The revision the resident engine was started on."""
    fingerprint: str
    """`<id>@<revision>`."""


class DecideResponse(_Strict):
    """A decision: one distribution per question."""

    model: ModelProvenance
    """Which weights answered (PHASE2-LLM.md section 5's triple)."""
    engine: str
    """The engine kind that answered: `vllm`, `llama-server`, `mlx-lm`."""
    answers: dict[str, Answer]
    """Question name to answer, in the request's question order."""
    timing_ms: DecideTiming
    """Crucible's clock, per request."""
    tokens: DecideTokens
    """Prompt sizes."""


# -------------------------------------------------------------------- plans


@dataclass(frozen=True)
class Plan:
    """One question, resolved: its letters and the legend the model reads."""

    name: str
    question: ChoiceQuestion | ScoreQuestion | YesNoQuestion
    #: `[(letter, option name)]`, in option order.
    labels: tuple[tuple[str, str], ...]
    #: `[(letter, text shown after it)]`.
    legend: tuple[tuple[str, str], ...]

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(letter for letter, _ in self.labels)


def assign_labels(names: list[str], question: str) -> tuple[tuple[str, str], ...]:
    """`[(letter, option name)]` in the order given. Past 26: `too_many_options`."""
    if len(names) > MAX_OPTIONS:
        raise ApiError(
            400,
            "too_many_options",
            f"question {question!r} has {len(names)} options; the label set is "
            f"A..Z ({MAX_OPTIONS}). Split it into two questions",
            {"question": question, "options": len(names), "max_options": MAX_OPTIONS},
        )
    return tuple(zip(LETTERS, names))


def plan(name: str, question: ChoiceQuestion | ScoreQuestion | YesNoQuestion) -> Plan:
    """Letters and legend for one question. Raises before anything is sent."""
    if isinstance(question, ChoiceQuestion):
        labels = assign_labels(list(question.options), name)
        legend = tuple(
            (letter, f"{option}: {question.options[option]}") for letter, option in labels
        )
    elif isinstance(question, ScoreQuestion):
        labels = assign_labels(list(question.levels), name)
        legend = labels
    else:
        labels = tuple(zip(LETTERS, YESNO_OPTIONS))
        legend = labels
    return Plan(name=name, question=question, labels=labels, legend=legend)


def plan_all(request: DecideRequest) -> list[Plan]:
    """Every question resolved BEFORE the first forward pass, so one bad question
    refuses the whole decision instead of after GPU time was spent on the rest."""
    return [plan(name, question) for name, question in request.questions.items()]


def check_image_count(images: list[str] | None) -> int:
    """How many images, or `too_many_images` by name."""
    count = len(images or [])
    if count > MAX_IMAGES:
        raise ApiError(
            400,
            "too_many_images",
            f"the request carries {count} images; a decision reads at most "
            f"{MAX_IMAGES}",
            {"images": count, "max_images": MAX_IMAGES},
        )
    return count


def top_k(n_labels: int, max_logprobs: int | None) -> int:
    """How many top tokens to ask the engine for: the labels plus the margin,
    never more than the engine returns. `max_logprobs` None is an engine with no
    small cap (llama-server: bounded only by the vocabulary)."""
    wanted = n_labels + LABEL_MARGIN
    return wanted if max_logprobs is None else min(wanted, max_logprobs)


# ----------------------------------------------------------------- messages


def render_state(state: Any) -> str:
    """Strings pass through verbatim; any other JSON value becomes compact JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def question_block(kind: str, instructions: str, legend: tuple[tuple[str, str], ...]) -> str:
    """The question-specific tail of the user message. snap's words, unchanged."""
    if kind == "yesno":
        head = f"Statement: {instructions}\nIs this statement true of the state above?"
    else:
        head = f"Question: {instructions}"
    lines = "\n".join(f"{letter}. {text}" for letter, text in legend)
    return f"{head}\nOptions:\n{lines}\nAnswer with the letter only."


def image_format(raw: bytes) -> str | None:
    """`png`/`jpeg`/`gif`/`webp` by the file's own first bytes, else None."""
    for magic, offset, name in IMAGE_SIGNATURES:
        if raw[offset : offset + len(magic)] == magic:
            if name == "webp" and raw[:4] != b"RIFF":
                continue
            return name
    return None


def image_part(encoded: str) -> dict[str, Any]:
    """One OpenAI `image_url` content part, the encoding `pages.py` sends."""
    kind = image_format(base64.b64decode(encoded, validate=True))
    if kind is None:  # pragma: no cover - DecideRequest refuses it first
        raise ApiError(400, "invalid_request", "an image is not a known format")
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{encoded}"}}


def _user_content(
    state_text: str, images: list[str], block: str | None
) -> str | list[dict[str, Any]]:
    """The user turn: `State:`, the images, the state text, then the question.

    Text only, it is ONE string — `State:\\n<state>` and, for a question,
    `\\n\\n<block>` — exactly snap's layout, so a prime's content is a string
    prefix of every question's. With images it is a list of OpenAI content
    parts in the same order: a `State:` text part, the images, then one text
    part carrying the state text and the question block. The prime's parts are
    then the question's parts with the last text part cut back to the state
    text (or absent, when the state is empty), which is the prefix property
    in the only form content parts can have it.
    """
    tail_parts = [part for part in (state_text, block) if part]
    tail = "\n\n".join(tail_parts)
    if not images:
        return "State:\n" + tail if tail else "State:"
    parts: list[dict[str, Any]] = [{"type": "text", "text": "State:"}]
    parts.extend(image_part(encoded) for encoded in images)
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts


def messages(state_text: str, images: list[str], block: str | None) -> list[dict[str, Any]]:
    """System + user. `block` None is the prime: the shared prefix alone."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_content(state_text, images, block)},
    ]


def question_messages(state_text: str, images: list[str], item: Plan) -> list[dict[str, Any]]:
    block = question_block(item.question.type, item.question.instructions, item.legend)
    return messages(state_text, images, block)


def prime_messages(state_text: str, images: list[str]) -> list[dict[str, Any]]:
    return messages(state_text, images, None)


def request_body(
    engine_model_name: str, msgs: list[dict[str, Any]], k: int | None
) -> dict[str, Any]:
    """The engine's chat body for one forward pass. `k` None is the prime.

    Every knob a reading depends on is STATED, and no manifest `[defaults]` is
    applied, for a measured reason: mlx-lm computes its logprobs AFTER its
    logits processors (`mlx_lm/generate.py` L409-420, 0.31.3), so a manifest's
    `repetition_penalty` would move the letters — and the letters appear in the
    legend, which is exactly the context a repetition penalty punishes.
    `enable_thinking: false` is stated for the chat door's reason (Qwen3.5
    spends a bounded budget on reasoning otherwise), and a stated key is one no
    manifest default can override (`crucible/sampling.py`).

    The prime asks for no logprobs at all: its reply is never read for letters
    (PHASE22 section 2.5), and asking would only spend the engine's cap.
    """
    body: dict[str, Any] = {
        "model": engine_model_name,
        "messages": msgs,
        "max_tokens": 1,
    }
    if k is not None:
        body["logprobs"] = True
        body["top_logprobs"] = k
    body["temperature"] = 0
    body["chat_template_kwargs"] = {"enable_thinking": False}
    body["stream"] = False
    return body


# -------------------------------------------------------------------- reply


@dataclass(frozen=True)
class Reading:
    """What one engine reply said."""

    prompt_tokens: int
    cached_tokens: int | None
    #: `[(token string, probability)]` in the engine's order; None for a prime.
    top: tuple[tuple[str, float], ...] | None


def _require(container: Any, key: str, kind: Any, where: str, engine: str) -> Any:
    if not isinstance(container, dict) or key not in container:
        raise _engine_error(engine, f"{where}: {key!r} is missing")
    value = container[key]
    if isinstance(value, bool) and kind is not bool:
        raise _engine_error(engine, f"{where}.{key} is a bool, expected {kind}")
    if not isinstance(value, kind):
        raise _engine_error(
            engine, f"{where}.{key} is {type(value).__name__}, expected {kind}"
        )
    return value


def _engine_error(engine: str, detail: str) -> ApiError:
    return ApiError(
        502,
        "engine_error",
        f"the {engine} engine's reply cannot be read as a decision: {detail}",
        {"engine": engine},
    )


def read_reply(data: Any, engine: str, *, want_probs: bool) -> Reading:
    """`usage` and, for a question, `choices[0].logprobs.content[0].top_logprobs`.

    ONE PARSER FOR THREE ENGINES, because they answer in one shape (PHASE22
    section 1): vLLM 0.29.0's `ChatCompletionLogProbsContent` (`{token,
    logprob, bytes, top_logprobs}`), llama-server b10970's
    `probs_vector_to_json` (the same plus `id`), and mlx-lm 0.31.3's
    `dict(i[0], top_logprobs=i)` (`{id, token, logprob, top_logprobs}`, no
    `bytes`). Only `token` and `logprob` are read.

    `cached_tokens` is null when the engine did not say: vLLM without
    `--enable-prompt-tokens-details` omits `prompt_tokens_details`, and vLLM
    WITH it can still send `cached_tokens: null` (`PromptTokenUsageInfo`).
    """
    usage = _require(data, "usage", dict, "reply", engine)
    prompt_tokens = _require(usage, "prompt_tokens", int, "usage", engine)
    details = usage.get("prompt_tokens_details")
    if details is None:
        cached: int | None = None
    elif isinstance(details, dict):
        value = details.get("cached_tokens")
        if value is None:
            cached = None
        elif isinstance(value, int) and not isinstance(value, bool):
            cached = value
        else:
            raise _engine_error(
                engine,
                f"usage.prompt_tokens_details.cached_tokens is "
                f"{type(value).__name__}, expected an integer or null",
            )
    else:
        raise _engine_error(
            engine,
            f"usage.prompt_tokens_details is {type(details).__name__}, expected an "
            "object or null",
        )
    if not want_probs:
        return Reading(prompt_tokens=prompt_tokens, cached_tokens=cached, top=None)

    choices = _require(data, "choices", list, "reply", engine)
    if not choices:
        raise _engine_error(engine, "'choices' is empty")
    logprobs = _require(choices[0], "logprobs", dict, "choices[0]", engine)
    content = _require(logprobs, "content", list, "choices[0].logprobs", engine)
    if not content:
        raise _engine_error(
            engine, "choices[0].logprobs.content is empty (no token was generated)"
        )
    tops = _require(
        content[0], "top_logprobs", list, "choices[0].logprobs.content[0]", engine
    )
    top: list[tuple[str, float]] = []
    for index, entry in enumerate(tops):
        where = f"top_logprobs[{index}]"
        token = _require(entry, "token", str, where, engine)
        logprob = _require(entry, "logprob", (int, float), where, engine)
        top.append((token, math.exp(logprob)))
    return Reading(prompt_tokens=prompt_tokens, cached_tokens=cached, top=tuple(top))


def label_distribution(
    top: tuple[tuple[str, float], ...], item: Plan, engine: str
) -> tuple[dict[str, float], float]:
    """`({option: renormalised p}, label_mass)`, labels matched BY TOKEN STRING.

    A label is the entry whose token string IS the letter — `"A"`, never `" A"`.
    Two entries with one string are possible in general (byte-level pieces
    decode alike), so only a LABEL's string appearing twice is refused: that
    one would make the answer ambiguous. Renormalising p_i / Σp over the labels
    is a softmax over the label logits.
    """
    letters = set(item.letters)
    by_token: dict[str, float] = {}
    for token, probability in top:
        if token in by_token and token in letters:
            raise _engine_error(
                engine, f"label token {token!r} appears twice in top_logprobs"
            )
        by_token.setdefault(token, probability)
    raw: dict[str, float] = {}
    for letter, option in item.labels:
        if letter not in by_token:
            raise ApiError(
                502,
                "label_not_in_probs",
                f"question {item.name!r}: label {letter!r} (option {option!r}) is not "
                f"among the top {len(top)} tokens the {engine} engine returned. The "
                "model wanted to say something else strongly enough to push a "
                "letter out; fewer options, or plainer ones, is the repair",
                {"question": item.name, "letter": letter, "option": option,
                 "engine": engine, "top_k": len(top)},
            )
        raw[option] = by_token[letter]
    mass = sum(raw.values())
    if mass <= 0.0:
        raise ApiError(
            502,
            "label_not_in_probs",
            f"question {item.name!r}: every label has probability 0 on the {engine} "
            "engine",
            {"question": item.name, "letter": None, "option": None,
             "engine": engine, "top_k": len(top)},
        )
    return {option: p / mass for option, p in raw.items()}, mass


def answer(
    item: Plan, probabilities: dict[str, float], mass: float
) -> ChoiceAnswer | ScoreAnswer | YesNoAnswer:
    """snap's answer shapes and arithmetic, unchanged (`snap/decide.py _answer`)."""
    question = item.question
    if isinstance(question, YesNoQuestion):
        return YesNoAnswer(p=probabilities["Yes"], label_mass=mass)
    best = max(probabilities, key=probabilities.__getitem__)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=best,
            probabilities=probabilities,
            confidence=probabilities[best],
            label_mass=mass,
        )
    score = sum(
        (index + 1) * probabilities[level] for index, (_, level) in enumerate(item.labels)
    )
    return ScoreAnswer(
        score=score,
        level=best,
        probabilities=probabilities,
        confidence=probabilities[best],
        label_mass=mass,
    )


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecideRequest",
    "DecideResponse",
    "DecideTiming",
    "DecideTokens",
    "ForwardTiming",
    "LABEL_MARGIN",
    "LETTERS",
    "MAX_IMAGES",
    "MAX_LEVELS",
    "MAX_OPTIONS",
    "ModelProvenance",
    "Plan",
    "Reading",
    "SYSTEM_PROMPT",
    "ScoreAnswer",
    "ScoreQuestion",
    "UNSTATED_ENGINE_CONCURRENCY",
    "YESNO_OPTIONS",
    "YesNoAnswer",
    "YesNoQuestion",
    "answer",
    "assign_labels",
    "check_image_count",
    "image_format",
    "image_part",
    "label_distribution",
    "messages",
    "plan",
    "plan_all",
    "prime_messages",
    "question_block",
    "question_messages",
    "read_reply",
    "render_state",
    "request_body",
    "top_k",
]
