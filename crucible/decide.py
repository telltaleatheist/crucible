from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import string
import time
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_serializer,
    model_validator,
)

from .errors import ApiError
from .queuerequest import QueueChoice, queue_field

LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)

MAX_OPTIONS = len(LETTERS)

MAX_LEVELS = 10

YESNO_OPTIONS: tuple[str, str] = ("Yes", "No")

MAX_IMAGES = 8

LABEL_MARGIN = 4

SYSTEM_PROMPT = (
    "You are a precise classifier. You are shown a state and one question about it, with "
    "lettered options. Reply with the single letter of the best option and nothing else."
)

STATE_HEADER = "\n\nState:\n"

IMAGES_NOTE = "The state's images open the user message."

PRIME_USER_TEXT = "The questions follow."

MAX_CANDIDATES = 26

MAX_CANDIDATE_TOKENS = 256

LIKELIHOOD_SYSTEM_PROMPT = (
    "You are shown a state and one request about it. Reply with exactly what the "
    "request asks for and nothing else."
)

IMAGE_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "png"),
    (b"\xff\xd8\xff", 0, "jpeg"),
    (b"GIF87a", 0, "gif"),
    (b"GIF89a", 0, "gif"),
    (b"WEBP", 8, "webp"),
)


_NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class _Strict(BaseModel):
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


class LikelihoodQuestion(_Strict):
    """Score free-text candidate replies by how likely the model is to say each,
    instead of generating one. Nothing is decoded, so nothing can fail to parse
    or run away."""

    type: Literal["likelihood"]
    """`likelihood`."""
    instructions: _NonEmpty
    """The request the candidates answer, as the user turn says it: "Write out
    the second sentence of the passage, spelled as it should be." Each candidate
    is scored as the start of the model's reply to it (thinking off)."""
    candidates: dict[_NonEmpty, _NonEmpty] = Field(min_length=2)
    """Candidate name to the reply text to score, in the order the answer lists
    them. 2 to 26 (`too_many_candidates`), each text unique, starting and
    ending on a non-space (a chat template's open reply drops trailing space),
    at most 256 tokens as it tokenizes after the context (`candidate_too_long`)."""
    rank_by: Literal["total", "mean"] = "total"
    """Which measure picks the `winner`. `total` (the default): the summed
    log-probability, the model's probability of the whole reply; right for
    variants of the same content ("cooperate" against "co-operate", OCR
    readings of one line), where the mean would reward a variant for being
    split into more, individually likely tokens. `mean`: the log-probability
    per token, for candidates whose lengths differ by content (chapter titles),
    where the total penalises every extra token."""

    @field_validator("candidates")
    @classmethod
    def _replies(cls, value: dict[str, str]) -> dict[str, str]:
        texts = list(value.values())
        if len(set(texts)) != len(texts):
            raise ValueError("candidate texts must be unique")
        for name, text in value.items():
            if text != text.strip():
                raise ValueError(
                    f"candidate {name!r} starts or ends with whitespace; the chat "
                    "template renders an open reply without its trailing space, so it "
                    "would not be the text that is scored"
                )
        return value


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, YesNoQuestion, LikelihoodQuestion],
    Field(discriminator="type"),
]


_Options = Annotated[dict[_NonEmpty, _NonEmpty], Field(min_length=2)]


class DecideItem(_Strict):
    """One item of the items form: its text, and optionally its own options."""

    text: _NonEmpty
    """The item, quoted verbatim into its own question (never referred to by
    number): a transcript passage, or a question about the image."""
    options: _Options | None = None
    """This item's own options (name to one-line description, letters A, B, C…
    in the order given, 2 to 26), rendered inline under the item. Absent: the
    request's `options`."""


class DecideRequest(_Strict):
    """`POST /v1/decide`: one forward pass per question at the resident model, or a list of items about one state; nothing decoded or loaded."""

    model: _NonEmpty | None = None
    """The Crucible model id. One that is not resident is loaded for the
    decision while it waits in the line (`409 model_not_resident` with
    `"queue": false`). An upstream id (`<upstream>/<id>`) is
    refused `400 decide_needs_logprobs`: no upstream returns a distribution.
    Absent: the model this server registered for `decide` (`GET /v1/capability`,
    the decide row's `selected`), or, when `images` are sent, the one it
    registered for a decision with images (that row's `with_images`: the vision
    form of the same weights when it fits, else the largest model that reads
    images and fits at or below decide's 9B goal). The answer's `model` names
    which one served it."""
    form: _NonEmpty | None = None
    """Which form of the model serves the decision, for a model whose block here
    states more than one (`GET /v1/models`, the row's `forms`). Absent: whichever
    form is resident, and the form this server's card takes when one is loaded.
    Another form of the same model on the card is a reload; a name the model does
    not have is refused `400 unknown_form` before anything waits."""
    state: Any
    """What the questions are about: a string, used verbatim, or any other JSON
    value, serialised as compact JSON. Required and never null; may be `""` only
    when `images` carry the state."""
    questions: Annotated[dict[_NonEmpty, Question], Field(min_length=1)] | None = None
    """Question name to question. Names are single path members (no `/`, `\\`,
    leading dot) and key the answers. Answers come back in this order. Exactly
    one of `questions` and `items` is sent."""
    instructions: _NonEmpty | None = None
    """The items form's ask, written under every item's text: "Which of the
    categories listed above does the speaker do in this passage?". Each item's
    question is `text`, a newline, then this; absent, `text` alone. Refused with
    `questions`."""
    options: _Options | None = None
    """The items form's shared options (name to one-line description, letters
    A, B, C… in the order given, 2 to 26). An item without its own `options`
    uses these."""
    items: Annotated[list[DecideItem], Field(min_length=1)] | None = None
    """The items form: an ordered list of choice questions about ONE state,
    each answered exactly as a lone choice question would be (it sees the state
    and its own question, never another item), in one request: on the Mac the
    shared state runs once and every item continues from its cache. Answers
    come back as a list in this order. At most 512 (`too_many_items`); token
    caps in docs/internals/api.md."""
    images: list[str] | None = None
    """Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no
    whitespace, no `data:` prefix), read as part of the state, after its text.
    At most 8 (`too_many_images`). A named `model` must serve `image` on this
    backend (`400 model_text_only` otherwise); with no `model`, the server
    takes the model it registered for a decision with images, and refuses
    `409 no_image_model_fits` when nothing that reads images fits its card.
    `[]` is the same as none."""
    missing: Literal["refuse", "report"] = "refuse"
    """What to do when a label is not among the top tokens the engine returned.
    `refuse` (the default): the decision is `502 label_not_in_probs` naming the
    question and the letter. `report`: the door never invents a number — that
    option's probability and log-probability are null, it is named in the
    answer's `missing_labels`, and the renormalisation, `confidence`, `score`
    and `label_mass` run over the letters actually returned. A question whose
    EVERY label is missing is refused in both modes: there is no answer to
    report."""
    queue: QueueChoice = queue_field()
    """Absent: while the model is not resident, or every slot on its engine is
    taken, the request is held open in the server's queue (docs/QUEUE.md) up to
    an hour, and the model is loaded for it when its turn comes.
    `{"max_wait_s": N}` changes the wait. `false`: refused at once instead
    (`409 model_not_resident`, `503 chat_queue_full`, `409 session_open`)."""

    @field_validator("state")
    @classmethod
    def _state_not_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("state is required and may not be null")
        return value

    @field_validator("questions")
    @classmethod
    def _names_are_path_members(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        from .jobs.base import validate_member_name

        for name in value or {}:
            validate_member_name(name)
        return value

    @field_validator("images")
    @classmethod
    def _images_are_strict_base64(cls, value: list[str] | None) -> list[str] | None:
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

    @model_validator(mode="after")
    def _one_form(self) -> "DecideRequest":
        if (self.questions is None) == (self.items is None):
            raise ValueError(
                "send exactly one of `questions` (named questions) and `items` "
                "(an ordered list read in one pass)"
            )
        if self.questions is not None:
            stray = [name for name in ("instructions", "options") if getattr(self, name) is not None]
            if stray:
                raise ValueError(f"{stray} belong to the items form; with `questions` "
                                 "each question carries its own")
            return self
        unlabelled = [i for i, item in enumerate(self.items or []) if item.options is None]
        if unlabelled and self.options is None:
            raise ValueError(
                f"items {unlabelled[:10]} carry no `options` and the request has no "
                "shared `options`; send `options` once, or on each item"
            )
        return self


class _Answer(_Strict):
    @model_serializer(mode="wrap")
    def _missing_only_when_reported(self, handler):
        data = handler(self)
        if getattr(self, "missing_labels") is None:
            data.pop("missing_labels", None)
        return data


class ChoiceAnswer(_Answer):
    """A choice question's distribution."""

    type: Literal["choice"] = "choice"
    """`choice`."""
    choice: str
    """The most probable option (of those returned, in report mode)."""
    probabilities: dict[str, float | None]
    """Option name to probability, in option order, renormalised over the
    letters so they sum to 1 (a softmax over the label logits). Null only for
    an option reported missing."""
    logprobs: dict[str, float | None]
    """Option name to ln of its `probabilities` entry, in option order; add
    ln `label_mass` (multiply the probability by `label_mass`) for the
    un-renormalised mass. NOT calibrated: one forward pass's reading, not a
    measured frequency. Null where the probability is null, or exactly 0
    (`-Infinity` is not JSON)."""
    confidence: float
    """The largest renormalised probability."""
    label_mass: float
    """The raw probability the option letters held together before renormalising
    (over the letters RETURNED, in report mode). Low means the model wanted to
    say something that is not an option. A renormalised probability times it
    is the un-renormalised mass."""
    missing_labels: list[str] | None = None
    """Present only when the request said `missing: "report"` — absent, not
    null, otherwise: the options whose letter was not among the top tokens the
    engine returned, in option order, `[]` when none was. Nothing is invented for
    them; their `probabilities` and `logprobs` are null."""


class ScoreAnswer(_Answer):
    """A score question's distribution and its expected level."""

    type: Literal["score"] = "score"
    """`score`."""
    score: float
    """Σ (1-based level index × p) over the levels returned: 1.0 is certainly
    the lowest level."""
    level: str
    """The most probable level (of those returned, in report mode)."""
    probabilities: dict[str, float | None]
    """Level to renormalised probability, lowest level first. Null only for a
    level reported missing."""
    logprobs: dict[str, float | None]
    """Level to ln of its `probabilities` entry, lowest first; add ln
    `label_mass` for the un-renormalised mass. NOT calibrated. Null where the
    probability is null, or exactly 0 (`-Infinity` is not JSON)."""
    confidence: float
    """The largest renormalised probability."""
    label_mass: float
    """The raw probability the level letters held together before renormalising
    (over the letters RETURNED, in report mode). Low means the model wanted to
    say something that is not an option. A renormalised probability times it
    is the un-renormalised mass."""
    missing_labels: list[str] | None = None
    """Present only when the request said `missing: "report"` — absent, not
    null, otherwise: the levels whose letter was not among the top tokens the
    engine returned, in level order, `[]` when none was. Nothing is invented for
    them; their `probabilities` and `logprobs` are null."""


class YesNoAnswer(_Answer):
    """A yesno question's probability."""

    type: Literal["yesno"] = "yesno"
    """`yesno`."""
    p: float
    """Renormalised P(Yes). In report mode with `A` or `B` missing it is the
    returned one renormalised alone — 1.0 or 0.0, which is honest and useless:
    gate on `label_mass`."""
    logprob: float | None
    """ln `p`; add ln `label_mass` for the un-renormalised mass. NOT
    calibrated. Null when `p` is exactly 0 (`-Infinity` is not JSON) — a
    report-mode answer with `Yes` missing."""
    label_mass: float
    """The raw probability the letters `A` and `B` held together before renormalising
    (over the letters RETURNED, in report mode). Low means the model wanted to
    say something that is not an option. A renormalised probability times it
    is the un-renormalised mass."""
    missing_labels: list[str] | None = None
    """Present only when the request said `missing: "report"` — absent, not
    null, otherwise: `["Yes"]` or `["No"]` when that letter was not among the
    top tokens the engine returned, `[]` when both were (both missing is
    refused)."""


class CandidateScore(_Strict):
    """One candidate's reading."""

    logprob: float
    """The summed log-probability of its tokens: ln P(this reply | the context).
    NOT calibrated."""
    tokens: int
    """How many tokens were scored: the candidate as it tokenizes after the
    context, plus the context's last token when the candidate's first
    characters merged into it (`boundary_tokens`)."""
    mean_logprob: float
    """`logprob` / `tokens`."""
    probability: float
    """A softmax over the candidates' `logprob` totals: the model's probability of
    this reply renormalised over the replies offered. Always over totals,
    whatever `rank_by` says, because that is the quantity with a meaning."""


class LikelihoodAnswer(_Strict):
    """A likelihood question's reading: every candidate's log-likelihood."""

    type: Literal["likelihood"] = "likelihood"
    """`likelihood`."""
    winner: str
    """The candidate with the largest `logprob` (`rank_by: "total"`) or
    `mean_logprob` (`"mean"`); on a tie, the first in request order. With
    `"mean"` it need not hold the largest `probability`."""
    rank_by: Literal["total", "mean"]
    """The measure `winner` was picked by, echoed from the question."""
    candidates: dict[str, CandidateScore]
    """Candidate name to its reading, in the request's order."""
    context_tokens: int
    """The context's tokens: the state, the request and the opened reply."""
    boundary_tokens: int
    """Context tokens read again with the candidates because a candidate's
    first characters merged with the context's last token: 0 or 1."""


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, YesNoAnswer, LikelihoodAnswer],
    Field(discriminator="type"),
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
    than one question, null when it had one, and null when the engine read every
    question in one batched request (mlx-lm: each question's timing is then that
    one request)."""


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
    """Which weights answered (`{id, revision, fingerprint}`, docs/internals/jobs-runtime.md "Provenance sidecars")."""
    engine: str
    """The engine kind that answered: `vllm`, `llama-server`, `mlx-lm`."""
    answers: dict[str, Answer]
    """Question name to answer, in the request's question order."""
    timing_ms: DecideTiming
    """Crucible's clock, per request."""
    tokens: DecideTokens
    """Prompt sizes."""


class ItemsTiming(_Strict):
    """Where the items form's time went."""

    total: float
    """The whole decision, ms, Crucible's clock."""
    engine_requests: int
    """1 when the engine read every item in one batched request (mlx-lm,
    mlx-vlm); otherwise one per item plus the shared prefix sent first."""


class ItemsTokens(_Strict):
    """How big the items form's prompts were."""

    shared: int | None
    """The tokens every item's prompt shares (system, state, images), run once;
    null where each item went as its own request."""
    per_item: list[int]
    """Each item's whole prompt (shared part included), in item order."""
    images: int
    """How many images every item's prompt carried."""


class DecideItemsResponse(_Strict):
    """An items-form decision: one choice distribution per item, in item order."""

    model: ModelProvenance
    """Which weights answered (`{id, revision, fingerprint}`)."""
    engine: str
    """The engine kind that answered: `vllm`, `mlx-lm`, `mlx-vlm`."""
    answers: list[ChoiceAnswer]
    """One choice answer per item, in the request's item order, in the shape
    a lone choice question answers with."""
    timing_ms: ItemsTiming
    """Crucible's clock."""
    tokens: ItemsTokens
    """Prompt sizes."""


@dataclass(frozen=True)
class Plan:
    name: str
    question: ChoiceQuestion | ScoreQuestion | YesNoQuestion | LikelihoodQuestion
    labels: tuple[tuple[str, str], ...]
    legend: tuple[tuple[str, str], ...]

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(letter for letter, _ in self.labels)


def assign_labels(names: list[str], question: str) -> tuple[tuple[str, str], ...]:
    if len(names) > MAX_OPTIONS:
        raise ApiError(
            400,
            "too_many_options",
            f"question {question!r} has {len(names)} options; the label set is "
            f"A..Z ({MAX_OPTIONS}). Split it into two questions",
            {"question": question, "options": len(names), "max_options": MAX_OPTIONS},
        )
    return tuple(zip(LETTERS, names))


def check_candidate_count(question: LikelihoodQuestion, name: str) -> None:
    count = len(question.candidates)
    if count > MAX_CANDIDATES:
        raise ApiError(
            400,
            "too_many_candidates",
            f"question {name!r} has {count} candidates; a likelihood question scores "
            f"at most {MAX_CANDIDATES}. Split it, or drop the candidates no reading "
            "would choose",
            {"question": name, "candidates": count, "max_candidates": MAX_CANDIDATES},
        )


def plan(
    name: str, question: ChoiceQuestion | ScoreQuestion | YesNoQuestion | LikelihoodQuestion
) -> Plan:
    if isinstance(question, LikelihoodQuestion):
        check_candidate_count(question, name)
        return Plan(name=name, question=question, labels=(), legend=())
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
    return [plan(name, question) for name, question in (request.questions or {}).items()]


def check_image_count(images: list[str] | None) -> int:
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
    wanted = n_labels + LABEL_MARGIN
    return wanted if max_logprobs is None else min(wanted, max_logprobs)


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def question_block(kind: str, instructions: str, legend: tuple[tuple[str, str], ...]) -> str:
    if kind == "yesno":
        head = f"Statement: {instructions}\nIs this statement true of the state above?"
    else:
        head = f"Question: {instructions}"
    lines = "\n".join(f"{letter}. {text}" for letter, text in legend)
    return f"{head}\nOptions:\n{lines}\nAnswer with the letter only."


def image_format(raw: bytes) -> str | None:
    for magic, offset, name in IMAGE_SIGNATURES:
        if raw[offset : offset + len(magic)] == magic:
            if name == "webp" and raw[:4] != b"RIFF":
                continue
            return name
    return None


def image_part(encoded: str) -> dict[str, Any]:
    kind = image_format(base64.b64decode(encoded, validate=True))
    if kind is None:
        raise ApiError(400, "invalid_request", "an image is not a known format")
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{encoded}"}}


def system_content(state_text: str, has_images: bool, prompt: str = SYSTEM_PROMPT) -> str:
    content = prompt + STATE_HEADER + state_text
    if has_images:
        content += ("\n\n" if state_text else "") + IMAGES_NOTE
    return content


def _user_content(images: list[str], text: str) -> str | list[dict[str, Any]]:
    if not images:
        return text
    parts: list[dict[str, Any]] = [image_part(encoded) for encoded in images]
    parts.append({"type": "text", "text": text})
    return parts


def messages(state_text: str, images: list[str], block: str | None) -> list[dict[str, Any]]:
    user_text = PRIME_USER_TEXT if block is None else block
    return [
        {"role": "system", "content": system_content(state_text, bool(images))},
        {"role": "user", "content": _user_content(images, user_text)},
    ]


def question_messages(state_text: str, images: list[str], item: Plan) -> list[dict[str, Any]]:
    block = question_block(item.question.type, item.question.instructions, item.legend)
    return messages(state_text, images, block)


def prime_messages(state_text: str, images: list[str]) -> list[dict[str, Any]]:
    return messages(state_text, images, None)


def request_body(
    engine_model_name: str, msgs: list[dict[str, Any]], k: int | None
) -> dict[str, Any]:
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


@dataclass(frozen=True)
class Reading:
    prompt_tokens: int
    cached_tokens: int | None
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


@dataclass(frozen=True)
class Distribution:
    probabilities: dict[str, float | None]
    mass: float
    missing: tuple[str, ...]


def label_distribution(
    top: tuple[tuple[str, float], ...],
    item: Plan,
    engine: str,
    missing: Literal["refuse", "report"] = "refuse",
) -> Distribution:
    letters = set(item.letters)
    by_token: dict[str, float] = {}
    for token, probability in top:
        if token in by_token and token in letters:
            raise _engine_error(
                engine, f"label token {token!r} appears twice in top_logprobs"
            )
        by_token.setdefault(token, probability)
    raw: dict[str, float | None] = {}
    absent: list[str] = []
    for letter, option in item.labels:
        if letter in by_token:
            raw[option] = by_token[letter]
            continue
        if missing == "refuse":
            raise ApiError(
                502,
                "label_not_in_probs",
                f"question {item.name!r}: label {letter!r} (option {option!r}) is not "
                f"among the top {len(top)} tokens the {engine} engine returned. The "
                "model wanted to say something else strongly enough to push a "
                "letter out; fewer options, or plainer ones, is the repair (or "
                "`missing: \"report\"`, which answers over the letters returned)",
                {"question": item.name, "letter": letter, "option": option,
                 "engine": engine, "top_k": len(top)},
            )
        raw[option] = None
        absent.append(option)
    if len(absent) == len(item.labels):
        raise ApiError(
            502,
            "label_not_in_probs",
            f"question {item.name!r}: none of its labels "
            f"{'/'.join(item.letters)} is among the top {len(top)} tokens the "
            f"{engine} engine returned, so there is no answer to report. Fewer "
            "options, or plainer ones, is the repair",
            {"question": item.name, "letter": None, "option": None,
             "engine": engine, "top_k": len(top)},
        )
    mass = sum(p for p in raw.values() if p is not None)
    if mass <= 0.0:
        raise ApiError(
            502,
            "label_not_in_probs",
            f"question {item.name!r}: every label the {engine} engine returned has "
            "probability 0",
            {"question": item.name, "letter": None, "option": None,
             "engine": engine, "top_k": len(top)},
        )
    return Distribution(
        probabilities={
            option: None if p is None else p / mass for option, p in raw.items()
        },
        mass=mass,
        missing=tuple(absent),
    )


def _ln(p: float | None) -> float | None:
    return None if p is None or p <= 0.0 else math.log(p)


def answer(
    item: Plan, dist: Distribution, missing: Literal["refuse", "report"]
) -> ChoiceAnswer | ScoreAnswer | YesNoAnswer:
    question = item.question
    probabilities = dist.probabilities
    missing_labels = list(dist.missing) if missing == "report" else None
    if isinstance(question, YesNoQuestion):
        p_yes, p_no = probabilities["Yes"], probabilities["No"]
        if p_yes is not None:
            p = p_yes
        else:
            assert p_no is not None
            p = 1.0 - p_no
        return YesNoAnswer(
            p=p, logprob=_ln(p), label_mass=dist.mass, missing_labels=missing_labels
        )
    returned = {option: p for option, p in probabilities.items() if p is not None}
    best = max(returned, key=returned.__getitem__)
    logprobs = {option: _ln(p) for option, p in probabilities.items()}
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=best,
            probabilities=probabilities,
            logprobs=logprobs,
            confidence=returned[best],
            label_mass=dist.mass,
            missing_labels=missing_labels,
        )
    score = sum(
        (index + 1) * returned[level]
        for index, (_, level) in enumerate(item.labels)
        if level in returned
    )
    return ScoreAnswer(
        score=score,
        level=best,
        probabilities=probabilities,
        logprobs=logprobs,
        confidence=returned[best],
        label_mass=dist.mass,
        missing_labels=missing_labels,
    )


EnginePost = Callable[[dict[str, Any]], Awaitable[Any]]


def decide_not_served(
    resident: Any, reason: str, extra: dict[str, Any] | None
) -> ApiError:
    return ApiError(
        503,
        "decide_not_served",
        f"the {resident.engine} engine serving {resident.model_id!r} cannot serve "
        f"this decision: {reason}. Nothing was sent to it",
        {"model": resident.model_id, "engine": resident.engine, "reason": reason,
         **(extra or {})},
    )


def _image_models_offer(backend_kind: str, image_models: list[str]) -> str:
    if not image_models:
        return (
            f" No model this build ships answers images on {backend_kind}; send "
            "the decision without `images`, or to a Crucible on another backend"
        )
    load = json.dumps({"type": "load-model", "model": image_models[0]})
    return (
        f" The models that answer images on {backend_kind}, largest first, are "
        f"{image_models}: load one (POST /v1/jobs {load}) and send the decision "
        "to it"
    )


def refuse_images_not_served(
    model_id: str,
    manifest: Any,
    backend_kind: str,
    n_images: int,
    image_models: Callable[[], list[str]] = list,
) -> None:
    served = manifest.serves(backend_kind)
    if "image" in served:
        return
    offered = image_models()
    raise ApiError(
        400,
        "model_text_only",
        f"{model_id!r} is served {list(served)} on "
        f"{backend_kind} (its weights accept "
        f"{list(manifest.modalities)}) and this decision carries "
        f"{n_images} image(s). Whether a model answers images HERE is "
        "its manifest's backend block (`serves`, docs/internals/engines-and-capability.md \"The decision door\")."
        + _image_models_offer(backend_kind, offered),
        {"model": model_id, "backend": backend_kind,
         "serves": list(served),
         "modalities": list(manifest.modalities),
         "images": n_images,
         "image_models": offered},
    )


def label_plans(plans: list[Plan]) -> list[Plan]:
    return [item for item in plans if not isinstance(item.question, LikelihoodQuestion)]


def likelihood_plans(plans: list[Plan]) -> list[Plan]:
    return [item for item in plans if isinstance(item.question, LikelihoodQuestion)]


def refuse_unreadable_labels(resident: Any, reading: Any, plans: list[Plan]) -> None:
    plans = label_plans(plans)
    if not plans:
        return
    if not reading.served:
        raise decide_not_served(resident, reading.basis, None)
    widest = max(plans, key=lambda item: len(item.labels))
    if reading.max_logprobs is not None and len(widest.labels) > reading.max_logprobs:
        raise decide_not_served(
            resident,
            f"{resident.engine} returns at most {reading.max_logprobs} top "
            f"logprobs and question {widest.name!r} has {len(widest.labels)} "
            f"options ({reading.basis})",
            {"question": widest.name, "options": len(widest.labels),
             "max_logprobs": reading.max_logprobs},
        )


async def _gathered_in_order(tasks: list[asyncio.Task[Any]]) -> list[Any]:
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if task.cancelled():
                continue
            failure = task.exception()
            if failure is not None:
                raise failure from None
        raise


def _timing(reading: Reading, wall_ms: float) -> ForwardTiming:
    return ForwardTiming(
        wall_ms=round(wall_ms, 1),
        prompt_tokens=reading.prompt_tokens,
        cached_tokens=reading.cached_tokens,
    )


async def decide_on_engine(
    post: EnginePost,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
    *,
    max_logprobs: int | None,
    concurrency: int,
) -> DecideResponse:
    started = time.perf_counter()
    state_text = render_state(body.state)
    images = list(body.images or [])
    engine = resident.engine

    async def forward(msgs: list[dict[str, Any]], k: int | None) -> tuple[Reading, float]:
        sent = time.perf_counter()
        data = await post(request_body(resident.engine_model_name, msgs, k))
        wall_ms = (time.perf_counter() - sent) * 1000.0
        return read_reply(data, engine, want_probs=k is not None), wall_ms

    prime: ForwardTiming | None = None
    if len(plans) > 1:
        prime = _timing(*await forward(prime_messages(state_text, images), None))

    gate = asyncio.Semaphore(concurrency)

    async def ask(item: Plan) -> tuple[Any, ForwardTiming, int]:
        k = top_k(len(item.labels), max_logprobs)
        async with gate:
            reading, wall_ms = await forward(question_messages(state_text, images, item), k)
        assert reading.top is not None
        dist = label_distribution(reading.top, item, engine, missing=body.missing)
        return answer(item, dist, body.missing), _timing(reading, wall_ms), reading.prompt_tokens

    results = await _gathered_in_order([asyncio.create_task(ask(item)) for item in plans])

    answers = {}
    per_question = {}
    tokens = {}
    for item, (answered, timed, prompt_tokens) in zip(plans, results):
        answers[item.name] = answered
        per_question[item.name] = timed
        tokens[item.name] = prompt_tokens
    return DecideResponse(
        model=ModelProvenance(
            id=resident.model_id,
            revision=resident.revision,
            fingerprint=resident.fingerprint,
        ),
        engine=engine,
        answers=answers,
        timing_ms=DecideTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            per_question=per_question,
            prime=prime,
        ),
        tokens=DecideTokens(per_question=tokens, images=len(images)),
    )


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "CandidateScore",
    "ChoiceQuestion",
    "DecideItem",
    "DecideItemsResponse",
    "DecideRequest",
    "DecideResponse",
    "DecideTiming",
    "DecideTokens",
    "Distribution",
    "EnginePost",
    "ForwardTiming",
    "IMAGES_NOTE",
    "ItemsTiming",
    "ItemsTokens",
    "LABEL_MARGIN",
    "LETTERS",
    "LIKELIHOOD_SYSTEM_PROMPT",
    "LikelihoodAnswer",
    "LikelihoodQuestion",
    "MAX_CANDIDATES",
    "MAX_CANDIDATE_TOKENS",
    "MAX_IMAGES",
    "MAX_LEVELS",
    "MAX_OPTIONS",
    "ModelProvenance",
    "PRIME_USER_TEXT",
    "Plan",
    "Reading",
    "STATE_HEADER",
    "SYSTEM_PROMPT",
    "ScoreAnswer",
    "ScoreQuestion",
    "YESNO_OPTIONS",
    "YesNoAnswer",
    "YesNoQuestion",
    "answer",
    "assign_labels",
    "check_candidate_count",
    "check_image_count",
    "decide_not_served",
    "decide_on_engine",
    "image_format",
    "image_part",
    "label_distribution",
    "label_plans",
    "likelihood_plans",
    "messages",
    "plan",
    "plan_all",
    "prime_messages",
    "question_block",
    "question_messages",
    "read_reply",
    "refuse_images_not_served",
    "refuse_unreadable_labels",
    "render_state",
    "request_body",
    "system_content",
    "top_k",
]
