"""`POST /v1/decide`, end to end through the fake engine (PHASE22-DECIDE.md).

No GPU and no weights: the env, the weights and the engine are the chat door's
fixtures, and the fake engine answers chat logprobs in vLLM 0.29.0's exact shape
(`tests/fake_engine.py`, `probs_for`). Nothing about the door is faked — the act
header, the resident check, the admission, the in-flight record, the prime, the
fan-out and the reading are what runs on the card.

"Before any forward pass" is asserted the only way it can be: the fake engine's
own request log is EMPTY after the refusal. A refusal that half-ran would have
spent the card on a decision the client is about to make again.
"""

from __future__ import annotations

import base64
import re
import threading
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import residency as residency_module
from crucible.decide import LABEL_MARGIN, SYSTEM_PROMPT, UNSTATED_ENGINE_CONCURRENCY
from crucible.engines import ENGINES

from .fake_engine import FakeEngine
from .test_llm_api import fake_env, llm_client, run_job  # noqa: F401 - fixtures

MODEL = "qwen3.5-9b"
PAGE_MODEL = "dots-ocr"

EXAMPLE: dict[str, Any] = {
    "model": MODEL,
    "state": "Hi, I was charged twice for my subscription this month. Please fix it today.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "options": {"billing": "Payment and invoice issues",
                             "technical": "Bugs and errors", "other": "Anything else"}},
        "anger": {"type": "score", "instructions": "How frustrated is the customer?",
                  "levels": ["Calm", "Frustrated but civil", "Very angry"]},
        "urgent": {"type": "yesno", "instructions": "The message conveys urgency"},
    },
}

#: Raw (pre-renormalisation) letter probabilities per question, snap's numbers:
#: each renormalises to the contract's worked answer and leaves `label_mass`.
EXAMPLE_RAW = {
    "Which team should handle this?": {"A": 0.91 * 0.998, "B": 0.07 * 0.998, "C": 0.02 * 0.998},
    "How frustrated is the customer?": {"A": 0.62 * 0.997, "B": 0.36 * 0.997, "C": 0.02 * 0.997},
    "The message conveys urgency": {"A": 0.83 * 0.99, "B": 0.17 * 0.99},
}

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n a page").decode()


def _user_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part["type"] == "text")


def question_of(messages: list[dict[str, Any]]) -> str | None:
    """The question line, or None for a prime (the shared prefix has none)."""
    found = re.search(r"(?:Question|Statement): (.*)\n", _user_text(messages))
    return found.group(1) if found else None


def example_probs(messages: list[dict[str, Any]]) -> dict[str, float]:
    question = question_of(messages)
    return {} if question is None else EXAMPLE_RAW[question]


def yes_mostly(messages: list[dict[str, Any]]) -> dict[str, float]:
    return {"A": 0.7, "B": 0.2}


def _decide(client: TestClient, auth: dict[str, str], body: dict[str, Any],
            **headers: str) -> Any:
    return client.post("/v1/decide", headers={**auth, **headers}, json=body)


@pytest.fixture
def loaded(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> Callable[..., FakeEngine]:
    """Load MODEL on a fake engine built with these options; return the engine."""

    def load(model: str = MODEL, **options: Any) -> FakeEngine:
        built = engine_factory(**options)
        fake_weights(model)
        events = run_job(llm_client, auth, type="load-model", model=model)
        assert events[-1]["event"] == "done", events[-1]
        return built[-1]

    return load


# --------------------------------------------------------------- the example


def test_the_worked_example_end_to_end(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=example_probs)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 200, response.text
    body = response.json()

    team, anger, urgent = (body["answers"][k] for k in ("team", "anger", "urgent"))
    assert team["type"] == "choice" and team["choice"] == "billing"
    assert team["probabilities"] == pytest.approx(
        {"billing": 0.91, "technical": 0.07, "other": 0.02})
    assert team["confidence"] == pytest.approx(0.91)
    assert team["label_mass"] == pytest.approx(0.998)
    assert anger["type"] == "score" and anger["score"] == pytest.approx(1.4)
    assert anger["level"] == "Calm" and anger["label_mass"] == pytest.approx(0.997)
    assert urgent == {"type": "yesno", "p": pytest.approx(0.83),
                      "label_mass": pytest.approx(0.99)}
    assert list(body["answers"]) == ["team", "anger", "urgent"]

    assert body["engine"] == "vllm"
    # A prime and three questions, and nothing else.
    assert len(engine.requests) == 4
    prime = body["timing_ms"]["prime"]
    assert prime is not None and prime["prompt_tokens"] > 0
    for name in ("team", "anger", "urgent"):
        timed = body["timing_ms"]["per_question"][name]
        assert timed["wall_ms"] >= 0.0
        assert timed["prompt_tokens"] == body["tokens"]["per_question"][name]
        # Every question reused the primed prefix, and the engine said so.
        assert timed["cached_tokens"] > 0
    assert body["tokens"]["images"] == 0
    assert body["timing_ms"]["total"] >= 0.0


def test_the_model_triple_is_the_load_s_own_provenance(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """A decision names the weights that made it, from the same source the
    artifact sidecar uses — not a second spelling of it."""
    loaded(probs_for=example_probs)
    body = _decide(llm_client, auth, EXAMPLE).json()
    store = llm_client.app.state.store
    load_job = store.create("load-model", MODEL, {})
    assert body["model"] == store.provenance(load_job)["model"]
    assert body["model"]["fingerprint"] == f"{MODEL}@{body['model']['revision']}"


def test_an_engine_that_does_not_report_its_cache_says_null_not_zero(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """vLLM without `--enable-prompt-tokens-details`. Zero would be a
    measurement nobody made."""
    loaded(probs_for=example_probs, report_cached=False)
    body = _decide(llm_client, auth, EXAMPLE).json()
    assert body["timing_ms"]["prime"]["cached_tokens"] is None
    for name in ("team", "anger", "urgent"):
        timed = body["timing_ms"]["per_question"][name]
        assert timed["cached_tokens"] is None
        assert timed["prompt_tokens"] > 0, "the prompt size is still known"


# ------------------------------------------------------- what reaches the engine


def test_the_prime_goes_first_and_every_question_extends_it(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=example_probs)
    assert _decide(llm_client, auth, EXAMPLE).status_code == 200
    prime, *questions = engine.requests

    assert question_of(prime["messages"]) is None
    assert prime["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert prime["messages"][1]["content"] == f"State:\n{EXAMPLE['state']}"
    # A prime is not an answer: it asks for no logprobs.
    assert "logprobs" not in prime and "top_logprobs" not in prime
    assert prime["max_tokens"] == 1

    expected_k = {"Which team should handle this?": 3 + LABEL_MARGIN,
                  "How frustrated is the customer?": 3 + LABEL_MARGIN,
                  "The message conveys urgency": 2 + LABEL_MARGIN}
    assert sorted(question_of(q["messages"]) for q in questions) == sorted(expected_k)
    for sent in questions:
        assert sent["messages"][0] == prime["messages"][0]
        assert sent["messages"][1]["content"].startswith(
            prime["messages"][1]["content"] + "\n\n")
        assert sent["model"] == MODEL
        assert sent["max_tokens"] == 1 and sent["temperature"] == 0
        assert sent["logprobs"] is True
        assert sent["top_logprobs"] == expected_k[question_of(sent["messages"])]
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["stream"] is False

    # The prime FINISHED before any question left.
    assert engine.events[:2] == [("start", 0), ("end", 0)]


def test_one_question_is_not_primed(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=example_probs)
    body = {**EXAMPLE, "questions": {"urgent": EXAMPLE["questions"]["urgent"]}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 200, response.text
    assert response.json()["timing_ms"]["prime"] is None
    assert len(engine.requests) == 1
    assert question_of(engine.requests[0]["messages"]) == "The message conveys urgency"


def test_the_engine_is_sent_its_own_name_for_the_model(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx-lm answers to its weights directory, not the Crucible id; the chat
    door substitutes the one field and so does a decision."""
    built = engine_factory(probs_for=example_probs)
    monkeypatch.setattr(
        residency_module, "engine_model_name",
        lambda engine_name, model_dir, model_id: f"/weights/{model_id}",
    )
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 200, response.text
    assert {sent["model"] for sent in built[0].requests} == {f"/weights/{MODEL}"}
    assert response.json()["model"]["id"] == MODEL


def test_questions_go_out_together_up_to_the_unstated_ceiling(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """vLLM states no admission, so a decision's own ceiling is Crucible's
    number. Twenty questions that each take a moment: never more than sixteen
    open at once, and genuinely more than one."""
    engine = loaded(probs_for=yes_mostly, answer_delay=0.3)
    questions = {f"q{i}": {"type": "yesno", "instructions": f"statement {i}"}
                 for i in range(20)}
    response = _decide(llm_client, auth, {**EXAMPLE, "questions": questions})
    assert response.status_code == 200, response.text
    assert list(response.json()["answers"]) == list(questions)
    assert 1 < engine.max_in_flight <= UNSTATED_ENGINE_CONCURRENCY


def test_a_serial_engine_is_asked_no_more_than_its_admission(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama-server behind `--parallel 1`, or mlx-lm: admission is 2, and a
    decision's fan-out is held to it rather than queueing inside the engine."""
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency", 1, raising=False)
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_basis", "one slot",
                        raising=False)
    engine = loaded(probs_for=yes_mostly, answer_delay=0.2)
    questions = {f"q{i}": {"type": "yesno", "instructions": f"statement {i}"}
                 for i in range(6)}
    response = _decide(llm_client, auth, {**EXAMPLE, "questions": questions})
    assert response.status_code == 200, response.text
    assert engine.max_in_flight == 2


# ---------------------------------------------------------- engine-side faults


def test_a_letter_outside_the_top_k_is_label_not_in_probs(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    def probs(messages: list[dict[str, Any]]) -> dict[str, float]:
        if question_of(messages) == "How frustrated is the customer?":
            return {"A": 0.6, "B": 0.3}  # C (Very angry) never appears
        return example_probs(messages)

    loaded(probs_for=probs)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "label_not_in_probs"
    assert "'anger'" in error["message"] and "'C'" in error["message"]
    assert error["details"]["question"] == "anger" and error["details"]["letter"] == "C"


def test_an_engine_refusal_is_engine_error_quoting_the_engine(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fake refuses `top_logprobs` past ITS cap with vLLM's own 400, which
    is what the card does to a reader that asks for more than the engine was
    started with. Here the class says 32 and the engine says 20."""
    engine = loaded(probs_for=yes_mostly, max_logprobs=20)
    options = {f"o{i}": "d" for i in range(17)}  # 17 + 4 = 21 > 20
    body = {**EXAMPLE, "questions": {"wide": {"type": "choice", "instructions": "x",
                                              "options": options}}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "engine_error"
    assert error["details"]["engine"] == "vllm" and error["details"]["status"] == 400
    assert "greater than max allowed: 20" in error["message"]
    assert engine.requests[0]["top_logprobs"] == 21


# ---------------------------------------- refusals made before any forward pass


def test_too_many_options_is_refused_before_anything_is_sent(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=yes_mostly)
    body = {**EXAMPLE, "questions": {
        "ok": {"type": "yesno", "instructions": "x"},
        "big": {"type": "choice", "instructions": "y",
                "options": {f"o{i}": "d" for i in range(27)}}}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "too_many_options"
    assert engine.requests == []


def test_too_many_images_is_refused_before_anything_is_sent(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "images": [PNG] * 9})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "too_many_images"
    assert engine.requests == []


def test_images_on_a_text_model_are_model_text_only(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """`qwen3.5-9b` is declared text on every backend today (PHASE22 section
    2.7). Sending its images would have them dropped or refused inside the
    engine; the door says which manifest decides it instead."""
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "images": [PNG]})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_text_only"
    assert error["details"] == {"model": MODEL, "modalities": ["text"], "images": 1}
    assert engine.requests == []


def test_an_upstream_model_is_decide_needs_logprobs(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "model": "anthropic/claude"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "decide_needs_logprobs"
    assert engine.requests == []


def test_an_engine_that_returns_no_logprobs_is_decide_not_served(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["vllm"], "decide_logprobs", False)
    monkeypatch.setattr(ENGINES["vllm"], "max_logprobs", None)
    monkeypatch.setattr(ENGINES["vllm"], "decide_basis", "this build reads none")
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "decide_not_served"
    assert error["details"]["engine"] == "vllm"
    assert "this build reads none" in error["message"]
    assert engine.requests == []


def test_more_options_than_the_engine_s_cap_is_decide_not_served(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx-lm returns at most 11. A 12-option question cannot be read there,
    and asking would be the engine's own 400 after the card was spent on the
    prime."""
    monkeypatch.setattr(ENGINES["vllm"], "max_logprobs", 11)
    engine = loaded(probs_for=yes_mostly)
    body = {**EXAMPLE, "questions": {
        "ok": {"type": "yesno", "instructions": "x"},
        "wide": {"type": "choice", "instructions": "y",
                 "options": {f"o{i}": "d" for i in range(12)}}}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "decide_not_served"
    assert error["details"]["question"] == "wide"
    assert error["details"]["max_logprobs"] == 11
    assert engine.requests == []


def test_an_unknown_act_is_refused_before_the_work(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, EXAMPLE, **{"X-Crucible-Act": "analysys"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_act"
    assert engine.requests == []


def test_a_model_that_is_not_resident_gets_the_chat_door_s_409(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """Crucible never loads a model to answer a decision."""
    empty = _decide(llm_client, auth, EXAMPLE)
    assert empty.status_code == 409
    assert empty.json()["error"]["details"] == {"requested": MODEL, "resident": None}

    engine = loaded(probs_for=yes_mostly)
    wrong = _decide(llm_client, auth, {**EXAMPLE, "model": "qwen3.8-27b-4bit"})
    chat = llm_client.post(
        "/v1/openai/chat/completions", headers=auth,
        json={"model": "qwen3.8-27b-4bit", "messages": [{"role": "user", "content": "hi"}]})
    assert wrong.status_code == chat.status_code == 409
    mine, theirs = wrong.json()["error"], chat.json()["error"]
    assert mine["code"] == theirs["code"] == "model_not_resident"
    assert mine["details"] == theirs["details"]
    assert mine["message"] == theirs["message"].replace("a chat request", "a decision")
    assert engine.requests == []


def test_a_full_door_is_chat_queue_full(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency", 1, raising=False)
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_basis", "one slot",
                        raising=False)
    engine = loaded(probs_for=yes_mostly)
    inflight = llm_client.app.state.inflight
    held = [inflight.open(act=None, model=MODEL, client="a-test") for _ in range(2)]
    try:
        response = _decide(llm_client, auth, EXAMPLE)
    finally:
        for entry in held:
            inflight.close(entry)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "chat_queue_full"
    assert error["details"]["max_in_flight"] == 2
    assert engine.requests == []


def test_the_door_needs_auth(llm_client: TestClient) -> None:  # noqa: F811
    response = llm_client.post("/v1/decide", json=EXAMPLE)
    assert response.status_code == 401


def test_a_malformed_body_is_invalid_request_naming_the_field(
    llm_client: TestClient, auth: dict[str, str]  # noqa: F811
) -> None:
    response = _decide(llm_client, auth, {**EXAMPLE, "state": None})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert ["body", "state"] in [p["location"] for p in error["details"]["problems"]]


# ------------------------------------------------------------ the record


def test_a_decision_in_flight_is_on_the_activity_bench_and_then_gone(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """A decision IS a completion of one token, and the bench says so — under
    the act the client named."""
    seen: list[dict[str, Any]] = []
    looked = threading.Event()

    def look() -> None:
        if not looked.is_set():
            looked.set()
            seen.append(llm_client.get("/v1/activity", headers=auth).json())

    loaded(probs_for=example_probs, on_post=look)
    response = _decide(llm_client, auth, EXAMPLE, **{"X-Crucible-Act": "analysis"})
    assert response.status_code == 200, response.text

    chat = seen[0]["chat"]
    assert chat["in_flight"] == 1
    assert chat["rows"][0]["act"] == "analysis"
    assert chat["rows"][0]["model"] == MODEL
    # A decision takes no lane, as a chat takes none.
    assert seen[0]["running"] == []

    after = llm_client.get("/v1/activity", headers=auth).json()
    assert after["chat"]["in_flight"] == 0 and after["chat"]["rows"] == []


# ------------------------------------------------------------ images


def test_images_travel_as_data_uri_parts_ahead_of_the_text(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    """On a model whose manifest declares `image` (the page reader is the one
    that does today), every request carries the same images first."""
    engine = loaded(model=PAGE_MODEL, probs_for=example_probs)
    body = {**EXAMPLE, "model": PAGE_MODEL, "state": "", "images": [PNG]}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 200, response.text
    assert response.json()["tokens"]["images"] == 1
    prime, *questions = engine.requests
    part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    assert prime["messages"][1]["content"] == [{"type": "text", "text": "State:"}, part]
    for sent in questions:
        content = sent["messages"][1]["content"]
        assert content[:2] == [{"type": "text", "text": "State:"}, part]
        assert content[2]["type"] == "text"
