from __future__ import annotations

import base64
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import engines as engines_module
from crucible.decide import (
    LABEL_MARGIN,
    PRIME_USER_TEXT,
    SYSTEM_PROMPT,
)
from crucible.engines import ENGINES

from .fake_engine import END_OF_TURN, FakeEngine, _rendered, system_segment
from .test_llm_api import fake_env, llm_client, run_job

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
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> Callable[..., FakeEngine]:

    def load(model: str = MODEL, **options: Any) -> FakeEngine:
        built = engine_factory(**options)
        fake_weights(model)
        events = run_job(llm_client, auth, type="load-model", model=model)
        assert events[-1]["event"] == "done", events[-1]
        return built[-1]

    return load


def test_the_worked_example_end_to_end(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
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
                      "logprob": pytest.approx(math.log(0.83)),
                      "label_mass": pytest.approx(0.99)}
    assert list(body["answers"]) == ["team", "anger", "urgent"]
    assert list(team["logprobs"]) == ["billing", "technical", "other"]
    assert team["logprobs"] == pytest.approx(
        {k: math.log(v) for k, v in team["probabilities"].items()})
    assert list(anger["logprobs"]) == ["Calm", "Frustrated but civil", "Very angry"]
    for answer in (team, anger, urgent):
        assert "missing_labels" not in answer

    assert body["engine"] == "vllm"
    assert len(engine.requests) == 4
    prime = body["timing_ms"]["prime"]
    assert prime is not None and prime["prompt_tokens"] > 0
    for name in ("team", "anger", "urgent"):
        timed = body["timing_ms"]["per_question"][name]
        assert timed["wall_ms"] >= 0.0
        assert timed["prompt_tokens"] == body["tokens"]["per_question"][name]
        assert timed["cached_tokens"] > 0
    assert body["tokens"]["images"] == 0
    assert body["timing_ms"]["total"] >= 0.0


def test_the_model_triple_is_the_load_s_own_provenance(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    loaded(probs_for=example_probs)
    body = _decide(llm_client, auth, EXAMPLE).json()
    store = llm_client.app.state.store
    load_job = store.create("load-model", MODEL, {})
    assert body["model"] == store.provenance(load_job)["model"]
    assert body["model"]["fingerprint"] == f"{MODEL}@{body['model']['revision']}"


def test_an_engine_that_does_not_report_its_cache_says_null_not_zero(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    loaded(probs_for=example_probs, report_cached=False)
    body = _decide(llm_client, auth, EXAMPLE).json()
    assert body["timing_ms"]["prime"]["cached_tokens"] is None
    for name in ("team", "anger", "urgent"):
        timed = body["timing_ms"]["per_question"][name]
        assert timed["cached_tokens"] is None
        assert timed["prompt_tokens"] > 0, "the prompt size is still known"


def test_the_prime_goes_first_and_every_question_extends_it(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=example_probs)
    assert _decide(llm_client, auth, EXAMPLE).status_code == 200
    prime, *questions = engine.requests

    assert question_of(prime["messages"]) is None
    assert prime["messages"][0] == {
        "role": "system", "content": f"{SYSTEM_PROMPT}\n\nState:\n{EXAMPLE['state']}"}
    assert prime["messages"][1] == {"role": "user", "content": PRIME_USER_TEXT}
    assert "logprobs" not in prime and "top_logprobs" not in prime
    assert prime["max_tokens"] == 1

    expected_k = {"Which team should handle this?": 3 + LABEL_MARGIN,
                  "How frustrated is the customer?": 3 + LABEL_MARGIN,
                  "The message conveys urgency": 2 + LABEL_MARGIN}
    assert sorted(question_of(q["messages"]) for q in questions) == sorted(expected_k)
    for sent in questions:
        assert sent["messages"][0] == prime["messages"][0]
        assert sent["messages"][1]["content"].startswith(("Question: ", "Statement: "))
        assert EXAMPLE["state"] not in sent["messages"][1]["content"]
        assert sent["model"] == MODEL
        assert sent["max_tokens"] == 1 and sent["temperature"] == 0
        assert sent["logprobs"] is True
        assert sent["top_logprobs"] == expected_k[question_of(sent["messages"])]
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["stream"] is False

    assert engine.events[:2] == [("start", 0), ("end", 0)]


LONG_STATE = " ".join(
    f"Speaker {i % 3}: sentence number {i} of the transcript." for i in range(300)
)


def test_on_mlx_lm_s_exact_prefix_rule_every_question_reuses_the_primed_state(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=example_probs, prefix_cache="segments")
    response = _decide(llm_client, auth, {**EXAMPLE, "state": LONG_STATE})
    assert response.status_code == 200, response.text
    timing = response.json()["timing_ms"]
    assert timing["prime"]["cached_tokens"] == 0

    segment = system_segment(engine.requests[0]["messages"])
    assert segment is not None and LONG_STATE in segment
    assert len(timing["per_question"]) == 3
    for name, timed in timing["per_question"].items():
        assert timed["cached_tokens"] == len(segment) // 4, name
        assert timed["cached_tokens"] >= len(LONG_STATE) // 4
        assert timed["prompt_tokens"] - timed["cached_tokens"] < 100


def test_the_old_layout_reused_only_the_frame_under_the_same_rule() -> None:
    frame = {"role": "system", "content": SYSTEM_PROMPT}
    prime = [frame, {"role": "user", "content": f"State:\n{LONG_STATE}"}]
    question = [frame, {"role": "user",
                        "content": f"State:\n{LONG_STATE}\n\nQuestion: Which?"}]
    frame_segment = system_segment(prime)
    assert frame_segment is not None
    saved = [frame_segment, _rendered(prime) + END_OF_TURN]
    text = _rendered(question)
    reused = max(len(key) for key in saved if text.startswith(key))
    assert reused == len(frame_segment) < len(LONG_STATE) // 10


def test_one_question_is_not_primed(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=example_probs)
    body = {**EXAMPLE, "questions": {"urgent": EXAMPLE["questions"]["urgent"]}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 200, response.text
    assert response.json()["timing_ms"]["prime"] is None
    assert len(engine.requests) == 1
    assert question_of(engine.requests[0]["messages"]) == "The message conveys urgency"


def test_the_engine_is_sent_its_own_name_for_the_model(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = engine_factory(probs_for=example_probs)
    monkeypatch.setattr(
        engines_module, "engine_model_name",
        lambda engine_name, model_dir, model_id: f"/weights/{model_id}",
    )
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 200, response.text
    assert {sent["model"] for sent in built[0].requests} == {f"/weights/{MODEL}"}
    assert response.json()["model"]["id"] == MODEL


def test_questions_go_out_together_up_to_the_engines_admission(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=yes_mostly, answer_delay=0.3)
    questions = {f"q{i}": {"type": "yesno", "instructions": f"statement {i}"}
                 for i in range(20)}
    response = _decide(llm_client, auth, {**EXAMPLE, "questions": questions})
    assert response.status_code == 200, response.text
    assert list(response.json()["answers"]) == list(questions)
    assert 1 < engine.max_in_flight <= 17


def test_a_serial_engine_is_asked_no_more_than_its_admission(
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_flag", None)
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency", 1, raising=False)
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_basis", "one slot",
                        raising=False)
    engine = loaded(probs_for=yes_mostly, answer_delay=0.2)
    questions = {f"q{i}": {"type": "yesno", "instructions": f"statement {i}"}
                 for i in range(6)}
    response = _decide(llm_client, auth, {**EXAMPLE, "questions": questions})
    assert response.status_code == 200, response.text
    assert engine.max_in_flight == 2


def test_a_letter_outside_the_top_k_is_label_not_in_probs(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    def probs(messages: list[dict[str, Any]]) -> dict[str, float]:
        if question_of(messages) == "How frustrated is the customer?":
            return {"A": 0.6, "B": 0.3}
        return example_probs(messages)

    loaded(probs_for=probs)
    response = _decide(llm_client, auth, EXAMPLE)
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "label_not_in_probs"
    assert "'anger'" in error["message"] and "'C'" in error["message"]
    assert error["details"]["question"] == "anger" and error["details"]["letter"] == "C"


def _c_below_the_top_k(messages: list[dict[str, Any]]) -> dict[str, float]:
    if question_of(messages) == "How frustrated is the customer?":
        return {"A": 0.30, "B": 0.20, "so": 0.09, "I": 0.08, "Um": 0.07,
                "Well": 0.06, "It": 0.05, "C": 0.001}
    return example_probs(messages)


@pytest.mark.parametrize("mode", [{}, {"missing": "refuse"}], ids=["default", "stated"])
def test_refuse_mode_is_unchanged_for_a_letter_ranked_below_k(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine],
    mode: dict[str, str],
) -> None:
    engine = loaded(probs_for=_c_below_the_top_k)
    response = _decide(llm_client, auth, {**EXAMPLE, **mode})
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "label_not_in_probs"
    assert error["details"]["question"] == "anger" and error["details"]["letter"] == "C"
    assert all("missing" not in sent for sent in engine.requests)


def test_report_mode_end_to_end_with_a_letter_ranked_below_k(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=_c_below_the_top_k)
    response = _decide(llm_client, auth, {**EXAMPLE, "missing": "report"})
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    anger_sent = next(sent for sent in engine.requests
                      if question_of(sent["messages"]) == "How frustrated is the customer?")
    assert anger_sent["top_logprobs"] == 3 + LABEL_MARGIN

    anger = answers["anger"]
    assert anger["missing_labels"] == ["Very angry"]
    assert anger["probabilities"] == pytest.approx(
        {"Calm": 0.6, "Frustrated but civil": 0.4, "Very angry": None})
    assert anger["logprobs"]["Very angry"] is None
    assert anger["logprobs"]["Calm"] == pytest.approx(math.log(0.6))
    assert anger["label_mass"] == pytest.approx(0.5)
    assert anger["score"] == pytest.approx(1 * 0.6 + 2 * 0.4)
    assert anger["level"] == "Calm" and anger["confidence"] == pytest.approx(0.6)
    assert answers["team"]["missing_labels"] == []
    assert answers["urgent"]["missing_labels"] == []
    assert answers["team"]["choice"] == "billing"


def test_report_mode_still_refuses_a_question_with_no_label_returned(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    def probs(messages: list[dict[str, Any]]) -> dict[str, float]:
        if question_of(messages) == "The message conveys urgency":
            return {"So": 0.4, "I": 0.3, "Um": 0.1, "It": 0.05, "Well": 0.04,
                    "Hmm": 0.009, "A": 0.0005, "B": 0.0005}
        return example_probs(messages)

    loaded(probs_for=probs)
    response = _decide(llm_client, auth, {**EXAMPLE, "missing": "report"})
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "label_not_in_probs"
    assert error["details"]["question"] == "urgent" and error["details"]["letter"] is None


@pytest.mark.parametrize("bad", ["ignore", "REPORT", None, 1])
def test_an_unknown_missing_mode_is_invalid_request(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine],
    bad: Any,
) -> None:
    engine = loaded(probs_for=example_probs)
    response = _decide(llm_client, auth, {**EXAMPLE, "missing": bad})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert ["body", "missing"] in [p["location"] for p in error["details"]["problems"]]
    assert engine.requests == []


def test_an_engine_refusal_is_engine_error_quoting_the_engine(
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = loaded(probs_for=yes_mostly, max_logprobs=20)
    options = {f"o{i}": "d" for i in range(17)}
    body = {**EXAMPLE, "questions": {"wide": {"type": "choice", "instructions": "x",
                                              "options": options}}}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "engine_error"
    assert error["details"]["engine"] == "vllm" and error["details"]["status"] == 400
    assert "greater than max allowed: 20" in error["message"]
    assert engine.requests[0]["top_logprobs"] == 21


def test_too_many_options_is_refused_before_anything_is_sent(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
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
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "images": [PNG] * 9})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "too_many_images"
    assert engine.requests == []


def test_images_on_a_text_model_are_model_text_only(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "images": [PNG]})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_text_only"
    assert error["details"] == {
        "model": MODEL, "backend": "cuda-linux", "serves": ["text"],
        "modalities": ["text"], "images": 1,
        "image_models": ["qwen3.8-27b-4bit-vl", "qwen3.5-9b-vl", "qwen3.5-4b",
                         "qwen3.5-2b", "qwen3.5-0.8b"],
    }
    assert '{"type": "load-model", "model": "qwen3.8-27b-4bit-vl"}' in error["message"]
    assert engine.requests == []


def test_model_text_only_reads_what_the_backend_SERVES_not_the_weights(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "sees-elsewhere.toml").write_text(
        """
[model]
id = "sees-elsewhere"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text", "image"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
serves = ["text"]
engine_args = ["--language-model-only"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        built = engine_factory(probs_for=yes_mostly)
        fake_weights("sees-elsewhere")
        events = run_job(client, auth, type="load-model", model="sees-elsewhere")
        assert events[-1]["event"] == "done", events[-1]
        response = _decide(
            client, auth, {**EXAMPLE, "model": "sees-elsewhere", "images": [PNG]}
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_text_only"
    assert error["details"]["serves"] == ["text"]
    assert error["details"]["modalities"] == ["text", "image"]
    assert built[-1].requests == []


def test_the_models_row_says_what_this_host_serves(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows["qwen3.5-4b"]["modalities"] == ["text", "image"]
    assert rows["qwen3.5-4b"]["serves"] == ["text", "image"]
    assert rows[MODEL]["serves"] == ["text"]
    assert rows[PAGE_MODEL]["serves"] == ["text", "image"]


def test_a_small_tier_that_serves_images_here_answers_an_image_decision(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(model="qwen3.5-4b", probs_for=yes_mostly)
    body = {
        "model": "qwen3.5-4b",
        "state": "",
        "images": [PNG],
        "questions": {"urgent": EXAMPLE["questions"]["urgent"]},
    }
    response = _decide(llm_client, auth, body)
    assert response.status_code == 200, response.text
    sent = engine.requests[-1]["messages"][-1]["content"]
    assert any(part["type"] == "image_url" for part in sent)


def test_an_upstream_model_is_decide_needs_logprobs(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, {**EXAMPLE, "model": "anthropic/claude"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "decide_needs_logprobs"
    assert engine.requests == []


def test_an_engine_that_returns_no_logprobs_is_decide_not_served(
    llm_client: TestClient,
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
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=yes_mostly)
    response = _decide(llm_client, auth, EXAMPLE, **{"X-Crucible-Act": "analysys"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_act"
    assert engine.requests == []


def test_a_model_that_is_not_resident_gets_the_chat_door_s_409(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
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
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ENGINES["vllm"], "chat_concurrency_flag", None)
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


def test_the_door_needs_auth(llm_client: TestClient) -> None:
    response = llm_client.post("/v1/decide", json=EXAMPLE)
    assert response.status_code == 401


def test_a_malformed_body_is_invalid_request_naming_the_field(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = _decide(llm_client, auth, {**EXAMPLE, "state": None})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert ["body", "state"] in [p["location"] for p in error["details"]["problems"]]
    assert "POST /v1/decide" in error["message"]
    assert "job request" not in error["message"]


def test_a_decision_in_flight_is_on_the_activity_bench_and_then_gone(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
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
    assert seen[0]["running"] == []

    after = llm_client.get("/v1/activity", headers=auth).json()
    assert after["chat"]["in_flight"] == 0 and after["chat"]["rows"] == []


def test_images_travel_as_data_uri_parts_ahead_of_the_text(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(model=PAGE_MODEL, probs_for=example_probs)
    body = {**EXAMPLE, "model": PAGE_MODEL, "state": "", "images": [PNG]}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 200, response.text
    assert response.json()["tokens"]["images"] == 1
    prime, *questions = engine.requests
    part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    assert isinstance(prime["messages"][0]["content"], str)
    assert prime["messages"][1]["content"] == [
        part, {"type": "text", "text": PRIME_USER_TEXT}]
    for sent in questions:
        assert sent["messages"][0] == prime["messages"][0]
        content = sent["messages"][1]["content"]
        assert content[0] == part and len(content) == 2
        assert content[1]["type"] == "text"
        assert content[1]["text"].startswith(("Question: ", "Statement: "))


ITEMS_EXAMPLE: dict[str, Any] = {
    "model": MODEL,
    "state": "The first sentence.\nThe second sentence.\nThe third sentence.",
    "instructions": "Which of the categories listed above does the speaker do in this passage?",
    "options": {"hate": "Hate", "conspiracy": "Conspiracy", "none": "None of these"},
    "items": [
        {"text": 'Passage: "The first sentence."'},
        {"text": 'Passage: "The second sentence."'},
        {"text": 'Passage: "The third sentence."'},
    ],
}

ITEM_RAW = {
    "first": {"A": 0.1, "B": 0.1, "C": 0.78},
    "second": {"A": 0.7, "B": 0.2, "C": 0.08},
    "third": {"A": 0.05, "B": 0.9, "C": 0.04},
}


def item_probs(messages: list[dict[str, Any]]) -> dict[str, float]:
    question = question_of(messages) or ""
    return next((row for word, row in ITEM_RAW.items() if word in question), {})


def test_the_items_form_answers_a_list_in_order_in_the_choice_shape(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=item_probs)
    response = _decide(llm_client, auth, ITEMS_EXAMPLE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [a["choice"] for a in body["answers"]] == ["none", "hate", "conspiracy"]
    first = body["answers"][0]
    assert first["type"] == "choice" and "missing_labels" not in first
    assert first["probabilities"] == pytest.approx({"hate": 0.1 / 0.98, "conspiracy": 0.1 / 0.98, "none": 0.78 / 0.98})
    assert first["label_mass"] == pytest.approx(0.98)
    assert body["engine"] == "vllm" and body["timing_ms"]["engine_requests"] == 4
    assert body["tokens"]["shared"] is None and len(body["tokens"]["per_item"]) == 3
    questions = [question_of(r["messages"]) for r in engine.requests[1:]]
    assert sorted(questions) == sorted(f'Passage: "The {w} sentence."' for w in ("first", "second", "third"))
    assert all(ITEMS_EXAMPLE["instructions"] in _user_text(r["messages"]) for r in engine.requests[1:])


def test_every_item_goes_as_the_lone_choice_question_would(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    from crucible import decide as decide_core

    engine = loaded(probs_for=item_probs)
    assert _decide(llm_client, auth, ITEMS_EXAMPLE).status_code == 200
    lone = [
        decide_core.question_messages(ITEMS_EXAMPLE["state"], [], decide_core.plan(
            "q", decide_core.ChoiceQuestion(
                type="choice", instructions=f"{item['text']}\n{ITEMS_EXAMPLE['instructions']}",
                options=ITEMS_EXAMPLE["options"])))
        for item in ITEMS_EXAMPLE["items"]
    ]
    sent = [r["messages"] for r in engine.requests[1:]]
    assert sorted(map(str, sent)) == sorted(map(str, lone))
    assert engine.requests[0]["messages"] == decide_core.prime_messages(ITEMS_EXAMPLE["state"], [])


def test_too_many_items_is_refused_before_anything_is_sent(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=item_probs)
    body = {**ITEMS_EXAMPLE, "items": [{"text": f"t{i}"} for i in range(513)]}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "too_many_items"
    assert error["details"] == {"items": 513, "max_items": 512}
    assert engine.requests == []


def test_an_item_with_too_many_options_is_refused_by_its_index(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]
) -> None:
    engine = loaded(probs_for=item_probs)
    body = {**ITEMS_EXAMPLE, "items": [
        {"text": "a"}, {"text": "b", "options": {f"o{i}": "d" for i in range(27)}}]}
    response = _decide(llm_client, auth, body)
    assert response.status_code == 400
    assert response.json()["error"]["details"]["question"] == "items[1]"
    assert engine.requests == []


def test_an_items_engine_started_before_its_route_says_to_load_again(
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.api.routes import decide as decide_route
    from crucible.engines import DecideItemsReading

    monkeypatch.setattr(
        decide_route, "decide_items_reading",
        lambda engine: DecideItemsReading(batched=True, basis="test: batched"),
    )
    loaded(probs_for=item_probs)
    response = _decide(llm_client, auth, ITEMS_EXAMPLE)
    assert response.status_code == 503, response.text
    error = response.json()["error"]
    assert error["code"] == "decide_not_served"
    assert "/v1/crucible/items" in error["message"] and "load-model" in error["message"]
    assert error["details"]["form"] == "items"


def test_on_an_engine_that_reads_questions_as_items_the_question_form_takes_the_items_route(
    llm_client: TestClient,
    auth: dict[str, str],
    loaded: Callable[..., FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.api.routes import decide as decide_route
    from crucible.engines import DecideItemsReading

    monkeypatch.setattr(
        decide_route, "decide_items_reading",
        lambda engine: DecideItemsReading(batched=True, basis="test: batched", questions=True),
    )
    engine = loaded(probs_for=example_probs)
    response = _decide(llm_client, auth, EXAMPLE)
    # The fake engine has no items route: reaching it proves where the question
    # form went, and no chat request (no prime, no question) was sent.
    assert response.status_code == 503, response.text
    assert response.json()["error"]["details"]["form"] == "items"
    assert not any(
        SYSTEM_PROMPT in str(request.get("messages")) for request in engine.requests
    ), "no decision prompt went to the chat route"
