from __future__ import annotations

import asyncio
import base64
import math
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from crucible import decide, decide_items
from crucible.decide import DecideRequest
from crucible.engines import decide_items_reading, items_forward
from crucible.engines.items_forward import ItemsRefusal
from crucible.errors import ApiError

FLAGS = {"hate": "Hate", "conspiracy": "Conspiracy", "none": "None of these"}

ASK = "Which of the categories listed above does the speaker do in this passage?"

ITEMS_BODY: dict[str, Any] = {
    "model": "qwen3.5-9b",
    "state": "first sentence\nsecond sentence",
    "instructions": ASK,
    "options": FLAGS,
    "items": [
        {"text": 'Passage from the transcript above: "first sentence"'},
        {"text": 'Passage from the transcript above: "second sentence"'},
    ],
}

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n a frame").decode()

RESIDENT = SimpleNamespace(
    model_id="qwen3.5-9b", engine="mlx-lm", engine_model_name="/w/qwen", revision="r1",
    fingerprint="qwen3.5-9b@r1", max_model_len=16384,
)


def _request(**changes: Any) -> DecideRequest:
    return DecideRequest.model_validate({**ITEMS_BODY, **changes})


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_the_items_form_validates_and_the_questions_form_is_unchanged() -> None:
    body = _request()
    assert body.items is not None and body.questions is None
    questions = DecideRequest.model_validate({
        "model": "m", "state": "s",
        "questions": {"q": {"type": "yesno", "instructions": "It is"}},
    })
    assert questions.items is None and questions.instructions is None
    assert decide.plan_all(questions)[0].name == "q"


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"questions": {"q": {"type": "yesno", "instructions": "x"}}}, "exactly one"),
        ({"items": None}, "exactly one"),
        ({"options": None}, "carry no `options`"),
        ({"items": []}, "at least 1"),
        ({"items": [{"text": ""}]}, "at least 1 character"),
        ({"items": [{"text": "x", "options": {"only": "one"}}]}, "at least 2"),
        ({"items": [{"text": "x", "why": "y"}]}, "Extra inputs"),
    ],
)
def test_a_malformed_items_body_is_refused_by_the_schema(changes: dict, needle: str) -> None:
    with pytest.raises(ValidationError) as caught:
        _request(**changes)
    assert needle in str(caught.value)


def test_instructions_or_options_with_questions_are_refused() -> None:
    with pytest.raises(ValidationError) as caught:
        DecideRequest.model_validate({
            "model": "m", "state": "s", "options": FLAGS,
            "questions": {"q": {"type": "yesno", "instructions": "x"}},
        })
    assert "belong to the items form" in str(caught.value)


def test_one_item_past_the_cap_is_too_many_items_with_the_numbers() -> None:
    items = [{"text": f"t{i}"} for i in range(decide_items.MAX_ITEMS + 1)]
    with pytest.raises(ApiError) as caught:
        decide_items.check_item_count(_request(items=items).items or [])
    assert caught.value.status_code == 400 and caught.value.code == "too_many_items"
    assert caught.value.details == {"items": 513, "max_items": 512}
    assert "Split the list into 2 requests" in caught.value.message
    assert decide_items.check_item_count([{}] * decide_items.MAX_ITEMS) == 512


def test_every_item_is_the_lone_choice_question_with_its_text_inline() -> None:
    body = _request()
    first, second = decide_items.item_plans(body)
    assert first.name == "items[0]" and second.name == "items[1]"
    assert first.question.instructions == f'Passage from the transcript above: "first sentence"\n{ASK}'
    lone = decide.plan("q", decide.ChoiceQuestion(
        type="choice", instructions=first.question.instructions, options=FLAGS))
    assert first.legend == lone.legend and first.labels == lone.labels
    assert decide_items.blocks([first])[0] == decide.question_block("choice", lone.question.instructions, lone.legend)


def test_without_instructions_the_item_text_is_the_whole_question() -> None:
    (only,) = decide_items.item_plans(_request(instructions=None, items=[{"text": "Is a face visible?"}]))
    assert only.question.instructions == "Is a face visible?"


def test_an_item_s_own_options_replace_the_shared_ones() -> None:
    body = _request(options=None, items=[
        {"text": "Is a face visible?", "options": {"yes": "Yes", "no": "No"}},
        {"text": "How busy is the frame?", "options": {str(n): f"level {n}" for n in range(1, 6)}},
    ])
    first, second = decide_items.item_plans(body)
    assert first.letters == ("A", "B") and second.letters == ("A", "B", "C", "D", "E")
    assert "E. 5: level 5" in decide_items.blocks([second])[0]


def test_filling_the_open_user_turn_gives_the_lone_question_s_messages() -> None:
    body = _request(images=[PNG])
    plans = decide_items.item_plans(body)
    opened = decide_items.open_messages(decide.render_state(body.state), [PNG])
    for item, block in zip(plans, decide_items.blocks(plans)):
        filled = items_forward.item_messages(opened, block)
        assert filled == decide.question_messages(decide.render_state(body.state), [PNG], item)
    plain = decide_items.open_messages("s", [])
    assert items_forward.item_messages(plain, "q") == decide.question_messages("s", [], decide.plan(
        "x", decide.YesNoQuestion(type="yesno", instructions="unused")))[:1] + [{"role": "user", "content": "q"}]


def test_the_prompt_cap_is_the_smaller_of_the_ceiling_and_the_load_window() -> None:
    assert decide_items.prompt_cap(16384) == 16383
    assert decide_items.prompt_cap(65536) == decide_items.MAX_PROMPT_TOKENS


def test_the_shared_part_is_the_common_token_prefix_and_every_item_keeps_a_tail() -> None:
    split = items_forward.split_shared([[1, 2, 3, 9], [1, 2, 3, 8, 7], [1, 2, 4]], 100, 10)
    assert split.shared == [1, 2]
    assert split.suffixes == [[3, 9], [3, 8, 7], [4]]
    lone = items_forward.split_shared([[5, 6, 7]], 100, 10)
    assert lone.shared == [5, 6] and lone.suffixes == [[7]]
    same = items_forward.split_shared([[5, 6], [5, 6]], 100, 10)
    assert same.shared == [5] and same.suffixes == [[6], [6]]


def test_an_item_over_its_cap_is_refused_naming_the_item_and_the_numbers() -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.split_shared([[1, 2], [1, 3, 4, 5]], 100, 2)
    assert caught.value.code == "item_too_long"
    assert caught.value.details == {"item": 1, "tokens": 3, "max_tokens": 2}
    error = decide_items.refusal_error(caught.value)
    assert error.status_code == 400 and "Shorten that item" in error.message


def test_a_prompt_over_its_cap_is_refused_with_the_numbers() -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.split_shared([[1] * 6 + [2], [1] * 6 + [3, 3, 3]], 8, 10)
    assert caught.value.code == "item_prompt_too_long"
    assert caught.value.details == {"item": 1, "tokens": 9, "shared_tokens": 6, "max_tokens": 8}
    assert "load the model with a longer context" in decide_items.refusal_error(caught.value).message


def _ask(**changes: Any) -> dict[str, Any]:
    return {"model": "w", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": ""}],
            "questions": ["ab", "c"], "top_logprobs": 3, "max_prompt_tokens": 50,
            "max_item_tokens": 5, **changes}


def test_answer_items_tokenizes_each_lone_question_and_reads_each_tail() -> None:
    ask = items_forward.parse_items(_ask(), ("w",))
    seen: list[Any] = []

    def tokenize(messages: list[dict[str, Any]]) -> list[int]:
        return [0, 0] + [ord(ch) for ch in messages[-1]["content"]]

    def read(split: Any) -> list[list[tuple[int, float]]]:
        seen.append(split)
        return [[(65, math.log(0.7)), (66, math.log(0.2))], [(66, math.log(0.9)), (1, -math.inf)]]

    document = items_forward.answer_items(ask, tokenize, read, chr)
    assert seen[0].shared == [0, 0] and seen[0].suffixes == [[97, 98], [99]]
    assert document["shared_tokens"] == 2 and document["item_tokens"] == [2, 1]
    assert document["slots"][0]["top_logprobs"][0] == {"token": "A", "logprob": pytest.approx(math.log(0.7))}
    assert document["slots"][1]["top_logprobs"] == [{"token": "B", "logprob": pytest.approx(math.log(0.9))}]


@pytest.mark.parametrize(
    "spoil,code",
    [
        ({"extra": 1}, "unknown_field"),
        ({"model": "other"}, "model_not_found"),
        ({"questions": []}, "bad_questions"),
        ({"messages": [{"role": "user", "content": "already asked"}]}, "open_user_turn"),
        ({"top_logprobs": 41}, "too_many_top_logprobs"),
        ({"max_prompt_tokens": 0}, "bad_max_prompt_tokens"),
        ({"chat_template_kwargs": {"tools": []}}, "unknown_template_kwarg"),
    ],
)
def test_an_items_request_the_engine_cannot_read_is_refused_by_name(spoil: dict, code: str) -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.parse_items(_ask(**spoil), ("w",))
    assert caught.value.code == code


def _items_reply(rows: list[dict[str, float]], shared: int = 5000) -> dict[str, Any]:
    return {
        "object": "crucible.items", "shared_tokens": shared, "item_tokens": [40] * len(rows),
        "slots": [
            {"top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in row.items()]}
            for row in rows
        ],
    }


def _batched(body: DecideRequest, reply: Any, sent: list | None = None) -> Any:
    async def call(path: str, wire: dict) -> Any:
        if sent is not None:
            sent.append((path, wire))
        return reply

    async def post(wire: dict) -> Any:
        raise AssertionError("the batched path sends no chat request")

    return _run(decide_items.decide_items_on_engine(
        call, post, RESIDENT, body, decide_items.item_plans(body),
        batched=True, max_logprobs=40, concurrency=2,
    ))


def test_the_batched_route_gets_the_open_turn_and_one_question_per_item() -> None:
    sent: list[tuple[str, dict]] = []
    body = _request()
    response = _batched(body, _items_reply([{"A": 0.6, "C": 0.3, "B": 0.05}, {"C": 0.9, "A": 0.05, "B": 0.03}]), sent)
    ((path, wire),) = sent
    assert path == items_forward.ITEMS_PATH
    plans = decide_items.item_plans(body)
    assert wire["questions"] == decide_items.blocks(plans)
    assert wire["messages"] == decide_items.open_messages(decide.render_state(body.state), [])
    assert wire["top_logprobs"] == 7 and wire["max_item_tokens"] == 1024
    assert wire["max_prompt_tokens"] == 16383
    assert wire["model"] == "/w/qwen" and wire["chat_template_kwargs"] == {"enable_thinking": False}
    first, second = response.answers
    assert first.choice == "hate" and second.choice == "none"
    assert first.probabilities == pytest.approx({"hate": 0.6 / 0.95, "conspiracy": 0.05 / 0.95, "none": 0.3 / 0.95})
    assert first.label_mass == pytest.approx(0.95)
    dumped = response.model_dump(mode="json")
    assert dumped["answers"][0]["type"] == "choice" and "missing_labels" not in dumped["answers"][0]
    assert dumped["tokens"] == {"shared": 5000, "per_item": [5040, 5040], "images": 0}
    assert dumped["timing_ms"]["engine_requests"] == 1
    assert dumped["engine"] == "mlx-lm" and dumped["model"]["fingerprint"] == "qwen3.5-9b@r1"


def test_report_mode_names_a_letter_the_item_did_not_return() -> None:
    first, second = _batched(
        _request(missing="report"), _items_reply([{"A": 0.5, "C": 0.4}, {"A": 0.2, "B": 0.2, "C": 0.5}])
    ).answers
    assert first.missing_labels == ["conspiracy"] and first.probabilities["conspiracy"] is None
    assert second.missing_labels == []


def test_refuse_mode_names_the_item_whose_letter_is_missing() -> None:
    with pytest.raises(ApiError) as caught:
        _batched(_request(), _items_reply([{"A": 0.5, "B": 0.1, "C": 0.3}, {"A": 0.5, "C": 0.4}]))
    assert caught.value.code == "label_not_in_probs"
    assert caught.value.details["question"] == "items[1]"


def test_a_reply_with_the_wrong_count_is_engine_error() -> None:
    with pytest.raises(ApiError) as caught:
        decide_items.read_items_reply(_items_reply([{"A": 0.5}]), "mlx-vlm", 2)
    assert caught.value.status_code == 502 and caught.value.code == "engine_error"


def test_an_engine_refusal_by_name_becomes_the_door_s_refusal() -> None:
    error = decide_items.engine_refusal(400, {"error": {
        "code": "item_too_long", "message": "item 3 is 2000 tokens past the shared state",
        "details": {"item": 3, "tokens": 2000, "max_tokens": 1024}}})
    assert error is not None and error.code == "item_too_long" and error.details["item"] == 3
    assert decide_items.engine_refusal(400, {"error": {"code": "bad_json"}}) is None
    assert decide_items.engine_refusal(500, {"error": {"code": "item_too_long"}}) is None


def test_without_a_batched_route_every_item_is_its_own_request_after_the_prime() -> None:
    sent: list[dict] = []
    letters = [{"A": 0.1, "B": 0.1, "C": 0.75}, {"A": 0.7, "B": 0.2, "C": 0.05}]

    async def post(wire: dict) -> Any:
        sent.append(wire)
        if not wire.get("logprobs"):
            return {"usage": {"prompt_tokens": 50}}
        text = wire["messages"][-1]["content"]
        row = letters[0] if "first sentence" in text else letters[1]
        tops = [{"token": t, "logprob": math.log(p)} for t, p in row.items()]
        return {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": tops}]}}],
                "usage": {"prompt_tokens": 60, "prompt_tokens_details": {"cached_tokens": 48}}}

    async def call(path: str, wire: dict) -> Any:
        raise AssertionError("no batched route on this engine")

    body = _request()
    resident = SimpleNamespace(**{**vars(RESIDENT), "engine": "vllm"})
    response = _run(decide_items.decide_items_on_engine(
        call, post, resident, body, decide_items.item_plans(body),
        batched=False, max_logprobs=32, concurrency=4,
    ))
    assert len(sent) == 3 and "logprobs" not in sent[0]
    assert [a.choice for a in response.answers] == ["none", "hate"]
    assert response.timing_ms.engine_requests == 3
    assert response.tokens.shared is None and response.tokens.per_item == [60, 60]
    plans = decide_items.item_plans(body)
    assert sent[1]["messages"] == decide.question_messages(decide.render_state(body.state), [], plans[0])


def test_only_the_mac_engines_read_items_in_one_request() -> None:
    assert decide_items_reading("mlx-lm").batched is True
    assert decide_items_reading("mlx-vlm").batched is True
    assert decide_items_reading("vllm").batched is False
    assert decide_items_reading("llama-server").batched is False
