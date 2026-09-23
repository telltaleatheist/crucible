"""The decision's reading, without a socket (PHASE22-DECIDE.md section 3).

snap's pure-function tests, ported: the legend, the letters, yes/no as A/B, the
renormalisation, the expected-value score, the image layout, and the refusals a
caller can act on. snap measured these semantics on a card (2026-09-22); what is
asserted here is that Crucible's port did not move them.
"""

from __future__ import annotations

import base64
import math

import pytest
from pydantic import ValidationError

from crucible import decide
from crucible.decide import (
    ChoiceQuestion,
    DecideRequest,
    ScoreQuestion,
    YesNoQuestion,
)
from crucible.errors import ApiError

#: snap's worked example (`tests/unit/fake_llama.py` EXAMPLE_REQUEST).
EXAMPLE = {
    "model": "qwen3.5-9b",
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

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n first image").decode()
JPEG = base64.b64encode(b"\xff\xd8\xff\xe0 second image").decode()


def _top(pairs: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(pairs.items())


def _plans(body: dict = EXAMPLE) -> dict[str, decide.Plan]:
    return {p.name: p for p in decide.plan_all(DecideRequest.model_validate(body))}


# ------------------------------------------------------------------ letters


def test_26_options_are_a_to_z() -> None:
    labels = decide.assign_labels([f"opt{i}" for i in range(26)], "q")
    assert labels[0] == ("A", "opt0") and labels[-1] == ("Z", "opt25")


def test_27_options_are_too_many_options_by_name() -> None:
    body = {**EXAMPLE, "questions": {"big": {
        "type": "choice", "instructions": "y",
        "options": {f"o{i}": "d" for i in range(27)}}}}
    request = DecideRequest.model_validate(body)  # the schema lets it through...
    with pytest.raises(ApiError) as caught:
        decide.plan_all(request)  # ...and the plan names it, before any pass
    assert caught.value.status_code == 400
    assert caught.value.code == "too_many_options"
    assert "'big'" in caught.value.message
    assert caught.value.details == {"question": "big", "options": 27, "max_options": 26}


def test_yesno_is_a_yes_b_no() -> None:
    plan = _plans()["urgent"]
    assert plan.labels == (("A", "Yes"), ("B", "No"))


def test_choice_letters_follow_the_options_insertion_order() -> None:
    plan = _plans()["team"]
    assert plan.labels == (("A", "billing"), ("B", "technical"), ("C", "other"))
    assert plan.legend[0] == ("A", "billing: Payment and invoice issues")


# ------------------------------------------------------------------ prompt


def test_state_first_question_last_and_the_words_are_snaps() -> None:
    plan = _plans()["team"]
    messages = decide.question_messages("the STATE text", [], plan)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == decide.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert user == (
        "State:\nthe STATE text\n\n"
        "Question: Which team should handle this?\nOptions:\n"
        "A. billing: Payment and invoice issues\nB. technical: Bugs and errors\n"
        "C. other: Anything else\nAnswer with the letter only."
    )


def test_a_yesno_block_is_a_statement() -> None:
    block = decide.question_block("yesno", "It is raining", (("A", "Yes"), ("B", "No")))
    assert block == (
        "Statement: It is raining\nIs this statement true of the state above?\n"
        "Options:\nA. Yes\nB. No\nAnswer with the letter only."
    )


def test_every_question_extends_the_prime_verbatim() -> None:
    prime = decide.prime_messages("a long shared state", [])
    assert prime[1]["content"] == "State:\na long shared state"
    for plan in _plans().values():
        question = decide.question_messages("a long shared state", [], plan)
        assert question[0] == prime[0]
        assert question[1]["content"].startswith(prime[1]["content"] + "\n\n")


def test_render_state_non_string_is_compact_json() -> None:
    assert decide.render_state({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'
    assert decide.render_state("verbatim  text") == "verbatim  text"
    assert decide.render_state({"é": "ü"}) == '{"é":"ü"}'


def test_images_come_first_as_content_parts_then_the_text() -> None:
    plan = _plans()["urgent"]
    user = decide.question_messages("the STATE text", [PNG, JPEG], plan)[1]["content"]
    assert user[0] == {"type": "text", "text": "State:"}
    assert user[1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    assert user[2] == {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{JPEG}"}}
    assert user[3]["type"] == "text"
    assert user[3]["text"].startswith("the STATE text\n\nStatement: The message conveys urgency")
    assert len(user) == 4


def test_the_prime_with_images_is_the_question_cut_back_to_the_state() -> None:
    plan = _plans()["urgent"]
    for state in ("some text", ""):
        prime = decide.prime_messages(state, [PNG])[1]["content"]
        question = decide.question_messages(state, [PNG], plan)[1]["content"]
        if state:
            assert question[:-1] == prime[:-1]
            assert question[-1]["text"].startswith(prime[-1]["text"] + "\n\n")
        else:
            # An empty state adds no part of its own: the question's block is
            # the one part the prime does not have.
            assert question[:-1] == prime
            assert question[-1]["text"].startswith("Statement:")


def test_the_image_media_type_is_read_off_the_bytes() -> None:
    assert decide.image_format(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert decide.image_format(b"\xff\xd8\xff\xdb") == "jpeg"
    assert decide.image_format(b"GIF89a....") == "gif"
    assert decide.image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert decide.image_format(b"not an image") is None


# ------------------------------------------------------------ the request


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"state": None}, "state"),
        ({"state": ""}, "state may not be empty"),
        ({"state": "   "}, "state may not be empty"),
        ({"questions": {}}, "questions"),
        ({"extra": 1}, "extra"),
        ({"model": ""}, "model"),
        ({"images": ["not base64!"]}, "images[0] is not base64"),
        ({"images": [base64.b64encode(b"plain text").decode()]}, "images[0] is not a PNG"),
        ({"questions": {"a/b": {"type": "yesno", "instructions": "x"}}}, "invalid name"),
        ({"questions": {".hidden": {"type": "yesno", "instructions": "x"}}}, "dot"),
        ({"questions": {"s": {"type": "score", "instructions": "x", "levels": ["a", "a"]}}},
         "unique"),
        ({"questions": {"s": {"type": "score", "instructions": "x",
                              "levels": [f"l{i}" for i in range(11)]}}}, "levels"),
        ({"questions": {"c": {"type": "choice", "instructions": "x",
                              "options": {"only": "one"}}}}, "options"),
    ],
)
def test_the_caller_s_mistakes_are_refused_naming_the_field(change, fragment) -> None:
    with pytest.raises(ValidationError) as caught:
        DecideRequest.model_validate({**EXAMPLE, **change})
    assert fragment in str(caught.value)


def test_images_carry_an_empty_state() -> None:
    request = DecideRequest.model_validate({**EXAMPLE, "state": "", "images": [PNG]})
    assert request.images == [PNG]


def test_nine_images_are_too_many_images_by_name() -> None:
    with pytest.raises(ApiError) as caught:
        decide.check_image_count([PNG] * 9)
    assert caught.value.code == "too_many_images" and caught.value.status_code == 400
    assert decide.check_image_count([PNG] * 8) == 8
    assert decide.check_image_count(None) == 0


# ---------------------------------------------------------------- the reading


def test_k_is_the_labels_plus_the_margin_clamped_to_the_engine() -> None:
    assert decide.top_k(3, 32) == 7
    assert decide.top_k(26, 32) == 30
    assert decide.top_k(10, 11) == 11
    assert decide.top_k(26, None) == 30


def test_the_example_math() -> None:
    plans = _plans()
    team, mass = decide.label_distribution(
        _top({"A": 0.91 * 0.998, "B": 0.07 * 0.998, "C": 0.02 * 0.998, "The": 0.002}),
        plans["team"], "vllm")
    assert team == pytest.approx({"billing": 0.91, "technical": 0.07, "other": 0.02})
    assert mass == pytest.approx(0.998)
    answer = decide.answer(plans["team"], team, mass)
    assert answer.choice == "billing" and answer.confidence == pytest.approx(0.91)

    anger, mass = decide.label_distribution(
        _top({"A": 0.62 * 0.997, "B": 0.36 * 0.997, "C": 0.02 * 0.997}), plans["anger"], "vllm")
    answer = decide.answer(plans["anger"], anger, mass)
    assert answer.score == pytest.approx(1.4)
    assert answer.level == "Calm" and answer.label_mass == pytest.approx(0.997)

    urgent, mass = decide.label_distribution(
        _top({"A": 0.83 * 0.99, "B": 0.17 * 0.99}), plans["urgent"], "vllm")
    answer = decide.answer(plans["urgent"], urgent, mass)
    assert answer.p == pytest.approx(0.83) and answer.label_mass == pytest.approx(0.99)


def test_labels_are_matched_by_the_exact_letter_string() -> None:
    """The filler " A" must never be read as label A."""
    probabilities, mass = decide.label_distribution(
        _top({" A": 0.5, "A": 0.1, "B": 0.1}), _plans()["urgent"], "vllm")
    assert probabilities["Yes"] == pytest.approx(0.5)
    assert mass == pytest.approx(0.2)


def test_a_missing_letter_is_label_not_in_probs_naming_question_and_letter() -> None:
    with pytest.raises(ApiError) as caught:
        decide.label_distribution(_top({"A": 0.6, "B": 0.3}), _plans()["anger"], "vllm")
    assert caught.value.status_code == 502
    assert caught.value.code == "label_not_in_probs"
    assert "'anger'" in caught.value.message and "'C'" in caught.value.message
    assert caught.value.details["letter"] == "C"


def test_a_label_reported_twice_is_an_engine_error() -> None:
    top = (("A", 0.4), ("A", 0.3), ("B", 0.1))
    with pytest.raises(ApiError) as caught:
        decide.label_distribution(top, _plans()["urgent"], "vllm")
    assert caught.value.code == "engine_error"


def test_every_label_at_zero_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        decide.label_distribution(_top({"A": 0.0, "B": 0.0}), _plans()["urgent"], "vllm")
    assert caught.value.code == "label_not_in_probs"


# ------------------------------------------------------------------ the reply


def _reply(tops: list[dict], usage: dict) -> dict:
    return {"choices": [{"logprobs": {"content": [{**tops[0], "top_logprobs": tops}]}}],
            "usage": usage}


VLLM_TOPS = [{"token": "A", "logprob": math.log(0.7), "bytes": [65]},
             {"token": "B", "logprob": math.log(0.2), "bytes": [66]}]
#: llama-server b10970's `probs_vector_to_json` adds `id`.
LLAMA_TOPS = [{"id": 32, "token": "A", "bytes": [65], "logprob": math.log(0.7)},
              {"id": 33, "token": "B", "bytes": [66], "logprob": math.log(0.2)}]
#: mlx-lm 0.31.3's `_format_top_logprobs` has no `bytes`.
MLX_TOPS = [{"id": 32, "token": "A", "logprob": math.log(0.7)},
            {"id": 33, "token": "B", "logprob": math.log(0.2)}]


@pytest.mark.parametrize("tops", [VLLM_TOPS, LLAMA_TOPS, MLX_TOPS], ids=["vllm", "llama", "mlx"])
def test_one_parser_reads_all_three_engines(tops) -> None:
    reading = decide.read_reply(
        _reply(tops, {"prompt_tokens": 90, "prompt_tokens_details": {"cached_tokens": 64}}),
        "x", want_probs=True)
    assert reading.prompt_tokens == 90 and reading.cached_tokens == 64
    assert [t for t, _ in reading.top] == ["A", "B"]
    assert reading.top[0][1] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": 9},
     {"prompt_tokens": 9, "prompt_tokens_details": None},
     {"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": None}}],
    ids=["absent", "null-details", "null-count"],
)
def test_an_unreported_cache_is_null_never_zero(usage) -> None:
    reading = decide.read_reply(_reply(VLLM_TOPS, usage), "vllm", want_probs=True)
    assert reading.cached_tokens is None and reading.prompt_tokens == 9


def test_a_prime_reply_is_not_read_for_letters() -> None:
    reading = decide.read_reply(
        {"choices": [{"logprobs": None}], "usage": {"prompt_tokens": 40}}, "vllm",
        want_probs=False)
    assert reading.top is None and reading.prompt_tokens == 40


@pytest.mark.parametrize(
    "reply, fragment",
    [
        ({"choices": []}, "'usage' is missing"),
        ({"usage": {}, "choices": []}, "'prompt_tokens' is missing"),
        ({"usage": {"prompt_tokens": 1}, "choices": []}, "'choices' is empty"),
        ({"usage": {"prompt_tokens": 1}, "choices": [{"logprobs": None}]}, "logprobs"),
        ({"usage": {"prompt_tokens": 1}, "choices": [{"logprobs": {"content": []}}]},
         "no token was generated"),
        ({"usage": {"prompt_tokens": 1, "prompt_tokens_details": 3},
          "choices": [{"logprobs": {"content": []}}]}, "prompt_tokens_details"),
    ],
)
def test_an_unreadable_reply_is_engine_error_naming_the_field(reply, fragment) -> None:
    with pytest.raises(ApiError) as caught:
        decide.read_reply(reply, "vllm", want_probs=True)
    assert caught.value.status_code == 502 and caught.value.code == "engine_error"
    assert fragment in caught.value.message
    assert "vllm" in caught.value.message


# ------------------------------------------------------------------- the body


def test_a_question_body_states_every_knob_a_reading_depends_on() -> None:
    msgs = decide.prime_messages("s", [])
    assert decide.request_body("served", msgs, 7) == {
        "model": "served",
        "messages": msgs,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 7,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }


def test_a_prime_body_asks_for_no_logprobs() -> None:
    body = decide.request_body("served", decide.prime_messages("s", []), None)
    assert "logprobs" not in body and "top_logprobs" not in body
    assert body["max_tokens"] == 1 and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_question_types_are_the_three_snap_has() -> None:
    assert {ChoiceQuestion, ScoreQuestion, YesNoQuestion} == {
        type(DecideRequest.model_validate(EXAMPLE).questions[name])
        for name in ("team", "anger", "urgent")
    }
