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


def test_the_state_is_in_the_system_turn_and_the_question_is_the_user_turn() -> None:
    plan = _plans()["team"]
    messages = decide.question_messages("the STATE text", [], plan)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == decide.SYSTEM_PROMPT + "\n\nState:\nthe STATE text"
    assert messages[1]["content"] == (
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


@pytest.mark.parametrize("images", [[], [PNG, JPEG]])
def test_the_prime_s_system_turn_is_every_question_s_byte_for_byte(images) -> None:
    """The prefix property, in the one form every engine's cache can use: the
    shared prefix is the WHOLE system turn, and it ends where the user turn
    begins (mlx-lm's system segment, llama-server's last-user checkpoint)."""
    state = "a long shared state " * 50
    prime = decide.prime_messages(state, images)
    assert [m["role"] for m in prime] == ["system", "user"]
    for plan in _plans().values():
        question = decide.question_messages(state, images, plan)
        assert question[0] == prime[0]
        assert question[0]["content"].encode("utf-8") == prime[0]["content"].encode("utf-8")
        assert question[1] != prime[1]


def _texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [part["text"] for part in content if part["type"] == "text"]


@pytest.mark.parametrize("images", [[], [PNG]])
def test_the_state_is_only_in_the_system_turn_and_the_block_only_in_the_user_turn(
    images,
) -> None:
    state = "THE-STATE-MARKER and what it says"
    for plan in _plans().values():
        block = decide.question_block(
            plan.question.type, plan.question.instructions, plan.legend
        )
        system, user = decide.question_messages(state, images, plan)
        assert state in system["content"] and block not in system["content"]
        assert all(state not in text for text in _texts(user["content"]))
        assert _texts(user["content"])[-1] == block
    system, user = decide.prime_messages(state, images)
    assert state in system["content"]
    assert all(state not in text for text in _texts(user["content"]))


def test_the_prime_s_user_turn_is_fixed_and_never_empty() -> None:
    """mlx-lm finds the system segment by rendering `system + [user ""]`; a
    prime whose user turn WERE empty would never differ from that render, and
    it would save no system segment at all."""
    assert decide.PRIME_USER_TEXT.strip()
    assert decide.prime_messages("x", [])[1]["content"] == decide.PRIME_USER_TEXT
    assert decide.prime_messages("y", [])[1] == decide.prime_messages("x", [])[1]


def test_render_state_non_string_is_compact_json() -> None:
    assert decide.render_state({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'
    assert decide.render_state("verbatim  text") == "verbatim  text"
    assert decide.render_state({"é": "ü"}) == '{"é":"ü"}'


def test_images_open_the_user_turn_as_content_parts_then_the_question() -> None:
    """Images cannot go in the system turn (Qwen3.5's template raises "System
    message cannot contain images."), so they open the user turn, before the
    question, and the system turn says where they are."""
    plan = _plans()["urgent"]
    system, user = decide.question_messages("the STATE text", [PNG, JPEG], plan)
    assert isinstance(system["content"], str)
    assert system["content"] == (
        decide.SYSTEM_PROMPT + "\n\nState:\nthe STATE text\n\n" + decide.IMAGES_NOTE
    )
    user = user["content"]
    assert user[0] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    assert user[1] == {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{JPEG}"}}
    assert user[2]["type"] == "text"
    assert user[2]["text"].startswith("Statement: The message conveys urgency")
    assert len(user) == 3


def test_the_prime_with_images_differs_from_a_question_only_in_its_text() -> None:
    plan = _plans()["urgent"]
    for state in ("some text", ""):
        prime = decide.prime_messages(state, [PNG])
        question = decide.question_messages(state, [PNG], plan)
        assert question[0] == prime[0]
        assert question[1]["content"][:-1] == prime[1]["content"][:-1]
        assert prime[1]["content"][-1] == {"type": "text", "text": decide.PRIME_USER_TEXT}
        assert question[1]["content"][-1]["text"].startswith("Statement:")
    # An empty state (the images carry it) leaves `State:` and the note alone.
    assert decide.prime_messages("", [PNG])[0]["content"] == (
        decide.SYSTEM_PROMPT + "\n\nState:\n" + decide.IMAGES_NOTE
    )


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
    team = decide.label_distribution(
        _top({"A": 0.91 * 0.998, "B": 0.07 * 0.998, "C": 0.02 * 0.998, "The": 0.002}),
        plans["team"], "vllm")
    assert team.probabilities == pytest.approx(
        {"billing": 0.91, "technical": 0.07, "other": 0.02})
    assert team.mass == pytest.approx(0.998) and team.missing == ()
    answer = decide.answer(plans["team"], team, "refuse")
    assert answer.choice == "billing" and answer.confidence == pytest.approx(0.91)

    anger = decide.label_distribution(
        _top({"A": 0.62 * 0.997, "B": 0.36 * 0.997, "C": 0.02 * 0.997}), plans["anger"], "vllm")
    answer = decide.answer(plans["anger"], anger, "refuse")
    assert answer.score == pytest.approx(1.4)
    assert answer.level == "Calm" and answer.label_mass == pytest.approx(0.997)

    urgent = decide.label_distribution(
        _top({"A": 0.83 * 0.99, "B": 0.17 * 0.99}), plans["urgent"], "vllm")
    answer = decide.answer(plans["urgent"], urgent, "refuse")
    assert answer.p == pytest.approx(0.83) and answer.label_mass == pytest.approx(0.99)


def test_labels_are_matched_by_the_exact_letter_string() -> None:
    """The filler " A" must never be read as label A."""
    dist = decide.label_distribution(
        _top({" A": 0.5, "A": 0.1, "B": 0.1}), _plans()["urgent"], "vllm")
    assert dist.probabilities["Yes"] == pytest.approx(0.5)
    assert dist.mass == pytest.approx(0.2)


# -------------------------------------------------------- the log-probabilities


def test_logprobs_are_ln_of_the_probabilities_in_option_order() -> None:
    plans = _plans()
    # Returned out of option order on purpose: the answer is in OPTION order.
    team = decide.label_distribution(
        _top({"C": 0.05, "A": 0.6, "The": 0.1, "B": 0.25}), plans["team"], "vllm")
    answer = decide.answer(plans["team"], team, "refuse")
    assert list(answer.probabilities) == ["billing", "technical", "other"]
    assert list(answer.logprobs) == ["billing", "technical", "other"]
    for option, p in answer.probabilities.items():
        assert answer.logprobs[option] == pytest.approx(math.log(p))
    # The un-renormalised mass is recoverable, as the contract says.
    assert math.exp(answer.logprobs["billing"]) * answer.label_mass == pytest.approx(0.6)

    anger = decide.label_distribution(
        _top({"B": 0.5, "A": 0.3, "C": 0.1}), plans["anger"], "vllm")
    answer = decide.answer(plans["anger"], anger, "refuse")
    assert list(answer.logprobs) == ["Calm", "Frustrated but civil", "Very angry"]
    assert answer.logprobs["Calm"] == pytest.approx(math.log(0.3 / 0.9))

    urgent = decide.label_distribution(_top({"A": 0.8, "B": 0.1}), plans["urgent"], "vllm")
    answer = decide.answer(plans["urgent"], urgent, "refuse")
    assert answer.logprob == pytest.approx(math.log(0.8 / 0.9))


def test_a_refuse_mode_answer_carries_no_missing_labels_key() -> None:
    plans = _plans()
    for name, top in (("team", {"A": 0.5, "B": 0.3, "C": 0.2}),
                      ("anger", {"A": 0.5, "B": 0.3, "C": 0.2}),
                      ("urgent", {"A": 0.5, "B": 0.5})):
        answer = decide.answer(
            plans[name], decide.label_distribution(_top(top), plans[name], "vllm"), "refuse")
        assert "missing_labels" not in answer.model_dump(mode="json")


# ---------------------------------------------------------------- report mode


def test_report_mode_with_one_label_missing_never_invents_a_number() -> None:
    plans = _plans()
    # anger: B ("Frustrated but civil") is outside the top-K.
    dist = decide.label_distribution(
        _top({"A": 0.3, "The": 0.4, "C": 0.1}), plans["anger"], "vllm", missing="report")
    assert dist.missing == ("Frustrated but civil",)
    assert dist.mass == pytest.approx(0.4)  # the RETURNED letters' raw mass
    answer = decide.answer(plans["anger"], dist, "report")
    wire = answer.model_dump(mode="json")
    assert wire["missing_labels"] == ["Frustrated but civil"]
    assert wire["probabilities"] == pytest.approx(
        {"Calm": 0.75, "Frustrated but civil": None, "Very angry": 0.25})
    assert list(wire["probabilities"]) == ["Calm", "Frustrated but civil", "Very angry"]
    assert wire["logprobs"]["Frustrated but civil"] is None
    assert wire["logprobs"]["Calm"] == pytest.approx(math.log(0.75))
    assert wire["label_mass"] == pytest.approx(0.4)
    # The expected value over the RETURNED levels: 1 x 0.75 + 3 x 0.25.
    assert wire["score"] == pytest.approx(1.5)
    assert wire["level"] == "Calm" and wire["confidence"] == pytest.approx(0.75)


def test_report_mode_with_several_labels_missing() -> None:
    body = {**EXAMPLE, "questions": {"pick": {
        "type": "choice", "instructions": "x",
        "options": {"one": "d", "two": "d", "three": "d", "four": "d", "five": "d"}}}}
    item = _plans(body)["pick"]
    dist = decide.label_distribution(
        _top({"D": 0.2, "B": 0.1, "Hmm": 0.5, "Well": 0.1}), item, "vllm", missing="report")
    answer = decide.answer(item, dist, "report")
    assert answer.missing_labels == ["one", "three", "five"]  # option order
    assert answer.label_mass == pytest.approx(0.3)
    assert answer.probabilities == pytest.approx(
        {"one": None, "two": 1 / 3, "three": None, "four": 2 / 3, "five": None})
    assert [k for k, v in answer.logprobs.items() if v is None] == ["one", "three", "five"]
    assert answer.choice == "four" and answer.confidence == pytest.approx(2 / 3)
    returned = [p for p in answer.probabilities.values() if p is not None]
    assert sum(returned) == pytest.approx(1.0)


def test_report_mode_with_nothing_missing_says_an_empty_list() -> None:
    item = _plans()["team"]
    dist = decide.label_distribution(
        _top({"A": 0.5, "B": 0.3, "C": 0.2}), item, "vllm", missing="report")
    wire = decide.answer(item, dist, "report").model_dump(mode="json")
    assert wire["missing_labels"] == []


@pytest.mark.parametrize("missing", ["refuse", "report"])
def test_every_label_missing_is_refused_in_both_modes(missing) -> None:
    with pytest.raises(ApiError) as caught:
        decide.label_distribution(
            _top({"The": 0.6, "I": 0.3}), _plans()["anger"], "vllm", missing=missing)
    assert caught.value.status_code == 502 and caught.value.code == "label_not_in_probs"
    assert "'anger'" in caught.value.message


@pytest.mark.parametrize(
    "top, p, logprob, missing_labels",
    [({"B": 0.4, "The": 0.5}, 0.0, None, ["Yes"]),
     ({"A": 0.4, "The": 0.5}, 1.0, 0.0, ["No"]),
     ({"A": 0.3, "B": 0.1}, 0.75, math.log(0.75), [])],
    ids=["yes-missing", "no-missing", "none-missing"],
)
def test_yesno_report_mode_is_the_returned_one_renormalised_alone(
    top, p, logprob, missing_labels
) -> None:
    item = _plans()["urgent"]
    dist = decide.label_distribution(_top(top), item, "vllm", missing="report")
    wire = decide.answer(item, dist, "report").model_dump(mode="json")
    assert wire["p"] == pytest.approx(p)
    assert wire["logprob"] == (None if logprob is None else pytest.approx(logprob))
    assert wire["missing_labels"] == missing_labels
    assert wire["label_mass"] == pytest.approx(sum(v for k, v in top.items() if k in "AB"))


def test_missing_must_be_one_of_the_two_words() -> None:
    assert DecideRequest.model_validate(EXAMPLE).missing == "refuse"
    assert DecideRequest.model_validate({**EXAMPLE, "missing": "report"}).missing == "report"
    for bad in ("ignore", None, True, ""):
        with pytest.raises(ValidationError) as caught:
            DecideRequest.model_validate({**EXAMPLE, "missing": bad})
        assert "missing" in str(caught.value)


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
