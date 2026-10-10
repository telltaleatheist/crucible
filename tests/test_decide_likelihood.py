"""Likelihood questions: free-text candidates scored by the log-probability of
every one of their tokens, never generated. The pure pieces (the boundary, the
math, the readers), the Mac's items pass against a fake mlx, the vLLM path
against a fake engine, and the door end to end."""

from __future__ import annotations

import asyncio
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from crucible import decide, decide_likelihood
from crucible.decide import DecideRequest, LikelihoodQuestion
from crucible.engines import likelihood_reading, items_forward
from crucible.engines.items_forward import ItemsRefusal, likelihood_split
from crucible.errors import ApiError

from .fake_engine import FakeEngine, GENERATION_PROMPT, _rendered, fake_token_logprob
from .test_decide_mac_floor import Arr, FakeCache
from .test_llm_api import fake_env, llm_client, run_job  # noqa: F401 - fixtures this module uses

MODEL = "qwen3.5-9b"

SPELLING: dict[str, Any] = {
    "type": "likelihood",
    "instructions": "Write out the second sentence, spelled as it should be.",
    "candidates": {"plain": "They agreed to cooperate.", "hyphen": "They agreed to co-operate."},
}

RESIDENT = SimpleNamespace(
    model_id=MODEL, engine="mlx-lm", engine_model_name="/w/qwen", revision="r1",
    fingerprint=f"{MODEL}@r1", max_model_len=16384,
)


def _request(**questions: Any) -> DecideRequest:
    return DecideRequest.model_validate(
        {"model": MODEL, "state": "First. They agreed to co-operate.", "questions": questions}
    )


# --- the question -------------------------------------------------------------------


def test_a_likelihood_question_validates_with_total_as_its_default_ranking() -> None:
    question = _request(spelling=SPELLING).questions["spelling"]
    assert isinstance(question, LikelihoodQuestion) and question.rank_by == "total"
    assert question.normalize == "softmax"
    (item,) = decide.plan_all(_request(spelling=SPELLING))
    assert item.labels == () and decide.likelihood_plans([item]) == [item]
    assert decide.label_plans([item]) == []


@pytest.mark.parametrize(
    "change,fragment",
    [
        ({"candidates": {"a": "x"}}, "at least 2"),
        ({"candidates": {"a": "x", "b": "x"}}, "unique"),
        ({"candidates": {"a": "x ", "b": "y"}}, "whitespace"),
        ({"candidates": {"a": " x", "b": "y"}}, "whitespace"),
        ({"candidates": {"a": "", "b": "y"}}, "at least 1 character"),
        ({"rank_by": "median"}, "'total' or 'mean'"),
        ({"normalize": "sigmoid"}, "'softmax' or 'none'"),
        ({"options": {"a": "x", "b": "y"}}, "Extra inputs"),
    ],
)
def test_a_malformed_likelihood_question_is_refused_by_the_schema(change: dict, fragment: str) -> None:
    with pytest.raises(ValidationError) as caught:
        _request(q={**SPELLING, **change})
    assert fragment in str(caught.value)


def test_two_hundred_fifty_seven_candidates_are_too_many_by_name() -> None:
    body = _request(q={**SPELLING, "candidates": {f"c{i}": f"text {i}" for i in range(257)}})
    with pytest.raises(ApiError) as caught:
        decide.plan_all(body)
    assert caught.value.code == "too_many_candidates"
    assert caught.value.details == {"question": "q", "candidates": 257, "max_candidates": 256}
    decide.plan_all(_request(q={**SPELLING, "candidates": {f"c{i}": f"t{i}" for i in range(256)}}))


def test_the_context_asks_for_a_reply_not_a_letter() -> None:
    msgs = decide_likelihood.likelihood_messages("the state", [], "Write it out.")
    assert msgs[0]["content"].startswith(decide.LIKELIHOOD_SYSTEM_PROMPT)
    assert msgs[0]["content"].endswith(decide.STATE_HEADER + "the state")
    assert "letter" not in decide.LIKELIHOOD_SYSTEM_PROMPT
    assert msgs[1] == {"role": "user", "content": "Write it out."}
    opened = decide_likelihood.open_messages("the state", [])
    assert items_forward.item_messages(opened, "Write it out.") == msgs, (
        "the items route rebuilds the vLLM path's context exactly"
    )


# --- the boundary -------------------------------------------------------------------


def test_candidates_are_scored_from_where_their_shared_context_ends() -> None:
    context = [1, 2, 3, 4]
    split = likelihood_split([context], [[[1, 2, 3, 4, 7, 8], [1, 2, 3, 4, 9]]], 100, 100, 10)
    assert split.boundaries == [4] and split.context_tokens == [4]
    assert split.shared == [1, 2, 3], "the row reads the hidden state at the boundary's last token"
    assert split.suffixes == [[4, 7, 8], [4, 9]]
    assert [(s.start, s.targets) for s in split.scored] == [(0, [7, 8]), (0, [9])]


def test_a_first_token_that_merges_into_the_context_s_last_is_scored_with_it() -> None:
    # "...\n\n" then "!x": the tokenizer makes "\n\n!" one token (60) for one candidate.
    split = likelihood_split([[1, 2, 6]], [[[1, 2, 60, 7], [1, 2, 6, 8]]], 100, 100, 10)
    assert split.boundaries == [2], "every candidate is scored from the same boundary"
    assert [s.targets for s in split.scored] == [[60, 7], [6, 8]]


def test_a_template_that_does_not_continue_the_opened_reply_is_refused() -> None:
    with pytest.raises(ItemsRefusal) as caught:
        likelihood_split([[1, 2, 5, 6]], [[[1, 2, 7], [1, 2, 8]]], 100, 100, 10)
    assert caught.value.code == "candidate_not_a_reply"
    assert caught.value.details == {"group": 0, "context_tokens": 4, "common_tokens": 2}


def test_a_long_candidate_and_a_long_prompt_are_refused_naming_them() -> None:
    with pytest.raises(ItemsRefusal) as caught:
        likelihood_split([[1, 2]], [[[1, 2, 3], [1, 2, 3, 4, 5, 6]]], 100, 100, 3)
    assert caught.value.code == "candidate_too_long"
    assert caught.value.details == {"group": 0, "candidate": 1, "tokens": 4, "max_tokens": 3}
    with pytest.raises(ItemsRefusal) as caught:
        likelihood_split([[1, 2]], [[[1, 2, 3], [1, 2, 3, 4]]], 3, 100, 10)
    assert caught.value.code == "item_prompt_too_long" and caught.value.details["candidate"] == 1


def test_the_shared_part_stops_before_the_earliest_group_s_boundary() -> None:
    contexts = [[1, 2, 3, 4], [1, 2, 3, 5, 6]]
    prompts = [[[1, 2, 3, 4, 9], [1, 2, 3, 4, 8]], [[1, 2, 3, 5, 6, 7], [1, 2, 3, 5, 6, 9, 9]]]
    split = likelihood_split(contexts, prompts, 100, 100, 10)
    assert split.boundaries == [4, 5] and split.shared == [1, 2, 3]
    assert [(s.group, s.start, s.targets) for s in split.scored] == [
        (0, 0, [9]), (0, 0, [8]), (1, 1, [7]), (1, 1, [9, 9]),
    ]


# --- the math -----------------------------------------------------------------------


def _answer(
    rows: list[list[float]], rank_by: str = "total", normalize: str = "softmax", **names: str
) -> Any:
    candidates = names or {"a": "x", "b": "y"}
    (item,) = decide.plan_all(_request(q={
        **SPELLING, "candidates": candidates, "rank_by": rank_by, "normalize": normalize}))
    return decide_likelihood.likelihood_answer(item, rows, 40, 39)


def test_totals_means_and_a_softmax_over_the_totals() -> None:
    answer = _answer([[-2.0], [-5.0, -0.01, -0.01]])
    a, b = answer.candidates["a"], answer.candidates["b"]
    assert (a.logprob, a.tokens, a.mean_logprob) == (-2.0, 1, -2.0)
    assert b.logprob == pytest.approx(-5.02) and b.tokens == 3
    assert b.mean_logprob == pytest.approx(-5.02 / 3)
    assert a.probability == pytest.approx(1 / (1 + math.exp(-3.02)))
    assert a.probability + b.probability == pytest.approx(1.0)
    assert answer.winner == "a" and answer.rank_by == "total" and answer.normalize == "softmax"
    assert answer.context_tokens == 40 and answer.boundary_tokens == 1


def test_normalize_none_is_each_reply_s_own_probability_summed_with_nothing() -> None:
    answer = _answer([[-2.0], [-5.0, -0.01, -0.01]], normalize="none")
    a, b = answer.candidates["a"], answer.candidates["b"]
    assert a.probability == pytest.approx(math.exp(-2.0))
    assert b.probability == pytest.approx(math.exp(-5.02))
    assert a.probability + b.probability < 0.2, "independent: nothing makes them sum to 1"
    assert answer.normalize == "none" and answer.winner == "a", "the winner does not move"
    assert (a.logprob, b.logprob) == (-2.0, pytest.approx(-5.02))


def test_ranking_by_the_mean_can_pick_what_the_total_does_not() -> None:
    answer = _answer([[-2.0], [-5.0, -0.01, -0.01]], rank_by="mean")
    assert answer.winner == "b", "the split variant's near-certain tokens lift its mean"
    assert answer.candidates["a"].probability > answer.candidates["b"].probability, (
        "the probabilities stay a softmax over the totals"
    )


def test_a_tie_goes_to_the_first_candidate_in_request_order() -> None:
    assert _answer([[-1.0], [-0.5, -0.5]]).winner == "a"
    assert _answer([[-1.0], [-1.0]], c="z", d="y").winner == "c"


def test_the_softmax_is_stable_far_from_zero() -> None:
    assert decide_likelihood.softmax([-2000.0, -2001.0]) == pytest.approx(
        [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))]
    )


# --- which engine scores -------------------------------------------------------------


def test_each_engine_states_whether_and_how_it_scores_candidates() -> None:
    assert likelihood_reading("vllm").route == "prompt-logprobs"
    assert likelihood_reading("vllm").images is False
    assert likelihood_reading("mlx-lm").route == "items"
    assert likelihood_reading("mlx-vlm").route == "items" and likelihood_reading("mlx-vlm").images
    llama = likelihood_reading("llama-server")
    assert llama.route == "forced-tokens" and llama.images is False
    assert "post_sampling_probs" in llama.basis and "b10970" in llama.basis


def test_an_engine_that_cannot_score_is_refused_by_name_and_images_where_it_reads_text_only() -> None:
    cannot = SimpleNamespace(route=None, images=False, basis="it returns no log-probabilities")
    with pytest.raises(ApiError) as caught:
        decide_likelihood.refuse_unscorable("m", "some-engine", cannot, 0)
    assert caught.value.status_code == 400
    assert caught.value.code == "likelihood_unsupported_on_engine"
    with pytest.raises(ApiError) as caught:
        decide_likelihood.refuse_unscorable("m", "llama-server", likelihood_reading("llama-server"), 1)
    assert caught.value.code == "likelihood_images_unsupported_on_engine"
    with pytest.raises(ApiError) as caught:
        decide_likelihood.refuse_unscorable("m", "vllm", likelihood_reading("vllm"), 1)
    assert caught.value.code == "likelihood_images_unsupported_on_engine"
    decide_likelihood.refuse_unscorable("m", "mlx-vlm", likelihood_reading("mlx-vlm"), 2)


# --- the items route's request (mlx-lm, mlx-vlm) ------------------------------------


def _likely_ask(**changes: Any) -> dict[str, Any]:
    return {
        "model": "w",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": ""}],
        "candidates": [{"question": "q", "texts": ["a", "b"]}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_prompt_tokens": 50, "max_item_tokens": 50, "max_candidate_tokens": 5, **changes,
    }


def test_the_items_route_reads_candidates_and_still_reads_questions() -> None:
    ask = items_forward.parse_request(_likely_ask(), ("w",))
    assert isinstance(ask, items_forward.LikelihoodAsk)
    assert ask.groups == [items_forward.CandidateGroup(question="q", texts=["a", "b"])]
    questions = items_forward.parse_request(
        {"model": "w", "messages": _likely_ask()["messages"], "questions": ["x"],
         "top_logprobs": 3, "max_prompt_tokens": 5, "max_item_tokens": 5}, ("w",))
    assert isinstance(questions, items_forward.ItemsAsk)


@pytest.mark.parametrize(
    "spoil,code",
    [
        ({"top_logprobs": 3}, "unknown_field"),
        ({"candidates": []}, "bad_candidates"),
        ({"candidates": [{"question": "q", "texts": []}]}, "bad_candidates"),
        ({"candidates": [{"question": "q", "texts": ["a"], "why": 1}]}, "bad_candidates"),
        ({"max_candidate_tokens": 0}, "bad_max_candidate_tokens"),
        ({"model": "other"}, "model_not_found"),
    ],
)
def test_a_likelihood_request_the_engine_cannot_read_is_refused_by_name(spoil: dict, code: str) -> None:
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.parse_request(_likely_ask(**spoil), ("w",))
    assert caught.value.code == code


def test_the_door_sends_one_items_request_and_reads_every_group() -> None:
    body = _request(spelling=SPELLING, title={
        "type": "likelihood", "instructions": "Name the chapter.", "rank_by": "mean",
        "candidates": {"one": "The Long Road", "two": "Road", "three": "A Road"},
    })
    plans = decide.plan_all(body)
    sent: list[tuple[str, dict]] = []
    reply = {
        "object": "crucible.likelihood", "shared_tokens": 300, "cached_tokens": 280,
        "groups": [
            {"context_tokens": 330, "boundary": 330, "read_tokens": 50,
             "candidates": [{"logprobs": [-0.5, -0.1]}, {"logprobs": [-2.0, -0.1, -0.1]}]},
            {"context_tokens": 320, "boundary": 319, "read_tokens": 19,
             "candidates": [{"logprobs": [-1.0, -1.0, -1.0]}, {"logprobs": [-2.5]},
                            {"logprobs": [-1.2, -1.2]}]},
        ],
    }

    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        return reply

    scored = asyncio.run(decide_likelihood.score_on_items(call, RESIDENT, body, plans))
    ((path, wire),) = sent
    assert path == items_forward.ITEMS_PATH
    assert wire["candidates"] == [
        {"question": SPELLING["instructions"], "texts": list(SPELLING["candidates"].values())},
        {"question": "Name the chapter.", "texts": ["The Long Road", "Road", "A Road"]},
    ]
    assert wire["messages"] == decide_likelihood.open_messages(decide.render_state(body.state), [])
    assert wire["max_candidate_tokens"] == decide.MAX_CANDIDATE_TOKENS
    assert wire["max_prompt_tokens"] == wire["max_item_tokens"] == 16383
    assert wire["model"] == "/w/qwen" and wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert items_forward.parse_request(wire, ("/w/qwen",)).groups[1].texts[2] == "A Road"
    spelling, timing, tokens = scored["spelling"]
    assert spelling.winner == "plain" and spelling.candidates["plain"].logprob == pytest.approx(-0.6)
    assert spelling.boundary_tokens == 0
    # Each candidate's prompt is its boundary, as llama-server is sent it; the 300-token
    # state was 280 held and 20 read, plus the question's own 30: 50 read, 610 cached.
    assert tokens == 330 * 2 and timing.prompt_tokens == 660
    assert timing.cached_tokens == 660 - 50
    assert scored["title"][1].prompt_tokens == 319 * 3
    assert scored["title"][1].cached_tokens == 319 * 3 - 19
    title = scored["title"][0]
    assert title.winner == "one", "by the mean: -1.0 a token beats -1.2 and -2.5"
    assert title.boundary_tokens == 1


def test_an_engine_refusal_about_a_group_names_the_question_and_candidate() -> None:
    body = _request(spelling=SPELLING)
    plans = decide.plan_all(body)

    async def call(path: str, wire: dict) -> Any:
        raise decide_likelihood.refusal_error(ItemsRefusal(
            400, "candidate_too_long", "group 0's candidate 1 is 300 tokens",
            {"group": 0, "candidate": 1, "tokens": 300, "max_tokens": 256}))

    with pytest.raises(ApiError) as caught:
        asyncio.run(decide_likelihood.score_on_items(call, RESIDENT, body, plans))
    assert caught.value.code == "candidate_too_long"
    assert caught.value.details["question"] == "spelling"
    assert caught.value.details["candidate_name"] == "hyphen"
    assert "question 'spelling', candidate 'hyphen'" in caught.value.message
    assert "Shorten that candidate" in caught.value.message


def test_a_reply_with_the_wrong_shape_is_engine_error() -> None:
    plans = decide.plan_all(_request(spelling=SPELLING))
    for reply in (
        {"groups": []},
        {"groups": [{"context_tokens": 3, "boundary": 3, "read_tokens": 3,
                     "candidates": [{"logprobs": [-1.0]}]}]},
        {"groups": [{"context_tokens": 3, "boundary": 3, "read_tokens": 3,
                     "candidates": [{"logprobs": [-1.0]}, {"logprobs": []}]}]},
        {"groups": [{"context_tokens": 3, "boundary": 3,
                     "candidates": [{"logprobs": [-1.0]}, {"logprobs": [-1.0]}]}]},
        {"groups": [{"context_tokens": 3, "boundary": 3, "read_tokens": 7,
                     "candidates": [{"logprobs": [-1.0]}, {"logprobs": [-1.0]}]}]},
    ):
        with pytest.raises(ApiError) as caught:
            decide_likelihood.read_likelihood_reply(reply, "mlx-lm", plans)
        assert caught.value.code == "engine_error"


# --- the mlx-lm likelihood pass, against a fake mlx --------------------------------


class Positions:
    """A batch's hidden states as each position's whole context, so a read at
    [row, a:b] says exactly which tokens each scored position had seen."""

    def __init__(self, contexts: list[list[int]], base: int) -> None:
        self.contexts = contexts
        self.base = base

    def __getitem__(self, index: Any) -> list[list[int]]:
        row, span = index
        return [self.contexts[row][: self.base + i + 1] for i in range(span.start, span.stop)]


class LikelyModel:
    def __init__(self) -> None:
        self.forwards: list[tuple[int, int]] = []
        self.model = self._inner
        self.lm_head = lambda hidden: hidden

    def _inner(self, ids: Arr, cache: list[FakeCache]) -> Positions:
        self.forwards.append((len(ids.rows), len(ids.rows[0])))
        base = len(cache[0].rows[0])
        for entry in cache:
            if len(entry.rows) == 1 and len(ids.rows) > 1:
                entry.rows = entry.rows * len(ids.rows)
            entry.rows = [old + new for old, new in zip(entry.rows, ids.rows)]
        return Positions([list(row) for row in cache[0].rows], base)


class LikelyTokenizer:
    """`[1] state [2 3] question [4]`, the opened reply `[5 6]`, then the reply's
    characters; a reply that opens with `!` merges with the 6 into one 60."""

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> list[int]:
        assert kwargs["tokenize"] and kwargs["enable_thinking"] is False
        reply = None
        if kwargs.get("continue_final_message"):
            assert not kwargs["add_generation_prompt"]
            reply = messages[-1]["content"]
            messages = messages[:-1]
        else:
            assert kwargs["add_generation_prompt"]
        ids = [1, *(ord(c) for c in messages[0]["content"]), 2, 3,
               *(ord(c) for c in messages[-1]["content"]), 4, 5, 6]
        if reply is None:
            return ids
        if reply.startswith("!"):
            return [*ids[:-1], 60, *(ord(c) for c in reply[1:])]
        return [*ids, *(ord(c) for c in reply)]


def _fake_score(contexts: list[list[int]], targets: list[int]) -> list[float]:
    return [fake_token_logprob(context, target) for context, target in zip(contexts, targets)]


@pytest.fixture
def likely_mlx(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    core = types.ModuleType("mlx.core")
    core.array = lambda value: Arr(value.rows) if isinstance(value, Arr) else (
        Arr(value) if value and isinstance(value[0], list) else Arr([value]))
    core.eval = lambda *args: None
    core.repeat = lambda arr, n, axis: Arr(arr.rows * n)
    core.concatenate = lambda parts, axis: [context for part in parts for context in part]
    core.clear_cache = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = core
    cache_module = types.ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda model: [FakeCache(), FakeCache()]
    for name, module in {
        "mlx": mlx, "mlx.core": core, "mlx_lm": types.ModuleType("mlx_lm"),
        "mlx_lm.models": types.ModuleType("mlx_lm.models"), "mlx_lm.models.cache": cache_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(items_forward, "token_logprobs",
                        lambda head, hidden, targets: _fake_score(head(hidden), targets))
    monkeypatch.setattr(items_forward, "next_logprobs", lambda head, hidden, targets: [
        fake_token_logprob(head(hidden)[0], target) for target in targets])
    monkeypatch.setattr(items_forward, "STATES", items_forward.StateCache(4, 1 << 30))
    model = LikelyModel()
    return SimpleNamespace(model=model, provider=SimpleNamespace(
        load=lambda *a: (model, LikelyTokenizer()),
        cli_args=SimpleNamespace(prefill_step_size=2048),
    ))


def _likely_job(state: str, groups: list[tuple[str, list[str]]]) -> items_forward.MlxLmItemsJob:
    return items_forward.MlxLmItemsJob(items_forward.parse_request(_likely_ask(
        messages=[{"role": "system", "content": state}, {"role": "user", "content": ""}],
        candidates=[{"question": q, "texts": t} for q, t in groups],
        max_prompt_tokens=500, max_item_tokens=100, max_candidate_tokens=20,
    ), ("w",)))


def _whole(state: str, question: str, reply: str) -> list[int]:
    return LikelyTokenizer().apply_chat_template(
        [{"role": "system", "content": state}, {"role": "user", "content": question},
         {"role": "assistant", "content": reply}],
        add_generation_prompt=False, continue_final_message=True, tokenize=True,
        enable_thinking=False,
    )


def test_every_candidate_token_is_scored_at_its_own_position_in_one_batched_forward(
    likely_mlx: Any,
) -> None:
    groups = [("aq", ["cat", "co-op"]), ("bq", ["!x", "yz", "w"])]
    document = items_forward.mlx_lm_answer(likely_mlx.provider, _likely_job("state", groups))
    assert document["object"] == "crucible.likelihood"
    context_a = len("state") + 3 + 2 + 3
    assert [g["context_tokens"] for g in document["groups"]] == [context_a, context_a]
    assert [g["boundary"] for g in document["groups"]] == [context_a, context_a - 1], (
        "bq's `!x` merged into the opened reply's last token"
    )
    for (question, texts), group in zip(groups, document["groups"]):
        for text, row in zip(texts, group["candidates"]):
            whole = _whole("state", question, text)
            boundary = group["boundary"]
            assert row["logprobs"] == pytest.approx(
                [fake_token_logprob(whole[:p], whole[p]) for p in range(boundary, len(whole))]
            ), f"{text!r} is read at its own positions, not the padding"
    shared = len("state") + 3
    tail_a = context_a - shared
    assert likely_mlx.model.forwards == [
        (1, shared),
        (1, tail_a), (2, len("co-op") - 1),
        (1, tail_a - 1), (3, 2),
    ], (
        "the state once; per question its context once, then every candidate but "
        "its first token and its never-read last token as a row ('cat' and 'co-op' "
        "read 2 and 4 tokens; bq re-reads the opened reply's 6, so 'yz' reads [6, y], "
        "'w' [6], '!x' [60])"
    )
    assert document["shared_tokens"] == shared and document["cached_tokens"] == 0
    assert [g["read_tokens"] for g in document["groups"]] == [shared + tail_a, tail_a - 1], (
        "what each question's pass read, as the forwards above: the state once, with "
        "the first question, then each question's own tail; candidates are not prompt"
    )


def test_one_token_candidates_cost_no_forward_past_their_question(likely_mlx: Any) -> None:
    document = items_forward.mlx_lm_answer(likely_mlx.provider, _likely_job("state", [("q", ["a", "b", "c"])]))
    shared = len("state") + 3
    assert likely_mlx.model.forwards == [(1, shared), (1, len("q") + 3)], (
        "the state, then the question's tail; every first token read off its last position"
    )
    for text, row in zip("abc", document["groups"][0]["candidates"]):
        whole = _whole("state", "q", text)
        assert row["logprobs"] == pytest.approx([fake_token_logprob(whole[:-1], whole[-1])])


def test_the_next_likelihood_pass_on_the_same_state_skips_its_prefill(likely_mlx: Any) -> None:
    items_forward.mlx_lm_answer(likely_mlx.provider, _likely_job("state", [("aq", ["a", "b"]), ("bq", ["c", "d"])]))
    likely_mlx.model.forwards.clear()
    document = items_forward.mlx_lm_answer(
        likely_mlx.provider, _likely_job("state", [("cq", ["e", "f"]), ("dq", ["g", "h"])])
    )
    assert document["cached_tokens"] == len("state") + 3
    assert likely_mlx.model.forwards == [(1, len("cq") + 3), (1, len("dq") + 3)], (
        "no prefill: each question's tail straight away, and one-token candidates need no row"
    )
    assert [g["read_tokens"] for g in document["groups"]] == [len("cq") + 3, len("dq") + 3], (
        "the held state is read by no question, so each reads only its own tail"
    )


def test_a_candidate_past_its_cap_is_refused_before_any_forward(likely_mlx: Any) -> None:
    job = _likely_job("state", [("qa", ["a", "b" * 30])])
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.mlx_lm_answer(likely_mlx.provider, job)
    assert caught.value.code == "candidate_too_long" and caught.value.details["candidate"] == 1
    assert likely_mlx.model.forwards == []


def test_each_reader_counts_what_its_passes_read_per_question() -> None:
    # Two questions over a 4-token shared part; boundaries 6 and 7; two candidates each.
    split = likelihood_split(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 8, 9, 10]],
        [[[1, 2, 3, 4, 5, 6, 20], [1, 2, 3, 4, 5, 6, 21]],
         [[1, 2, 3, 4, 8, 9, 10, 20], [1, 2, 3, 4, 8, 9, 10, 21]]],
        100, 100, 10,
    )
    assert split.boundaries == [6, 7] and len(split.shared) == 4
    assert items_forward.question_read_tokens(split, 4, 0) == [4 + 2, 3]
    assert items_forward.question_read_tokens(split, 4, 3) == [1 + 2, 3], "3 held"
    assert items_forward.row_read_tokens(split) == [4 + 2 * 2, 3 * 2], (
        "mlx-vlm reads the shared part once and every candidate's own row to its boundary"
    )


def test_a_non_finite_log_probability_is_engine_error_not_a_number() -> None:
    split = likelihood_split([[1, 2]], [[[1, 2, 3], [1, 2, 4]]], 10, 10, 10)
    with pytest.raises(ItemsRefusal) as caught:
        items_forward.likelihood_document(split, [[-1.0], [-math.inf]], [2])
    assert caught.value.code == "engine_error" and caught.value.details == {"group": 0}


# --- the vLLM path, against the fake engine's template ------------------------------


def _vllm_call(
    sent: list[tuple[str, dict]], spoil: Callable[[dict], dict] | None = None
) -> Callable[[str, dict], Any]:
    from .fake_engine import _Handler, fake_prompt_ids

    handler = type("H", (_Handler,), {"served_name": "/w/qwen"})

    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        if path == decide_likelihood.TOKENIZE_PATH:
            return {"count": 0, "max_model_len": 8192, "tokens": fake_prompt_ids(wire)}
        reply = object.__new__(handler)._prompt_logprobs_reply(wire)
        return reply if spoil is None else spoil(reply)

    return call


def _vllm(body: DecideRequest, sent: list, spoil: Callable[[dict], dict] | None = None) -> Any:
    resident = SimpleNamespace(**{**vars(RESIDENT), "engine": "vllm"})
    return asyncio.run(decide_likelihood.score_on_prompt_logprobs(
        _vllm_call(sent, spoil), resident, body, decide.plan_all(body), concurrency=4,
    ))


def test_vllm_tokenizes_everything_first_then_scores_each_candidate_s_prompt() -> None:
    body = _request(spelling=SPELLING)
    sent: list[tuple[str, dict]] = []
    scored = _vllm(body, sent)
    paths = [path for path, _ in sent]
    assert paths == ["/tokenize"] * 3 + ["/v1/chat/completions"] * 2, "no forward before every check"
    context, *_ = [wire for path, wire in sent if path == "/tokenize"]
    assert context["add_generation_prompt"] is True and context["continue_final_message"] is False
    chat = [wire for path, wire in sent if path == "/v1/chat/completions"]
    for wire, text in zip(chat, SPELLING["candidates"].values()):
        assert wire["messages"][-1] == {"role": "assistant", "content": text}
        assert wire["continue_final_message"] is True and wire["add_generation_prompt"] is False
        assert wire["prompt_logprobs"] == 0 and wire["return_token_ids"] is True
        assert wire["chat_template_kwargs"] == {"enable_thinking": False}
        assert wire["max_tokens"] == 1 and wire["temperature"] == 0 and wire["stream"] is False
    answer, timing, tokens = scored["spelling"]
    msgs = decide_likelihood.likelihood_messages(decide.render_state(body.state), [], SPELLING["instructions"])
    prefix = [ord(c) for c in _rendered(msgs) + GENERATION_PROMPT]
    assert answer.context_tokens == len(prefix) and answer.boundary_tokens == 0
    for name, text in SPELLING["candidates"].items():
        whole = prefix + [ord(c) for c in text]
        expected = [fake_token_logprob(whole[:p], whole[p]) for p in range(len(prefix), len(whole))]
        assert answer.candidates[name].logprob == pytest.approx(math.fsum(expected))
        assert answer.candidates[name].tokens == len(text)
    assert tokens == sum(len(prefix) + len(t) for t in SPELLING["candidates"].values())
    assert timing.cached_tokens == 0


def test_vllm_refuses_a_long_candidate_before_any_forward() -> None:
    long = {**SPELLING, "candidates": {"a": "x", "b": "y" * 300}}
    sent: list[tuple[str, dict]] = []
    with pytest.raises(ApiError) as caught:
        _vllm(_request(q=long), sent)
    assert caught.value.code == "candidate_too_long" and caught.value.details["candidate_name"] == "b"
    assert all(path == "/tokenize" for path, _ in sent)


def test_vllm_prompt_ids_that_disagree_with_tokenize_are_engine_error() -> None:
    def shifted(reply: dict) -> dict:
        return {**reply, "prompt_token_ids": [0, *reply["prompt_token_ids"][1:]]}

    with pytest.raises(ApiError) as caught:
        _vllm(_request(spelling=SPELLING), [], shifted)
    assert caught.value.code == "engine_error" and "/tokenize" in caught.value.message


def test_vllm_s_clamped_minus_infinity_is_not_read_as_a_number() -> None:
    def impossible(reply: dict) -> dict:
        entries = list(reply["prompt_logprobs"])
        last = entries[-1]
        entries[-1] = {token: {**value, "logprob": -9999.0} for token, value in last.items()}
        return {**reply, "prompt_logprobs": entries}

    with pytest.raises(ApiError) as caught:
        _vllm(_request(spelling=SPELLING), [], impossible)
    assert caught.value.code == "engine_error" and "no probability" in caught.value.message


# --- the llama-server path: forced continuations, against a fake engine -------------


LLAMA_OPEN = "<a>"


def _llama_render(wire: dict) -> str:
    """A template whose open reply is the generation prompt plus the content."""
    msgs = wire["messages"]
    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    if wire["continue_final_message"]:
        assert wire["add_generation_prompt"] is False
        *turns, last = msgs
        return "".join(f"[{m['content']}]" for m in turns) + LLAMA_OPEN + last["content"]
    assert wire["add_generation_prompt"] is True
    return "".join(f"[{m['content']}]" for m in msgs) + LLAMA_OPEN


def _llama_call(
    sent: list[tuple[str, dict]], spoil: Callable[[dict, dict], dict] | None = None
) -> Callable[[str, dict], Any]:
    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        if path == decide_likelihood.APPLY_TEMPLATE_PATH:
            return {"prompt": _llama_render(wire)}
        if path == decide_likelihood.TOKENIZE_PATH:
            assert wire["add_special"] is True and wire["parse_special"] is True
            return {"tokens": [ord(c) for c in wire["content"]]}
        assert path == decide_likelihood.COMPLETION_PATH
        prompt, grammar = list(wire["prompt"]), wire["grammar"]
        forced = [int(t) for t in grammar.removeprefix("root ::= ").replace("<[", "").replace("]>", "").split()]
        reply = {
            "tokens": forced,
            "completion_probabilities": [
                {"id": t, "logprob": fake_token_logprob(prompt + forced[:i], t), "top_logprobs": []}
                for i, t in enumerate(forced)
            ],
            "timings": {"cache_n": len(prompt) - 4, "prompt_n": 4},
        }
        return reply if spoil is None else spoil(wire, reply)

    return call


def _llama(body: DecideRequest, sent: list, spoil: Callable[[dict, dict], dict] | None = None) -> Any:
    resident = SimpleNamespace(**{**vars(RESIDENT), "engine": "llama-server"})
    return asyncio.run(decide_likelihood.score_on_forced_tokens(
        _llama_call(sent, spoil), resident, body, decide.plan_all(body),
    ))


def test_llama_server_tokenizes_everything_first_then_forces_each_candidate() -> None:
    body = _request(spelling=SPELLING)
    sent: list[tuple[str, dict]] = []
    scored = _llama(body, sent)
    paths = [path for path, _ in sent]
    assert paths == ["/apply-template", "/tokenize"] * 3 + ["/completion"] * 2, (
        "no forward before every check"
    )
    msgs = decide_likelihood.likelihood_messages(decide.render_state(body.state), [], SPELLING["instructions"])
    prefix = [ord(c) for c in "".join(f"[{m['content']}]" for m in msgs) + LLAMA_OPEN]
    completions = [wire for path, wire in sent if path == "/completion"]
    answer, timing, tokens = scored["spelling"]
    for wire, (name, text) in zip(completions, SPELLING["candidates"].items()):
        targets = [ord(c) for c in text]
        assert wire["prompt"] == prefix, "the context up to the boundary, as token ids"
        assert wire["grammar"] == "root ::= " + " ".join(f"<[{t}]>" for t in targets)
        assert wire["n_predict"] == len(targets) and wire["n_probs"] == 1
        assert wire["post_sampling_probs"] is False and wire["cache_prompt"] is True
        assert wire["return_tokens"] is True and wire["stream"] is False
        whole = prefix + targets
        expected = [fake_token_logprob(whole[:p], whole[p]) for p in range(len(prefix), len(whole))]
        assert answer.candidates[name].logprob == pytest.approx(math.fsum(expected))
        assert answer.candidates[name].tokens == len(text)
    assert answer.context_tokens == len(prefix) and answer.boundary_tokens == 0
    assert tokens == 2 * len(prefix) and timing.prompt_tokens == tokens
    assert timing.cached_tokens == 2 * (len(prefix) - 4)


def test_llama_server_refuses_a_long_candidate_before_any_forward() -> None:
    long = {**SPELLING, "candidates": {"a": "x", "b": "y" * 300}}
    sent: list[tuple[str, dict]] = []
    with pytest.raises(ApiError) as caught:
        _llama(_request(q=long), sent)
    assert caught.value.code == "candidate_too_long" and caught.value.details["candidate_name"] == "b"
    assert all(path != "/completion" for path, _ in sent)


@pytest.mark.parametrize(
    "spoil,fragment",
    [
        (lambda wire, reply: {**reply, "tokens": reply["tokens"][:-1]}, "did not hold the reply"),
        (lambda wire, reply: {**reply, "completion_probabilities": reply["completion_probabilities"][1:]},
         "completion_probabilities"),
        (lambda wire, reply: {**reply, "completion_probabilities": [
            {**entry, "logprob": -3.4028234663852886e38} for entry in reply["completion_probabilities"]]},
         "no probability at all"),
        (lambda wire, reply: {**reply, "timings": {}}, "cache_n"),
    ],
)
def test_a_llama_server_reply_that_is_not_the_forced_candidate_is_engine_error(
    spoil: Callable[[dict, dict], dict], fragment: str
) -> None:
    with pytest.raises(ApiError) as caught:
        _llama(_request(spelling=SPELLING), [], spoil)
    assert caught.value.code == "engine_error" and fragment in caught.value.message


# --- the door, end to end ------------------------------------------------------------


@pytest.fixture
def loaded(
    llm_client: TestClient,  # noqa: F811
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


def _yes(messages: list[dict[str, Any]]) -> dict[str, float]:
    return {"A": 0.8, "B": 0.15}


def test_a_likelihood_question_beside_a_label_question_end_to_end(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=_yes, scores_prompts=True)
    response = llm_client.post("/v1/decide", headers=auth, json={
        "model": MODEL, "state": "First. They agreed to co-operate.",
        "questions": {
            "urgent": {"type": "yesno", "instructions": "It is urgent"},
            "spelling": SPELLING,
        },
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["answers"]) == ["urgent", "spelling"]
    assert body["answers"]["urgent"]["p"] == pytest.approx(0.8 / 0.95)
    spelling = body["answers"]["spelling"]
    assert spelling["type"] == "likelihood" and spelling["rank_by"] == "total"
    assert list(spelling["candidates"]) == ["plain", "hyphen"]
    totals = {name: c["logprob"] for name, c in spelling["candidates"].items()}
    assert spelling["winner"] == max(totals, key=totals.__getitem__)
    assert sum(c["probability"] for c in spelling["candidates"].values()) == pytest.approx(1.0)
    assert set(body["timing_ms"]["per_question"]) == {"urgent", "spelling"}
    assert body["timing_ms"]["prime"] is None, "one label question sends no prime"
    assert len(engine.tokenized) == 3
    scored = [r for r in engine.requests if "prompt_logprobs" in r]
    assert len(scored) == 2 and all("logprobs" not in r for r in scored)


def test_a_likelihood_question_for_an_engine_that_cannot_score_is_refused_before_it_waits(
    llm_client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    from crucible.engines.llama_server import LlamaServerEngine

    monkeypatch.setattr(LlamaServerEngine, "decide_likelihood_route", None)
    response = llm_client.post("/v1/decide", headers=auth, json={
        "model": "qwen3.5-4b-bside", "state": "s", "questions": {"spelling": SPELLING},
    })
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "likelihood_unsupported_on_engine"
    assert error["details"] == {"model": "qwen3.5-4b-bside", "engine": "llama-server"}


def test_an_older_items_route_is_named_with_the_load_that_fixes_it() -> None:
    from crucible.api.routes import decide as route

    older = route._items_route_older(RESIDENT, {"error": {
        "code": "unknown_field", "message": "an items request does not carry ['candidates']"}})
    assert older is not None and older.code == "decide_not_served"
    assert '"load-model"' in older.message and "candidates" in older.message
    assert route._items_route_older(RESIDENT, {"error": {"code": "bad_json"}}) is None


def test_a_decision_of_likelihood_questions_alone_sends_no_label_request(
    llm_client: TestClient, auth: dict[str, str], loaded: Callable[..., FakeEngine]  # noqa: F811
) -> None:
    engine = loaded(probs_for=_yes, scores_prompts=True)
    title = {"type": "likelihood", "instructions": "Name the chapter.", "rank_by": "mean",
             "candidates": {"long": "The Long Road", "short": "Road"}}
    response = llm_client.post("/v1/decide", headers=auth, json={
        "model": MODEL, "state": "A chapter.", "questions": {"spelling": SPELLING, "title": title},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["answers"]) == ["spelling", "title"]
    assert body["answers"]["title"]["rank_by"] == "mean"
    assert body["timing_ms"]["prime"] is None
    assert len(engine.requests) == 4 and all("prompt_logprobs" in r for r in engine.requests)
    assert body["tokens"]["per_question"]["title"] == body["timing_ms"]["per_question"]["title"]["prompt_tokens"]
