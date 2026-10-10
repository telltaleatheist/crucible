"""The Mac's decide floor: the question form through the items route, and the
items engine's batched rows and held states (engines/items_forward.py), read
with fakes for mlx so the logic runs on any machine."""

from __future__ import annotations

import asyncio
import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import decide, decide_items
from crucible.decide import DecideRequest
from crucible.engines import decide_items_reading, items_forward
from crucible.errors import ApiError

RESIDENT = SimpleNamespace(
    model_id="qwen3.5-9b", engine="mlx-lm", engine_model_name="/w/qwen", revision="r1",
    fingerprint="qwen3.5-9b@r1", max_model_len=16384,
)

QUESTIONS: dict[str, Any] = {
    "team": {"type": "choice", "instructions": "Which team?",
             "options": {"billing": "Payment", "technical": "Bugs", "other": "Else"}},
    "urgent": {"type": "yesno", "instructions": "The message conveys urgency"},
    "anger": {"type": "score", "instructions": "How frustrated?",
              "levels": ["Calm", "Civil", "Angry"]},
}


def _questions_request(**changes: Any) -> DecideRequest:
    return DecideRequest.model_validate(
        {"model": "qwen3.5-9b", "state": {"msg": "charged twice"}, "questions": QUESTIONS, **changes}
    )


def _reply(rows: list[dict[str, float]], cached: Any = "absent") -> dict[str, Any]:
    reply: dict[str, Any] = {
        "object": "crucible.items", "shared_tokens": 300, "item_tokens": [30 + i for i in range(len(rows))],
        "slots": [
            {"top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in row.items()]}
            for row in rows
        ],
    }
    if cached != "absent":
        reply["cached_tokens"] = cached
    return reply


def _on_items(body: DecideRequest, reply: Any, sent: list) -> Any:
    async def call(path: str, wire: dict) -> Any:
        sent.append((path, wire))
        return reply

    return asyncio.run(decide_items.decide_questions_on_items(
        call, RESIDENT, body, decide.plan_all(body), max_logprobs=40,
    ))


def test_the_question_form_is_one_items_request_with_the_chat_path_s_prompts() -> None:
    body = _questions_request()
    sent: list[tuple[str, dict]] = []
    rows = [{"A": 0.7, "B": 0.2, "C": 0.05}, {"A": 0.8, "B": 0.15}, {"C": 0.6, "A": 0.3, "B": 0.05}]
    response = _on_items(body, _reply(rows, cached=261), sent)
    ((path, wire),) = sent
    assert path == items_forward.ITEMS_PATH
    plans = decide.plan_all(body)
    state = decide.render_state(body.state)
    for item, question in zip(plans, wire["questions"]):
        assert items_forward.item_messages(wire["messages"], question) == decide.question_messages(
            state, [], item
        ), "every row is the prompt the chat path sends for that question"
    assert wire["top_logprobs"] == decide.top_k(3, 40)
    assert wire["max_item_tokens"] == wire["max_prompt_tokens"] == 16383
    assert response.answers["team"].choice == "billing"
    assert response.answers["urgent"].p == pytest.approx(0.8 / 0.95)
    assert response.answers["anger"].probabilities["Angry"] == pytest.approx(0.6 / 0.95)
    assert list(response.answers) == ["team", "urgent", "anger"]
    timing = response.timing_ms
    assert timing.prime is None
    assert [t.prompt_tokens for t in timing.per_question.values()] == [330, 331, 332]
    assert {t.cached_tokens for t in timing.per_question.values()} == {261}
    assert len({t.wall_ms for t in timing.per_question.values()}) == 1, "one request, one clock"
    assert response.tokens.per_question == {"team": 330, "urgent": 331, "anger": 332}


def test_an_engine_that_does_not_say_what_it_reused_reports_null_not_zero() -> None:
    response = _on_items(_questions_request(questions={"urgent": QUESTIONS["urgent"]}),
                         _reply([{"A": 0.5, "B": 0.5}]), [])
    assert response.timing_ms.per_question["urgent"].cached_tokens is None


def test_a_cached_count_that_is_not_an_integer_is_engine_error() -> None:
    with pytest.raises(ApiError) as caught:
        decide_items.read_items_reply(_reply([{"A": 1.0}], cached="261"), "mlx-lm", 1)
    assert caught.value.code == "engine_error" and "cached_tokens" in caught.value.message


def test_a_missing_label_is_refused_by_question_name() -> None:
    with pytest.raises(ApiError) as caught:
        _on_items(_questions_request(), _reply([{"A": 0.7, "B": 0.2, "C": 0.05}, {"A": 0.9}, {"A": 0.3, "B": 0.3, "C": 0.3}]), [])
    assert caught.value.code == "label_not_in_probs" and caught.value.details["question"] == "urgent"


def test_only_mlx_lm_reads_the_question_form_as_items() -> None:
    assert decide_items_reading("mlx-lm").questions is True
    for engine in ("mlx-vlm", "vllm", "llama-server"):
        assert decide_items_reading(engine).questions is False


def test_a_question_batching_engine_without_an_items_route_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.engines import EngineError, engine_class

    cls = engine_class("vllm")
    monkeypatch.setattr(cls, "decide_questions_batched", True)
    monkeypatch.setattr(cls, "decide_questions_basis", "test")
    with pytest.raises(EngineError) as caught:
        decide_items_reading("vllm")
    assert "no batched items route" in str(caught.value)


# --- the items engine's pure pieces -------------------------------------------------


def test_the_common_prefix_may_be_a_whole_prompt() -> None:
    assert items_forward.common_prefix([[1, 2, 3], [1, 2, 4]]) == 2
    assert items_forward.common_prefix([[1, 2], [1, 2, 3]]) == 2
    assert items_forward.common_prefix([[5], [6]]) == 0


def test_rows_per_pass_is_bounded_by_bytes_tokens_and_the_ceiling() -> None:
    row = items_forward.ROW_BYTES // 10
    assert items_forward.rows_per_pass(row, 30) == 10
    assert items_forward.rows_per_pass(1, 30) == items_forward.MAX_ROWS
    assert items_forward.rows_per_pass(1, 1024) == items_forward.CHUNK_TOKENS // 1024
    assert items_forward.rows_per_pass(items_forward.ROW_BYTES * 4, 30) == 1, "never zero"
    assert items_forward.rows_per_pass(0, 5000) == 1


def test_row_groups_take_the_longest_first_and_cover_every_item_once() -> None:
    suffixes = [[1] * n for n in (3, 9, 1, 9, 5)]
    groups = items_forward.row_groups(suffixes, lambda longest: 2)
    assert groups == [[1, 3], [4, 0], [2]]
    assert sorted(i for group in groups for i in group) == list(range(5))
    sized = items_forward.row_groups(suffixes, lambda longest: 1 if longest > 5 else 3)
    assert sized == [[1], [3], [4, 0, 2]]


def test_padding_is_on_the_right_and_each_row_is_read_at_its_own_end() -> None:
    rows, lasts = items_forward.padded([[7, 8, 9], [4], [5, 6]])
    assert rows == [[7, 8, 9], [4, 4, 4], [5, 6, 6]]
    assert lasts == [2, 0, 1]


def test_a_held_state_is_reused_only_as_a_prefix_and_only_for_its_model() -> None:
    model, other = object(), object()
    states = items_forward.StateCache(entries=4, max_bytes=1000)
    states.keep(model, [1, 2, 3], ["c123"], 10)
    states.keep(model, [1, 2], ["c12"], 10)
    assert states.nearest(model, [1, 2, 3, 4]).cache == ["c123"], "the longest prefix"
    assert states.nearest(model, [1, 2, 9]).cache == ["c12"]
    assert states.nearest(model, [1, 9]) is None, "a held state is never trimmed back"
    assert states.nearest(model, [1, 2]).cache == ["c12"], "an exact state is a prefix"
    assert states.nearest(other, [1, 2, 3]) is None
    states.keep(other, [1, 2, 3], ["o"], 10)
    assert len(states) == 1, "another model's states are dropped when it keeps one"


def test_held_states_are_bounded_by_count_and_bytes_oldest_first() -> None:
    model = object()
    states = items_forward.StateCache(entries=2, max_bytes=100)
    states.keep(model, [1], ["a"], 10)
    states.keep(model, [2], ["b"], 10)
    states.nearest(model, [1, 5])
    states.keep(model, [3], ["c"], 10)
    assert states.nearest(model, [2]) is None, "the least recently used went"
    assert states.nearest(model, [1]) is not None and states.nearest(model, [3]) is not None
    states.keep(model, [4], ["d"], 95)
    assert len(states) == 1 and states.nbytes == 95
    states.keep(model, [5], ["e"], 101)
    assert states.nearest(model, [5]) is None, "a state over the budget is read and not kept"
    states.keep(model, [4], ["d2"], 95)
    assert len(states) == 1 and states.nearest(model, [4]).cache == ["d2"], "re-kept, not doubled"


# --- the mlx-lm items pass, run against a fake mlx ----------------------------------


class Arr:
    """A token list standing in for an mx.array: rows of token ids."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(row) for row in rows]

    @property
    def nbytes(self) -> int:
        return 4 * sum(len(row) for row in self.rows)

    def __getitem__(self, index: Any) -> "Arr":
        first, span = index
        if first is None:
            return Arr([self.rows[0][span]])
        assert first == slice(None), "rows are only ever sliced whole"
        return Arr([row[span] for row in self.rows])


class Hidden:
    """Each row's whole context; indexing it at (rows, lasts) gives what each row
    was read at: the shared state plus its suffix up to its own last token."""

    def __init__(self, contexts: list[list[int]], base: int) -> None:
        self.contexts = contexts
        self.base = base

    def __getitem__(self, index: Any) -> list[list[int]]:
        rows, lasts = index
        return [self.contexts[row][: self.base + 1 + last] for row, last in zip(rows, lasts.rows[0])]


class FakeCache:
    def __init__(self) -> None:
        self.rows: list[list[int]] = [[]]

    @property
    def state(self) -> list[Arr]:
        return [Arr(self.rows)]

    @state.setter
    def state(self, value: list[Arr]) -> None:
        (arr,) = value
        self.rows = [list(row) for row in arr.rows]


class FakeModel:
    def __init__(self) -> None:
        self.forwards: list[tuple[int, int]] = []
        self.model = self._inner
        self.lm_head = lambda hidden: hidden

    def _inner(self, ids: Arr, cache: list[FakeCache]) -> Hidden:
        self.forwards.append((len(ids.rows), len(ids.rows[0])))
        base = len(cache[0].rows[0])
        for entry in cache:
            if len(entry.rows) == 1 and len(ids.rows) > 1:
                entry.rows = entry.rows * len(ids.rows)
            entry.rows = [old + new for old, new in zip(entry.rows, ids.rows)]
        return Hidden([list(row) for row in cache[0].rows], base)


class FakeTokenizer:
    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> list[int]:
        assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
        state = [ord(ch) for ch in messages[0]["content"]]
        question = [ord(ch) for ch in messages[-1]["content"]]
        return [1, *state, 2, 3, *question, 4, 5]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(ids[0])]


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    core = types.ModuleType("mlx.core")
    core.array = lambda value: Arr(value.rows) if isinstance(value, Arr) else (
        Arr(value) if value and isinstance(value[0], list) else Arr([value]))
    core.eval = lambda *args: None
    core.repeat = lambda arr, n, axis: Arr(arr.rows * n)
    core.arange = lambda n: list(range(n))
    core.clear_cache = lambda: None
    # Pieces of one forward: the last piece's contexts are the whole rows (the cache
    # holds everything read), positions counted from where the first piece began.
    core.concatenate = lambda pieces, axis: Hidden(pieces[-1].contexts, pieces[0].base)
    mlx = types.ModuleType("mlx")
    mlx.core = core
    cache_module = types.ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda model: [FakeCache(), FakeCache()]
    for name, module in {
        "mlx": mlx, "mlx.core": core, "mlx_lm": types.ModuleType("mlx_lm"),
        "mlx_lm.models": types.ModuleType("mlx_lm.models"), "mlx_lm.models.cache": cache_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def top_of(head: Any, contexts: list[list[int]], k: int) -> list[list[tuple[int, float]]]:
        return [[(len(context), 0.0), (sum(context) % 997, -1.0)] for context in contexts]

    monkeypatch.setattr(items_forward, "top_of", top_of)
    monkeypatch.setattr(items_forward, "STATES", items_forward.StateCache(4, 1 << 30))
    model = FakeModel()
    cli_args = SimpleNamespace(prefill_step_size=2048)
    provider = SimpleNamespace(load=lambda *args: (model, FakeTokenizer()), cli_args=cli_args)
    return SimpleNamespace(model=model, provider=provider)


def _job(state: str, questions: list[str]) -> items_forward.MlxLmItemsJob:
    return items_forward.MlxLmItemsJob(items_forward.parse_items({
        "model": "w", "messages": [{"role": "system", "content": state}, {"role": "user", "content": ""}],
        "questions": questions, "top_logprobs": 2, "max_prompt_tokens": 500, "max_item_tokens": 100,
    }, ("w",)))


def _expected(state: str, question: str) -> list[int]:
    return FakeTokenizer().apply_chat_template(
        [{"role": "system", "content": state}, {"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=True,
    )


def test_every_item_is_read_at_its_whole_prompt_in_one_batched_forward(fake_mlx: Any) -> None:
    questions = ["ab", "c", "defg"]
    document = items_forward.mlx_lm_answer(fake_mlx.provider, _job("state", questions))
    for question, slot in zip(questions, document["slots"]):
        whole = _expected("state", question)
        assert slot["top_logprobs"][0]["token"] == str(len(whole))
        assert slot["top_logprobs"][1]["token"] == str(sum(whole) % 997), "read at its own end, not the padding"
    shared = len("state") + 3
    assert fake_mlx.model.forwards == [(1, shared), (3, 6)], "the state once, then one forward of 3 rows"
    assert document["cached_tokens"] == 0 and document["shared_tokens"] == shared


def test_the_next_decision_on_the_same_state_skips_its_prefill(fake_mlx: Any) -> None:
    # The two questions share their first character too: the state is kept where
    # the STATE ends, not where these questions stop agreeing.
    items_forward.mlx_lm_answer(fake_mlx.provider, _job("state", ["qa", "qbc"]))
    shared = len("state") + 3
    assert fake_mlx.model.forwards == [(1, shared), (1, 1), (2, 4)]
    fake_mlx.model.forwards.clear()
    document = items_forward.mlx_lm_answer(fake_mlx.provider, _job("state", ["xyz"]))
    assert document["cached_tokens"] == shared
    # One question: past the held state, the question and the template's tail are
    # read in ONE forward, its row; nothing is read twice and no forward is spent
    # on a prefix no other row shares.
    assert fake_mlx.model.forwards == [(1, 3 + 2)]
    assert document["shared_tokens"] == shared and document["item_tokens"] == [5]
    whole = _expected("state", "xyz")
    assert document["slots"][0]["top_logprobs"][0]["token"] == str(len(whole))
    assert document["slots"][0]["top_logprobs"][1]["token"] == str(sum(whole) % 997)


def test_a_different_state_is_read_whole(fake_mlx: Any) -> None:
    items_forward.mlx_lm_answer(fake_mlx.provider, _job("state", ["ab", "c"]))
    fake_mlx.model.forwards.clear()
    document = items_forward.mlx_lm_answer(fake_mlx.provider, _job("other", ["ab", "c"]))
    assert document["cached_tokens"] == 0, "a held state that does not open this one is not used"
    assert fake_mlx.model.forwards[0] == (1, len("other") + 3)


def test_no_forward_reads_more_than_the_engines_prefill_step(fake_mlx: Any) -> None:
    # The engine's --prefill-step-size bounds every evaluation of the items pass:
    # the state is read in steps, rows per forward are step // longest, and a row
    # longer than the step is read in pieces that continue its cache.
    fake_mlx.provider.cli_args.prefill_step_size = 4
    questions = ["abcdefghij", "k", "lm"]
    document = items_forward.mlx_lm_answer(fake_mlx.provider, _job("statestate", questions))
    shared = len("statestate") + 3
    assert fake_mlx.model.forwards[:4] == [(1, 4), (1, 4), (1, 4), (1, 1)], "the state in steps"
    assert all(rows * tokens <= 4 for rows, tokens in fake_mlx.model.forwards)
    assert fake_mlx.model.forwards[4:] == [(1, 4), (1, 4), (1, 4), (1, 4), (1, 3)], (
        "the 12-token row in three pieces of 4 over its own cache, then the others "
        "one row each (4 // 4 and 4 // 3 rows)"
    )
    for question, slot in zip(questions, document["slots"]):
        whole = _expected("statestate", question)
        assert slot["top_logprobs"][0]["token"] == str(len(whole)), question
        assert slot["top_logprobs"][1]["token"] == str(sum(whole) % 997), question
    assert document["shared_tokens"] == shared


def test_rows_are_split_when_the_state_is_too_big_to_repeat(
    fake_mlx: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(items_forward, "ROW_BYTES", 2 * 4 * 2 * (len("state") + 3))
    questions = ["a", "bb", "ccc", "dddd", "e"]
    document = items_forward.mlx_lm_answer(fake_mlx.provider, _job("state", questions))
    assert [rows for rows, _ in fake_mlx.model.forwards[1:]] == [2, 2, 1]
    for question, slot in zip(questions, document["slots"]):
        whole = _expected("state", question)
        assert slot["top_logprobs"][0]["token"] == str(len(whole))
        assert slot["top_logprobs"][1]["token"] == str(sum(whole) % 997)
