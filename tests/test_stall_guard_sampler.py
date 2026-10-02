from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from crucible import stallguard

from .test_stall_guard import (
    BOC_ID,
    DEFAULT_ENV,
    EOC_ID,
    FIXTURES,
    MALFORMED,
    N,
    V,
    make_env,
    package_of,
    run_script,
    texts,
)

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------- the patched sampler



def _stub_modules() -> dict[str, types.ModuleType]:
    def top_k_renorm(probs: Any, k: Any) -> Any:
        kth = probs.sort(dim=-1, descending=True).values.gather(1, (k.long() - 1).view(-1, 1))
        kept = torch.where(probs >= kth, probs, torch.zeros_like(probs))
        return kept / kept.sum(dim=-1, keepdim=True)

    def top_p_renorm(probs: Any, p: Any) -> Any:
        values, order = probs.sort(dim=-1, descending=True)
        before = values.cumsum(dim=-1) - values
        keep_sorted = before < p.view(-1, 1)
        keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
        kept = torch.where(keep, probs, torch.zeros_like(probs))
        return kept / kept.sum(dim=-1, keepdim=True)

    def multinomial_with_seed(logprobs: Any, seeds: Any, positions: Any) -> Any:
        return torch.zeros((logprobs.shape[0], 1), dtype=torch.long)

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "sgl_kernel", "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.sampler", "sglang_omni", "sglang_omni.models",
        "sglang_omni.models.higgs_tts", "sglang_omni.models.higgs_tts.utils",
    ):
        modules[name] = types.ModuleType(name)
    modules["sgl_kernel"].top_k_renorm_prob = top_k_renorm
    modules["sgl_kernel"].top_p_renorm_prob = top_p_renorm
    modules["sglang.srt.layers.sampler"].multinomial_with_seed = multinomial_with_seed
    modules["sglang_omni.models.higgs_tts.utils"].BOC_ID = BOC_ID
    modules["sglang_omni.models.higgs_tts.utils"].EOC_ID = EOC_ID
    return modules


@pytest.fixture
def load_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Any]]:
    for name, module in _stub_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    patched = make_env(tmp_path / "patched")
    assert run_script(patched).returncode == 0
    counter = [0]

    def load(guard: str | None, *, stock_file: bool = False) -> Any:
        if guard is None:
            monkeypatch.delenv("HIGGS_STALL_GUARD", raising=False)
        else:
            monkeypatch.setenv("HIGGS_STALL_GUARD", guard)
        counter[0] += 1
        name = f"_stall_sampler_{counter[0]}"
        path = (FIXTURES / "sampler.py.txt") if stock_file else package_of(patched) / "sampler.py"
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        return module

    yield load


def sampling(B: int, temperature: float) -> dict[str, Any]:
    return {
        "temperature": torch.full((B,), temperature),
        "top_p": torch.full((B,), 0.95),
        "top_k_buf": torch.full((B,), 50, dtype=torch.long),
    }


def silence_logits(B: int, *, stuck: float = 10.0, exit_gap: float = 5.0, noise: bool = False):
    logits = torch.randn(B, N, V) if noise else torch.zeros(B, N, V)
    logits[:, :, 7] += stuck
    logits[:, 0, 300] += stuck - exit_gap
    return logits


def test_import_reads_the_variable_and_refuses_a_malformed_one(load_sampler: Any) -> None:
    assert load_sampler(None).STALL_GUARD is None
    assert load_sampler("off").STALL_GUARD is None
    on = load_sampler(DEFAULT_ENV)
    assert (on.STALL_GUARD.frames, on.STALL_GUARD.rate, on.STALL_GUARD.max,
            on.STALL_GUARD.window) == (37, 1.0, 20.0, 16)
    assert on.STALL_RING_WIDTH == 16
    with pytest.raises(ValueError) as caught:
        load_sampler("37,0.5,20")
    assert "HIGGS_STALL_GUARD" in str(caught.value)


@pytest.mark.parametrize("raw", MALFORMED + [DEFAULT_ENV, "off", " off ", "1,0.001,1000,64"])
def test_the_server_and_crucible_read_the_variable_alike(load_sampler: Any, raw: str) -> None:
    S = load_sampler(None)
    try:
        ours = stallguard.parse_env(raw)
    except stallguard.StallGuardError:
        with pytest.raises(ValueError):
            S.parse_stall_guard(raw)
        return
    theirs = S.parse_stall_guard(raw)
    if ours is None:
        assert theirs is None
    else:
        assert (theirs.frames, theirs.rate, theirs.max, theirs.window) == (
            ours.frames, ours.rate, ours.max, ours.window,
        )


def greedy_run(S: Any, frames: int, **logit_kw: Any) -> tuple[list[int], Any, list[Any]]:
    state = S.HiggsBatchedSamplerState(1, N, device="cpu")
    rows = torch.arange(1)
    cb0: list[int] = []
    after: list[Any] = []
    for _ in range(frames):
        logits = silence_logits(1, **logit_kw)
        codes = S.batched_step(logits, state, rows, **sampling(1, 0.0))
        cb0.append(int(codes[0, 0]))
        after.append(logits)
    return cb0, state, after


def test_off_a_greedy_silence_never_leaves(load_sampler: Any) -> None:
    cb0, _, after = greedy_run(load_sampler("off"), 200)
    assert set(cb0[N:]) == {7}
    assert all(torch.equal(x, silence_logits(1)) for x in after)


def expected_exit_step(frames: int, rate: float, cap: float, gap: float) -> int | None:
    # Steady frame s (s = 0 is the first past the delay window): the ring is empty
    # at s = 0, so the run after frame s is s; the penalty at frame s reads the
    # run frame s - 1 left.
    for s in range(1, 10_000):
        run = s - 1
        pen = min(cap, rate * max(0, run - frames))
        if pen > gap:
            return s
    return None


@pytest.mark.parametrize("frames, rate", [(37, 0.5), (5, 1.0), (20, 0.25)])
def test_the_penalty_ramps_from_the_stated_frame_and_greedy_is_covered(
    load_sampler: Any, frames: int, rate: float
) -> None:
    S = load_sampler(f"{frames},{rate},20,8")
    cb0, _, _ = greedy_run(S, 200)
    steady = cb0[N:]
    s = expected_exit_step(frames, rate, 20.0, 5.0)
    assert steady[:s] == [7] * s
    assert steady[s] == 300


def test_the_penalty_is_capped_at_max(load_sampler: Any) -> None:
    S = load_sampler("5,1,2,8")
    cb0, state, after = greedy_run(S, 120)
    assert set(cb0[N:]) == {7}, "a cap below the gap must never force the exit"
    assert int(state.stall_run[0]) == 120 - N - 1
    lowered = silence_logits(1)[0, 0, 7] - after[-1][0, 0, 7]
    assert float(lowered) == pytest.approx(2.0)


def test_each_code_in_the_ring_is_lowered_once_and_nothing_else_moves(load_sampler: Any) -> None:
    S = load_sampler("3,1,10,4")
    state = S.HiggsBatchedSamplerState(1, N, device="cpu")
    rows = torch.arange(1)
    state.delay_count[0] = N
    state.stall_ring[0] = torch.tensor([7, 7, 9, 7])
    state.stall_run[0] = 6
    logits = torch.zeros(1, N, V)
    logits[0, :, 7] = 50.0
    before = logits.clone()
    S.batched_step(logits, state, rows, **sampling(1, 0.0))
    delta = before - logits
    assert float(delta[0, 0, 7]) == pytest.approx(3.0)
    assert float(delta[0, 0, 9]) == pytest.approx(3.0)
    delta[0, 0, 7] = 0
    delta[0, 0, 9] = 0
    assert torch.count_nonzero(delta) == 0, "only cb0's ring codes may move"


def test_a_loop_hopping_among_codes_is_one_run(load_sampler: Any) -> None:
    S = load_sampler("1000,1,10,4")
    state = S.HiggsBatchedSamplerState(1, N, device="cpu")
    rows = torch.arange(1)
    hops = [7, 8, 9]
    runs: list[int] = []
    for t in range(40):
        logits = torch.zeros(1, N, V)
        logits[0, 0, hops[t % 3]] = 50.0
        S.batched_step(logits, state, rows, **sampling(1, 0.0))
        runs.append(int(state.stall_run[0]))
    steady = runs[N:]
    assert steady[:3] == [0, 0, 0]
    assert steady[3:] == list(range(1, len(steady) - 2))


def test_a_new_code_outside_the_ring_ends_the_run(load_sampler: Any) -> None:
    S = load_sampler("1000,1,10,2")
    state = S.HiggsBatchedSamplerState(1, N, device="cpu")
    rows = torch.arange(1)
    sequence = [5] * 12 + [6, 5, 7, 8, 5]
    runs: list[int] = []
    for code in sequence:
        logits = torch.zeros(1, N, V)
        logits[0, 0, code] = 50.0
        S.batched_step(logits, state, rows, **sampling(1, 0.0))
        runs.append(int(state.stall_run[0]))
    # steady from index N: 5,5,5,5 -> 0,1,2,3; 6 -> 0 (not in [5,5]); 5 -> 1 (in [5,6]);
    # 7 -> 0; 8 -> 0; 5 -> 0 (the window of 2 holds [7, 8]).
    assert runs[N:] == [0, 1, 2, 3, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("where", ["delay", "winddown", "done"])
def test_rows_outside_the_counted_window_are_untouched_and_reset(
    load_sampler: Any, where: str
) -> None:
    S = load_sampler("3,1,10,4")
    state = S.HiggsBatchedSamplerState(2, N, device="cpu")
    rows = torch.arange(2)
    for row in (0, 1):
        state.delay_count[row] = N
        state.stall_ring[row] = torch.tensor([7, 7, 7, 7])
        state.stall_run[row] = 50
    if where == "delay":
        state.delay_count[1] = 2
    elif where == "winddown":
        state.eoc_countdown[1] = 3
    else:
        state.generation_done[1] = True
    logits = torch.zeros(2, N, V)
    logits[:, :, 7] = 50.0
    before = logits.clone()
    S.batched_step(logits, state, rows, **sampling(2, 0.0))
    assert torch.equal(logits[1], before[1]), f"a {where} row's logits moved"
    assert float((before - logits)[0, 0, 7]) == pytest.approx(10.0)
    assert int(state.stall_run[1]) == 0
    assert state.stall_ring[1].tolist() == [-1, -1, -1, -1]
    assert int(state.stall_run[0]) == 51


def test_reset_row_empties_the_guard(load_sampler: Any) -> None:
    S = load_sampler(DEFAULT_ENV)
    state = S.HiggsBatchedSamplerState(2, N, device="cpu")
    state.stall_run[1] = 9
    state.stall_ring[1] = 7
    state.reset_row(1)
    assert int(state.stall_run[1]) == 0
    assert state.stall_ring[1].tolist() == [-1] * 16


def test_the_graph_path_updates_its_shadow_buffers_in_place(load_sampler: Any) -> None:
    S = load_sampler("3,1,10,4")
    pool = 4
    run_buf = torch.zeros(pool, dtype=torch.long)
    ring_buf = torch.full((pool, S.STALL_RING_WIDTH), S.STALL_RING_EMPTY, dtype=torch.long)
    delay = torch.full((2,), N, dtype=torch.long)
    eoc = torch.full((2,), -1, dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    last = torch.zeros(2, N, dtype=torch.long)
    for _ in range(6):
        logits = torch.zeros(2, N, V)
        logits[:, :, 7] = 50.0
        S.batched_step_direct(
            logits, delay, eoc, done, last,
            seeds=torch.full((2,), -1, dtype=torch.long),
            step_count=torch.zeros(2, dtype=torch.long),
            stall_run=run_buf[:2], stall_ring=ring_buf[:2],
            **sampling(2, 0.0),
        )
    assert run_buf.tolist() == [5, 5, 0, 0]
    assert ring_buf[:2].tolist() == [[7, 7, 7, 7]] * 2
    assert ring_buf[2:].tolist() == [[-1] * 4] * 2


@pytest.mark.parametrize("guard", ["off", DEFAULT_ENV])
def test_a_render_that_never_stalls_draws_exactly_what_stock_draws(
    load_sampler: Any, guard: str
) -> None:
    stock_sampler = load_sampler(None, stock_file=True)
    patched = load_sampler(guard)
    draws: list[list[list[int]]] = []
    for S in (stock_sampler, patched):
        torch.manual_seed(1234)
        state = S.HiggsBatchedSamplerState(3, N, device="cpu")
        rows = torch.arange(3)
        out = []
        for _ in range(60):
            logits = torch.randn(3, N, V) * 3
            out.append(S.batched_step(logits, state, rows, **sampling(3, 0.8)).tolist())
        draws.append(out)
    assert draws[0] == draws[1]


def test_a_sampled_silence_is_left_soon_after_the_stated_frame(load_sampler: Any) -> None:
    for guard, should_leave in (("off", False), (DEFAULT_ENV, True)):
        S = load_sampler(guard)
        torch.manual_seed(0)
        state = S.HiggsBatchedSamplerState(1, N, device="cpu")
        rows = torch.arange(1)
        left = None
        for t in range(300):
            logits = torch.randn(1, N, V)
            logits[:, :, 7] += 12.0
            logits[:, 0, 300] += 6.0
            codes = S.batched_step(logits, state, rows, **sampling(1, 0.8))
            if t >= N and int(codes[0, 0]) != 7:
                left = t - N
                break
        if should_leave:
            assert left is not None and 37 < left < 37 + 40, left
        else:
            assert left is None


def test_the_patched_model_files_carry_the_shadow_state(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert run_script(env).returncode == 0
    after = texts(env)
    assert "self._cg_active_stall_ring = torch.full(" in after["model.py"]
    assert "stall_run=self._cg_active_stall_run[:batch_size]," in after["model.py"]
    assert "model._cg_active_stall_ring[:bs] = pool.stall_ring[rows_t]" in after["model_runner.py"]
    assert "pool.stall_ring[rows_t] = model._cg_active_stall_ring[:n_real]" in after[
        "model_runner.py"
    ]
