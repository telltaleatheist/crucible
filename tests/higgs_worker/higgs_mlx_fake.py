"""A FAKE MLX, just wide enough to run `_generate_delayed_rows_batch`'s loop.

WHY A FAKE AND NOT THE REAL THING. The batched generator's row bookkeeping -
which row retires at which step, when `on_retire` fires, that a retired row is
stacked exactly once - is the part that can be wrong in a way no listener would
diagnose, and it is pure control flow. The arithmetic around it is already
tested against no model at all (`test_higgs_mlx_batch`). Running it on real MLX
would put a Metal allocation inside a unit test, make the file un-runnable on
the two thirds of this project's machines that have no MLX, and still not prove
the thing MLX decides (whether a left-padded batch SOUNDS like a solo row -
Owen measures that on the Mac).

So the three imports the loop makes inside itself -

    import mlx.core as mx
    from mlx_audio.tts.models.higgs_audio_v3.generation import HiggsSamplerState
    from mlx_lm.models.cache import BatchKVCache

- are answered from `sys.modules` by the objects below, numpy underneath, and
put back EXACTLY as they were on the way out. That last part matters:
`test_engine_lazy_imports` asserts this package imports with no mlx present, and
a leaked fake would make that test pass for the wrong reason.
"""
import sys
import types
from contextlib import contextmanager

import numpy as np


class FakeSampler:
    """`HiggsSamplerState` as the loop uses it: a `generation_done` flag that
    goes true after that ROW's own number of steps."""

    def __init__(self, done_at: int):
        self.done_at = int(done_at)
        self.steps = 0
        self.generation_done = False

    def step(self) -> None:
        self.steps += 1
        if self.steps >= self.done_at:
            self.generation_done = True


class FakeModel:
    """Every model attribute the batch loop touches, and nothing else.

    One `layers` entry (the loop builds one `BatchKVCache` per layer and the
    count is not what is under test), a backbone that answers a hidden state of
    the right RANK, and a batch sampler that emits one 8-code row per active
    sampler and retires it on schedule.
    """

    HIDDEN = 4

    def __init__(self, samplers):
        self.layers = [object()]
        self._samplers = samplers

    def _build_prompt_embeddings(self, text, references):
        raise AssertionError(
            'FakeModel._build_prompt_embeddings: the fake batch stubs '
            '_mlx_prompts_for, so this should never be reached')

    def backbone(self, ids, cache=None, input_embeddings=None):
        rows = int(np.asarray(ids).shape[0])
        return np.zeros((rows, 1, self.HIDDEN), dtype=np.float32)

    def _audio_logits(self, hidden):
        return hidden

    def _step_batch_sampler(self, logits, samplers, temperature=None,
                            top_p=None, top_k=None):
        out = []
        for sampler in samplers:
            sampler.step()
            # A distinguishable but meaningless code row: the codes' VALUES are
            # the codec's business, and no codec runs here.
            out.append(np.full((8,), sampler.steps, dtype=np.int64))
        return out

    def _embed_audio_codes(self, codes_batch):
        rows = int(np.asarray(codes_batch).shape[0])
        return np.zeros((rows, self.HIDDEN), dtype=np.float32)


class FakeBatchKVCache:
    """`filter(keep)` is the only method the loop calls, and re-slicing a cache
    that holds nothing is a no-op. What the loop must get RIGHT - that `keep`
    is a position in the live active list rather than an original row index -
    is asserted by the retirement ORDER the tests read back."""

    def __init__(self, left_padding):
        self.left_padding = list(left_padding)
        self.filters = []

    def filter(self, keep):
        self.filters.append(np.asarray(keep).tolist())


def _fake_mx():
    """`mlx.core`, numpy underneath. Only the names the loop calls."""
    mx = types.ModuleType('mlx.core')
    mx.int32 = np.int32
    mx.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    mx.array = lambda values, dtype=None: np.array(values, dtype=dtype)
    mx.stack = lambda arrays, axis=0: np.stack(arrays, axis=axis)
    mx.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
    mx.pad = lambda a, widths: np.pad(a, widths)
    mx.eval = lambda *args: None
    mx.random = types.SimpleNamespace(seed=lambda value: None)
    return mx


@contextmanager
def fake_mlx_batch(engine, done_at, positions=700, hidden=FakeModel.HIDDEN):
    """Run `engine._generate_delayed_rows_batch` against the fakes.

    `done_at[i]` is how many steps row i takes before its sampler says
    `generation_done`. `_mlx_prompts_for` is stubbed too, because building a
    prompt is the model's job and this fake has no weights to build one from.
    """
    samplers = [FakeSampler(d) for d in done_at]
    remaining = list(samplers)
    model = FakeModel(samplers)

    generation = types.ModuleType(
        'mlx_audio.tts.models.higgs_audio_v3.generation')
    generation.HiggsSamplerState = (
        lambda num_codebooks=8: remaining.pop(0))
    cache_mod = types.ModuleType('mlx_lm.models.cache')
    cache_mod.BatchKVCache = FakeBatchKVCache

    mx = _fake_mx()
    mlx_pkg = types.ModuleType('mlx')
    mlx_pkg.core = mx
    injected = {
        'mlx': mlx_pkg,
        'mlx.core': mx,
        'mlx_audio.tts.models.higgs_audio_v3.generation': generation,
        'mlx_lm.models.cache': cache_mod,
    }
    saved = {name: sys.modules.get(name, KeyError) for name in injected}
    sys.modules.update(injected)

    had_model = hasattr(engine, '_model')
    had_sampling = hasattr(engine, '_sampling')
    old_model = getattr(engine, '_model', None)
    old_sampling = getattr(engine, '_sampling', None)
    old_prompts = getattr(engine, '_mlx_prompts_for', None)
    engine._model = model
    engine._sampling = {'temperature': 0.8, 'top_p': 0.9, 'top_k': 50}
    engine._mlx_prompts_for = lambda texts: [
        (np.zeros((1, positions, hidden), dtype=np.float32), positions)
        for _text in texts]
    try:
        yield model
    finally:
        # PUT BACK EXACTLY WHAT WAS THERE. A leaked fake `mlx` would make
        # test_engine_lazy_imports pass for the wrong reason.
        for name, value in saved.items():
            if value is KeyError:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if had_model:
            engine._model = old_model
        else:
            del engine._model
        if had_sampling:
            engine._sampling = old_sampling
        else:
            del engine._sampling
        if old_prompts is None:
            del engine._mlx_prompts_for
        else:
            engine._mlx_prompts_for = old_prompts
