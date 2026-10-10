"""Under `[audio] low_vram`, YuE2's synthesizing stage holds the same weights on the card
whatever the song's length (Victoria's 3070 laptop, 2026-10-10: a song composed to 8,960
tokens ran out of its 6.8 GiB cap 4 s into synthesizing).

yue2-infer synthesizes a song as one chunk up to about 10,000 frames, and each chunk's
prefill runs the AR half over the prefix plus every codec token and keeps every layer's
keys and values for the solve. It used to do that with the whole AR half, embeddings and
lm_head on the card, so the keys grew on top of 4.03 GiB of weights. The worker now brings
the AR half one layer at a time for the prefill and the NAR half for the solve
(yue2_worker.own_residency, ar_one_layer_at_a_time).

torch is not in the server's env, so these drive own_residency with stand-ins shaped like
YuE2 (28 layers, the real part sizes) and a stand-in of yue2-infer's synthesis loop
(yue2/nar.py at the pinned commit: CachedNAR's prefill and solve, `_offload_ar` around
each chunk's solve, `chunk_ranges`). A module used while its weights are in host memory
raises, as torch does. The real model was measured in the YuE2 env
(docs/internals/audio.md, "Synthesizing under low_vram")."""
from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from .test_audio_instrumental import _load_worker

MB = 1_000_000
LAYERS = 28
CONTEXT = 24576
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * 8 * 128 * 2  # keys and values, 8 KV heads of 128, bf16
PREFIX = 4_522  # Victoria's song: 13,483 prefill tokens = 4,522 + 8,960 codec + MUSIC_END

# Bytes of each part, the real model's (bfloat16).
SIZES = {
    "input_layernorm": 4_096, "post_attention_layernorm": 4_096,
    "nar_input_layernorm": 4_096, "nar_pre_mlp_layernorm": 4_096,
    "self_attn": 25_166_336, "nar_self_attn": 25_166_336,
    "mlp": 75_497_472, "nar_mlp": 75_497_472,
    "embed_tokens": 756_547_584, "lm_head": 756_547_584,
    "norm": 4_096, "rotary_emb": 0,
    "vae2llm": 266_240, "llm2vae": 262_272, "time_embedder": 9_441_280,
    "latent_pos_embed": 100_663_296,
}


class Device:
    def __init__(self, spec: Any, index: int | None = None) -> None:
        kind, _, number = str(spec).partition(":")
        self.type = kind
        self.index = index if index is not None else (int(number) if number else None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class Storage:
    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return id(self)

    def nbytes(self) -> int:
        return self._nbytes


class Tensor:
    def __init__(self, device: Device, nbytes: int, storage: Storage | None = None) -> None:
        self.device = device
        self.nbytes = nbytes
        self._storage = storage or Storage(nbytes)

    def detach(self) -> "Tensor":
        return Tensor(self.device, self.nbytes, self._storage)

    def untyped_storage(self) -> Storage:
        return self._storage

    def to(self, device: Any) -> "Tensor":
        return Tensor(Device(device), self.nbytes)


class Parameter(Tensor):
    def __init__(self, nbytes: int) -> None:
        super().__init__(Device("cpu"), nbytes)

    @property
    def data(self) -> "Parameter":
        return self

    @data.setter
    def data(self, value: Tensor) -> None:
        self.device = value.device
        self._storage = value.untyped_storage()


class Handle:
    def __init__(self, hooks: list, hook: Any) -> None:
        self._hooks, self._hook = hooks, hook

    def remove(self) -> None:
        self._hooks.remove(self._hook)


class Module:
    def __init__(self, name: str, nbytes: int = 0, **children: "Module") -> None:
        self.name = name
        self._parameters: dict[str, Any] = {"weight": Parameter(nbytes)} if nbytes else {}
        self._buffers: dict[str, Any] = {}
        self._children = children
        self._pre_hooks: list = []
        for child_name, child in children.items():
            setattr(self, child_name, child)

    def modules(self):
        yield self
        for child in self._children.values():
            yield from child.modules()

    def named_children(self):
        return list(self._children.items())

    def named_parameters(self, prefix: str = ""):
        for name, tensor in self._parameters.items():
            yield prefix + name, tensor
        for child_name, child in self._children.items():
            yield from child.named_parameters(f"{prefix}{child_name}.")

    def register_forward_pre_hook(self, hook: Any) -> Handle:
        self._pre_hooks.append(hook)
        return Handle(self._pre_hooks, hook)

    def __call__(self, *args: Any) -> Any:
        for hook in list(self._pre_hooks):
            hook(self, args)
        self.require_on_card()
        return args[0] if args else None

    def require_on_card(self) -> None:
        for name, tensor in self._parameters.items():
            if tensor.device.type != "cuda":
                raise RuntimeError(
                    f"Expected all tensors to be on the same device: {self.name}.{name} is on cpu"
                )


class List(Module):
    def __init__(self, name: str, items: list) -> None:
        super().__init__(name, **{str(i): item for i, item in enumerate(items)})
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _layer(index: int) -> Module:
    return Module(f"layers.{index}", **{
        name: Module(f"layers.{index}.{name}", SIZES[name])
        for name in ("input_layernorm", "self_attn", "nar_input_layernorm", "nar_self_attn",
                     "post_attention_layernorm", "mlp", "nar_pre_mlp_layernorm", "nar_mlp")
    })


def _yue2() -> Module:
    backbone = Module(
        "model",
        embed_tokens=Module("embed_tokens", SIZES["embed_tokens"]),
        layers=List("layers", [_layer(i) for i in range(LAYERS)]),
        norm=Module("norm", SIZES["norm"]),
        rotary_emb=Module("rotary_emb"),
    )
    return Module(
        "YuE2ForCausalLM",
        model=backbone,
        lm_head=Module("lm_head", SIZES["lm_head"]),
        llm2vae=Module("llm2vae", SIZES["llm2vae"]),
        vae2llm=Module("vae2llm", SIZES["vae2llm"]),
        time_embedder=Module("time_embedder", SIZES["time_embedder"]),
        latent_pos_embed=Module("latent_pos_embed", SIZES["latent_pos_embed"]),
    )


class Card:
    """What is on the card at each moment yue2-infer's loop would allocate: every
    parameter on cuda plus the prefix keys and values alive. Its high mark is the stage's
    peak, split into the weights and the keys at that moment."""

    def __init__(self) -> None:
        self.model: Module | None = None
        self.kv_bytes = 0
        self.peak = (0, 0)

    def weights(self) -> int:
        assert self.model is not None
        return sum(t.nbytes for _, t in self.model.named_parameters() if t.device.type == "cuda")

    def mark(self) -> None:
        weights = self.weights()
        if weights + self.kv_bytes > sum(self.peak):
            self.peak = (weights, self.kv_bytes)


def _nar_module(card: Card) -> types.ModuleType:
    """yue2/nar.py's synthesis loop at the pinned commit, its memory and order of calls."""
    nar = types.ModuleType("yue2.nar")

    def chunk_ranges(frames: int, prefix_tokens: int) -> list[tuple[int, int]]:
        size = min((CONTEXT - prefix_tokens - 3) // 2, CONTEXT)
        return [(a, min(a + size, frames)) for a in range(0, frames, size)]

    class CachedNAR:
        def __init__(self, model: Module, ar_length: int, nar_length: int) -> None:
            self.model, self.ar_length, self.nar_length = model, ar_length, nar_length
            model.vae2llm.require_on_card()
            model.latent_pos_embed.require_on_card()
            self.cache: list[int] = []
            self._prefill()

        def _prefill(self) -> None:
            backbone = self.model.model
            backbone.rotary_emb()
            backbone.embed_tokens(self.ar_length)
            for layer in backbone.layers:
                layer.self_attn(layer.input_layernorm(self.ar_length))
                kv = self.ar_length * KV_BYTES_PER_TOKEN_PER_LAYER
                self.cache.append(kv)
                card.kv_bytes += kv
                layer.mlp(layer.post_attention_layernorm(self.ar_length))
                card.mark()

        def velocity(self) -> None:
            model = self.model
            model.vae2llm()
            model.time_embedder()
            for layer in model.model.layers:
                layer.nar_self_attn(layer.nar_input_layernorm())
                layer.nar_mlp(layer.nar_pre_mlp_layernorm())
            model.model.norm()
            model.llm2vae()
            card.mark()

        def solve(self, steps: int = 2) -> None:
            for _ in range(steps):
                self.velocity()
                self.velocity()

        def close(self) -> None:
            card.kv_bytes -= sum(self.cache)
            self.cache.clear()

    @contextlib.contextmanager
    def _offload_ar(model, enabled):
        raise AssertionError("the worker replaces yue2-infer's _offload_ar under low_vram")
        yield

    def synthesize(model: Module, prefix_tokens: int, frames: int, offload_ar: bool) -> None:
        for a, b in chunk_ranges(frames, prefix_tokens):
            engine = nar.CachedNAR(model, prefix_tokens + (b - a) + 1, (b - a) + 2)
            with nar._offload_ar(model, offload_ar):
                try:
                    engine.solve()
                finally:
                    engine.close()

    nar.CachedNAR = CachedNAR
    nar._offload_ar = _offload_ar
    nar.synthesize = synthesize
    nar.chunk_ranges = chunk_ranges
    return nar


class Pipe:
    """YuE2Pipeline as own_residency reads and wraps it."""

    quantization = "none"
    backend = "torch"
    model_dir = "model"
    vae_dir = "vae"

    def __init__(self, nar: types.ModuleType) -> None:
        self.device = Device("cuda", 0)
        self.offload_ar = False
        self._model = None
        self._vae = None
        self._nar = nar

    def _load_model(self, for_nar=False):
        raise AssertionError("the worker replaces _load_model")

    def decode(self, latents, *, full=False, vae=None):
        return latents

    def synthesize(self, prefix_tokens: int, frames: int) -> None:
        model = self._load_model(for_nar=True)
        self._nar.synthesize(model, prefix_tokens, frames, self.offload_ar)


def _torch() -> Any:
    return SimpleNamespace(
        device=Device,
        bfloat16="bfloat16",
        no_grad=contextlib.nullcontext,
        cuda=SimpleNamespace(current_device=lambda: 0, empty_cache=lambda: None),
    )


@pytest.fixture
def yue2(monkeypatch: pytest.MonkeyPatch) -> Any:
    worker = _load_worker(monkeypatch)
    card = Card()
    nar = _nar_module(card)

    def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
        card.model = _yue2()
        return SimpleNamespace(eval=lambda: card.model)

    package = types.ModuleType("yue2")
    package.nar = nar
    monkeypatch.setitem(sys.modules, "yue2", package)
    monkeypatch.setitem(sys.modules, "yue2.nar", nar)
    monkeypatch.setitem(sys.modules, "yue2.modeling_vae", SimpleNamespace(YuE2VAE=None))
    monkeypatch.setitem(
        sys.modules, "yue2.modeling_yue2",
        SimpleNamespace(YuE2ForCausalLM=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    pipe = Pipe(nar)
    worker.own_residency(pipe, _torch(), low_vram=True)
    return SimpleNamespace(worker=worker, card=card, nar=nar, pipe=pipe)


def _synthesizing_peak(yue2: Any, frames: int, prefix: int = PREFIX) -> tuple[int, int]:
    yue2.pipe._load_model()  # the composing stage's layout, which synthesizing starts from
    yue2.card.peak = (0, 0)
    yue2.pipe.synthesize(prefix, frames)
    return yue2.card.peak


def _ar_layer_bytes() -> int:
    return sum(SIZES[n] for n in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"))


def _nar_half_bytes() -> int:
    return LAYERS * sum(SIZES[n] for n in ("nar_input_layernorm", "nar_self_attn",
                                           "nar_pre_mlp_layernorm", "nar_mlp"))


def _adapter_bytes() -> int:
    return sum(SIZES[n] for n in ("norm", "vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"))


def test_the_weights_on_the_card_while_synthesizing_do_not_grow_with_the_song(yue2: Any) -> None:
    peaks = {frames: _synthesizing_peak(yue2, frames) for frames in (1_000, 4_000, 6_000, 8_960, 9_000)}
    weights = {frames: peak[0] for frames, peak in peaks.items()}
    assert set(weights.values()) == {_nar_half_bytes() + _adapter_bytes()}, weights
    for frames, (_, kv) in peaks.items():
        # What grows is the prefix keys and values the solve attends to, and only that.
        assert kv == LAYERS * (PREFIX + frames + 1) * KV_BYTES_PER_TOKEN_PER_LAYER


def test_victorias_song_now_peaks_below_the_whole_ar_half_it_used_to_carry(yue2: Any) -> None:
    """The incident's song: 8,960 codec tokens after a 4,522-token prefix. Before, the
    prefill's keys (1.44 GiB at the end) sat beside the AR half, embeddings and lm_head
    (4.33 GB) and the 0.11 GB of adapters; now the solve's NAR half (2.82 GB) is the
    largest set of weights beside them."""
    weights, kv = _synthesizing_peak(yue2, 8_960)
    ar_whole = LAYERS * _ar_layer_bytes() + SIZES["embed_tokens"] + SIZES["lm_head"]
    assert kv == LAYERS * 13_483 * KV_BYTES_PER_TOKEN_PER_LAYER
    assert weights + kv < ar_whole + _adapter_bytes() + kv - 1_400 * MB


def test_synthesizing_stays_below_the_composing_stage_at_any_length(yue2: Any) -> None:
    """The composing stage allocates its keys and values for the whole token budget
    (prefix + 9,000) before it writes a token, with the AR half on the card; that is fixed
    by the prompt, not by how long the song came out. Synthesizing must stay under it at
    every length the composing stage can produce."""
    ar_whole = LAYERS * _ar_layer_bytes() + SIZES["embed_tokens"] + SIZES["lm_head"]
    composing = ar_whole + _adapter_bytes() + LAYERS * (PREFIX + 9_000) * KV_BYTES_PER_TOKEN_PER_LAYER
    for frames in (200, 3_000, 6_000, 9_000):
        assert sum(_synthesizing_peak(yue2, frames)) < composing


def test_the_prefill_holds_one_ar_layer_at_a_time(yue2: Any) -> None:
    seen: list[int] = []
    prefill = yue2.nar.CachedNAR._prefill

    def watched(engine: Any) -> None:
        for layer in engine.model.model.layers:
            layer.mlp.register_forward_pre_hook(lambda *_: seen.append(yue2.card.weights()))
        prefill(engine)

    yue2.nar.CachedNAR._prefill = watched
    _synthesizing_peak(yue2, 9_000)
    assert seen and set(seen) == {_ar_layer_bytes() + _adapter_bytes()}


def test_after_synthesizing_both_halves_are_home_and_the_next_stage_brings_ar_back(yue2: Any) -> None:
    _synthesizing_peak(yue2, 3_000)
    assert yue2.card.weights() == _adapter_bytes()
    model = yue2.pipe._load_model()
    assert yue2.card.weights() == _adapter_bytes() + LAYERS * _ar_layer_bytes() + SIZES["embed_tokens"] + SIZES["lm_head"]
    # And no streaming hook is left behind to move layers during the next composition.
    assert all(not m._pre_hooks for m in model.modules())


def test_a_long_song_in_two_chunks_streams_each_chunks_prefill(yue2: Any) -> None:
    """A long prefix leaves chunks shorter than the song (`chunk_ranges`), so the AR half
    is streamed once per chunk and the weights stay the same."""
    weights, _ = _synthesizing_peak(yue2, 9_000, prefix=10_000)
    assert len(yue2.nar.chunk_ranges(9_000, 10_000)) == 2
    assert weights == _nar_half_bytes() + _adapter_bytes()


def test_a_prefill_that_skips_a_layer_is_refused_by_name(yue2: Any) -> None:
    def skipping(self: Any) -> None:
        backbone = self.model.model
        backbone.embed_tokens()
        for layer in list(backbone.layers)[:-1]:
            layer.input_layernorm()

    yue2.nar.CachedNAR._prefill = skipping
    yue2.worker.own_residency(yue2.pipe, _torch(), low_vram=True)
    with pytest.raises(RuntimeError, match="streams YuE2's AR half in the order embed_tokens"):
        yue2.pipe.synthesize(PREFIX, 1_000)


def test_a_yue2_infer_without_the_prefill_it_streams_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_worker(monkeypatch)
    nar = _nar_module(Card())
    del nar.CachedNAR._prefill
    monkeypatch.setitem(sys.modules, "yue2", types.ModuleType("yue2"))
    monkeypatch.setitem(sys.modules, "yue2.nar", nar)
    monkeypatch.setitem(sys.modules, "yue2.modeling_vae", SimpleNamespace(YuE2VAE=None))
    monkeypatch.setitem(sys.modules, "yue2.modeling_yue2", SimpleNamespace(YuE2ForCausalLM=None))
    with pytest.raises(RuntimeError, match=r"nar.CachedNAR._prefill\(self\)"):
        worker.own_residency(Pipe(nar), _torch(), low_vram=True)
