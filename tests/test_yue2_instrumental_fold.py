"""yue2-3b-instrumental's adapter is folded into the backbone only while YuE2 composes:
the score is planned on the base weights, then W += B @ A for every target linear through
the AR pass that writes the song's tokens and the synthesis that prefills the same
backbone, and afterwards every weight is the base again, exactly - also when composing
raises (docs/AUDIO.md "yue2-3b-instrumental").

torch is not in the server's env, so these drive the real worker code (lorafold,
HostHomes.folded, own_residency, YuE2Engine._stages) with stand-ins: numpy arrays for the
weights, a three-layer YuE2 with every target linear, and a pipeline that records the
weights each stage computes with. Integer-valued arrays keep the arithmetic exact, so a
restored weight either equals the base bit for bit or the test fails."""
from __future__ import annotations

import contextlib
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from crucible.jobs.audio import lorafold

from .test_audio_instrumental import PLANNING, SCORE, TAGS, _load_worker

LAYERS = 3
HIDDEN, KV, INTER, RANK = 4, 2, 6, 2
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SHAPES = {  # [out, in], YuE2's layout at toy size
    "self_attn.q_proj": (HIDDEN, HIDDEN), "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN), "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (INTER, HIDDEN), "mlp.up_proj": (INTER, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTER),
}
TIMING = {"output_tokens": 10, "execution": "cuda_graph", "attention": "sdpa", "prefix_tokens": 5,
          "cfg_branches": 1, "seconds": 1.0, "prefill_seconds": 0.1, "output_tps": 10.0}


# --- stand-ins: torch's few tensor and module methods the worker uses, over numpy -------


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
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def data_ptr(self) -> int:
        return id(self.array)

    def nbytes(self) -> int:
        return self.array.nbytes


class Tensor:
    dtype = "bfloat16"

    def __init__(self, array: np.ndarray, device: Device, storage: Storage | None = None) -> None:
        self.device = device
        self._storage = storage or Storage(array)

    @property
    def array(self) -> np.ndarray:
        return self._storage.array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def detach(self) -> "Tensor":
        return Tensor(self.array, self.device, self._storage)

    def untyped_storage(self) -> Storage:
        return self._storage

    def to(self, device: Any) -> "Tensor":
        target = device if isinstance(device, Device) else Device(device)
        return self if target == self.device else Tensor(self.array.copy(), target)

    def addmm_(self, mat1: "Tensor", mat2: "Tensor", *, alpha: float = 1.0) -> "Tensor":
        if not self.device == mat1.device == mat2.device:
            raise RuntimeError("Expected all tensors to be on the same device")
        self.array[...] += alpha * (mat1.array @ mat2.array)
        return self


class Parameter(Tensor):
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
    def __init__(self, name: str, weight: np.ndarray | None = None, **children: "Module") -> None:
        self.name = name
        self._parameters: dict[str, Any] = (
            {} if weight is None else {"weight": Parameter(weight, Device("cpu"))}
        )
        self._buffers: dict[str, Any] = {}
        self._children = children
        self._pre_hooks: list = []
        for child_name, child in children.items():
            setattr(self, child_name, child)

    @property
    def weight(self) -> Parameter:
        return self._parameters["weight"]

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
        for name, tensor in self._parameters.items():
            if tensor.device.type != "cuda":
                raise RuntimeError(f"{self.name}.{name} is on cpu while it computes")
        return None


class List(Module):
    def __init__(self, name: str, items: list) -> None:
        super().__init__(name, **{str(i): item for i, item in enumerate(items)})
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _values(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(seed).integers(-4, 5, size=shape).astype(np.float64)


def _layer(index: int) -> Module:
    def linear(path: str) -> Module:
        return Module(f"layers.{index}.{path}", _values(index * 10 + TARGETS.index(path), SHAPES[path]))

    def block(name: str) -> Module:
        return Module(f"layers.{index}.{name}", **{
            path.split(".")[1]: linear(path) for path in TARGETS if path.startswith(name + ".")
        })

    return Module(f"layers.{index}", **{
        "input_layernorm": Module("ln", np.ones(HIDDEN)),
        "self_attn": block("self_attn"),
        "post_attention_layernorm": Module("ln", np.ones(HIDDEN)),
        "mlp": block("mlp"),
        "nar_input_layernorm": Module("nar ln", np.ones(HIDDEN)),
        "nar_self_attn": Module("nar_self_attn", _values(index + 99, (HIDDEN, HIDDEN))),
        "nar_pre_mlp_layernorm": Module("nar ln", np.ones(HIDDEN)),
        "nar_mlp": Module("nar_mlp", _values(index + 199, (HIDDEN, INTER))),
    })


def _yue2() -> Module:
    backbone = Module(
        "model",
        embed_tokens=Module("embed_tokens", np.ones((8, HIDDEN))),
        layers=List("layers", [_layer(i) for i in range(LAYERS)]),
        norm=Module("norm", np.ones(HIDDEN)),
        rotary_emb=Module("rotary_emb"),
    )
    return Module(
        "YuE2ForCausalLM", model=backbone, lm_head=Module("lm_head", np.ones((8, HIDDEN))),
        llm2vae=Module("llm2vae", np.ones(2)), vae2llm=Module("vae2llm", np.ones(2)),
        time_embedder=Module("time_embedder", np.ones(2)),
        latent_pos_embed=Module("latent_pos_embed", np.ones(2)),
    )


def _adapter_tensors(layers: int = LAYERS) -> dict[str, Tensor]:
    tensors = {}
    for i in range(layers):
        for path in TARGETS:
            out, inner = SHAPES[path]
            seed = 1000 + i * 20 + TARGETS.index(path) * 2
            tensors[f"layers.{i}.{path}.lora_A"] = Tensor(_values(seed, (RANK, inner)), Device("cpu"))
            tensors[f"layers.{i}.{path}.lora_B"] = Tensor(_values(seed + 1, (out, RANK)), Device("cpu"))
    return tensors


IDENTITY = {"name": "instrumental-lora", "hf_repo": "o/r", "revision": "a" * 40,
            "file": "lora.safetensors", "sha256": "b" * 64, "bytes": 10, "rank": RANK, "scale": 1.0}


def _delta(scale: float = 1.0) -> lorafold.Delta:
    tensors = _adapter_tensors()
    targets = lorafold.targets_of({k: v.shape for k, v in tensors.items()},
                                  layers=LAYERS, names=TARGETS, rank=RANK)
    return lorafold.Delta(tensors, targets, layers=LAYERS, scale=scale, applies_to=("compose",),
                          identity={**IDENTITY, "scale": scale})


def _torch() -> Any:
    return SimpleNamespace(
        device=Device, bfloat16="bfloat16", no_grad=contextlib.nullcontext,
        cuda=SimpleNamespace(current_device=lambda: 0, empty_cache=lambda: None,
                             max_memory_reserved=lambda: 0, reset_peak_memory_stats=lambda: None),
    )


# --- the fold's arithmetic ----------------------------------------------------------------


def test_the_fold_adds_b_at_a_to_a_small_weight_and_nothing_else() -> None:
    weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Device("cuda", 0))
    a = Tensor(np.array([[1.0, 0.0, 2.0]]), Device("cpu"))       # [rank 1, in 3]
    b = Tensor(np.array([[3.0], [-1.0]]), Device("cpu"))         # [out 2, rank 1]
    delta = lorafold.Delta({}, (), layers=1, scale=1.0, applies_to=("compose",), identity={})
    delta.fold_into(weight, (a, b))
    assert weight.array.tolist() == [[4.0, 2.0, 9.0], [3.0, 5.0, 4.0]]
    half = lorafold.Delta({}, (), layers=1, scale=0.5, applies_to=("compose",), identity={})
    half.fold_into(weight, (a, b))
    assert weight.array.tolist() == [[5.5, 2.0, 12.0], [2.5, 5.0, 3.0]]
    assert a.device == Device("cpu"), "A and B stay in host memory; only a copy goes to the card"


def test_the_real_adapter_layout_reads_as_196_linears_at_rank_64() -> None:
    """ar_lora_inst_v3abc.bf16.safetensors' header (392 tensors), read from a local copy."""
    shapes = {}
    real = {"self_attn.q_proj": (2048, 2048), "self_attn.k_proj": (1024, 2048),
            "self_attn.v_proj": (1024, 2048), "self_attn.o_proj": (2048, 2048),
            "mlp.gate_proj": (6144, 2048), "mlp.up_proj": (6144, 2048), "mlp.down_proj": (2048, 6144)}
    for i in range(28):
        for path, (out, inner) in real.items():
            shapes[f"layers.{i}.{path}.lora_A"] = (64, inner)
            shapes[f"layers.{i}.{path}.lora_B"] = (out, 64)
    targets = lorafold.targets_of(shapes, layers=28, names=tuple(real), rank=64)
    assert len(shapes) == 392 and len(targets) == 196
    assert targets[0].module_path.startswith("model.layers.0.")
    assert {(t.out_features, t.in_features) for t in targets} == set(real.values())


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda s: s.pop("layers.2.mlp.down_proj.lora_A"), "only lora_B for layers.2.mlp.down_proj"),
        (lambda s: [s.pop(k) for k in list(s) if k.startswith("layers.1.self_attn.k_proj")],
         "changes no layers.1.self_attn.k_proj"),
        (lambda s: s.update({"layers.0.mlp.gate_proj.lora_A": (RANK + 1, HIDDEN)}), "rank 2"),
        (lambda s: s.update({"layers.3.mlp.gate_proj.lora_A": (RANK, HIDDEN)}), "layers 0 to 2"),
        (lambda s: s.update({"layers.0.nar_mlp.up_proj.lora_A": (RANK, HIDDEN)}), "outside what"),
        (lambda s: s.update({"vae2llm.weight": (2, 2)}), "not layers.<i>"),
    ],
)
def test_an_adapter_file_unlike_its_manifest_is_refused_by_name(change: Any, said: str) -> None:
    shapes = {k: v.shape for k, v in _adapter_tensors().items()}
    change(shapes)
    with pytest.raises(lorafold.AdapterError, match=said):
        lorafold.targets_of(shapes, layers=LAYERS, names=TARGETS, rank=RANK)


def test_binding_to_a_model_of_another_shape_is_refused_before_anything_changes() -> None:
    model = _yue2()
    model.model.layers._items[1].mlp.up_proj._parameters["weight"] = Parameter(
        np.zeros((INTER + 1, HIDDEN)), Device("cpu"))
    with pytest.raises(lorafold.AdapterError, match=r"layers.1.mlp.up_proj.weight is \[7, 4\]"):
        _delta().bind(model)
    short = _yue2()
    short.model.layers._items.pop()
    with pytest.raises(lorafold.AdapterError, match="has 2 layers and the adapter is for 3"):
        _delta().bind(short)


# --- the worker: plan on the base, compose and synthesize with the delta, base after -------


def _nar_module(seen: dict) -> types.ModuleType:
    """yue2/nar.py as own_residency wraps it: a prefill walking the embeddings and each
    layer (its input_layernorm first), and `_offload_ar` around the solve."""
    nar = types.ModuleType("yue2.nar")

    class CachedNAR:
        def __init__(self, model: Module) -> None:
            self.model = model
            self._prefill()

        def _prefill(self) -> None:
            backbone = self.model.model
            backbone.embed_tokens()
            for index, layer in enumerate(backbone.layers):
                layer.input_layernorm()
                seen["synthesizing"].append(_snapshot(self.model, layers=[index]))

    @contextlib.contextmanager
    def _offload_ar(model, enabled):
        yield

    nar.CachedNAR = CachedNAR
    nar._offload_ar = _offload_ar
    return nar


def _linear(model: Module, layer: int, path: str) -> Module:
    return lorafold.submodule(model, f"model.layers.{layer}.{path}")


def _snapshot(model: Module, layers: list[int] | None = None) -> dict:
    """Every target weight a stage computes with, which must be on the card to compute."""
    found = {}
    for i in range(LAYERS) if layers is None else layers:
        for path in TARGETS:
            weight = _linear(model, i, path).weight
            if weight.device.type != "cuda":
                raise RuntimeError(f"layers.{i}.{path} computed from host memory")
            found[(i, path)] = weight.array.copy()
    return found


class Pipe:
    """YuE2Pipeline as the worker drives it, recording the target weights each stage sees."""

    quantization = "none"
    backend = "torch"
    model_dir = "model"
    vae_dir = "vae"

    def __init__(self, nar: types.ModuleType, seen: dict) -> None:
        self.device = Device("cuda", 0)
        self.offload_ar = False
        self._model = None
        self._vae = None
        self._nar = nar
        self.seen = seen
        self.fail_composing = False
        self.generation_config = SimpleNamespace(abc=SimpleNamespace(max_tokens=4096),
                                                 semantic=SimpleNamespace(max_tokens=9000))

    def _load_model(self, for_nar=False):
        raise AssertionError("the worker replaces _load_model")

    def decode(self, latents, *, full=False, vae=None):
        return np.zeros((4, 2))

    def plan(self, tags, lyrics, **kwargs):
        self.seen["scoring"].append(_snapshot(self._load_model()))
        return SimpleNamespace(abc=kwargs.get("abc", SCORE), truncated=False, timing=TIMING)

    def generate_semantic(self, plan, *, cancelled, on_token):
        self.seen["composing"].append(_snapshot(self._load_model()))
        if self.fail_composing:
            raise RuntimeError("CUDA out of memory (stand-in, while composing)")
        return SimpleNamespace(timing=TIMING, truncated=False)

    def synthesize(self, semantic, *, cancelled):
        model = self._load_model(for_nar=True)
        engine = self._nar.CachedNAR(model)
        with self._nar._offload_ar(model, self.offload_ar):
            return engine


class Progress:
    asked_to_stop = False

    def __init__(self) -> None:
        self.stages: list[str] = []

    def enter(self, stage: str, steps=None) -> None:
        self.stages.append(stage)

    def note(self, text: str) -> None:
        ...


def _engine(monkeypatch: pytest.MonkeyPatch, low_vram: bool, delta: Any) -> Any:
    worker = _load_worker(monkeypatch)
    seen: dict = {"scoring": [], "composing": [], "synthesizing": []}
    nar = _nar_module(seen)
    model = _yue2()

    package = types.ModuleType("yue2")
    package.nar = nar
    monkeypatch.setitem(sys.modules, "yue2", package)
    monkeypatch.setitem(sys.modules, "yue2.nar", nar)
    vae = SimpleNamespace(from_pretrained=lambda *a, **k: Module("vae", np.ones(2)))
    monkeypatch.setitem(sys.modules, "yue2.modeling_vae", SimpleNamespace(YuE2VAE=vae))
    monkeypatch.setitem(sys.modules, "yue2.modeling_yue2", SimpleNamespace(
        YuE2ForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(eval=lambda: model))))
    engine = worker.YuE2Engine.__new__(worker.YuE2Engine)
    engine._torch = _torch()
    engine.device = "cuda"
    engine.low_vram = low_vram
    engine._pipe = Pipe(nar, seen)
    engine.host_homes, engine._homes = worker.own_residency(engine._pipe, engine._torch, low_vram)
    engine._delta = delta
    base = {(i, p): _linear(model, i, p).weight.array.copy() for i in range(LAYERS) for p in TARGETS}
    return SimpleNamespace(worker=worker, engine=engine, model=model, seen=seen, base=base)


def _job(tmp_path: Path) -> Any:
    return SimpleNamespace(
        tags=TAGS, lyrics=None, planning_lyrics=PLANNING, seed=7, cfg=1.0, instrumental=True,
        cover=None, min_duration_s=None, max_duration_s=None, longest_s=360,
        planning_resizable=False, output_path=str(tmp_path / "audio.flac"),
    )


def _folded(base: dict, delta: lorafold.Delta) -> dict:
    found = {}
    for target in delta.targets:
        a, b = (delta.tensors[target.a].array, delta.tensors[target.b].array)
        found[(target.layer, target.name)] = base[(target.layer, target.name)] + delta.scale * (b @ a)
    return found


def _same(seen: dict, wanted: dict) -> bool:
    return all(np.array_equal(value, wanted[key]) for key, value in seen.items())


def _restored(yue2: Any) -> None:
    """Every target weight is the base exactly, and so is every host home."""
    homes = yue2.engine._homes["model"]
    for (i, path), base in yue2.base.items():
        module = _linear(yue2.model, i, path)
        assert np.array_equal(module.weight.array, base), f"layers.{i}.{path} is not the base"
        assert np.array_equal(homes._home_of(module, "weight").array, base), "a home changed"
    assert homes._folding is None


@pytest.mark.parametrize("low_vram", [False, True])
def test_plan_sees_the_base_compose_and_synthesis_see_the_delta_and_the_base_comes_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, low_vram: bool
) -> None:
    yue2 = _engine(monkeypatch, low_vram, _delta())
    folded = _folded(yue2.base, yue2.engine._delta)
    for song in range(2):
        for stage in yue2.seen.values():
            stage.clear()
        score, audio = yue2.engine._stages(_job(tmp_path), Progress(), {})
        # The model plan and the instrumental's fixed re-plan, both on the base weights.
        assert len(yue2.seen["scoring"]) == 2
        assert all(_same(seen, yue2.base) for seen in yue2.seen["scoring"]), f"song {song + 1}"
        assert len(yue2.seen["composing"]) == 1 and _same(yue2.seen["composing"][0], folded)
        # Synthesis prefills the backbone layer by layer: under low_vram each layer arrives
        # fresh from its home and is folded as it arrives.
        assert len(yue2.seen["synthesizing"]) == LAYERS
        assert all(_same(seen, folded) for seen in yue2.seen["synthesizing"])
        _restored(yue2)
        assert yue2.engine.adapter == {**IDENTITY, "applied_to": ["compose"],
                                       "linears": LAYERS * len(TARGETS)}


@pytest.mark.parametrize("low_vram", [False, True])
def test_a_compose_that_raises_leaves_the_base_exactly_and_the_next_song_folds_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, low_vram: bool
) -> None:
    yue2 = _engine(monkeypatch, low_vram, _delta())
    yue2.engine._pipe.fail_composing = True
    with pytest.raises(RuntimeError, match="out of memory"):
        yue2.engine._stages(_job(tmp_path), Progress(), {})
    assert _same(yue2.seen["composing"][0], _folded(yue2.base, yue2.engine._delta))
    _restored(yue2)
    yue2.engine._pipe.fail_composing = False
    yue2.engine._stages(_job(tmp_path), Progress(), {})
    assert _same(yue2.seen["composing"][-1], _folded(yue2.base, yue2.engine._delta))
    _restored(yue2)


def test_a_model_without_an_adapter_composes_on_the_base_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yue2 = _engine(monkeypatch, False, None)
    yue2.engine._stages(_job(tmp_path), Progress(), {})
    for stage in ("scoring", "composing", "synthesizing"):
        assert all(_same(seen, yue2.base) for seen in yue2.seen[stage]), stage
    assert yue2.engine.adapter is None


def test_the_nar_half_and_everything_but_the_targets_never_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yue2 = _engine(monkeypatch, True, _delta())
    before = {name: tensor.array.copy() for name, tensor in yue2.model.named_parameters()}
    targets = {f"model.layers.{i}.{p}.weight" for i in range(LAYERS) for p in TARGETS}
    observed: dict = {}
    original = yue2.engine._pipe.generate_semantic

    def watch(plan, **kwargs):
        observed.update({n: t.array.copy() for n, t in yue2.model.named_parameters()})
        return original(plan, **kwargs)

    yue2.engine._pipe.generate_semantic = watch
    yue2.engine._stages(_job(tmp_path), Progress(), {})
    changed = {name for name, value in observed.items() if not np.array_equal(value, before[name])}
    assert changed == targets


def test_a_fold_never_reaches_a_host_home(monkeypatch: pytest.MonkeyPatch) -> None:
    yue2 = _engine(monkeypatch, False, _delta())
    yue2.engine._pipe._load_model()
    homes = yue2.engine._homes["model"]
    module = _linear(yue2.model, 0, "self_attn.q_proj")
    home = homes._home_of(module, "weight")
    homes._folding = (yue2.engine._delta, yue2.engine._delta.bind(yue2.model))
    with pytest.raises(RuntimeError, match="folded into a host home"):
        homes._fold_arrival(module, home, home)


# --- the load: the adapter file is checked before the worker is ready -----------------------


def _request(path: Path, sha256: str) -> dict:
    return {"adapter": {**IDENTITY, "sha256": sha256, "path": str(path), "layers": LAYERS,
                        "targets": list(TARGETS), "applies_to": ["compose"]}}


def test_the_adapter_is_rehashed_and_read_into_host_memory_at_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = _load_worker(monkeypatch)
    path = tmp_path / "lora.safetensors"
    path.write_bytes(b"adapter bytes")
    sha = hashlib.sha256(b"adapter bytes").hexdigest()
    tensors = _adapter_tensors()
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch",
                        SimpleNamespace(load_file=lambda p, device: tensors))
    delta = worker.load_adapter(_request(path, sha), _torch(), {"model": object()})
    assert len(delta.targets) == LAYERS * len(TARGETS) and delta.applies_to == ("compose",)
    assert delta.identity["sha256"] == sha
    with pytest.raises(RuntimeError, match="hashes to .* not the 0000"):
        worker.load_adapter(_request(path, "0" * 64), _torch(), {"model": object()})
    with pytest.raises(RuntimeError, match="through the CUDA worker's host homes"):
        worker.load_adapter(_request(path, sha), _torch(), None)
    assert worker.load_adapter({"adapter": None}, _torch(), None) is None
    with pytest.raises(RuntimeError, match="has no 'adapter'"):
        worker.load_adapter({}, _torch(), None)
    tensors["layers.0.mlp.up_proj.lora_A"].dtype = "float32"
    with pytest.raises(RuntimeError, match="is float32; YuE2 runs bfloat16"):
        worker.load_adapter(_request(path, sha), _torch(), {"model": object()})
