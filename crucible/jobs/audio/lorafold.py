"""A low-rank adapter folded into a model's linears in memory: W += scale * B @ A for every
target linear, never on disk (docs/AUDIO.md "yue2-3b-instrumental").

Pure: no torch import. The worker hands it torch's tensors, and the tests hand it stand-ins
with the same few methods (`shape`, `.to`, `addmm_`), so the fold's arithmetic and its
refusals are checked without the YuE2 env. Taking the delta out again is the host homes'
job (yue2_worker.HostHomes.folded): the base is restored from the copy it was never
folded into, not by subtracting, so the round trip is exact."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

# An adapter tensor's name: `layers.<i>.<block>.<linear>.lora_A` or `lora_B`, as
# ar_lora_inst_v3abc names its 392 (28 layers x 7 linears x A and B).
KEY = re.compile(r"^layers\.(\d+)\.([a-z_]+\.[a-z_]+)\.lora_([AB])$")

HASH_CHUNK = 8 * 1024 * 1024


class AdapterError(RuntimeError):
    ...


@dataclass(frozen=True)
class Target:
    """One linear the adapter changes: its layer, its name inside the layer, the keys of
    its A ([rank, in]) and B ([out, rank]), and the weight shape they make ([out, in])."""

    layer: int
    name: str
    a: str
    b: str
    out_features: int
    in_features: int

    @property
    def module_path(self) -> str:
        return f"model.layers.{self.layer}.{self.name}"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(shapes: dict[str, tuple[int, ...]], layers: int, names: tuple[str, ...]) -> dict:
    pairs: dict[tuple[int, str], dict[str, tuple[str, tuple[int, ...]]]] = {}
    for key, shape in sorted(shapes.items()):
        found = KEY.match(key)
        if found is None:
            raise AdapterError(
                f"the adapter holds {key!r}, which is not layers.<i>.<block>.<linear>.lora_A "
                "or lora_B; nothing else is folded, so nothing else may be in the file"
            )
        layer, name, half = int(found[1]), found[2], found[3]
        if name not in names or not 0 <= layer < layers:
            raise AdapterError(
                f"the adapter holds {key!r}, outside what its manifest declares: layers 0 to "
                f"{layers - 1} and the linears {list(names)}"
            )
        if len(shape) != 2:
            raise AdapterError(f"the adapter's {key!r} has shape {list(shape)}; a lora half is 2-D")
        pairs.setdefault((layer, name), {})[half] = (key, tuple(shape))
    return pairs


def targets_of(
    shapes: dict[str, tuple[int, ...]], *, layers: int, names: tuple[str, ...], rank: int
) -> tuple[Target, ...]:
    """Every linear the adapter file changes, read from its tensors' names and shapes and
    held to the manifest: exactly `layers` x `names`, each with both halves at `rank`.
    Refused by name before anything is folded."""
    pairs = _pairs(shapes, layers, names)
    missing = sorted(
        f"layers.{i}.{name}" for i in range(layers) for name in names if (i, name) not in pairs
    )
    if missing:
        raise AdapterError(
            f"the adapter changes no {missing[0]} ({len(missing)} linear(s) missing); its "
            f"manifest declares all {layers} layers x {list(names)}"
        )
    found = []
    for (layer, name), halves in sorted(pairs.items()):
        if set(halves) != {"A", "B"}:
            raise AdapterError(f"the adapter has only lora_{''.join(halves)} for layers.{layer}.{name}")
        (a_key, a_shape), (b_key, b_shape) = halves["A"], halves["B"]
        if a_shape[0] != rank or b_shape[1] != rank:
            raise AdapterError(
                f"the adapter's layers.{layer}.{name} is A {list(a_shape)}, B {list(b_shape)}; "
                f"its manifest declares rank {rank} (A [rank, in], B [out, rank])"
            )
        found.append(Target(layer, name, a_key, b_key, b_shape[0], a_shape[1]))
    return tuple(found)


def submodule(root: Any, path: str) -> Any:
    found = root
    for part in path.split("."):
        found = getattr(found, part, None)
        if found is None:
            raise AdapterError(f"the model has no {path!r} (no {part!r}) for the adapter to fold into")
    return found


class Delta:
    """An adapter in host memory: its A and B for every target, the scale, and the stages
    it is folded around. `bind` finds each target's weight in a loaded model and checks
    its shape; `fold_into` adds the delta to one copy of a weight in place."""

    def __init__(self, tensors: dict[str, Any], targets: tuple[Target, ...], *, layers: int,
                 scale: float, applies_to: tuple[str, ...], identity: dict[str, Any]) -> None:
        self.tensors = tensors
        self.targets = targets
        self.layers = layers
        self.scale = scale
        self.applies_to = applies_to
        self.identity = identity

    def bind(self, model: Any) -> dict[Any, tuple[Any, Any]]:
        """Each target linear of `model` (YuE2ForCausalLM: `model.layers.<i>.<name>`) to
        its (A, B), every shape checked first, so a mismatch changes nothing."""
        layers = len(submodule(model, "model.layers"))
        if layers != self.layers:
            raise AdapterError(
                f"the model has {layers} layers and the adapter is for {self.layers}; it is "
                "another model's adapter"
            )
        bound = {}
        for target in self.targets:
            module = submodule(model, target.module_path)
            shape = tuple(module.weight.shape)
            if shape != (target.out_features, target.in_features):
                raise AdapterError(
                    f"{target.module_path}.weight is {list(shape)} and the adapter's delta for it "
                    f"is [{target.out_features}, {target.in_features}]; it is another model's adapter"
                )
            bound[module] = (self.tensors[target.a], self.tensors[target.b])
        return bound

    def fold_into(self, weight: Any, pair: tuple[Any, Any]) -> None:
        """weight += scale * B @ A, in place on this copy of the weight, in its own dtype
        on its own device: one bfloat16 GEMM that accumulates in float32 and rounds once,
        with no float32 copy of the weight beside it. A and B come to the weight's device
        for the call (at most 0.8 MB each for YuE2) and go when it returns."""
        a, b = pair
        weight.addmm_(b.to(weight.device), a.to(weight.device), alpha=self.scale)

    def record(self, applied_to: list[str], linears: int) -> dict[str, Any]:
        return {**self.identity, "applied_to": list(applied_to), "linears": linears}


__all__ = ["AdapterError", "Delta", "Target", "file_sha256", "submodule", "targets_of"]
