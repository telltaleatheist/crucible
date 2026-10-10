"""A YuE2 worker that makes song after song holds the same host memory after each one
(Victoria's 3070 laptop, 2026-10-10: OOM-killed at 15.5 GB anonymous on track 12 of an
album). YuE2's backbone moves between the card and host memory several times a song, and
`Module.to("cpu")` allocated a fresh host copy on every move, which glibc's heap kept
once freed. yue2_worker.HostHomes keeps the one host copy the load made and points
every module back at it, so a move to host memory allocates nothing.

torch is not in the server's env, so these drive HostHomes with a stand-in that records
every host allocation; the real model was measured in the YuE2 env (docs/internals/audio.md,
"Host memory")."""
from __future__ import annotations

import contextlib
import itertools
from types import SimpleNamespace
from typing import Any

import pytest

from .test_audio_instrumental import _load_worker

_POINTERS = itertools.count(1)


class Device:
    def __init__(self, spec: Any, index: int | None = None) -> None:
        if isinstance(spec, Device):
            spec = str(spec)
        if index is not None:
            spec = f"{spec}:{index}"
        kind, _, index = str(spec).partition(":")
        self.type = kind
        self.index = int(index) if index else None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class Storage:
    def __init__(self, nbytes: int) -> None:
        self._pointer = next(_POINTERS)
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._nbytes


class Tensor:
    """Records every tensor made in host memory, so a test can say none was."""

    host_allocations: list["Tensor"] = []

    def __init__(self, device: Device, nbytes: int, storage: Storage | None = None) -> None:
        self.device = device
        self.nbytes = nbytes
        self._storage = storage or Storage(nbytes)
        if device.type == "cpu" and storage is None:
            Tensor.host_allocations.append(self)

    def detach(self) -> "Tensor":
        return Tensor(self.device, self.nbytes, self._storage)

    def untyped_storage(self) -> Storage:
        return self._storage

    def to(self, device: Any) -> "Tensor":
        return Tensor(Device(device), self.nbytes)


class Parameter(Tensor):
    def __init__(self, device: Device, nbytes: int) -> None:
        super().__init__(device, nbytes)
        self.data = self

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data" and value is not self:
            # nn.Parameter.data = x: the parameter now IS x's storage, on x's device.
            object.__setattr__(self, "device", value.device)
            object.__setattr__(self, "_storage", value.untyped_storage())
            return
        object.__setattr__(self, name, value)


class Module:
    def __init__(self, **children: "Module") -> None:
        self._parameters: dict[str, Any] = {}
        self._buffers: dict[str, Any] = {}
        self._children = children

    def modules(self):
        yield self
        for child in self._children.values():
            yield from child.modules()


def _torch() -> Any:
    return SimpleNamespace(
        device=Device,
        no_grad=contextlib.nullcontext,
        cuda=SimpleNamespace(current_device=lambda: 0),
    )


def _model() -> tuple[Module, Module, Module]:
    host = Device("cpu")
    ar, nar = Module(), Module()
    ar._parameters["weight"] = Parameter(host, 4_000)
    nar._parameters["weight"] = Parameter(host, 3_000)
    nar._buffers["pe"] = Tensor(host, 10)
    nar._parameters["bias"] = None
    return Module(ar=ar, nar=nar), ar, nar


@pytest.fixture
def homes(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load_worker(monkeypatch)
    Tensor.host_allocations = []
    root, ar, nar = _model()
    taken = module.HostHomes(root, _torch(), "the backbone")
    Tensor.host_allocations = []
    return SimpleNamespace(module=module, homes=taken, root=root, ar=ar, nar=nar)


def test_the_homes_hold_each_tensor_the_load_made_once(homes: Any) -> None:
    assert homes.homes.bytes == 4_000 + 3_000 + 10


def test_a_move_to_host_memory_points_back_at_the_home_and_allocates_nothing(homes: Any) -> None:
    weight = homes.ar._parameters["weight"]
    home_storage = weight.untyped_storage()
    for song in range(6):
        homes.homes.place([homes.ar], "cuda")
        assert weight.device == Device("cuda:0")
        assert weight.untyped_storage() is not home_storage
        homes.homes.place([homes.ar], "cpu")
        assert weight.device == Device("cpu")
        assert weight.untyped_storage() is home_storage, f"song {song + 1} made a new host copy"
    assert Tensor.host_allocations == []


def test_a_buffer_comes_home_too(homes: Any) -> None:
    home = homes.nar._buffers["pe"]
    homes.homes.place([homes.nar], "cuda")
    assert homes.nar._buffers["pe"].device == Device("cuda:0")
    homes.homes.place([homes.nar], "cpu")
    assert homes.nar._buffers["pe"].untyped_storage() is home.untyped_storage()
    assert Tensor.host_allocations == []


def test_a_part_already_where_it_is_sent_is_not_copied_again(homes: Any) -> None:
    homes.homes.place([homes.ar], "cuda")
    on_card = homes.ar._parameters["weight"].untyped_storage()
    homes.homes.place([homes.ar], "cuda:0")
    assert homes.ar._parameters["weight"].untyped_storage() is on_card


def test_the_pipelines_own_moves_go_through_the_homes(homes: Any) -> None:
    """yue2-infer's decode moves the backbone with `.to("cpu")` and the VAE with
    `.to(device)` and back: the routed `.to` places every part from its home."""
    assert homes.root.to(Device("cuda:0")) is homes.root
    assert homes.nar._parameters["weight"].device == Device("cuda:0")
    homes.root.to("cpu")
    assert homes.nar._parameters["weight"].device == Device("cpu")
    assert Tensor.host_allocations == []


def test_a_move_that_is_not_a_device_is_refused_by_name(homes: Any) -> None:
    with pytest.raises(RuntimeError, match="only between the card and its host home"):
        homes.root.to("cuda", "bfloat16")
    with pytest.raises(RuntimeError, match="only between the card and its host home"):
        homes.root.to(device="cuda")


def test_a_module_added_after_the_homes_were_taken_is_refused_by_name(homes: Any) -> None:
    stranger = Module()
    stranger._parameters["weight"] = Parameter(Device("cpu"), 1)
    with pytest.raises(RuntimeError, match="has no home to come back to"):
        homes.homes.place([stranger], "cuda")


def test_homes_are_taken_only_from_host_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_worker(monkeypatch)
    root = Module()
    root._parameters["weight"] = Parameter(Device("cuda:0"), 1)
    with pytest.raises(RuntimeError, match="is on cuda:0 as its host home is taken"):
        module.HostHomes(root, _torch(), "the backbone")
