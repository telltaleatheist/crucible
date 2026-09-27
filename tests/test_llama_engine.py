from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from crucible import accelerator, llamacpp, weights
from crucible.backend import Backend, Gpu
from crucible.config import Config
from crucible.engines import ENGINES, LlamaServerEngine, engine_model_name
from crucible.engines.llama_server import (
    GRACEFUL_STOP_SECONDS,
    PAGES_ENGINE_FAILED,
    PORT_IN_USE,
    fatal_reason,
)
from crucible.errors import ApiError
from crucible.jobs.llm import LoadModelJobType
from crucible.manifests import load_manifest
from crucible.residency import Residency
from crucible.weights import WeightsError

from tests.fake_hub import FakeHub

GIB = 1024 ** 3
LLAMA_WINDOWS = "llama-windows"


def _config(home: Path) -> Config:
    home.mkdir(parents=True, exist_ok=True)
    return Config(
        path=home / "config.toml",
        home=home,
        name="crucible@staged",
        host="127.0.0.1",
        port=7101,
        token="t",
        backend_kind=LLAMA_WINDOWS,
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=3 * GIB,
        desktop_allowance_basis="stated",
        capability=None,
    )


def _zip_bytes(names: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in names.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


class FakeRelease:

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.asked: list[str] = []

    def fetch(self, url: str, destination: Path, on_progress) -> None:
        self.asked.append(url)
        name = url.rsplit("/", 1)[-1]
        payload = self.payloads.get(name)
        if payload is None:
            raise AssertionError(f"the fake release was asked for {name!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        if on_progress is not None:
            on_progress(len(payload), len(payload), name)


def test_the_pin_is_one_constant_and_the_three_published_assets() -> None:
    assert llamacpp.LLAMA_CPP_RELEASE == "b10970"
    cuda = {asset.name: asset for asset in llamacpp.CUDA_ASSETS}
    assert set(cuda) == {
        "llama-b10970-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    }
    assert cuda["llama-b10970-bin-win-cuda-12.4-x64.zip"].bytes == 254_074_942
    assert cuda["llama-b10970-bin-win-cuda-12.4-x64.zip"].sha256 == (
        "78c878ae30622a9e4be09e3831066454668ca70398114f23bf74ac814e52dad8"
    )
    assert cuda["cudart-llama-bin-win-cuda-12.4-x64.zip"].bytes == 391_443_627
    assert cuda["cudart-llama-bin-win-cuda-12.4-x64.zip"].sha256 == (
        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    )
    (cpu,) = llamacpp.CPU_ASSETS
    assert cpu.name == "llama-b10970-bin-win-cpu-x64.zip"
    assert cpu.bytes == 18_428_751
    assert cpu.sha256 == (
        "2c6d6516c04e95caa080d8eb917743e71858c73985acbb6739ad61b14e68b298"
    )


def test_an_nvidia_machine_takes_two_assets_and_a_cardless_one_takes_the_cpu_build() -> None:
    assert llamacpp.build_for("nvidia") == llamacpp.CUDA_BUILD
    assert llamacpp.build_for("cpu") == llamacpp.CPU_BUILD
    assert len(llamacpp.assets_for(llamacpp.CUDA_BUILD)) == 2
    assert len(llamacpp.assets_for(llamacpp.CPU_BUILD)) == 1
    assert llamacpp.expected_bytes(llamacpp.CUDA_BUILD) == 254_074_942 + 391_443_627


def test_a_build_this_server_does_not_know_is_refused_by_name() -> None:
    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.assets_for("rocm")
    assert caught.value.code == "engine_download_failed"


def test_every_asset_url_is_the_pinned_tag_and_nothing_is_listed() -> None:
    for asset in (*llamacpp.CUDA_ASSETS, *llamacpp.CPU_ASSETS):
        assert asset.url.startswith(
            f"https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{llamacpp.LLAMA_CPP_RELEASE}/"
        )
    source = (Path(llamacpp.__file__)).read_text(encoding="utf-8")
    assert "releases/latest" not in source
    assert "per_page" not in source


def _stage(tmp_path: Path, monkeypatch, payloads: dict[str, bytes]) -> tuple[Config, FakeRelease]:
    config = _config(tmp_path / "home")
    release = FakeRelease(payloads)
    assets = tuple(
        llamacpp.Asset(
            name=name,
            bytes=len(payload),
            sha256=weights.sha256_of(_write(tmp_path / "hash" / name, payload)),
        )
        for name, payload in payloads.items()
    )
    monkeypatch.setattr(llamacpp, "CUDA_ASSETS", assets)
    monkeypatch.setattr(llamacpp, "CPU_ASSETS", assets)
    return config, release


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_a_pull_verifies_both_zips_and_unpacks_them_into_one_directory(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"}),
        "cudart.zip": _zip_bytes({"cudart64_12.dll": b"MZ the runtime"}),
    }
    config, release = _stage(tmp_path, monkeypatch, payloads)
    found = llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)

    assert found.path == llamacpp.engine_dir(config)
    assert llamacpp.server_path(config).is_file()
    assert (llamacpp.engine_dir(config) / "cudart64_12.dll").is_file()
    assert len(release.asked) == 2
    record = json.loads(llamacpp.stamp_path(config).read_text(encoding="utf-8"))
    assert record["tag"] == llamacpp.LLAMA_CPP_RELEASE
    assert record["build"] == llamacpp.CUDA_BUILD


def test_a_digest_that_does_not_match_is_engine_sha_mismatch_and_places_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    release.payloads["server.zip"] = _zip_bytes({"llama-server.exe": b"different"})

    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert caught.value.code == "engine_sha_mismatch"
    assert not llamacpp.engine_dir(config).exists()
    assert llamacpp.installed(config, llamacpp.CUDA_BUILD) is None


def test_the_cudart_half_failing_leaves_no_half_installed_engine(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"}),
        "cudart.zip": _zip_bytes({"cudart64_12.dll": b"MZ the runtime"}),
    }
    config, release = _stage(tmp_path, monkeypatch, payloads)
    release.payloads["cudart.zip"] = b"not a zip and not the digest either"

    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert caught.value.code == "engine_sha_mismatch"
    assert not llamacpp.engine_dir(config).exists()


def test_a_zip_that_reaches_out_of_its_own_directory_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"../escaped.exe": b"MZ"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert caught.value.code == "engine_download_failed"
    assert not (tmp_path / "home" / "engines" / "escaped.exe").exists()


def test_a_release_with_no_llama_server_in_it_is_refused_rather_than_stamped(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"readme.txt": b"nothing useful"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert caught.value.code == "engine_download_failed"
    assert "llama-server.exe" in caught.value.message


def test_a_pull_is_idempotent_and_force_replaces(tmp_path: Path, monkeypatch) -> None:
    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert len(release.asked) == 1
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    assert len(release.asked) == 1, "a second pull fetched nothing"
    llamacpp.pull(config, llamacpp.CUDA_BUILD, force=True, fetch=release.fetch)
    assert len(release.asked) == 2


def test_an_engine_stamped_for_another_build_is_not_installed(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    llamacpp.pull(config, llamacpp.CPU_BUILD, fetch=release.fetch)
    assert llamacpp.installed(config, llamacpp.CPU_BUILD) is not None
    assert llamacpp.installed(config, llamacpp.CUDA_BUILD) is None


def test_a_stamp_beside_an_emptied_directory_is_not_installed(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    llamacpp.server_path(config).unlink()
    assert llamacpp.installed(config, llamacpp.CUDA_BUILD) is None


def test_remove_deletes_the_directory_and_says_which(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)
    gone = llamacpp.remove(config)
    assert gone == llamacpp.engine_dir(config)
    assert not gone.exists()


def test_a_llama_windows_row_pulls_ONLY_its_named_files(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path / "home")
    manifest = load_manifest("qwen3.5-9b")
    spec = manifest.spec(LLAMA_WINDOWS)
    assert spec.files == ("Qwen3.5-9B-Q8_0.gguf",)

    hub = FakeHub(chunks=2)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", hub.snapshot_download, raising=False
    )
    found = weights.pull(config, manifest, spec)
    assert hub.allowed == [["Qwen3.5-9B-Q8_0.gguf"]]
    assert (found.path / "Qwen3.5-9B-Q8_0.gguf").is_file()
    record = json.loads((found.path / weights.STAMP_NAME).read_text("utf-8"))
    assert record["files"] == ["Qwen3.5-9B-Q8_0.gguf"]


def test_a_safetensors_backend_still_pulls_the_whole_repo(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path / "home")
    manifest = load_manifest("qwen3.5-9b")
    spec = manifest.spec("cuda-linux")
    assert spec.files == ()

    hub = FakeHub(chunks=2)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", hub.snapshot_download, raising=False
    )
    found = weights.pull(config, manifest, spec)
    assert hub.allowed == [None]
    assert (found.path / "model.safetensors").is_file()


def test_dots_ocr_with_the_text_tower_and_no_mmproj_is_NOT_installed(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path / "home")
    manifest = load_manifest("dots-ocr")
    spec = manifest.spec(LLAMA_WINDOWS)
    assert spec.files == ("dots.ocr-Q8_0.gguf", "mmproj-dots.ocr-Q8_0.gguf")

    hub = FakeHub(chunks=1)
    hub.absent = {"mmproj-dots.ocr-Q8_0.gguf"}
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", hub.snapshot_download, raising=False
    )
    with pytest.raises(WeightsError) as caught:
        weights.pull(config, manifest, spec)
    assert "mmproj-dots.ocr-Q8_0.gguf" in str(caught.value)
    assert weights.installed(config, manifest, spec) is None


def test_a_named_file_deleted_after_the_pull_makes_the_subject_not_installed(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path / "home")
    manifest = load_manifest("dots-ocr")
    spec = manifest.spec(LLAMA_WINDOWS)
    hub = FakeHub(chunks=1)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", hub.snapshot_download, raising=False
    )
    found = weights.pull(config, manifest, spec)
    assert weights.installed(config, manifest, spec) is not None

    (found.path / "mmproj-dots.ocr-Q8_0.gguf").unlink()
    assert weights.installed(config, manifest, spec) is None
    with pytest.raises(WeightsError) as caught:
        weights.require_installed(config, manifest, spec)
    assert "mmproj-dots.ocr-Q8_0.gguf" in str(caught.value)
    assert "now pins" not in str(caught.value)


def test_missing_files_is_empty_for_a_spec_that_names_none(tmp_path: Path) -> None:
    manifest = load_manifest("qwen3.5-9b")
    assert weights.missing_files(tmp_path, manifest.spec("cuda-linux")) == ()


def _windows_backend(vendor: str = "nvidia") -> Backend:
    return Backend(
        kind=LLAMA_WINDOWS,
        platform="windows",
        arch="AMD64",
        gpu=Gpu(vendor=vendor, name="RTX 3090 Ti" if vendor == "nvidia" else "cpu",
                vram_bytes=24 * GIB),
        detail="llama.cpp cuda build",
    )


def test_the_engine_is_a_catalog_subject_on_this_backend_and_only_this_one(
    tmp_path: Path,
) -> None:
    from crucible import catalog

    config = _config(tmp_path / "home")
    rows = catalog.subjects(config, _windows_backend())
    engines = [row for row in rows if row.kind == "engine"]
    assert [row.id for row in engines] == ["llama-cpp"]
    (engine,) = engines
    assert engine.job_type == "llm"
    assert engine.expected_bytes == llamacpp.expected_bytes(llamacpp.CUDA_BUILD)
    assert engine.source.startswith("github:ggml-org/llama.cpp@")
    assert engine.installed() is None

    linux = Backend(
        kind="cuda-linux",
        platform="linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="RTX 3090 Ti", vram_bytes=24 * GIB),
        detail="nvidia-smi",
    )
    assert not [row for row in catalog.subjects(config, linux) if row.kind == "engine"]


def test_engine_is_a_subject_kind_and_it_is_last(tmp_path: Path) -> None:
    from crucible import catalog

    assert "engine" in catalog.KINDS
    assert catalog.KINDS[-1] == "engine"
    assert catalog.declared_ids()["engine"] == ["llama-cpp"]


def test_the_catalog_subject_pulls_and_reports_installed(
    tmp_path: Path, monkeypatch
) -> None:
    from crucible import catalog

    payloads = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"})}
    config, release = _stage(tmp_path, monkeypatch, payloads)
    subject = catalog.find(config, _windows_backend(), "engine", "llama-cpp")
    assert subject is not None
    assert subject.installed() is None
    monkeypatch.setattr(llamacpp, "download", release.fetch)
    subject.pull(force=False, on_line=None, on_progress=None)
    assert subject.installed() is not None


def test_llama_server_is_in_the_engine_table_and_names_the_crucible_id() -> None:
    assert ENGINES["llama-server"] is LlamaServerEngine
    assert engine_model_name("llama-server", Path("/anywhere"), "dots-ocr") == "dots-ocr"


def test_the_spawn_line_is_facts_3_and_the_alias_decision(tmp_path: Path) -> None:
    manifest = load_manifest("dots-ocr")
    spec = manifest.spec(LLAMA_WINDOWS)
    weights_dir = tmp_path / "dots"
    args = Residency._engine_args(
        manifest, spec, weights_dir, None, context=manifest.context_for(LLAMA_WINDOWS)
    )
    assert args[:2] == ["-m", str(weights_dir / "dots.ocr-Q8_0.gguf")]
    assert "--mmproj" in args
    assert args[args.index("--mmproj") + 1] == str(
        weights_dir / "mmproj-dots.ocr-Q8_0.gguf"
    )
    assert args[args.index("-c") + 1] == "16384"
    assert "--parallel" in args and args[args.index("--parallel") + 1] == "1"

    engine = LlamaServerEngine(
        python=tmp_path / "llama-server.exe", log_path=tmp_path / "log"
    )
    command = engine.command(weights_dir, "dots-ocr", 51234, args)
    assert command[0].endswith("llama-server.exe")
    assert command[command.index("--alias") + 1] == "dots-ocr"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "51234"
    assert "8000" not in command


def test_a_text_model_gets_no_mmproj(tmp_path: Path) -> None:
    manifest = load_manifest("qwen3.5-9b")
    spec = manifest.spec(LLAMA_WINDOWS)
    args = Residency._engine_args(
        manifest, spec, tmp_path, None, context=manifest.context_for(LLAMA_WINDOWS)
    )
    assert "--mmproj" not in args
    assert args[args.index("-c") + 1] == str(manifest.context_for(LLAMA_WINDOWS))


def test_the_fatal_lines_end_the_wait_early_and_each_has_a_name() -> None:
    assert fatal_reason("llama.cpp: hello")is None
    oom = fatal_reason("ggml_cuda_host_malloc: CUDA error: out of memory")
    assert oom is not None and oom[0] == PAGES_ENGINE_FAILED
    dll = fatal_reason("cudart64_12.dll was not found")
    assert dll is not None and dll[0] == PAGES_ENGINE_FAILED
    gguf = fatal_reason("error loading model: unable to read tensors")
    assert gguf is not None and gguf[0] == PAGES_ENGINE_FAILED
    port = fatal_reason("bind: address already in use")
    assert port is not None and port[0] == PORT_IN_USE


def test_a_fatal_line_in_the_log_raises_before_the_probe(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(
        "llama_model_load: loading\nCUDA error: out of memory\n", encoding="utf-8"
    )
    engine = LlamaServerEngine(python=tmp_path / "llama-server.exe", log_path=log)
    engine._port = 51234
    engine._served_name = "dots-ocr"
    with pytest.raises(Exception) as caught:
        engine.announced_ready()
    assert PAGES_ENGINE_FAILED in str(caught.value)
    assert "out of memory" in str(caught.value)


def test_a_taken_port_is_port_in_use_and_never_an_invitation_to_adopt(
    tmp_path: Path,
) -> None:
    log = tmp_path / "engine.log"
    log.write_text("server: bind: address already in use\n", encoding="utf-8")
    engine = LlamaServerEngine(python=tmp_path / "llama-server.exe", log_path=log)
    engine._port = 51234
    engine._served_name = "dots-ocr"
    with pytest.raises(Exception) as caught:
        engine.announced_ready()
    said = str(caught.value)
    assert PORT_IN_USE in said
    assert "never adopts" in said


def test_the_stop_clock_is_thirty_seconds_and_the_deviation_is_written_down() -> None:
    assert GRACEFUL_STOP_SECONDS == 30.0
    source = Path(
        __import__("crucible.engines.llama_server", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "DEVIATION" in source
    assert "does not run inside WSL2" in source


def test_stopping_an_engine_that_never_started_does_nothing(tmp_path: Path) -> None:
    engine = LlamaServerEngine(
        python=tmp_path / "llama-server.exe", log_path=tmp_path / "log"
    )
    engine.stop()
    assert engine.pids == frozenset()


def _fake_card(monkeypatch, *, free: int, total: int, apps=()) -> None:
    monkeypatch.setattr(
        accelerator, "nvidia_smi_path", lambda: r"C:\Windows\System32\nvidia-smi.exe"
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free, total))


def test_the_guard_reads_a_llama_windows_card_instead_of_denying_the_backend(
    monkeypatch,
) -> None:
    _fake_card(monkeypatch, free=20 * GIB, total=24 * GIB)
    state = accelerator.guard(
        LLAMA_WINDOWS,
        model_id="dots-ocr",
        need_bytes=6 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )
    assert state.backend == LLAMA_WINDOWS
    assert state.free_bytes == 20 * GIB
    assert state.total_bytes == 24 * GIB
    assert "20.0 GiB free of 24.0 GiB" in state.detail


def test_the_refusal_for_a_backend_that_really_is_not_one_names_all_three() -> None:
    with pytest.raises(accelerator.ProbeError) as caught:
        accelerator.read_state("cuda-windows", 0)
    said = str(caught.value)
    assert "cuda-linux" in said and "mlx-darwin" in said and LLAMA_WINDOWS in said


def test_a_cardless_windows_host_measures_system_memory(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "nvidia_smi_path", lambda: None)
    monkeypatch.setattr(
        accelerator, "probe_system_memory", lambda: (18 * GIB, 32 * GIB)
    )
    state = accelerator.read_state(LLAMA_WINDOWS, 3 * GIB)
    assert (state.free_bytes, state.total_bytes) == (18 * GIB, 32 * GIB)
    assert state.compute_apps == ()
    assert "system memory" in state.detail and "CPU" in state.detail


def test_a_windows_driver_that_will_not_answer_is_unreadable_and_never_ram(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        accelerator, "nvidia_smi_path", lambda: r"C:\Windows\System32\nvidia-smi.exe"
    )

    def refuses() -> tuple[int, int]:
        raise accelerator.ProbeError("nvidia-smi exited 255: driver/library mismatch")

    monkeypatch.setattr(accelerator, "probe_vram", refuses)
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    called: list[str] = []
    monkeypatch.setattr(
        accelerator,
        "probe_system_memory",
        lambda: called.append("ram") or (18 * GIB, 32 * GIB),
    )
    with pytest.raises(ApiError) as caught:
        accelerator.guard(
            LLAMA_WINDOWS, model_id="dots-ocr", need_bytes=6 * GIB
        )
    assert caught.value.code == "accelerator_unreadable"
    assert "driver/library mismatch" in caught.value.message
    assert called == []


def test_a_full_windows_card_is_refused_for_the_room_and_not_for_the_company(
    monkeypatch,
) -> None:
    _fake_card(
        monkeypatch,
        free=2 * GIB,
        total=24 * GIB,
        apps=[accelerator.ComputeApp(pid=4242, name="python.exe", used_bytes=20 * GIB)],
    )
    with pytest.raises(ApiError) as caught:
        accelerator.guard(
            LLAMA_WINDOWS, model_id="dots-ocr", need_bytes=6 * GIB
        )
    assert caught.value.code == "insufficient_memory"
    assert "needs 6.0 GiB" in caught.value.message
    assert "2.0 GiB free of 24.0 GiB" in caught.value.message
    assert caught.value.details["processes"][0]["pid"] == 4242
    assert "4242" not in caught.value.message


def test_a_windows_card_with_room_beside_the_desktop_loads(monkeypatch) -> None:
    _fake_card(
        monkeypatch,
        free=20 * GIB,
        total=24 * GIB,
        apps=[
            accelerator.ComputeApp(
                pid=1460, name="[Insufficient Permissions]", used_bytes=None
            ),
            accelerator.ComputeApp(
                pid=11208, name=r"C:\WINDOWS\explorer.exe", used_bytes=None
            ),
        ],
    )
    state = accelerator.guard(
        LLAMA_WINDOWS,
        model_id="dots-ocr",
        need_bytes=6 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )
    assert len(state.compute_apps) == 2


def test_a_llama_server_of_ours_that_outlived_its_run_is_accelerator_busy(
    monkeypatch,
) -> None:
    _fake_card(
        monkeypatch,
        free=20 * GIB,
        total=24 * GIB,
        apps=[
            accelerator.ComputeApp(
                pid=11208, name=r"C:\WINDOWS\explorer.exe", used_bytes=None
            ),
            accelerator.ComputeApp(
                pid=31337,
                name=r"C:\Users\tellt\AppData\Local\Crucible\engine\llama-server.exe",
                used_bytes=6 * GIB,
            ),
        ],
    )
    with pytest.raises(ApiError) as caught:
        accelerator.guard(
            LLAMA_WINDOWS, model_id="dots-ocr", need_bytes=6 * GIB
        )
    assert caught.value.code == "accelerator_busy"
    assert "pid 31337" in caught.value.message
    assert "never evicts" in caught.value.message
    assert [row["pid"] for row in caught.value.details["processes"]] == [31337]


def test_a_load_preflight_passes_on_a_fake_llama_windows_accelerator(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "server.zip": _zip_bytes({"llama-server.exe": b"MZ the server"}),
        "cudart.zip": _zip_bytes({"cudart64_12.dll": b"MZ the runtime"}),
    }
    config, release = _stage(tmp_path, monkeypatch, payloads)
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=release.fetch)

    manifest = load_manifest("dots-ocr")
    hub = FakeHub(chunks=1)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", hub.snapshot_download, raising=False
    )
    weights.pull(config, manifest, manifest.spec(LLAMA_WINDOWS))

    _fake_card(monkeypatch, free=20 * GIB, total=24 * GIB)
    backend = _windows_backend()
    job_type = LoadModelJobType(config, backend, Residency(config))
    assert job_type.check(backend).ready
    job_type.preflight("dots-ocr", {})


def test_a_loads_context_is_llama_servers_c(tmp_path: Path) -> None:
    manifest = load_manifest("qwen3.8-27b-4bit")
    spec = manifest.spec(LLAMA_WINDOWS)
    args = Residency._engine_args(manifest, spec, tmp_path, None, context=65536)
    assert args.count("-c") == 1
    assert args[args.index("-c") + 1] == "65536"


def test_mlx_lm_is_handed_no_context_flag(tmp_path: Path) -> None:
    manifest = load_manifest("qwen3.8-27b-8bit")
    spec = manifest.spec("mlx-darwin")
    args = Residency._engine_args(manifest, spec, tmp_path, None, context=131072)
    assert "131072" not in args
    assert "--max-model-len" not in args and "-c" not in args
