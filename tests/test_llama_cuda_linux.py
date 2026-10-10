"""llama-server on cuda-linux: the pinned binary from Crucible's tools release, the llm env's
CUDA libraries, the manifest rule that a GGUF block runs on it, and a load that starts it.

Owen, 2026-10-09: B-Sides only ever sends one request at a time, so its models are served on
cuda-linux by llama.cpp from GGUF instead of by vLLM, whose compile and graph capture took
minutes on every swap on an 8 GiB card."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

import pytest

from crucible import hosttools, jobenv, llamacpp, weights
from crucible.backend import Backend, Gpu
from crucible.config import Config
from crucible.engines import build_engine, engine_load_args
from crucible.engines.base import LIBRARY_PATH_ENV, library_path
from crucible.engines.llama_server import PAGES_ENGINE_FAILED, fatal_reason
from crucible.errors import ApiError
from crucible.jobs.llm import LoadModelJobType, llm_engine_status, model_rows
from crucible.manifests import (
    GGUF_ENGINE,
    ManifestError,
    block_engine,
    load_manifest,
    parse_manifest,
)
from crucible.residency import Residency
from tests.fake_hub import FakeHub

CUDA_LINUX = "cuda-linux"
GIB = 1024 ** 3
TOOLS_RELEASE = "https://github.com/telltaleatheist/crucible/releases/download/tools/"
BSIDE = "qwen3.5-4b-bside"


def _config(home: Path) -> Config:
    home.mkdir(parents=True, exist_ok=True)
    return Config(
        path=home / "config.toml",
        home=home,
        name="crucible@staged",
        host="127.0.0.1",
        port=7101,
        token="t",
        backend_kind=CUDA_LINUX,
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=1 * GIB,
        desktop_allowance_basis="stated",
        capability=None,
    )


def _backend(vram: int = 8 * GIB) -> Backend:
    return Backend(
        kind=CUDA_LINUX,
        platform="linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3070", vram_bytes=vram),
        detail="test",
    )


# ---- the pin -----------------------------------------------------------------------------


def test_the_linux_build_is_pinned_on_our_tools_release_with_a_full_digest() -> None:
    build = hosttools.LLAMA_SERVER_BUILDS["linux-x86_64"]
    assert build.url.startswith(TOOLS_RELEASE)
    assert build.url.endswith(".tar.xz")
    assert re.fullmatch(r"[0-9a-f]{64}", build.sha256)
    assert build.bytes > 1_000_000
    assert build.root == build.url.rsplit("/", 1)[-1].removesuffix(".tar.xz")
    assert build.version in build.root
    assert llamacpp.LLAMA_CPP_RELEASE in build.version, (
        "one llama.cpp across Crucible: the Linux build is the tag Windows pins"
    )
    assert "scripts/build-llama-server-linux.sh" in build.provenance


def test_the_build_script_builds_the_pinned_tag_against_the_env_s_cuda() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build-llama-server-linux.sh"
    ).read_text(encoding="utf-8")
    recipe = (
        Path(__file__).resolve().parents[1] / "crucible" / "envs" / "llm" / "cuda-linux.txt"
    ).read_text(encoding="utf-8")
    assert f"TAG={llamacpp.LLAMA_CPP_RELEASE}" in script
    for package in ("nvidia-cuda-runtime", "nvidia-cublas"):
        pinned = re.search(rf"^{package}==(\S+)$", recipe, re.M)
        assert pinned is not None, f"the llm env no longer pins {package}"
        assert f'"{package}=={pinned.group(1)}"' in script, (
            f"the binary is built against a different {package} than the env it loads from"
        )
    assert "86-real" in script, "the 3070 and the 3090 Ti are sm_86"


def _archive(root: str, body: bytes = b"\x7fELF the server") -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:xz") as bundle:
        info = tarfile.TarInfo(f"{root}/bin/llama-server")
        info.size = len(body)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(body))
    return raw.getvalue()


def _pinned(monkeypatch: pytest.MonkeyPatch, archive: bytes, *, sha256: str | None = None) -> hosttools.ToolBuild:
    build = hosttools.ToolBuild(
        version="b10970-cuda13.0",
        url=TOOLS_RELEASE + "llama-server-b10970-cuda13.0-test.tar.xz",
        sha256=sha256 or hashlib.sha256(archive).hexdigest(),
        bytes=len(archive),
        root="llama-server-b10970-cuda13.0-test",
        provenance="a test archive",
    )
    monkeypatch.setattr(hosttools, "llama_server_build", lambda platform_key=None: build)
    return build


def _fetch_from(archive: bytes):
    def fetch(url: str, destination: Path) -> str:
        destination.write_bytes(archive)
        return hashlib.sha256(archive).hexdigest()

    return fetch


def test_the_pinned_archive_is_placed_and_stamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive("llama-server-b10970-cuda13.0-test")
    build = _pinned(monkeypatch, archive)
    said = hosttools.ensure_llama_server(tmp_path, fetch=_fetch_from(archive))
    assert "placed at" in said
    placed = hosttools.llama_server_path(tmp_path)
    assert placed.read_bytes() == b"\x7fELF the server"
    stamp = json.loads(hosttools.llama_server_stamp(tmp_path).read_text())
    assert stamp["sha256"] == build.sha256
    assert hosttools.llama_server_placed(tmp_path, build)
    assert "already at" in hosttools.ensure_llama_server(tmp_path, fetch=_fetch_from(archive))


def test_the_install_says_the_download_and_the_unpacked_size_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Victoria's laptop, 2026-10-09: "fetching ... (107 MB)" then a 139.5 MB file on
    # disk. Both figures are right; the lines say which is which.
    body = bytes(range(256)) * 9_000
    archive = _archive("llama-server-b10970-cuda13.0-test", body)
    _pinned(monkeypatch, archive)
    lines: list[str] = []
    said = hosttools.ensure_llama_server(tmp_path, fetch=_fetch_from(archive), on_line=lines.append)
    assert lines == [
        f"fetching llama-server-b10970-cuda13.0-test.tar.xz "
        f"({len(archive) / 1e6:.0f} MB to download)"
    ]
    assert f"{len(body) / 1e6:.1f} MB unpacked" in said
    assert f"{len(archive) / 1e6:.1f} MB download" in said


def test_bytes_that_do_not_match_the_pin_place_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive("llama-server-b10970-cuda13.0-test")
    _pinned(monkeypatch, archive, sha256="0" * 64)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_llama_server(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_sha_mismatch"
    assert not hosttools.llama_server_path(tmp_path).exists()
    assert not hosttools.llama_server_stamp(tmp_path).exists()


def test_an_archive_without_the_server_is_refused_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive("some-other-root")
    _pinned(monkeypatch, archive)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_llama_server(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_unpack_failed"
    assert "bin/llama-server" in refused.value.message
    assert not list((tmp_path / hosttools.TOOLS_DIR_NAME / "bin").iterdir())


def test_a_platform_with_no_pinned_build_is_refused_not_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "host_platform", lambda: "linux-arm64")
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_llama_server(tmp_path)
    assert refused.value.code == "tool_unpinned"
    assert "linux-arm64" in refused.value.message


# ---- the engine: binary plus the env's CUDA libraries -------------------------------------


def _llm_env(home: Path, monkeypatch: pytest.MonkeyPatch, *, installed: bool = True,
             libraries: tuple[str, ...] = llamacpp.CUDA_LINUX_LIBRARIES) -> Path:
    spec = jobenv.llm_env(CUDA_LINUX)
    root = jobenv.env_dir(home, spec)
    lib = root / "lib" / "python3.11" / "site-packages" / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    for name in libraries:
        (lib / name).write_bytes(b"\x7fELF")
    python = jobenv.env_python(home, spec)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n")

    def status(where: Path, which: jobenv.EnvSpec, backend_kind: str) -> jobenv.EnvStatus:
        return jobenv.EnvStatus(
            installed=installed,
            path=jobenv.env_dir(where, which),
            detail="vllm 0.29.0" if installed else "no env at all",
            python_version="3.11.16",
            packages={},
        )

    monkeypatch.setattr(jobenv, "env_status", status)
    return lib


def _placed(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive("llama-server-b10970-cuda13.0-test")
    _pinned(monkeypatch, archive)
    hosttools.ensure_llama_server(home, fetch=_fetch_from(archive))


def test_the_engine_is_the_placed_binary_with_the_env_s_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = _llm_env(tmp_path, monkeypatch)
    _placed(tmp_path, monkeypatch)
    found = llamacpp.cuda_linux_engine(tmp_path)
    assert found.installed, found.detail
    assert found.executable == hosttools.llama_server_path(tmp_path)
    assert found.library_dirs == (lib,)


def test_no_env_means_no_engine_and_says_the_env_is_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_env(tmp_path, monkeypatch, installed=False)
    _placed(tmp_path, monkeypatch)
    found = llamacpp.cuda_linux_engine(tmp_path)
    assert not found.installed
    assert "llm env" in found.detail and "no env at all" in found.detail


def test_an_env_without_cublas_is_named_and_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_env(tmp_path, monkeypatch, libraries=("libcudart.so.13",))
    _placed(tmp_path, monkeypatch)
    found = llamacpp.cuda_linux_engine(tmp_path)
    assert not found.installed
    assert "libcublas.so.13" in found.detail and "--force" in found.detail


def test_an_env_with_no_binary_names_the_install_that_places_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_env(tmp_path, monkeypatch)
    _pinned(monkeypatch, _archive("x"))
    found = llamacpp.cuda_linux_engine(tmp_path)
    assert not found.installed
    assert "`crucible install llm`" in found.detail
    assert str(hosttools.llama_server_path(tmp_path)) in found.detail


def test_the_libraries_go_ahead_of_what_the_server_inherited() -> None:
    joined = library_path((Path("/env/nvidia/cu13/lib"),), "/usr/lib/wsl/lib")
    assert joined.split(os.pathsep) == [str(Path("/env/nvidia/cu13/lib")), "/usr/lib/wsl/lib"]
    assert library_path((Path("/a"),), "") == str(Path("/a"))


def test_an_engine_built_with_library_dirs_starts_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    class Spawned:
        pid = 4242

        def poll(self) -> int | None:
            return None

    def popen(command, env, **_):
        seen.update(env)
        return Spawned()

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setenv(LIBRARY_PATH_ENV, "/usr/lib/wsl/lib")
    binary = tmp_path / "llama-server"
    binary.write_text("")
    lib = tmp_path / "cu13" / "lib"
    engine = build_engine(GGUF_ENGINE, binary, tmp_path / "log", library_dirs=(lib,))
    engine.start(tmp_path, BSIDE, 51234, [])
    assert seen[LIBRARY_PATH_ENV].split(os.pathsep) == [str(lib), "/usr/lib/wsl/lib"]
    engine._process = None
    engine._close_log()


def test_a_missing_shared_library_ends_the_wait_with_its_name() -> None:
    line = (
        "llama-server: error while loading shared libraries: libcublas.so.13: "
        "cannot open shared object file: No such file or directory"
    )
    found = fatal_reason(line)
    assert found is not None and found[0] == PAGES_ENGINE_FAILED
    assert "llm env" in found[1]


# ---- the manifest rule ---------------------------------------------------------------------

DEMO = """
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text"]

[backends.{kind}]
engine = "{engine}"
hf_repo = "demo/Demo-1B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
{extra}
"""


def _demo(kind: str, engine: str, extra: str = ""):
    text = DEMO.format(kind=kind, engine=engine, extra=extra)
    return parse_manifest(text, Path("demo-1b.toml"), "demo-1b")


def test_a_gguf_block_on_cuda_linux_runs_on_llama_server() -> None:
    spec = _demo(CUDA_LINUX, "llama-server", 'file = "Demo-1B-Q8_0.gguf"').spec(CUDA_LINUX)
    assert spec.engine == GGUF_ENGINE
    assert spec.files == ("Demo-1B-Q8_0.gguf",)
    assert block_engine(CUDA_LINUX, ("text",), gguf=True) == GGUF_ENGINE
    assert block_engine(CUDA_LINUX, ("text",), gguf=False) == "vllm"


def test_a_gguf_named_for_vllm_is_refused_naming_the_weights_form() -> None:
    with pytest.raises(ManifestError) as refused:
        _demo(CUDA_LINUX, "vllm", 'file = "Demo-1B-Q8_0.gguf"')
    said = str(refused.value)
    assert "names a GGUF `file`" in said and "'llama-server'" in said


def test_llama_server_on_a_whole_repo_is_refused() -> None:
    with pytest.raises(ManifestError) as refused:
        _demo(CUDA_LINUX, "llama-server")
    assert "names no `file`" in str(refused.value)


def test_mlx_has_no_gguf_engine() -> None:
    with pytest.raises(ManifestError):
        _demo("mlx-darwin", "llama-server", 'file = "Demo-1B-Q8_0.gguf"')
    with pytest.raises(ManifestError) as refused:
        _demo("mlx-darwin", "mlx-lm", 'file = "Demo-1B-Q8_0.gguf"')
    assert "belong to a llama-windows block" in str(refused.value)


# ---- the B-Sides model ---------------------------------------------------------------------


def test_the_bside_model_is_gguf_on_llama_server_on_the_pc_and_mlx_on_the_mac() -> None:
    manifest = load_manifest(BSIDE)
    spec = manifest.spec(CUDA_LINUX)
    assert spec.engine == GGUF_ENGINE
    assert spec.file is not None and spec.file.endswith("Q8_0.gguf")
    assert spec.bits == 8
    assert spec.memory is not None and spec.memory.basis == "computed"
    assert manifest.spec("mlx-darwin").engine == "mlx-lm"
    assert manifest.defaults.thinking is False


def test_the_bside_spawn_line_offloads_every_layer_to_one_slot(tmp_path: Path) -> None:
    manifest = load_manifest(BSIDE)
    spec = manifest.spec(CUDA_LINUX)
    args = engine_load_args(
        manifest, spec, tmp_path, None, context=manifest.context_for(CUDA_LINUX)
    )
    assert args[:2] == ["-m", str(tmp_path / spec.file)]
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--n-gpu-layers") + 1] == "all"
    assert args[args.index("-c") + 1] == str(manifest.context_for(CUDA_LINUX))
    assert "--mmproj" not in args


def test_the_four_bit_vllm_form_is_gone() -> None:
    from crucible.manifests import load_all_manifests

    assert "qwen3.5-4b-bside-4bit" not in load_all_manifests()


# ---- a load ----------------------------------------------------------------------------------


def _pulled(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest(BSIDE)
    hub = FakeHub(chunks=1)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub.snapshot_download, raising=False)
    weights.pull(config, manifest, manifest.spec(CUDA_LINUX))


def test_a_bside_load_starts_the_binary_with_the_env_s_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "home")
    lib = _llm_env(config.home, monkeypatch)
    _placed(config.home, monkeypatch)
    _pulled(config, monkeypatch)
    backend = _backend()
    job_type = LoadModelJobType(config, backend, Residency(config))
    from crucible.jobs.llm import _require_loadable

    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)
    _, spec, (launch, _) = _require_loadable(config, backend, BSIDE)
    assert spec.engine == GGUF_ENGINE
    assert launch.executable == hosttools.llama_server_path(config.home)
    assert launch.library_dirs == (lib,)
    assert job_type.check(backend).ready


def test_a_bside_load_with_no_binary_is_env_missing_for_the_llm_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "home")
    _llm_env(config.home, monkeypatch)
    _pinned(monkeypatch, _archive("x"))
    _pulled(config, monkeypatch)
    backend = _backend()
    from crucible.jobs.llm import _require_loadable

    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)
    with pytest.raises(ApiError) as refused:
        _require_loadable(config, backend, BSIDE)
    assert refused.value.code == "env_missing"
    assert Path(refused.value.details["env"]).name == "llm", (
        "install-on-submit reads the env's name to know `crucible install llm` places it"
    )
    assert not llm_engine_status(config, backend).installed, (
        "the llm install is not whole without the engine its GGUF blocks run on"
    )
    assert llm_engine_status(config, backend, "vllm").installed
    rows = {row["id"]: row for row in model_rows(config, backend, Residency(config))}
    assert "llama-server" in rows[BSIDE]["reason"]
    assert "reason" not in rows["qwen3.5-4b-8bit"] or "llama" not in rows["qwen3.5-4b-8bit"]["reason"]


def test_a_platform_crucible_builds_no_server_for_keeps_its_llm_install_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "home")
    _llm_env(config.home, monkeypatch)
    monkeypatch.setattr(hosttools, "host_platform", lambda: "linux-arm64")
    backend = _backend()
    assert llm_engine_status(config, backend).installed, (
        "nothing is pinned to place here, so the vLLM env is the whole llm install"
    )
    gguf = llm_engine_status(config, backend, GGUF_ENGINE)
    assert not gguf.installed
    assert "no pinned build" in gguf.detail
