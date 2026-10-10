from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from crucible import hosttools, jobenv, workers
from crucible.backend import CUDA_LINUX
from crucible.cli import doctor, install
from crucible.engines.base import SubprocessEngine
from crucible.errors import EngineError

TOOLS_RELEASE = "https://github.com/telltaleatheist/crucible/releases/download/tools/"

ROOT = "zig-x86_64-linux-0.17.0"

# The real pinned archive, for the opt-in compile test below. Unit tests never
# download 57 MB; point this at the tarball to run that test (it is run in WSL).
REAL_ARCHIVE_ENV = "CRUCIBLE_TEST_ZIG_ARCHIVE"


def test_the_pin_is_on_our_tools_release_and_aims_at_zigs_own_glibc() -> None:
    assert sorted(hosttools.ZIG_BUILDS) == ["linux-x86_64"]
    build = hosttools.ZIG_BUILDS["linux-x86_64"]
    assert build.url == TOOLS_RELEASE + f"{ROOT}.tar.xz"
    assert build.root == ROOT
    assert build.sha256 == "1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026"
    assert build.bytes == 57_332_648
    assert build.target == "x86_64-linux-gnu.2.28"
    assert "MIT" in build.provenance


def test_mac_and_windows_have_no_compiler_pin() -> None:
    for key in ("darwin-arm64", "windows-x86_64", "windows-arm64"):
        assert hosttools.zig_build(key) is None


def _archive(root: str, *, with_zig: bool = True) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:xz") as bundle:
        members = {f"{root}/lib/libc/README": b"stubs\n"}
        if with_zig:
            members[f"{root}/zig"] = b"#!/bin/sh\necho fake zig \"$@\"\n"
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o755
            bundle.addfile(info, io.BytesIO(body))
    return raw.getvalue()


def _pinned(
    monkeypatch: pytest.MonkeyPatch, archive: bytes, *, sha256: str | None = None
) -> hosttools.CompilerBuild:
    build = hosttools.CompilerBuild(
        version="0.17.0",
        url=TOOLS_RELEASE + f"{ROOT}.tar.xz",
        sha256=sha256 or hashlib.sha256(archive).hexdigest(),
        bytes=len(archive),
        root=ROOT,
        provenance="a test archive",
        target="x86_64-linux-gnu.2.28",
    )
    monkeypatch.setattr(hosttools, "zig_build", lambda platform_key=None: build)
    return build


def _fetch_from(archive: bytes, calls: list[str] | None = None):
    def fetch(url: str, destination: Path) -> str:
        if calls is not None:
            calls.append(url)
        destination.write_bytes(archive)
        return hashlib.sha256(archive).hexdigest()

    return fetch


def _no_fetch(url: str, destination: Path) -> str:
    raise AssertionError(f"nothing should be fetched, but {url} was")


def test_the_pinned_zig_is_placed_whole_with_its_cc_and_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _pinned(monkeypatch, _archive(ROOT))
    said = hosttools.ensure_zig(tmp_path, fetch=_fetch_from(_archive(ROOT)))
    assert "0.17.0" in said and "placed" in said
    tree = tmp_path / "tools" / "zig" / "0.17.0"
    assert (tree / "zig").is_file() and (tree / "lib" / "libc" / "README").is_file()
    wrapper = tmp_path / "tools" / "bin" / "cc"
    text = wrapper.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert text == hosttools.c_compiler_wrapper(tmp_path, build)
    assert text.endswith(' cc -target x86_64-linux-gnu.2.28 -Wno-macro-redefined "$@"\n')
    assert " -w " not in text and "-Wno-everything" not in text, "only the one warning is off"
    assert str(tmp_path / "tools" / "zig" / "cache") in text
    stamp = json.loads(hosttools.zig_stamp(tmp_path).read_text())
    assert stamp["sha256"] == build.sha256 and stamp["target"] == build.target
    assert hosttools.zig_placed(tmp_path, build)
    leftovers = [p.name for p in (tmp_path / "tools").iterdir() if p.name.startswith(".zig-")]
    assert leftovers == [], "the staging directory is gone once the tree is placed"
    assert "already at" in hosttools.ensure_zig(tmp_path, fetch=_no_fetch)


def test_bytes_that_do_not_match_the_pin_place_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(ROOT)
    _pinned(monkeypatch, archive, sha256="0" * 64)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_zig(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_sha_mismatch"
    assert not (tmp_path / "tools" / "zig" / "0.17.0").exists()
    assert not (tmp_path / "tools" / "bin" / "cc").exists()
    assert not hosttools.zig_stamp(tmp_path).exists()


def test_an_archive_without_zig_is_refused_by_name_and_places_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(ROOT, with_zig=False)
    _pinned(monkeypatch, archive)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_zig(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_unpack_failed"
    assert f"{ROOT}/zig" in refused.value.message
    assert not (tmp_path / "tools" / "zig" / "0.17.0").exists()
    assert not (tmp_path / "tools" / "bin" / "cc").exists()


def test_a_wrong_cc_is_rewritten_without_fetching_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _pinned(monkeypatch, _archive(ROOT))
    hosttools.ensure_zig(tmp_path, fetch=_fetch_from(_archive(ROOT)))
    wrapper = tmp_path / "tools" / "bin" / "cc"
    wrapper.write_text("#!/bin/sh\nexec gcc \"$@\"\n", encoding="utf-8")
    assert not hosttools.zig_placed(tmp_path, build)
    said = hosttools.ensure_zig(tmp_path, fetch=_no_fetch)
    assert "rewritten" in said
    assert wrapper.read_text(encoding="utf-8") == hosttools.c_compiler_wrapper(tmp_path, build)


def test_a_tree_no_stamp_vouches_for_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _pinned(monkeypatch, _archive(ROOT))
    stale = tmp_path / "tools" / "zig" / "0.17.0"
    stale.mkdir(parents=True)
    (stale / "half-written").write_text("x")
    calls: list[str] = []
    hosttools.ensure_zig(tmp_path, fetch=_fetch_from(_archive(ROOT), calls))
    assert calls == [build.url]
    assert not (stale / "half-written").exists()
    assert (stale / "zig").is_file()


def test_no_pin_places_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hosttools, "zig_build", lambda platform_key=None: None)
    said = hosttools.ensure_zig(tmp_path, fetch=_no_fetch)
    assert said.startswith("c compiler: none is placed")
    assert not (tmp_path / "tools").exists()


def _env_with_triton(home: Path, key: str) -> Path:
    env = home / "envs" / key
    package = env / "lib" / "python3.12" / "site-packages" / "triton"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("")
    return env


def test_the_install_command_names_the_narrator_engine_for_a_tts_env() -> None:
    assert jobenv.install_command(jobenv.llm_env(CUDA_LINUX)) == "crucible install llm"
    assert (
        jobenv.install_command(jobenv.tts_env("higgs-v3", CUDA_LINUX))
        == "crucible install tts --narrator-engine higgs-v3"
    )
    assert jobenv.install_command(jobenv.worker_env("rvc", CUDA_LINUX)) == "crucible install rvc"


def test_cc_is_set_only_for_an_env_with_triton_on_a_pinned_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _pinned(monkeypatch, _archive(ROOT))
    plain = tmp_path / "envs" / "asr"
    (plain / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    assert hosttools.compiler_environment(plain, tmp_path) == {}
    llm = _env_with_triton(tmp_path, "llm")
    hosttools.ensure_zig(tmp_path, fetch=_fetch_from(_archive(ROOT)))
    assert hosttools.compiler_environment(llm, tmp_path) == {
        "CC": str(tmp_path / "tools" / "bin" / "cc")
    }
    assert hosttools.zig_placed(tmp_path, build)
    monkeypatch.setattr(hosttools, "zig_build", lambda platform_key=None: None)
    assert hosttools.compiler_environment(llm, tmp_path) == {}, "mlx-darwin / Windows"


def _unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(url: str, destination: Path) -> str:
        raise hosttools.HostToolError("tool_fetch_failed", f"{url} could not be reached")

    monkeypatch.setattr(hosttools, "_download", fetch)


def test_a_host_updated_without_an_install_places_the_compiler_on_first_need(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deploy runs no `crucible install`: the first engine or worker that needs the
    compiler places it, once."""
    archive = _archive(ROOT)
    build = _pinned(monkeypatch, archive)
    calls: list[str] = []
    monkeypatch.setattr(hosttools, "_download", _fetch_from(archive, calls))
    llm = _env_with_triton(tmp_path, "llm")
    assert hosttools.compiler_environment(llm, tmp_path) == {
        "CC": str(tmp_path / "tools" / "bin" / "cc")
    }
    assert hosttools.zig_placed(tmp_path, build)
    hosttools.compiler_environment(llm, tmp_path)
    assert len(calls) == 1, "placed once, then found"


def test_an_env_with_triton_and_no_compiler_is_refused_with_its_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a placement that fails is refused."""
    _pinned(monkeypatch, _archive(ROOT))
    _unreachable(monkeypatch)
    llm = _env_with_triton(tmp_path, "llm")
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.compiler_environment(llm, tmp_path)
    assert refused.value.code == "c_compiler_missing"
    assert "`crucible install llm`" in refused.value.message
    tts = _env_with_triton(tmp_path, "tts-higgs-v3")
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.compiler_environment(tts, tmp_path)
    assert "`crucible install tts --narrator-engine higgs-v3`" in refused.value.message


class _NeverSpawned(SubprocessEngine):
    name = "fake-vllm"

    def command(self, model_dir: Path, served_name: str, port: int, args: list[str]) -> list[str]:
        return [str(self._python), "-c", "pass"]


def test_an_engine_is_refused_before_it_launches(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pinned(monkeypatch, _archive(ROOT))
    _unreachable(monkeypatch)
    env = _env_with_triton(home, "llm")
    weights = home / "weights"
    weights.mkdir()

    def popen(*_a: object, **_k: object) -> None:
        raise AssertionError("the engine was launched without a compiler")

    monkeypatch.setattr(subprocess, "Popen", popen)
    engine = _NeverSpawned(python=env / "bin" / "python", log_path=home / "logs" / "e.log")
    with pytest.raises(EngineError) as refused:
        engine.start(weights, "m", 1, [])
    assert "c_compiler_missing" in str(refused.value)
    assert "crucible install llm" in str(refused.value)


def test_an_engine_runs_with_our_cc_over_an_inherited_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def compiler_environment(env_dir: Path, home: Path | None = None) -> dict[str, str]:
        seen.append(env_dir)
        return {"CC": "/crucible/tools/bin/cc"}

    monkeypatch.setattr(hosttools, "compiler_environment", compiler_environment)
    monkeypatch.setenv("CC", "gcc")
    captured: dict[str, str] = {}

    class _Process:
        pid = 1

        def poll(self) -> None:
            return None

    def popen(command: list[str], *, env: dict[str, str], **_k: object) -> _Process:
        captured.update(env)
        return _Process()

    monkeypatch.setattr(subprocess, "Popen", popen)
    python = tmp_path / "envs" / "llm" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    weights = tmp_path / "weights"
    weights.mkdir()
    engine = _NeverSpawned(python=python, log_path=tmp_path / "e.log")
    engine.start(weights, "m", 1, [])
    engine._close_log()
    assert seen == [tmp_path / "envs" / "llm"]
    assert captured["CC"] == "/crucible/tools/bin/cc"


def test_a_worker_runs_with_our_cc_over_an_inherited_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = Path(sys.executable)
    seen: list[Path] = []

    def compiler_environment(env_dir: Path, home: Path | None = None) -> dict[str, str]:
        seen.append(env_dir)
        return {"CC": "/crucible/tools/bin/cc"}

    monkeypatch.setattr(hosttools, "compiler_environment", compiler_environment)
    monkeypatch.setenv("CC", "gcc")
    script = tmp_path / "print_cc.py"
    script.write_text("import os, sys; sys.stdout.write(os.environ['CC'])\n")
    with (tmp_path / "w.log").open("ab") as log:
        process = workers._spawn(python, script, log, {"CC": "clang"})
        out, _ = process.communicate(timeout=60)
    assert seen == [python.parent.parent]
    assert out == "/crucible/tools/bin/cc"


def test_a_worker_with_triton_and_no_compiler_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compiler_environment(env_dir: Path, home: Path | None = None) -> dict[str, str]:
        raise hosttools.HostToolError("c_compiler_missing", "run `crucible install rvc`")

    monkeypatch.setattr(hosttools, "compiler_environment", compiler_environment)
    script = tmp_path / "w.py"
    script.write_text("")
    with (tmp_path / "w.log").open("ab") as log:
        with pytest.raises(workers.WorkerError) as refused:
            workers._spawn(Path(sys.executable), script, log, None)
    assert str(refused.value).startswith("c_compiler_missing: ")


def _tools_stubbed(monkeypatch: pytest.MonkeyPatch, placed: list[Path]) -> None:
    monkeypatch.setattr(hosttools, "ensure_ffmpeg", lambda home_dir, **_: "ffmpeg: ok")
    monkeypatch.setattr(hosttools, "ensure_silero_vad", lambda home_dir: "vad: ok")
    monkeypatch.setattr(
        hosttools, "ensure_zig", lambda home_dir, **_: placed.append(home_dir) or "cc: ok"
    )


class _Config:
    def __init__(self, home: Path) -> None:
        self.home = home


def test_install_places_the_compiler_when_any_env_here_runs_triton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pinned(monkeypatch, _archive(ROOT))
    placed: list[Path] = []
    _tools_stubbed(monkeypatch, placed)
    args = argparse.Namespace(verbose=False)
    assert install._ensure_tools(_Config(tmp_path), args) is None
    assert placed == [], "no env here holds Triton (an asr-only host)"
    _env_with_triton(tmp_path, "llm")
    assert install._ensure_tools(_Config(tmp_path), args) is None
    assert placed == [tmp_path]


def test_install_says_the_compiler_failed_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pinned(monkeypatch, _archive(ROOT))
    _tools_stubbed(monkeypatch, [])

    def refuse(home_dir: Path, **_: object) -> str:
        raise hosttools.HostToolError("tool_download_failed", "offline")

    monkeypatch.setattr(hosttools, "ensure_zig", refuse)
    _env_with_triton(tmp_path, "rvc")
    refusal = install._ensure_tools(_Config(tmp_path), argparse.Namespace(verbose=False))
    assert refusal is not None and "C compiler" in refusal and "tool_download_failed" in refusal


def test_install_on_an_unpinned_host_places_no_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "zig_build", lambda platform_key=None: None)
    placed: list[Path] = []
    _tools_stubbed(monkeypatch, placed)
    _env_with_triton(tmp_path, "llm")
    assert install._ensure_tools(_Config(tmp_path), argparse.Namespace(verbose=False)) is None
    assert placed == []


def test_doctor_names_the_missing_compiler_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _pinned(monkeypatch, _archive(ROOT))
    _env_with_triton(tmp_path, "llm")
    report = hosttools.c_compiler_report(tmp_path)
    assert report["placed"] is False and report["pinned_version"] == "0.17.0"
    findings = doctor._compiler_findings(report)
    assert [f.code for f in findings] == ["c_compiler_missing"]
    assert findings[0].fix == "crucible install llm"
    assert "NONE" in "".join(doctor.lines_c_compiler({"c_compiler": report}))
    hosttools.ensure_zig(tmp_path, fetch=_fetch_from(_archive(ROOT)))
    report = hosttools.c_compiler_report(tmp_path)
    assert report["placed"] and doctor._compiler_findings(report) == []
    assert build.version in "".join(doctor.lines_c_compiler({"c_compiler": report}))


@pytest.mark.skipif(
    hosttools.host_platform() != "linux-x86_64" or not os.environ.get(REAL_ARCHIVE_ENV),
    reason=f"compiles with the real pinned Zig: linux-x86_64 with ${REAL_ARCHIVE_ENV} only",
)
def test_the_real_cc_builds_a_shared_object_with_zigs_own_crt(tmp_path: Path) -> None:
    archive = Path(os.environ[REAL_ARCHIVE_ENV])

    def fetch(url: str, destination: Path) -> str:
        shutil.copyfile(archive, destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    hosttools.ensure_zig(tmp_path, fetch=fetch)
    source = tmp_path / "probe.c"
    source.write_text("int crucible_probe(int x) { return x + 1; }\n")
    shared = tmp_path / "probe.so"
    cc = hosttools.c_compiler_path(tmp_path)
    subprocess.run([str(cc), "-shared", "-fPIC", "-O2", str(source), "-o", str(shared)],
                   check=True, env={"PATH": "/usr/bin:/bin"})
    comment = subprocess.run(["readelf", "-p", ".comment", str(shared)],
                             check=True, capture_output=True, text=True).stdout
    assert "clang" in comment
    assert "GCC" not in comment, f"system crt leaked in: {comment}"


@pytest.mark.skipif(
    hosttools.host_platform() != "linux-x86_64" or not os.environ.get(REAL_ARCHIVE_ENV),
    reason=f"compiles with the real pinned Zig: linux-x86_64 with ${REAL_ARCHIVE_ENV} only",
)
def test_the_real_cc_is_quiet_about_pyconfigs_posix_level_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Triton's driver.c includes <dlfcn.h> before <Python.h>: zig's glibc defines
    _POSIX_C_SOURCE as 202405L and pyconfig.h defines it again as 200809L."""
    archive = Path(os.environ[REAL_ARCHIVE_ENV])

    def fetch(url: str, destination: Path) -> str:
        shutil.copyfile(archive, destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    hosttools.ensure_zig(tmp_path, fetch=fetch)
    cc = str(hosttools.c_compiler_path(tmp_path))
    launcher = tmp_path / "launcher.c"
    launcher.write_text(
        "#include <dlfcn.h>\n#define _POSIX_C_SOURCE 200809L\nint f(void) { return 0; }\n"
    )
    quiet = subprocess.run([cc, "-c", str(launcher), "-o", str(tmp_path / "l.o")],
                           check=True, capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin"})
    assert "macro redefined" not in quiet.stderr, quiet.stderr
    loud = tmp_path / "loud.c"
    loud.write_text("int g(void) { int u; return u; }\n")
    said = subprocess.run([cc, "-Wall", "-c", str(loud), "-o", str(tmp_path / "g.o")],
                          check=True, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})
    assert "-Wuninitialized" in said.stderr, "every other warning still prints"
