"""Environment packs: the manifest, the parts, the rename, and every refusal.

PHASE14-ENVPACKS.md. Nothing here builds a real pack — that takes an interpreter
download and a pip run, and `docs/PHASE14-ENVPACKS.md` section 7 records the one
that was built for real. What IS exercised here is everything between the
manifest and the stamp: parsing, the named refusals, the part concatenation and
its digest, the `.partial` rename and its cleanup, recipe drift, and the events
the install task emits while bytes arrive.

The archive-shaped tests need `tar` and `zstd`, because a pack IS a zstd tarball
and faking the archive would prove the test's own tar rather than the product's.
They skip with a reason when the tools are absent (`pytest -rs` prints it), and
CI's `envpacks.yml` is where a real pack is actually built and smoke-tested.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crucible import capability, cli, envpack, jobenv, workerenv
from crucible.envpack import PackEntry, PackError, PackManifest

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

VERSION = "9.9.9"

needs_tar_zstd = pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="a pack is a zstd tarball; this host has no tar and/or no zstd",
)


# ------------------------------------------------------------------- naming


def test_the_naming_rule_is_section_1s() -> None:
    name = envpack.pack_filename("tts-higgs-v3", "cuda-linux", "0.6.0")
    assert name == "crucible-env-tts-higgs-v3-cuda-linux-0.6.0.tar.zst"
    assert envpack.part_filename(name, 0).endswith(".tar.zst.part00")
    assert envpack.part_filename(name, 11).endswith(".tar.zst.part11")


def test_parts_are_zero_padded_so_they_sort_in_order() -> None:
    """The manifest carries the order, but a directory listing must agree.

    An operator who fetches the assets by hand and cats them in shell-glob
    order gets the right bytes only if `part9` and `part10` sort correctly,
    which is the whole reason for two digits.
    """
    names = [envpack.part_filename("a.tar.zst", index) for index in range(12)]
    assert names == sorted(names)


# ------------------------------------------------------------- the pack table


def test_each_backend_publishes_exactly_the_packs_its_recipes_describe() -> None:
    cuda = sorted(envpack.pack_targets("cuda-linux"))
    mac = sorted(envpack.pack_targets("mlx-darwin"))
    assert cuda == ["align", "asr", "llm", "rvc", "server", "tts-higgs-v3"]
    # Since 2026-09-14 the Mac has `align` and `asr` too: one is the same
    # engine on a different device, the other is a second engine with its own
    # recipe. A pack exists exactly when its `.txt` does, which is why adding
    # those two files was all it took.
    assert mac == ["align", "asr", "llm", "rvc", "server", "tts"]


def test_smoke_table_covers_every_installable_name() -> None:
    """R1: `cli.INSTALLER_FOR` and `envpack.SMOKE_IMPORT` cannot drift apart.

    `envpack` cannot import `cli` (a cycle), so the two tables are tied by
    this check instead of by an import. Every job type `crucible install` can
    install must, on every backend whose recipe exists, resolve to a pack with
    a smoke import — otherwise a release would publish a pack nothing proved
    can be imported, or fail to publish one for a type a machine can ask for.
    """
    for backend_kind in ("cuda-linux", "mlx-darwin"):
        targets = envpack.pack_targets(backend_kind)
        for job_type in cli.INSTALLABLE_JOB_TYPES:
            if job_type in workerenv.WORKER_JOB_TYPES:
                try:
                    workerenv.recipe_for(job_type, backend_kind)
                except workerenv.WorkerEnvError:
                    continue
                keys = [job_type]
            elif job_type == "llm":
                keys = [jobenv.llm_env(backend_kind).key]
            else:
                keys = sorted(
                    {
                        jobenv.tts_env(engine, backend_kind).key
                        for engine in ("higgs-v3",)
                    }
                )
            for key in keys:
                assert key in targets, f"{job_type}/{backend_kind} has no pack"
                assert targets[key].smoke_import, f"{key} has no smoke import"


def test_the_server_pack_is_smoke_tested_by_running_not_importing() -> None:
    target = envpack.pack_target("server", "cuda-linux")
    assert target.smoke_import is None
    assert target.job_type is None
    assert target.recipe.name == "pyproject.toml"


def test_the_server_pack_lands_beside_the_envs_not_inside_them() -> None:
    home = Path("/h")
    assert envpack.pack_target("server", "cuda-linux").env_dir(home) == home / "server"
    assert (
        envpack.pack_target("asr", "cuda-linux").env_dir(home) == home / "envs" / "asr"
    )


def test_every_pack_is_thirteen_rows_across_three_backends() -> None:
    """Ten, then eleven, then thirteen — and each number is a decision.

    PHASE15 4.4 added the `host` pack, which is the ONE row `llama-windows`
    publishes: that backend's engine is `llama-server` over GGUF (3.10), a
    binary Crucible spawns rather than a pip env it installs, so there are no
    job-type recipes for it. The Mac added `asr` and `align` on the same day
    (7c). `scripts/release.sh` and `envpacks.yml` both read `every_pack()`,
    so this count is what notices a backend that silently stopped publishing
    something.
    """
    rows = envpack.every_pack()
    assert len(rows) == len(set(rows)) == 13
    assert ("host", "llama-windows") in rows
    windows = [row for row in rows if row[1] == "llama-windows"]
    assert windows == [("host", "llama-windows")]


def test_a_pack_nobody_publishes_is_refused_by_name() -> None:
    """`tts-higgs-v3` is cuda-linux's pack name; the Mac's is plain `tts`,
    because on that backend every narrator engine resolves to one env."""
    with pytest.raises(PackError) as caught:
        envpack.pack_target("tts-higgs-v3", "mlx-darwin")
    assert caught.value.code == "pack_unknown"
    assert "'llm'" in caught.value.message


def test_a_backend_that_is_not_one_is_refused() -> None:
    with pytest.raises(PackError) as caught:
        envpack.pack_targets("rocm-linux")
    assert caught.value.code == "pack_not_buildable_here"


def test_the_pinned_interpreter_is_one_place_per_backend() -> None:
    for backend_kind, pin in envpack.STANDALONE_PYTHON.items():
        assert pin.python_version.startswith("3.11.")
        assert len(pin.sha256) == 64
        assert pin.release in pin.url and pin.asset in pin.url
        assert "install_only" in pin.asset
    assert "x86_64-unknown-linux-gnu" in envpack.STANDALONE_PYTHON["cuda-linux"].asset
    assert "aarch64-apple-darwin" in envpack.STANDALONE_PYTHON["mlx-darwin"].asset
    assert "x86_64-pc-windows-msvc" in envpack.STANDALONE_PYTHON["llama-windows"].asset


# --------------------------------------------------- the Windows host pack


def test_the_windows_pin_is_the_asset_that_exists_and_carries_no_shared_infix() -> None:
    """A REGRESSION PIN ON PHASE15 4.4's OWN CORRECTION.

    The doc first wrote the Windows interpreter as
    `x86_64-pc-windows-msvc-SHARED-install_only` and then corrected itself
    from the release's `SHA256SUMS`: there is no such asset on 20260901, the
    `-shared` infix is retired, and the Windows `install_only` build IS the
    shared one. A pin nobody can download fails on the `windows-latest`
    runner and nowhere else, hours after the tag, so the spelling is asserted
    here where it costs a second.
    """
    pin = envpack.STANDALONE_PYTHON[envpack.LLAMA_WINDOWS]
    assert pin.asset == (
        "cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz"
    )
    assert pin.sha256 == (
        "6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6"
    )
    assert "-shared" not in pin.asset
    # Same release and same CPython as the two backends: the host pack runs
    # the same server code, so a different 3.11 would resolve a different set.
    assert pin.release == envpack.STANDALONE_PYTHON["cuda-linux"].release
    assert pin.python_version == envpack.STANDALONE_PYTHON["cuda-linux"].python_version


@pytest.mark.parametrize("arch", ["AMD64", "x86_64"])
def test_a_windows_machine_builds_the_host_pack(
    monkeypatch: pytest.MonkeyPatch, arch: str
) -> None:
    """Both spellings, because both are seen.

    `platform.machine()` reads the registry on Windows and says `AMD64`;
    every other tool in this system says `x86_64` for the same silicon, and a
    build that answered `pack_not_buildable_here` on a perfectly good runner
    because of the case of four letters would be a very long afternoon.

    The platform is INJECTED rather than skipped: this suite runs in WSL, and
    a test that skips its subject is not a test of it (PHASE15 4.5).
    """
    monkeypatch.setattr(envpack.sys, "platform", "win32")
    monkeypatch.setattr(envpack.platform, "machine", lambda: arch)
    assert envpack.build_backend_kind() == envpack.LLAMA_WINDOWS


def test_arm_windows_builds_nothing_and_the_refusal_names_all_three_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no `aarch64-pc-windows` pin and no runner, so it is refused.

    The refusal must NOT say "Windows is never a backend" any more, and that
    is not a loosened assertion: PHASE15 section 0's amendment makes Windows
    the `llama-windows` backend, so the old sentence became false. A refusal
    that told an ARM Windows reader "Windows builds nothing" while an x86_64
    Windows machine builds the host pack is a refusal that sends them to the
    wrong question.
    """
    monkeypatch.setattr(envpack.sys, "platform", "win32")
    monkeypatch.setattr(envpack.platform, "machine", lambda: "ARM64")
    with pytest.raises(PackError) as caught:
        envpack.build_backend_kind()
    assert caught.value.code == "pack_not_buildable_here"
    assert "never a backend" not in caught.value.message
    for backend_kind in ("cuda-linux", "mlx-darwin", "llama-windows"):
        assert backend_kind in caught.value.message
    assert "win32/ARM64" in caught.value.message


def test_llama_windows_publishes_the_host_pack_and_nothing_else() -> None:
    """No job-type packs, because that backend's engine is not a pip env.

    `llama-windows` serves its llm classes and `pages` from `llama-server`
    children over GGUF (PHASE15 3.10) — a binary Crucible spawns, not
    something pip installs — and the Python job types need WSL2 there and say
    so. So there is nothing for an `envs/<type>/llama-windows.txt` to hold.

    And the traffic goes both ways — `host` must not appear on the two POSIX
    backends, or `crucible envpack list` would offer a Linux runner a pack
    with a tray in it.
    """
    assert set(envpack.pack_targets(envpack.LLAMA_WINDOWS)) == {"host"}
    assert "host" not in envpack.pack_targets("cuda-linux")
    assert "host" not in envpack.pack_targets("mlx-darwin")


def test_the_host_pack_is_built_from_pyproject_and_smoke_tested_by_running() -> None:
    target = envpack.pack_target("host", envpack.LLAMA_WINDOWS)
    assert target.recipe.name == "pyproject.toml"
    assert target.job_type is None
    # Like `server`: the thing most likely to be broken is the `.cmd` shim
    # that replaces pip's unrelocatable `.exe`, and an import would not touch
    # it.
    assert target.smoke_import is None


def test_asking_llama_windows_for_the_server_pack_names_what_it_does_publish() -> None:
    with pytest.raises(PackError) as caught:
        envpack.pack_target("server", envpack.LLAMA_WINDOWS)
    assert caught.value.code == "pack_unknown"
    assert "'host'" in caught.value.message


def test_the_host_pack_lands_beside_the_server_rather_than_under_envs() -> None:
    """`%LOCALAPPDATA%\\Crucible\\host\\` (PHASE15 4.4), and `<home>/host` here.

    Not under `envs/`, where `crucible doctor` reads every directory as a job
    env and would report the tray as a broken one.
    """
    home = Path("/h")
    assert (
        envpack.pack_target("host", envpack.LLAMA_WINDOWS).env_dir(home) == home / "host"
    )


def test_the_host_packs_archive_follows_the_one_naming_rule() -> None:
    """NO SPECIAL CASE, and the backend's name is why there needs to be none.

    Section 1's rule is `crucible-env-<name>-<backend>-<version>.tar.zst` and
    `pack_filename` is the one place that states it. An earlier draft called
    this backend `host-windows`, which made the asset
    `crucible-env-host-host-windows-…` and put the naming rule at odds with
    4.4's own sentence — the kind of disagreement that gets settled with a
    special case in the one function whose whole job is to state the rule,
    and then has to be mirrored byte-for-byte in
    `sdk/bootstrap/src/envpacks.ts`'s `packAssetName` or `install.ps1`
    downloads a 404. Renaming the backend to `llama-windows` (PHASE15 section
    0) dissolved it: the asset falls straight out of the rule.
    """
    assert (
        envpack.pack_filename("host", "llama-windows", "0.6.0")
        == "crucible-env-host-llama-windows-0.6.0.tar.zst"
    )
    assert envpack.pack_target("host", envpack.LLAMA_WINDOWS).archive_name(
        "0.6.0"
    ) == envpack.pack_filename("host", "llama-windows", "0.6.0")


def test_pack_python_knows_each_layout_and_refuses_one_it_does_not() -> None:
    """THE reason `pack_python` is public (PHASE15 4.4).

    python-build-standalone's Windows `install_only` tree is `python.exe`,
    `pythonw.exe`, `Scripts\\`, `Lib\\`, `DLLs\\` — there is no `bin/` at all.
    Three call sites spelling `bin/python` inline would be three places that
    have to learn this and two that will not.
    """
    root = Path("/p")
    assert envpack.pack_python(root, "cuda-linux") == root / "bin" / "python"
    assert envpack.pack_python(root, "mlx-darwin") == root / "bin" / "python"
    assert envpack.pack_python(root, envpack.LLAMA_WINDOWS) == root / "python.exe"
    with pytest.raises(PackError) as caught:
        envpack.pack_python(root, "rocm-linux")
    assert caught.value.code == "pack_not_buildable_here"


def test_the_tray_packages_are_named_once_and_are_not_wheel_dependencies() -> None:
    """Why they are here and not in `pyproject.toml`.

    `pyproject.toml` is what EVERY pack's server half is built from, so a
    tray dependency there would make every Linux and Mac server download and
    carry a GUI toolkit it can never open a window with.
    """
    assert envpack.HOST_EXTRA_PACKAGES == ("pystray", "pillow")
    pyproject = (envpack.repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    for package in envpack.HOST_EXTRA_PACKAGES:
        assert package not in pyproject


# ---------------------------------------------------------------- manifests


def one_entry(**overrides: Any) -> dict[str, Any]:
    entry = {
        "name": "asr",
        "backend": "cuda-linux",
        "python": "3.11.16",
        "bytes": 1234,
        "sha256": "a" * 64,
        "parts": ["crucible-env-asr-cuda-linux-9.9.9.tar.zst.part00"],
        "recipe_sha256": "b" * 64,
        "unpacked_bytes": 4321,
    }
    entry.update(overrides)
    return entry


def one_document(**overrides: Any) -> dict[str, Any]:
    document = {"schema": 1, "version": VERSION, "packs": [one_entry()]}
    document.update(overrides)
    return document


def test_a_manifest_round_trips() -> None:
    manifest = envpack.parse_manifest(json.dumps(one_document()))
    assert manifest.version == VERSION
    assert manifest.find("asr", "cuda-linux") is not None
    assert manifest.find("asr", "mlx-darwin") is None
    again = envpack.parse_manifest(manifest.dumps())
    assert again.packs == manifest.packs


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("not json at all", "is not JSON"),
        ("[]", "is a list, not an object"),
        (json.dumps({"version": VERSION, "packs": []}), "says schema None"),
        (json.dumps(one_document(schema=2)), "says schema 2"),
        (json.dumps(one_document(version=7)), "has no version string"),
        (json.dumps({"schema": 1, "version": VERSION}), "has no `packs` array"),
        (json.dumps(one_document(packs=["x"])), "packs[0] is a str"),
        (
            json.dumps({"schema": 1, "version": VERSION, "packs": [{"name": "asr"}]}),
            "has no 'backend'",
        ),
        (json.dumps(one_document(packs=[one_entry(bytes="big")])), ".bytes is 'big'"),
        # bool is an int in Python, and `bytes: true` must not read as 1.
        (json.dumps(one_document(packs=[one_entry(bytes=True)])), ".bytes is True"),
        (json.dumps(one_document(packs=[one_entry(parts=[])])), "parts is empty"),
        (
            json.dumps(one_document(packs=[one_entry(parts=[3])])),
            "not a filename",
        ),
        (
            json.dumps(one_document(packs=[one_entry(unpacked_bytes=0)])),
            "both are sizes",
        ),
        (
            json.dumps(one_document(packs=[one_entry(), one_entry()])),
            "names the same pack twice",
        ),
    ],
)
def test_every_unreadable_manifest_says_which_part(text: str, fragment: str) -> None:
    with pytest.raises(PackError) as caught:
        envpack.parse_manifest(text, source="envpacks.json")
    assert caught.value.code == "pack_manifest_unreadable"
    assert fragment in caught.value.message


def test_a_pack_absent_from_the_manifest_is_pack_not_published() -> None:
    manifest = envpack.parse_manifest(json.dumps(one_document()))
    with pytest.raises(PackError) as caught:
        manifest.require("llm", "cuda-linux")
    assert caught.value.code == "pack_not_published"
    # It names what IS there, so its reader can tell "not built" from "typo".
    assert "asr/cuda-linux" in caught.value.message
    assert "--build" in caught.value.message


def test_fragments_merge_into_one_manifest() -> None:
    left = json.dumps(one_document())
    right = json.dumps(
        one_document(packs=[one_entry(name="llm", backend="mlx-darwin")])
    )
    merged = envpack.merge_manifests(VERSION, [("a.json", left), ("b.json", right)])
    assert [(p.name, p.backend) for p in merged.packs] == [
        ("asr", "cuda-linux"),
        ("llm", "mlx-darwin"),
    ]


def test_a_fragment_from_another_release_is_never_merged() -> None:
    other = json.dumps(one_document(version="0.0.1"))
    with pytest.raises(PackError) as caught:
        envpack.merge_manifests(VERSION, [("stale.json", other)])
    assert caught.value.code == "pack_manifest_unreadable"
    assert "fragments from two builds are never merged" in caught.value.message


def test_two_fragments_carrying_one_pack_are_refused() -> None:
    text = json.dumps(one_document())
    with pytest.raises(PackError) as caught:
        envpack.merge_manifests(VERSION, [("a.json", text), ("b.json", text)])
    assert "two fragments carry the same pack" in caught.value.message


def test_the_manifest_url_is_this_versions_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(envpack.MANIFEST_URL_ENV, raising=False)
    url = envpack.manifest_url("0.6.0")
    assert url.endswith("/releases/download/v0.6.0/envpacks.json")


def test_the_override_is_an_option_and_it_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(envpack.MANIFEST_URL_ENV, "file:///tmp/x/envpacks.json")
    assert envpack.manifest_url("0.6.0") == "file:///tmp/x/envpacks.json"


def test_parts_are_fetched_from_beside_the_manifest() -> None:
    """One URL configures a mirror, not two. Section 3.1's `--manifest-url`."""
    assert (
        envpack.asset_url("https://x/y/envpacks.json", "a.part00")
        == "https://x/y/a.part00"
    )
    assert (
        envpack.asset_url("file:///tmp/p/envpacks.json", "a.part00")
        == "file:///tmp/p/a.part00"
    )


def test_an_unreadable_manifest_url_is_named(tmp_path: Path) -> None:
    missing = (tmp_path / "nope" / "envpacks.json").as_uri()
    with pytest.raises(PackError) as caught:
        envpack.read_manifest(missing)
    assert caught.value.code == "pack_manifest_unreadable"


# ---------------------------------------------------------------- progress


def test_the_progress_line_round_trips() -> None:
    line = envpack.progress_line(17, 100, "x.part00")
    assert envpack.parse_progress_line(line) == {
        "bytes_done": 17,
        "bytes_total": 100,
        "file": "x.part00",
    }
    assert envpack.parse_progress_line("Collecting torch==2.8.0") is None
    assert envpack.parse_progress_line(envpack.PROGRESS_PREFIX + "{oops") is None
    assert envpack.parse_progress_line(envpack.PROGRESS_PREFIX + '{"a": 1}') is None


def test_an_unknown_total_stays_null_rather_than_zero() -> None:
    """`weights.ProgressHook` says total may be None, and zero is a lie."""
    parsed = envpack.parse_progress_line(envpack.progress_line(5, None, "f"))
    assert parsed is not None and parsed["bytes_total"] is None


# ------------------------------------------------------------- disk + drift


def test_pack_disk_names_the_three_numbers(tmp_path: Path) -> None:
    entry = PackEntry(
        name="llm",
        backend="cuda-linux",
        python="3.11.16",
        bytes=3_000_000_000,
        sha256="a" * 64,
        parts=("a.part00", "a.part01"),
        recipe_sha256="b" * 64,
        unpacked_bytes=900_000_000_000_000,
    )
    with pytest.raises(PackError) as caught:
        envpack.check_disk(tmp_path, entry)
    assert caught.value.code == "pack_disk"
    assert "unpacked" in caught.value.message
    assert "archive" in caught.value.message
    assert "part in flight" in caught.value.message
    assert "Nothing was downloaded" in caught.value.message


def test_a_pack_built_from_another_recipe_is_pack_recipe_drift() -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    entry = PackEntry(
        name="asr",
        backend="cuda-linux",
        python="3.11.16",
        bytes=10,
        sha256="a" * 64,
        parts=("a.part00",),
        recipe_sha256="c" * 64,
        unpacked_bytes=20,
    )
    with pytest.raises(PackError) as caught:
        envpack.check_recipe(entry, target)
    assert caught.value.code == "pack_recipe_drift"
    assert "cccccccccccc" in caught.value.message
    assert "--build" in caught.value.message


def test_a_pack_built_from_this_recipe_passes() -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    entry = PackEntry(
        name="asr",
        backend="cuda-linux",
        python="3.11.16",
        bytes=10,
        sha256="a" * 64,
        parts=("a.part00",),
        recipe_sha256=jobenv.recipe_sha256(target.recipe),
        unpacked_bytes=20,
    )
    envpack.check_recipe(entry, target)  # no refusal


# ------------------------------------------------------- a pack, end to end


def build_fake_pack(
    tmp_path: Path,
    target: envpack.PackTarget,
    *,
    filler: int = 300_000,
    part_bytes: int = 64_000,
    recipe_sha256: str | None = None,
) -> tuple[Path, PackEntry]:
    """A real zstd tarball with a real manifest, small enough to be a test.

    Not a real env — building one costs a pip run — but a real ARCHIVE, split
    into real parts with a real digest, because the machinery under test is
    exactly the fetching, joining, hashing and unpacking of those bytes.
    """
    root = tmp_path / "tree"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_text("#!/bin/sh\necho 3.11.16\n", encoding="utf-8")
    (root / "bin" / "python").chmod(0o755)
    (root / "lib").mkdir()
    # Incompressible, so the archive really does exceed one part.
    (root / "lib" / "filler.bin").write_bytes(os.urandom(filler))

    out = tmp_path / "packs"
    out.mkdir(exist_ok=True)
    archive = out / target.archive_name(VERSION)
    envpack.create_archive(root, archive)
    unpacked_bytes = envpack.directory_bytes(root)
    size = archive.stat().st_size
    digest = envpack.sha256_of(archive)
    parts = envpack.split_archive(archive, part_bytes=part_bytes)
    entry = PackEntry(
        name=target.name,
        backend=target.backend_kind,
        python="3.11.16",
        bytes=size,
        sha256=digest,
        parts=tuple(part.name for part in parts),
        recipe_sha256=(
            recipe_sha256
            if recipe_sha256 is not None
            else jobenv.recipe_sha256(target.recipe)
        ),
        unpacked_bytes=unpacked_bytes,
    )
    envpack.write_manifest_entry(out, VERSION, entry)
    shutil.rmtree(root)
    return out, entry


@needs_tar_zstd
def test_the_parts_rejoin_to_exactly_the_archive_that_was_split(
    tmp_path: Path,
) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    assert len(entry.parts) > 1, "the fixture must exercise more than one part"
    parts = [out / name for name in entry.parts]
    # The digest of the whole, computed without ever reassembling it.
    assert envpack.sha256_of_parts(parts) == entry.sha256
    assert sum(part.stat().st_size for part in parts) == entry.bytes
    # Every part but the last is exactly one part long.
    sizes = [part.stat().st_size for part in parts]
    assert len(set(sizes[:-1])) == 1 and sizes[-1] <= sizes[0]


@needs_tar_zstd
def test_the_archive_holds_the_pack_at_top_level_with_no_wrapper_directory(
    tmp_path: Path,
) -> None:
    """A CONTRACT WITH `@crucible/bootstrap`, not an implementation detail.

    `envpack.create_archive` tars the CONTENTS of the pack (`-C <root> … .`),
    so unpacking into any directory fills it with `bin/` and `lib/`. Bootstrap
    unpacks the `server` pack into `~/.crucible/server/` and then runs
    `~/.crucible/server/bin/crucible` — a wrapper directory in the archive
    would make that `~/.crucible/server/python/bin/crucible` and every path
    bootstrap, the systemd unit and `install.ps1` state would be wrong at once,
    in a way no test on either side would otherwise notice.

    So the layout is asserted from OUTSIDE the code that produces it: the
    archive's members are read with `tar -t`.
    """
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target, filler=1000, part_bytes=1 << 20)
    joined = tmp_path / "whole.tar.zst"
    with joined.open("wb") as handle:
        for name in entry.parts:
            handle.write((out / name).read_bytes())

    import subprocess

    listing = subprocess.run(
        [shutil.which("tar") or "tar", "--zstd", "-tf", str(joined)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert listing.returncode == 0, listing.stderr
    members = [
        line.lstrip("./").rstrip("/")
        for line in listing.stdout.splitlines()
        if line.strip() not in ("", ".", "./")
    ]
    tops = {member.split("/", 1)[0] for member in members}
    assert tops == {"bin", "lib"}, f"a wrapper directory appeared: {sorted(tops)}"
    assert "bin/python" in members


@needs_tar_zstd
def test_download_joins_the_parts_and_deletes_each_one(tmp_path: Path) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    downloads = tmp_path / "downloads"
    archive = downloads / "joined.tar.zst"
    seen: list[tuple[int, int | None, str]] = []
    envpack.download_parts(
        [(out / name).as_uri() for name in entry.parts],
        archive,
        total_bytes=entry.bytes,
        downloads=downloads,
        on_progress=lambda done, total, name: seen.append((done, total, name)),
    )
    assert envpack.sha256_of(archive) == entry.sha256
    # Peak extra disk is one part: nothing is left behind but the archive.
    assert sorted(p.name for p in downloads.iterdir()) == ["joined.tar.zst"]
    assert seen and seen[-1][0] == entry.bytes
    assert {row[1] for row in seen} == {entry.bytes}


@needs_tar_zstd
def test_install_unpacks_stamps_and_renames(tmp_path: Path) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    home = tmp_path / "home"
    envpack.install_pack(
        home, target, VERSION, location=(out / "envpacks.json").as_uri()
    )
    final = target.env_dir(home)
    assert (final / "bin" / "python").is_file()
    assert (final / "lib" / "filler.bin").is_file()
    # `.partial` is gone, and so is the reassembled archive.
    assert not final.with_name(f"{final.name}.partial").exists()
    assert list((home / envpack.DOWNLOADS_DIRNAME).iterdir()) == []
    record = json.loads(
        (final / jobenv.ENV_STAMP_NAME).read_text(encoding="utf-8")
    )
    assert record["pack_sha256"] == entry.sha256
    assert record["recipe_sha256"] == entry.recipe_sha256
    assert record["backend"] == "cuda-linux"
    assert record["job_type"] == "asr"
    assert record["python_version"] == "3.11.16"


@needs_tar_zstd
def test_the_stamp_reaches_env_status_and_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pack-installed env reports WHICH WAY it got there. Section 5."""
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    home = tmp_path / "home"
    envpack.install_pack(
        home, target, VERSION, location=(out / "envpacks.json").as_uri()
    )
    pins = workerenv.recipe_pins(target.recipe)
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    status = workerenv.env_status(home, "asr", "cuda-linux")
    assert status.installed is True
    assert status.pack_sha256 == entry.sha256
    assert status.recipe_sha256 == entry.recipe_sha256
    assert status.to_dict()["pack_sha256"] == entry.sha256


@needs_tar_zstd
def test_a_leftover_partial_is_removed_and_the_old_env_survives_a_failure(
    tmp_path: Path,
) -> None:
    """Two halves of R6, in one place, because they meet at the rename.

    A `.partial` from an interrupted run is scratch and goes; the env that is
    already installed is NOT touched until a complete tree exists to replace
    it, so a failed install leaves the working env exactly as it was.
    """
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    home = tmp_path / "home"
    final = target.env_dir(home)
    final.mkdir(parents=True)
    (final / "the-old-env").write_text("still here", encoding="utf-8")
    partial = final.with_name(f"{final.name}.partial")
    partial.mkdir()
    (partial / "junk").write_text("half an unpack", encoding="utf-8")

    # A failure first: the manifest's digest is wrong, so nothing is replaced.
    broken = envpack.PackManifest(
        version=VERSION,
        packs=(
            PackEntry(**{**entry.to_dict(), "sha256": "d" * 64, "parts": list(entry.parts)}),
        ),
    )
    broken_path = out / "broken.json"
    broken_path.write_text(broken.dumps(), encoding="utf-8")
    with pytest.raises(PackError) as caught:
        envpack.install_pack(
            home, target, VERSION, location=broken_path.as_uri()
        )
    assert caught.value.code == "pack_sha_mismatch"
    assert (final / "the-old-env").is_file()
    # The bad archive is deleted rather than left for a resume to believe.
    assert list((home / envpack.DOWNLOADS_DIRNAME).iterdir()) == []

    envpack.install_pack(
        home, target, VERSION, location=(out / "envpacks.json").as_uri()
    )
    assert (final / "bin" / "python").is_file()
    assert not (final / "the-old-env").exists()
    assert not partial.exists()


@needs_tar_zstd
def test_a_short_download_is_refused_before_it_is_hashed(tmp_path: Path) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    # Truncate the last part: the bytes arrive, the length does not match.
    last = out / entry.parts[-1]
    last.write_bytes(last.read_bytes()[:-16])
    home = tmp_path / "home"
    with pytest.raises(PackError) as caught:
        envpack.install_pack(
            home, target, VERSION, location=(out / "envpacks.json").as_uri()
        )
    assert caught.value.code == "pack_download_failed"
    assert not target.env_dir(home).exists()


@needs_tar_zstd
def test_something_that_is_not_a_pack_is_refused_after_it_unpacks(
    tmp_path: Path,
) -> None:
    """A verified archive with no `bin/python` is not an env, and says so."""
    target = envpack.pack_target("asr", "cuda-linux")
    root = tmp_path / "tree"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "readme.txt").write_text("not a python", encoding="utf-8")
    out = tmp_path / "packs"
    out.mkdir()
    archive = out / target.archive_name(VERSION)
    envpack.create_archive(root, archive)
    size = archive.stat().st_size
    digest = envpack.sha256_of(archive)
    parts = envpack.split_archive(archive, part_bytes=1 << 20)
    envpack.write_manifest_entry(
        out,
        VERSION,
        PackEntry(
            name=target.name,
            backend=target.backend_kind,
            python="3.11.16",
            bytes=size,
            sha256=digest,
            parts=tuple(part.name for part in parts),
            recipe_sha256=jobenv.recipe_sha256(target.recipe),
            unpacked_bytes=envpack.directory_bytes(root),
        ),
    )
    home = tmp_path / "home"
    with pytest.raises(PackError) as caught:
        envpack.install_pack(
            home, target, VERSION, location=(out / "envpacks.json").as_uri()
        )
    assert caught.value.code == "pack_unpack_failed"
    assert not target.env_dir(home).exists()
    assert not target.env_dir(home).with_name("asr.partial").exists()


@needs_tar_zstd
def test_check_verifies_the_parts_and_the_recipe(tmp_path: Path) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, entry = build_fake_pack(tmp_path, target)
    assert envpack.check_pack(out, target, VERSION).sha256 == entry.sha256

    (out / entry.parts[0]).unlink()
    with pytest.raises(PackError) as caught:
        envpack.check_pack(out, target, VERSION)
    assert caught.value.code == "pack_not_published"


@needs_tar_zstd
def test_check_catches_a_pack_built_from_a_recipe_that_has_moved(
    tmp_path: Path,
) -> None:
    target = envpack.pack_target("asr", "cuda-linux")
    out, _ = build_fake_pack(tmp_path, target, recipe_sha256="e" * 64)
    with pytest.raises(PackError) as caught:
        envpack.check_pack(out, target, VERSION)
    assert caught.value.code == "pack_recipe_drift"


@needs_tar_zstd
def test_building_a_second_pack_merges_rather_than_replaces(tmp_path: Path) -> None:
    out, first = build_fake_pack(
        tmp_path / "a", envpack.pack_target("asr", "cuda-linux")
    )
    second = PackEntry(**{**first.to_dict(), "name": "llm", "parts": list(first.parts)})
    envpack.write_manifest_entry(out, VERSION, second)
    manifest = envpack.parse_manifest((out / "envpacks.json").read_text("utf-8"))
    assert sorted(p.name for p in manifest.packs) == ["asr", "llm"]
    # Writing the same pack again replaces its row rather than duplicating it.
    envpack.write_manifest_entry(out, VERSION, second)
    manifest = envpack.parse_manifest((out / "envpacks.json").read_text("utf-8"))
    assert sorted(p.name for p in manifest.packs) == ["asr", "llm"]


# ------------------------------------------------- the relocatable shebang


def write_script(path: Path, shebang: bytes, body: str = "import sys\nprint('ok')\n") -> None:
    path.write_bytes(shebang + body.encode("utf-8"))
    path.chmod(0o755)


def test_a_baked_in_shebang_is_rewritten_to_one_that_survives_a_move(
    tmp_path: Path,
) -> None:
    """The defect that killed the first real `server` build, 2026-09-14.

    pip writes each console script's shebang as the ABSOLUTE path of the
    interpreter that installed it. Move the tree and that path is gone, and
    `exec` answers ENOENT naming the SCRIPT — so the failure reads as "the
    file you are looking at does not exist".
    """
    root = tmp_path / "python"
    (root / "bin").mkdir(parents=True)
    script = root / "bin" / "crucible"
    write_script(script, f"#!{root}/bin/python3.11\n".encode("utf-8"))

    assert envpack.relocate_console_scripts(root) == ["crucible"]
    text = script.read_text(encoding="utf-8")
    assert str(root) not in text
    assert text.startswith(envpack.RELOCATABLE_SHEBANG)
    assert text.endswith("import sys\nprint('ok')\n")
    assert script.stat().st_mode & 0o111, "it must still be executable"


def test_the_rewritten_header_is_valid_python(tmp_path: Path) -> None:
    """The obvious form is a SyntaxError, and the smoke test is where it lands.

    `"exec" "$(dirname -- "$0")/python3" …` looks like implicit string
    concatenation and is not: the nested `"` inside the command substitution
    closes the Python string early. The single-quoted triple form is what
    makes the same two lines a shell command and a discarded docstring.
    """
    compile(
        envpack.RELOCATABLE_SHEBANG + "print('ok')\n", "crucible", "exec"
    )


def test_distlibs_long_path_wrapper_is_replaced_whole(tmp_path: Path) -> None:
    """pip writes a THREE-line wrapper when the interpreter path is long."""
    root = tmp_path / "python"
    (root / "bin").mkdir(parents=True)
    script = root / "bin" / "uvicorn"
    write_script(
        script,
        (
            "#!/bin/sh\n"
            f"'''exec' {root}/bin/python3.11 \"$0\" \"$@\"\n"
            "' '''\n"
        ).encode("utf-8"),
    )
    assert envpack.relocate_console_scripts(root) == ["uvicorn"]
    text = script.read_text(encoding="utf-8")
    assert str(root) not in text
    assert text.count("'''") == 2, "the old wrapper's quotes must be gone"
    compile(text, "uvicorn", "exec")


def test_a_shebang_that_is_not_ours_is_left_alone(tmp_path: Path) -> None:
    """`#!/usr/bin/env python3` is already relocatable; rewriting it would be
    this function inventing policy about somebody else's script."""
    root = tmp_path / "python"
    (root / "bin").mkdir(parents=True)
    write_script(root / "bin" / "theirs", b"#!/usr/bin/env python3\n")
    (root / "bin" / "python3").write_bytes(b"\x7fELF binary, not a script")
    assert envpack.relocate_console_scripts(root) == []
    assert (root / "bin" / "theirs").read_text("utf-8").startswith(
        "#!/usr/bin/env python3"
    )


@needs_tar_zstd
def test_a_relocated_script_actually_runs_from_somewhere_else(
    tmp_path: Path,
) -> None:
    """The whole point, proved by running it: build here, execute there.

    `bin/python3` is a shell script here rather than CPython, because what is
    under test is the HEADER finding its neighbour — not the interpreter.
    """
    if os.name != "posix":
        pytest.skip("the shebang is a POSIX sh polyglot")
    root = tmp_path / "python"
    (root / "bin").mkdir(parents=True)
    stub = root / "bin" / "python3"
    stub.write_text('#!/bin/sh\necho "ran $1"\n', encoding="utf-8")
    stub.chmod(0o755)
    write_script(root / "bin" / "crucible", f"#!{root}/bin/python3.11\n".encode())
    envpack.relocate_console_scripts(root)

    moved = tmp_path / "somewhere" / "else"
    moved.parent.mkdir()
    shutil.move(str(root), str(moved))
    import subprocess

    completed = subprocess.run(
        [str(moved / "bin" / "crucible"), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert str(moved / "bin" / "crucible") in completed.stdout


# ------------------------------------------------------- the Windows .cmd shim


def fake_windows_pack(
    tmp_path: Path, *, entry_points: str, scripts: tuple[str, ...] = ("crucible",)
) -> Path:
    """The SHAPE of a built Windows pack, without the 200 MB pip run.

    python-build-standalone's Windows `install_only` tree plus what pip leaves
    in it: `python.exe` at the root, an `.exe` launcher per console script in
    `Scripts\\`, and the distribution's own `entry_points.txt` under
    `Lib/site-packages`. Placeholders, because what is under test is the
    reading of the metadata and the bytes of the shim — not pip.
    """
    root = tmp_path / "pack"
    (root / "Scripts").mkdir(parents=True)
    (root / "python.exe").write_bytes(b"MZ not really a PE, and not read")
    for name in scripts:
        (root / "Scripts" / f"{name}.exe").write_bytes(b"MZ pip's launcher")
    dist_info = root / "Lib" / "site-packages" / "crucible-0.6.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return root


def test_the_cmd_shim_is_written_beside_python_exe_with_exactly_these_bytes(
    tmp_path: Path,
) -> None:
    """7.2a's defect, in its Windows shape, and the exact answer PHASE15 4.4 gives.

    pip does not write a shebang on Windows; it writes `Scripts\\<name>.exe`,
    a launcher BINARY with the building interpreter's absolute path compiled
    into it. A move breaks it and no shebang rewrite can reach it, so the
    pack ships a `.cmd` beside `python.exe` instead.

    The bytes are asserted whole rather than sniffed for a substring, because
    every part of them is load-bearing: `%~dp0` (the script's own directory,
    trailing backslash included — which is why there is no second backslash),
    the quotes (`%LOCALAPPDATA%` holds a user name, and "Owen Morgan" would
    otherwise split the command in two), `%*` (the arguments), and CRLF
    (cmd.exe's batch parser is line-oriented on CRLF; an LF-only `.cmd`
    misparses and the symptom is a shim that silently does nothing).
    """
    root = fake_windows_pack(
        tmp_path, entry_points="[console_scripts]\ncrucible = crucible.cli:main\n"
    )
    assert envpack.write_cmd_shims(root) == ["crucible"]
    shim = root / "crucible.cmd"
    assert shim.read_bytes() == (
        b'@echo off\r\n"%~dp0python.exe" -m crucible.cli %*\r\n'
    )
    # Beside python.exe, at the pack ROOT — not in Scripts\, or `%~dp0` would
    # point one directory below the interpreter and `install.ps1` would need a
    # subdirectory in every path it writes.
    assert shim.parent == root
    assert not (root / "Scripts" / "crucible.cmd").exists()


def test_a_console_script_whose_function_is_not_main_is_called_by_name(
    tmp_path: Path,
) -> None:
    """`-m` would run the module and NOT the function, and exit 0 doing nothing.

    `-m <module>` is right for `crucible.cli` because that module ends in
    `if __name__ == "__main__": raise SystemExit(main())` — running it as a
    script and calling its `main` are the same act. For any other function
    name that equivalence is gone, and the shim would be a command that
    succeeds having done none of the work. So the other form is spelled out.
    """
    root = fake_windows_pack(
        tmp_path,
        entry_points="[console_scripts]\ncrucible = crucible.cli:serve_forever\n",
    )
    assert envpack.write_cmd_shims(root) == ["crucible"]
    assert (root / "crucible.cmd").read_bytes() == (
        b"@echo off\r\n"
        b'"%~dp0python.exe" -c "import sys; from crucible.cli import '
        b'serve_forever; sys.exit(serve_forever())" %*\r\n'
    )


def test_the_shim_text_is_the_same_two_forms_asked_for_directly() -> None:
    """`cmd_shim_text` is the one owner of both forms, so both are pinned."""
    assert envpack.cmd_shim_text("crucible.cli", "main") == (
        '@echo off\r\n"%~dp0python.exe" -m crucible.cli %*\r\n'
    )
    assert "%~dp0python.exe" in envpack.cmd_shim_text("pkg.mod", "run")
    assert envpack.cmd_shim_text("pkg.mod", "run").endswith(" %*\r\n")


def test_every_launcher_pip_wrote_gets_a_shim(tmp_path: Path) -> None:
    """The NAMES come from `Scripts\\*.exe` — pip's own record of what it made."""
    root = fake_windows_pack(
        tmp_path,
        entry_points=(
            "[console_scripts]\n"
            "crucible = crucible.cli:main\n"
            "pip = pip._internal.cli.main:main\n"
            "pip3.11 = pip._internal.cli.main:main\n"
            "\n"
            "[gui_scripts]\n"
            "nothing = nowhere:main\n"
        ),
        scripts=("crucible", "pip", "pip3.11"),
    )
    assert envpack.write_cmd_shims(root) == ["crucible", "pip", "pip3.11"]
    # A `gui_scripts` entry pip did not write an `.exe` for gets no shim: the
    # launchers are the authority on what is in this tree.
    assert not (root / "nothing.cmd").exists()


def test_an_entry_point_that_is_not_module_colon_function_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A shim cannot be written for something nobody can read, and silence is worse.

    Writing no shim would ship a pack whose command is simply missing, and
    the person who discovers that is the operator, at the point of use.
    """
    root = fake_windows_pack(
        tmp_path, entry_points="[console_scripts]\ncrucible = crucible.cli\n"
    )
    with pytest.raises(PackError) as caught:
        envpack.write_cmd_shims(root)
    assert caught.value.code == "pack_build_failed"
    assert "module:function" in caught.value.message
    assert "crucible" in caught.value.message


def test_a_launcher_no_metadata_explains_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """The pack's contents and its metadata disagreeing is a defect, not a gap."""
    root = fake_windows_pack(
        tmp_path,
        entry_points="[console_scripts]\ncrucible = crucible.cli:main\n",
        scripts=("crucible", "mystery"),
    )
    with pytest.raises(PackError) as caught:
        envpack.write_cmd_shims(root)
    assert caught.value.code == "pack_build_failed"
    assert "mystery" in caught.value.message


def test_a_tree_pip_never_installed_into_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    with pytest.raises(PackError) as caught:
        envpack.write_cmd_shims(root)
    assert caught.value.code == "pack_build_failed"
    assert "Scripts" in caught.value.message


# ------------------------------------------------------ tar, on three platforms


def test_the_two_create_argvs_are_not_interchangeable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One level, two spellings, and NEITHER is a fallback for the other.

    GNU tar reaches zstd by launching the binary, and `-T0` is a `zstd` CLI
    flag that exists nowhere else. `tar.exe` is bsdtar with libzstd linked
    in — it compresses in-process, needs no `zstd.exe` (Windows ships none,
    which is the whole reason), and has `--options` where GNU tar has
    nothing. Sending either argv to the other tar fails.

    Both are asserted as data because that is what they are, and because a
    Windows runner is the only place the second one ever runs.
    """
    root, archive = tmp_path / "tree", tmp_path / "a.tar.zst"
    level = envpack.ZSTD_BUILD_LEVEL

    monkeypatch.setattr(envpack.sys, "platform", "linux")
    assert envpack.archive_argv("tar", root, archive) == [
        "tar",
        "--use-compress-program",
        f"zstd -T0 -{level}",
        "-C",
        str(root),
        "-cf",
        str(archive),
        ".",
    ]

    monkeypatch.setattr(envpack.sys, "platform", "win32")
    assert envpack.archive_argv("tar", root, archive) == [
        "tar",
        "--zstd",
        "--options",
        f"zstd:compression-level={level}",
        "-C",
        str(root),
        "-cf",
        str(archive),
        ".",
    ]


def test_the_read_argv_is_one_spelling_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--zstd -xf` costs nothing on either tar, so the read path does not fork.

    GNU tar launches the `zstd` that `require_zstd_tar` already proved is
    there; bsdtar decompresses in-process and accepts the flag in extract
    mode (measured 2026-09-14). One argv is one fewer thing that can differ
    between the machine that builds a pack and the machine that installs it.
    """
    archive, into = tmp_path / "a.tar.zst", tmp_path / "out"
    expected = ["tar", "--zstd", "-xf", str(archive), "-C", str(into)]
    for platform_name in ("linux", "darwin", "win32"):
        monkeypatch.setattr(envpack.sys, "platform", platform_name)
        assert envpack.extract_argv("tar", archive, into) == expected


def test_windows_needs_one_tool_and_asks_it_whether_it_carries_zstd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows ships `tar.exe` and NO `zstd.exe`, so demanding both is wrong.

    Demanding `zstd.exe` would refuse a build the stock machine can do —
    `C:\\Windows\\System32\\tar.exe` is bsdtar 3.8.1 with libzstd 1.5.5 linked
    in. What replaces the demand is a question rather than an assumption: an
    old libarchive, or a GNU tar first on PATH, is a real machine, and it must
    be turned away here and not at the compression step, which is after the
    interpreter download and the pip run.
    """
    monkeypatch.setattr(envpack.sys, "platform", "win32")
    monkeypatch.setattr(
        envpack.shutil, "which", lambda name: r"C:\Windows\System32\tar.exe"
        if name == "tar"
        else None,
    )
    bsdtar = (
        "bsdtar 3.8.1 - libarchive 3.8.1 zlib/1.2.13.1-motley liblzma/5.4.3 "
        "bz2lib/1.0.8 libzstd/1.5.5 cng/2.0 libb2/bundled\n"
    )
    monkeypatch.setattr(
        envpack.subprocess,
        "run",
        lambda *a, **k: _Completed(0, bsdtar),
    )
    envpack.require_zstd_tar()  # no refusal, and `zstd` was never looked for


def test_a_tar_that_does_not_carry_zstd_is_refused_by_name_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNU tar first on PATH: it would shell out to a `zstd` that is not there."""
    monkeypatch.setattr(envpack.sys, "platform", "win32")
    monkeypatch.setattr(envpack.shutil, "which", lambda name: "C:\\msys\\tar.exe")
    monkeypatch.setattr(
        envpack.subprocess,
        "run",
        lambda *a, **k: _Completed(0, "tar (GNU tar) 1.32\n"),
    )
    with pytest.raises(PackError) as caught:
        envpack.require_zstd_tar()
    assert caught.value.code == "pack_no_zstd"
    # It names what it FOUND, so its reader can tell "wrong tar first on PATH"
    # from "no tar at all".
    assert "GNU tar" in caught.value.message
    assert "libzstd" in caught.value.message
    assert "System32" in caught.value.message


class _Completed:
    """The two fields of `CompletedProcess` that `_require_tar_with_zstd` reads."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ------------------------------------------------------------------ the CLI


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


def test_envpack_list_names_every_pack(
    viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["envpack", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {(row["name"], row["backend"]) for row in rows} == set(
        envpack.every_pack()
    )
    assert cli.main(["envpack", "list", "--backend", "mlx-darwin", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in rows} == {
        "server",
        "llm",
        "rvc",
        "tts",
        "align",
        "asr",
    }


def test_a_host_with_no_zstd_is_refused_before_anything_is_downloaded(
    home: Path,
    viable: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The tool check is FIRST, and the order is the whole point of it.

    Discovering there is no `zstd` after three gigabytes have arrived is a
    refusal that cost somebody their evening. `install_pack` asks before it
    reads the manifest, so the machine that cannot unpack a pack is told so
    while it has spent nothing.
    """
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        envpack.shutil, "which", lambda name: None if name == "zstd" else "/bin/tar"
    )
    missing = (tmp_path / "nowhere" / "envpacks.json").as_uri()
    assert cli.main(["install", "asr", "--manifest-url", missing]) == 1
    assert "pack_no_zstd" in capsys.readouterr().err


@needs_tar_zstd
def test_install_refuses_a_manifest_that_is_not_there(
    home: Path, viable: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default is a download, and its failure has a name. No quiet build."""
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    missing = (tmp_path / "nowhere" / "envpacks.json").as_uri()
    assert cli.main(["install", "asr", "--manifest-url", missing]) == 1
    error = capsys.readouterr().err
    assert "pack_manifest_unreadable" in error
    assert not (home / "envs" / "asr").exists()


@needs_tar_zstd
def test_install_refuses_a_pack_this_release_does_not_publish(
    home: Path, viable: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    empty = tmp_path / "envpacks.json"
    empty.write_text(
        PackManifest(version=cli.VERSION, packs=()).dumps(), encoding="utf-8"
    )
    assert cli.main(["install", "asr", "--manifest-url", empty.as_uri()]) == 1
    assert "pack_not_published" in capsys.readouterr().err


def test_a_type_with_no_recipe_gets_the_MOST_SPECIFIC_refusal_it_has_earned(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Why the answer is resolved from the outside in.

    Three sentences are true of `crucible install align` on `llama-windows`
    and only one of them helps. "There is no pack called 'align'" is a fact
    about a table. "No align env recipe for backend 'llama-windows'" is a
    fact about this checkout. `needs_wsl` is the fact about the MACHINE —
    PHASE15-HOST.md 3.5 and 7.4's item 4 — and it is the one with something
    the operator can do in it, so it is asked first and it is the same
    sentence the capability row carries (`capability.NEEDS_WSL_REASON`).
    """
    windows = replace(FAKE_MAC_BACKEND, kind="llama-windows", platform="win32")
    monkeypatch.setattr(cli, "detect_backend", lambda: windows)
    assert cli.main(["init", "--enable-align"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "align"]) == 1
    error = capsys.readouterr().err
    assert "needs_wsl" in error
    assert capability.NEEDS_WSL_REASON in error
    # Neither of the two less useful truths reaches the operator.
    assert "pack_unknown" not in error
    assert "recipe" not in error
