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
from pathlib import Path
from typing import Any

import pytest

from crucible import cli, envpack, jobenv, workerenv
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
    # No `asr` and no `align` on the Mac, and those are facts rather than gaps:
    # `envs/asr/mlx-darwin.md` is prose explaining that CTranslate2 has no
    # Metal backend, and there is no `.txt` for either.
    assert mac == ["llm", "rvc", "server", "tts"]


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


def test_every_pack_is_ten_rows_across_two_backends() -> None:
    rows = envpack.every_pack()
    assert len(rows) == len(set(rows)) == 10


def test_a_pack_nobody_publishes_is_refused_by_name() -> None:
    with pytest.raises(PackError) as caught:
        envpack.pack_target("asr", "mlx-darwin")
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
    assert {row["name"] for row in rows} == {"server", "llm", "rvc", "tts"}


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


def test_the_mac_still_gets_the_recipe_refusal_not_a_pack_one(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Why the recipe is resolved before the pack table is asked.

    "There is no pack called 'align'" is true and useless; "CTranslate2 has no
    Metal backend, this build ships cuda-linux" is the answer.
    """
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-align"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "align"]) == 1
    error = capsys.readouterr().err
    assert "no align env recipe for backend 'mlx-darwin'" in error
    assert "pack_unknown" not in error
