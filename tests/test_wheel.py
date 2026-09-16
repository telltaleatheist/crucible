"""The wheel carries everything a server reads. Built here, not assumed.

THE DEFECT THIS EXISTS FOR, measured 2026-09-14: `pyproject.toml`'s
`packages.find = ["crucible*"]` shipped the package and `crucible/ui/` and
nothing else, while `models/ voices/ denoise/ rvc/ rvcbase/ align/ asr/ envs/`
sat BESIDE the package in the checkout. Every developer install is
`pip install -e .`, which makes the checkout the install and the gap
invisible; the first machine to meet a real wheel — the Mac, which unpacks
the phase-14 `server` pack — answered `catalog_unreadable` and
`denoise_manifests_unreadable`. The pack's own smoke test is
`crucible --version`, which does not read a manifest, so nothing noticed.

So this file BUILDS A WHEEL and looks inside it, and then INSTALLS one in a
throwaway venv and calls the readers. Not a glob over the source tree and not
a read of `pyproject.toml`: a glob that silently matches nothing is precisely
how this was broken, and a declaration is not a file.

It SKIPS by name when `python -m build` is not importable, because that is a
missing developer tool and not a defect in the package — CI installs it.
"""

from __future__ import annotations

import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every directory the server READS at run time, with one file each that the
#: reader owning it needs. One row per resolver in `crucible/*.py`:
#: `manifests_dir`, `voices_dir`, `denoise_manifests_dir`,
#: `rvc_manifests_dir`, `rvc_base_dir`, `align_manifests_dir`,
#: `asr_manifests_dir`, `recipes_dir`, and the operator page.
REQUIRED_TREES: tuple[tuple[str, str], ...] = (
    ("crucible/models", "qwen3.5-9b.toml"),
    ("crucible/voices", "higgs-default.toml"),
    ("crucible/denoise", "denoise-roformer.toml"),
    ("crucible/rvc", "sigma.toml"),
    ("crucible/rvcbase", "ultimate-rvc.toml"),
    ("crucible/align", "qwen3-aligner.toml"),
    ("crucible/asr", "faster-whisper-large-v3.toml"),
    ("crucible/envs", "llm/cuda-linux.txt"),
    ("crucible/ui", "index.html"),
)


def _require_build() -> None:
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip(
            "`python -m build` is not installed in this interpreter. The wheel "
            "test needs it and CI installs it; this is a missing developer "
            "tool, not a defect in the package"
        )


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One wheel, built once for this file. `--no-isolation`: no network."""
    _require_build()
    out = tmp_path_factory.mktemp("wheel")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "the wheel would not build:\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_names(built_wheel: Path) -> list[str]:
    with zipfile.ZipFile(built_wheel) as bundle:
        return bundle.namelist()


def test_core_wheel_installs_mlx_for_apple_silicon_backend_detection(built_wheel: Path) -> None:
    from email import message_from_bytes
    from packaging.requirements import Requirement
    with zipfile.ZipFile(built_wheel) as bundle:
        metadata = next(name for name in bundle.namelist() if name.endswith(".dist-info/METADATA"))
        requirements = [Requirement(value) for value in message_from_bytes(bundle.read(metadata)).get_all("Requires-Dist", [])]
    mlx = next(requirement for requirement in requirements if requirement.name == "mlx")
    assert mlx.marker is not None
    assert mlx.marker.evaluate({"sys_platform": "darwin", "platform_machine": "arm64"})
    for platform, machine in [("win32", "AMD64"), ("linux", "x86_64"), ("darwin", "x86_64")]:
        assert not mlx.marker.evaluate({"sys_platform": platform, "platform_machine": machine})


def test_every_directory_the_server_reads_is_inside_the_wheel(
    wheel_names: list[str],
) -> None:
    """The whole of the 2026-09-14 defect, as one row per reader."""
    missing: list[str] = []
    for directory, example in REQUIRED_TREES:
        if not any(name.startswith(f"{directory}/") for name in wheel_names):
            missing.append(f"{directory}/ (nothing at all)")
            continue
        wanted = f"{directory}/{example}"
        if wanted not in wheel_names:
            missing.append(wanted)
    assert not missing, (
        "a wheel built from this checkout is missing what the server reads: "
        + ", ".join(missing)
        + ". These are package data UNDER `crucible/`; a directory beside the "
        "package is not shipped however `package-data` declares it."
    )


def test_every_manifest_in_the_checkout_is_in_the_wheel(
    wheel_names: list[str],
) -> None:
    """Not a sample. A row added to `models/` and not shipped is the same bug."""
    inside = set(wheel_names)
    missing: list[str] = []
    for directory, _example in REQUIRED_TREES:
        source = REPO_ROOT / directory
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            name = path.relative_to(REPO_ROOT).as_posix()
            if name not in inside:
                missing.append(name)
    assert not missing, f"in the checkout and not in the wheel: {missing}"


def test_the_recipes_every_published_pack_builds_from_travel(
    wheel_names: list[str],
) -> None:
    """`envpack build` reads `envs/`, and a pack is built from an install."""
    from crucible import envpack, workerenv

    wanted: set[str] = set()
    for pack, backend_kind in envpack.every_pack():
        if pack in (envpack.SERVER_PACK, envpack.HOST_PACK):
            # `pyproject.toml` is their recipe, and that is the wheel's own
            # metadata rather than a file inside it.
            continue
        if pack in workerenv.WORKER_JOB_TYPES:
            recipe = workerenv.recipe_for(pack, backend_kind)
        else:
            recipe = envpack.pack_target(pack, backend_kind).recipe
        wanted.add(recipe.resolve().relative_to(REPO_ROOT).as_posix())
    missing = sorted(name for name in wanted if name not in wheel_names)
    assert not missing, f"recipes not in the wheel: {missing}"


def test_the_wheel_is_not_only_python_files(wheel_names: list[str]) -> None:
    """The SHAPE of the bug, so a future `packages.find` cannot repeat it.

    A wheel that is `.py` files and metadata and nothing else is exactly what
    shipped; this is the assertion that fails the instant that happens again,
    whatever the reason.
    """
    data = [
        name
        for name in wheel_names
        if name.startswith("crucible/")
        and not name.endswith(".py")
        and not name.endswith("/")
    ]
    assert len(data) > 20, (
        f"only {len(data)} non-Python files under crucible/ in the wheel; the "
        "manifests, the recipes and the operator page are all package data "
        "and all of them must travel"
    )


def test_an_INSTALLED_wheel_reads_its_own_catalog(
    built_wheel: Path, tmp_path: Path
) -> None:
    """The end-to-end half: a venv, the wheel, and the readers that broke.

    `pip install -e .` cannot prove this — under it the checkout IS the
    install, which is the whole reason nobody noticed for a year. So the
    wheel goes into a throwaway venv and the eight loaders are called there,
    with a working directory that is not this checkout.
    """
    environment = tmp_path / "venv"
    venv.create(environment, with_pip=True, symlinks=True)
    python = environment / "bin" / "python"
    if not python.exists():
        python = environment / "Scripts" / "python.exe"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-input", str(built_wheel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        pytest.skip(
            "the wheel's dependencies could not be installed in a fresh venv "
            f"(offline?): {install.stderr.strip()[:300]}"
        )
    probe = (
        "from crucible.manifests import load_all_manifests;"
        "from crucible.voices import load_all_voices;"
        "from crucible.denoisemodels import load_all_denoise_manifests;"
        "from crucible.rvcmodels import load_all_rvc_manifests;"
        "from crucible.rvcbase import load_rvc_base;"
        "from crucible.alignmodels import load_all_align_manifests;"
        "from crucible.asrmodels import load_all_asr_manifests;"
        "from crucible.jobenv import recipes_dir;"
        "print(len(load_all_manifests()), len(load_all_voices()),"
        " len(load_all_denoise_manifests()), len(load_all_rvc_manifests()),"
        " load_rvc_base().id, len(load_all_align_manifests()),"
        " len(load_all_asr_manifests()), recipes_dir('llm').is_dir())"
    )
    ran = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert ran.returncode == 0, (
        "an installed wheel cannot read its own catalog:\n" + ran.stderr
    )
    assert ran.stdout.split()[-1] == "True", ran.stdout
