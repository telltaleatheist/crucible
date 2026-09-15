"""A job type's env — `~/.crucible/envs/<key>/` (PHASE2-LLM.md section 2).

One env per job type, never one giant env (DESIGN.md section 5). Each is a venv
built from the server's own interpreter, with a recipe (`envs/<job type>/
<name>.txt`) installed from PyPI into it. The engines are then started as
subprocesses of that venv's python — the server process itself never imports
torch, vLLM, mlx or narrator.

Why an env is not simply one per job type
-----------------------------------------
`llm` is: one venv, `~/.crucible/envs/llm/`, whichever backend the host is.
`tts` is not, and the reason is in narrator's dependency matrix rather than in
Crucible's design (PHASE3-TTS.md section 4). Narrator engines pin conflicting
serving stacks against conflicting torches, and installing two of them into one
env resolves torch twice and breaks whichever loses. So on `cuda-linux` the env
is named for the engine and the voice manifest's `narrator_engine` picks which
one a load uses, while on `mlx-darwin` the engines share one and the env is
named for the backend the way `llm`'s is.

Since Owen's ruling of 2026-09-14 there is exactly ONE narrator engine
(`voices.NARRATOR_ENGINE_SAMPLING` carries it), so `cuda-linux` has one tts env
today — `tts-higgs-v3`. The engine stays in the NAME rather than collapsing to
`tts`, because the whole point of the naming rule is that the second engine
needs a second directory and not a rebuild of the first.

So an env is named by an `EnvSpec`, and each job type states its own naming rule
in its own constructor below — `llm_env()` and `tts_env()` — where the two can be
read against each other.

This file was `crucible/llmenv.py` until the `tts` job type needed the same
machinery. Nothing about the `llm` env's layout, stamp or refusals changed in the
move.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import narratorpatches
from .errors import CrucibleError

RECIPES_DIR_ENV = "CRUCIBLE_RECIPES_DIR"

#: The file that says an env finished installing, whichever way it was
#: installed. ONE name for both doors — `install_env` below writes it after pip
#: returns 0, and `crucible/envpack.py` writes it into the `.partial` tree
#: before the rename — because `env_status` is the only reader and there must
#: not be two answers to "is this env there".
ENV_STAMP_NAME = "crucible-env.json"

#: What `crucible doctor` and `crucible install llm` report the version of. The
#: engine module of each backend, so a wrong-backend env is obvious at a glance.
BACKEND_HEADLINE_PACKAGE: dict[str, str] = {
    "cuda-linux": "vllm",
    "mlx-darwin": "mlx-lm",
}

#: `tts`'s headline is the same package on both backends, because narrator is
#: what the env exists for on either — the engine underneath it (SGLang-Omni on
#: `cuda-linux`, mlx-audio on `mlx-darwin`) is narrator's own dependency and not
#: Crucible's, and a doctor line naming it would be reporting a level down.
NARRATOR_PACKAGE = "narrator"

#: WHICH SERVING STACK EACH `cuda-linux` tts env STARTS, keyed by narrator
#: engine. Read off the recipes, not chosen here:
#:
#:   higgs-v3   `envs/tts/higgs-v3-cuda-linux.txt` installs `vllm==0.28.0` and
#:              `vllm-omni==0.28.0` and no SGLang at all, so the only stack
#:              narrator can start out of that env is vllm-omni. (narrator has
#:              a second one, `sglang-omni`, which BookForge's own WSL env
#:              serves; a Crucible env that installed it would be a different
#:              recipe and a different value here.)
#:
#: A TABLE OF ONE, keyed by engine on purpose (Owen's ruling of 2026-09-14
#: removed `orpheus`; see `voices.NARRATOR_ENGINE_SAMPLING`). An engine ABSENT
#: from this table is one that starts no server underneath narrator — the
#: lookup below is `.get()` for that reason, and `None` is the answer rather
#: than a missing key.
#
# RULING OWED: THIS REPO SAYS "SGLang-Omni" IN SEVEN PLACES AND INSTALLS
# vllm-omni. `docs/PHASE3-TTS.md` section 4, `crucible/residency.py`'s warm-up
# comment, `crucible/voices.py`'s own header and every voice manifest's
# `estimate_note` describe narrator as starting SGLang-Omni on `cuda-linux`;
# `envs/tts/higgs-v3-cuda-linux.txt` is a FROZEN, resolved set that installs
# `vllm==0.28.0` + `vllm-omni==0.28.0` and no SGLang at all. The recipe is what
# runs, so `vllm-omni` is what is stated here — that is the only reading under
# which this file cannot lie.
#
# WHAT IS OWED IS WHICH ONE OWEN WANTS. BookForge's own catalog shipped
# `stack: "sglang-omni"` on 2026-09-06 on measurements that favour it heavily
# (same 50 chunks, one seed: vllm-omni at 16 in flight = 4 early stops, 13/50
# damaged, 6 sustained voice switches, 10,752 chars/min; SGLang-Omni at 16 = 0,
# 5/50, 0, 26,666). If Crucible is to match that, the recipe changes and this
# table with it; if it is not, the prose above is stale and should be corrected
# rather than left to disagree. The numbers that survive either way are the
# memory estimates: SGLang at --mem-fraction-static 0.60 holds ~19 GB and
# vllm-omni at 0.35 + 0.10 measured 18.7-19.2 GB, so the manifests' 19 GB is
# right for the wrong reason and is not a hazard tonight.
#
# ── IT IS ALSO THE ANSWER TO A QUESTION OWEN ASKED ON 2026-09-15 ─────────────
#
# "is wsl crucible using sglang with batching set to exactly what it was before
# we set up crucible?" — and the two halves of that have different answers, so
# they are written down separately rather than averaged into one.
#
# THE BATCHING: YES. `HIGGS_MAX_NUM_SEQS` is stage 0's admission width AND the
# width of narrator's own batch on BOTH stacks (`v3_served.serve_concurrency`,
# which `sgl_served.py` deliberately shares rather than naming a second
# variable). BookForge states 16 from its catalog; every voice manifest here
# states 16 from `[voice.serving]`, carrying BookForge's own measurement note
# verbatim. The number was ported, not re-derived, and nothing on this arm ever
# ran at a width nobody chose.
#
# THE STACK: NO. This is the OTHER one. On the shipped catalog BookForge renders
# Higgs on SGLang-Omni and has since 2026-09-06; a Crucible `tts` job on
# `cuda-linux` renders on vllm-omni, whose measured cost on the same 50 chunks
# is the line above — 4 early stops, 13/50 damaged and 6 sustained voice
# switches against 0, 5/50 and 0, at 40% of the throughput. That is a quality
# difference and not only a speed one, and it is the largest single divergence
# between the two narrator seams.
#
# TWO THINGS HAVE TO MOVE TOGETHER TO CLOSE IT, which is why it is a ruling and
# not a one-word edit here. The recipe must install the `sglomni` stack (python
# 3.12 + torch 2.13.0+cu130 + sglang-omni 0.1.4 — a SEPARATE env from
# vllm-omni's python 3.11 + vllm 0.28.0, which is why BookForge keeps two), and
# narrator must ship `serve_higgs_sgl.sh` the way it now ships
# `serve_higgs_v3.sh`: today that launcher exists only in BookForge's
# `electron/scripts/higgs/`, so `NARRATOR_HIGGS_SGL_SERVE_SCRIPT` would have to
# name a path into a BookForge checkout — the exact dependency BookForge
# 0eeb0267 removed for the other stack. The three `HIGGS_SGL_*` knobs are NOT
# the blocker: that script defaults `HIGGS_SGL_MEM_FRACTION` to 0.60,
# `HIGGS_SGL_MAX_NEW_TOKENS` to 7500 and `HIGGS_SGL_CUDA_GRAPH_MAX_BS` to
# `$HIGGS_MAX_NUM_SEQS` itself, which are the catalog's three values, so graphs
# would be captured at exactly the admitted width without Crucible saying a
# word.
CUDA_LINUX_SERVING_STACK: dict[str, str] = {
    "higgs-v3": "vllm-omni",
}


class EnvError(CrucibleError):
    """A job type's env is missing, or could not be built. Carries the reason."""


@dataclass(frozen=True)
class EnvSpec:
    """Which env, and which recipe builds it.

    `key` is the directory under `~/.crucible/envs/`; `recipe_name` is the
    `<name>.txt` inside `envs/<job_type>/`. Two fields rather than one because
    they answer two different questions — what is installed here, and what
    installs it — and `tts` on `cuda-linux` is where they differ.
    """

    job_type: str
    key: str
    recipe_name: str
    headline: str
    #: WHICH SERVING STACK narrator will start UNDERNEATH ITSELF out of this
    #: env, or None where it starts no server at all. `None` is not "unknown":
    #: it means this env's engine renders IN PROCESS (the Mac's mlx-audio) or
    #: has no stack concept (an engine that loads its own runtime).
    #:
    #: IT BELONGS TO THE RECIPE, which is why it is here rather than in the
    #: voice manifest. A Higgs v3 voice does not choose vllm-omni over
    #: SGLang-Omni — `higgs-v3-cuda-linux.txt` does, by installing
    #: `vllm-omni==0.28.0` and nothing else. narrator refuses by name when
    #: `HIGGS_STACK` is unset (`served_common.serving_stack`: the two stacks
    #: place sampling differently and size the frame cap against different
    #: context windows, so a guessed stack is a book rendered at sampling
    #: nobody chose), and this is the fact Crucible states it from.
    serving_stack: str | None = None


def llm_env(backend_kind: str) -> EnvSpec:
    """The one `llm` env. One per host, whichever backend it is."""
    if backend_kind not in BACKEND_HEADLINE_PACKAGE:
        raise EnvError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_HEADLINE_PACKAGE)}"
        )
    return EnvSpec(
        job_type="llm",
        key="llm",
        recipe_name=backend_kind,
        headline=BACKEND_HEADLINE_PACKAGE[backend_kind],
    )


def tts_env(narrator_engine: str, backend_kind: str) -> EnvSpec:
    """The `tts` env this narrator engine runs in on this backend.

    On `cuda-linux` two narrator engines cannot share a venv (see the module
    docstring), so the engine is in the env's name and in the recipe's. On
    `mlx-darwin` they can, so there is one env and one recipe, named for the
    backend the way `llm`'s are.
    """
    if backend_kind not in BACKEND_HEADLINE_PACKAGE:
        raise EnvError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_HEADLINE_PACKAGE)}"
        )
    if backend_kind == "cuda-linux":
        return EnvSpec(
            job_type="tts",
            key=f"tts-{narrator_engine}",
            recipe_name=f"{narrator_engine}-{backend_kind}",
            headline=NARRATOR_PACKAGE,
            serving_stack=CUDA_LINUX_SERVING_STACK.get(narrator_engine),
        )
    # mlx-darwin: NO SERVING STACK, and that is a fact about narrator rather
    # than a gap here. On darwin `narrator.engine.registry` builds
    # `HiggsV3MlxEngine` from `HiggsV3MlxConfig`, and neither reads
    # `HIGGS_STACK` — `serving_stack()` is called only by the SERVED arm's
    # `HiggsV3Engine.__post_init__` and its `detect_backend()`, while the MLX
    # class's `detect_backend()` returns 'mlx' off an import. Setting the
    # variable there would be a lever read by nothing, which is how a Mac spawn
    # ends up looking like a served one (BookForge's `higgsSpawnEnv` refuses
    # that shape by name for the same reason).
    return EnvSpec(
        job_type="tts",
        key="tts",
        recipe_name=backend_kind,
        headline=NARRATOR_PACKAGE,
    )


@dataclass(frozen=True)
class EnvStatus:
    """What `crucible doctor` prints for one env."""

    installed: bool
    path: Path
    detail: str
    python_version: str | None
    packages: dict[str, str]
    #: The sha256 of the PACK this env was unpacked from, or None when it was
    #: built here by `crucible install --build`. Not a second way of saying
    #: "installed": it says WHICH WAY, and `crucible doctor` prints it beside
    #: the recipe hash so an env that came off a release and an env somebody
    #: built at 2am are distinguishable without reading a JSON file.
    pack_sha256: str | None = None
    #: The sha256 of the recipe this env was installed from, as recorded at
    #: install time. `doctor` compares it against the recipe on disk NOW.
    recipe_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
            "pack_sha256": self.pack_sha256,
            "recipe_sha256": self.recipe_sha256,
        }


# ------------------------------------------------------------------- layout


def env_dir(home: Path, spec: EnvSpec) -> Path:
    return home / "envs" / spec.key


def env_python(home: Path, spec: EnvSpec) -> Path:
    """The venv interpreter the engines are spawned from."""
    return env_dir(home, spec) / "bin" / "python"


def stamp_path(home: Path, spec: EnvSpec) -> Path:
    """Written only after `pip install -r <recipe>` returns 0, or by a pack.

    Public because `crucible/envpack.py` is the second writer and the one place
    that must not guess this path.
    """
    return env_dir(home, spec) / ENV_STAMP_NAME


def recipes_dir(job_type: str) -> Path:
    """Where `envs/<job_type>/*.txt` live. Refuses by name if absent.

    `$CRUCIBLE_RECIPES_DIR`, when set, is the recipe ROOT and the job type is a
    directory under it — one variable for every job type, rather than one per
    type, which could point two halves of a build at two checkouts.
    """
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
        if not root.is_dir():
            raise EnvError(f"{RECIPES_DIR_ENV}={override!r} is not a directory")
        path = root / job_type
        if not path.is_dir():
            raise EnvError(
                f"{RECIPES_DIR_ENV}={override!r} holds no {job_type!r} directory"
            )
        return path
    path = Path(__file__).resolve().parent / "envs" / job_type
    if not path.is_dir():
        raise EnvError(
            f"no {job_type} env recipes at {path}; they are package data and "
            f"this install has lost them, or ${RECIPES_DIR_ENV} must point at them"
        )
    return path


def recipe_for(spec: EnvSpec) -> Path:
    """The recipe that builds this env, or a named refusal."""
    root = recipes_dir(spec.job_type)
    path = root / f"{spec.recipe_name}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.txt"))
        raise EnvError(
            f"no {spec.job_type} env recipe for {spec.recipe_name!r} at {path}; "
            f"this build ships recipes for {available}"
        )
    return path


#: A PEP 508 direct reference — `name @ url`, optionally with extras. This is how
#: `envs/tts/` pins narrator, which is not on PyPI: it lives in the BookForge
#: repo and is versioned with the app (PHASE3-TTS.md section 4 calls extracting
#: it an owed ruling for Owen). A `name==version` pin cannot express a git sha,
#: and `narrator==0.1.0` would be a pin that lets any commit through.
_DIRECT_REFERENCE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*@\s*(?P<url>\S+)\s*$"
)

#: The commit a direct reference names, taken from the `@<sha>` a pip VCS URL
#: puts after the repository and before any `#fragment`.
_VCS_COMMIT = re.compile(r"@(?P<sha>[0-9a-f]{40})(?:#|$)")


def _requirement_lines(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        yield stripped


def recipe_pins(path: Path) -> dict[str, str]:
    """The `name==version` pins in a recipe, by lower-cased name.

    Direct references are **skipped here and checked by
    `recipe_direct_references`** rather than refused: they are exact pins too,
    just of a commit rather than a version, and `pip list` reports the package's
    own metadata version for one, which would never match the sha. A line that is
    neither shape is still refused — every requirement in a recipe is pinned.
    """
    pins: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        if _DIRECT_REFERENCE.match(stripped):
            continue
        name, separator, version = stripped.partition("==")
        if separator != "==":
            raise EnvError(
                f"{path.name}: {stripped!r} is not a `name==version` pin or a "
                "`name @ url` direct reference; every requirement in a recipe is "
                "pinned exactly"
            )
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def recipe_direct_references(path: Path) -> dict[str, str]:
    """The commit each `name @ url` line pins, by lower-cased name.

    A direct reference whose URL carries no 40-character commit is refused: a
    branch or a tag is a moving target, and an env built from one cannot be said
    to match the recipe that built it.
    """
    references: dict[str, str] = {}
    for stripped in _requirement_lines(path):
        match = _DIRECT_REFERENCE.match(stripped)
        if match is None:
            continue
        commit = _VCS_COMMIT.search(match.group("url"))
        if commit is None:
            raise EnvError(
                f"{path.name}: {stripped!r} names no commit. A direct reference "
                "is pinned by `@<40-character sha>` before any `#fragment`; a "
                "branch name is not a pin"
            )
        name = match.group("name").strip().lower().replace("_", "-")
        references[name] = commit.group("sha")
    return references


def installed_direct_references(home: Path, spec: EnvSpec) -> dict[str, str]:
    """What commit each VCS-installed package in this venv actually came from.

    PEP 610: pip writes `direct_url.json` beside a distribution's metadata when
    it was installed from a URL rather than an index, and for a VCS install that
    file carries `vcs_info.commit_id` — the commit pip actually resolved. That is
    the only place the sha survives; `pip list` reports the package's declared
    version, which does not move when the commit does.
    """
    root = env_dir(home, spec) / "lib"
    found: dict[str, str] = {}
    if not root.is_dir():
        return found
    for record in root.glob("python*/site-packages/*.dist-info/direct_url.json"):
        try:
            document = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvError(f"could not read {record}: {exc}") from None
        commit = document.get("vcs_info", {}).get("commit_id")
        if not commit:
            continue
        name = record.parent.name.split("-")[0].lower().replace("_", "-")
        found[name] = commit
    return found


# ------------------------------------------------------------------- status


def installed_packages(home: Path, spec: EnvSpec) -> dict[str, str]:
    """`pip list` from this venv, by lower-cased name. {} if there is no venv."""
    python = env_python(home, spec)
    if not python.is_file():
        return {}
    completed = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise EnvError(
            f"`pip list` in {env_dir(home, spec)} exited {completed.returncode}: "
            f"{completed.stderr.strip() or 'no output'}"
        )
    return {
        entry["name"].lower().replace("_", "-"): entry["version"]
        for entry in json.loads(completed.stdout)
    }


def env_status(home: Path, spec: EnvSpec, backend_kind: str) -> EnvStatus:
    """Is this env there, and does it hold what the recipe pins?"""
    directory = env_dir(home, spec)
    python = env_python(home, spec)
    install = f"crucible install {spec.job_type}"
    if not python.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"no venv at {directory} — run `{install}`",
            python_version=None,
            packages={},
        )
    stamp = stamp_path(home, spec)
    if not stamp.is_file():
        return EnvStatus(
            installed=False,
            path=directory,
            detail=(
                f"{directory} exists but {stamp.name} does not: the last "
                f"`{install}` did not finish. Re-run it."
            ),
            python_version=None,
            packages={},
        )
    record = json.loads(stamp.read_text(encoding="utf-8"))
    # `.get` and not `[]` for these two alone: every stamp written before 0.6.0
    # predates env packs and carries neither key. Absent means "built here,
    # before anything recorded which recipe bytes it was built from", which is
    # exactly what `doctor` prints — not a default standing in for a fact.
    pack_sha256 = record.get("pack_sha256")
    recipe_sha256 = record.get("recipe_sha256")
    if record["backend"] != backend_kind:
        return EnvStatus(
            installed=False,
            path=directory,
            detail=(
                f"{directory} was installed for backend {record['backend']!r}, this "
                f"host is {backend_kind!r} — run `{install} --force`"
            ),
            python_version=record["python_version"],
            packages={},
            pack_sha256=pack_sha256,
            recipe_sha256=recipe_sha256,
        )

    present = installed_packages(home, spec)
    recipe = recipe_for(spec)
    pins = recipe_pins(recipe)
    wrong = sorted(
        f"{name} is {present.get(name, 'absent')}, recipe pins {version}"
        for name, version in pins.items()
        if present.get(name) != version
    )
    # A direct reference is checked against the COMMIT pip recorded, not against
    # a version: `narrator` is installed from a git sha and its metadata version
    # does not move when the sha does, so a version check here would call an env
    # built from last month's commit a match.
    built_from = installed_direct_references(home, spec)
    wrong += sorted(
        f"{name} was installed from "
        f"{built_from.get(name, 'no recorded commit')}, recipe pins {commit}"
        for name, commit in recipe_direct_references(recipe).items()
        if built_from.get(name) != commit
    )
    if wrong:
        return EnvStatus(
            installed=False,
            path=directory,
            detail=f"{directory} does not match {recipe.name}: " + "; ".join(wrong),
            python_version=record["python_version"],
            packages=present,
            pack_sha256=pack_sha256,
            recipe_sha256=recipe_sha256,
        )
    return EnvStatus(
        installed=True,
        path=directory,
        detail=(
            f"{spec.headline} {present[spec.headline]}, python "
            f"{record['python_version']}, {len(present)} packages"
        ),
        python_version=record["python_version"],
        packages=present,
        pack_sha256=pack_sha256,
        recipe_sha256=recipe_sha256,
    )


def require_env(home: Path, spec: EnvSpec, backend_kind: str) -> Path:
    """This venv's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, spec, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return env_python(home, spec)


# ------------------------------------------------------------------ install


def install_env(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Any = None,
) -> EnvStatus:
    """Create `~/.crucible/envs/<key>/` and install this env's recipe.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises EnvError naming what went wrong.
    """
    recipe = recipe_for(spec)
    directory = env_dir(home, spec)
    stamp = stamp_path(home, spec)

    if directory.exists() and not force:
        existing = env_status(home, spec, backend_kind)
        if existing.installed:
            return existing
        if stamp.is_file():
            raise EnvError(
                f"{directory} exists but does not match this host: {existing.detail}. "
                "Pass --force to rebuild it."
            )
        # A half-built venv from an interrupted install: no stamp, so nothing
        # downstream has ever trusted it. Rebuilding it is the only correct move.

    started = time.monotonic()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        shutil.rmtree(directory)

    _run(
        [sys.executable, "-m", "venv", str(directory)],
        f"could not create the venv at {directory}",
        on_line,
    )
    python = env_python(home, spec)
    if not python.is_file():
        raise EnvError(
            f"`python -m venv {directory}` returned 0 but there is no {python}"
        )
    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        f"could not upgrade pip in the {spec.job_type} env",
        on_line,
    )
    _run(
        [str(python), "-m", "pip", "install", "-r", str(recipe)],
        f"could not install {recipe} into {directory}",
        on_line,
    )

    # THE TWO SITE-PACKAGES PATCHES pip CANNOT EXPRESS, RE-APPLIED HERE.
    #
    # pip has just written vllm-omni's own `higgs_audio_v3.py` over the edit
    # narrator's sentinel proof reads, which is what made every Higgs load on
    # owens-pc fail from 07:46 on 2026-09-15 after a `--build --force` at 07:34
    # — the proof found a 0-byte report because the code that writes records was
    # gone. Before this call the recipe said the patches "must be re-applied"
    # and named nobody to do it; `crucible doctor` then reported them `missing`
    # from a command nobody runs after an install.
    #
    # ONLY FOR `tts`. Both patches edit the vLLM stack, and the `llm` env pins
    # `vllm` too — patching an LLM server's input processor to admit token -100
    # is not a thing anyone asked for. `narratorpatches` then selects again by
    # the recipe's own pins, so `mlx-darwin`'s tts env (no vllm, no vllm-omni)
    # runs neither and is not called broken for it.
    #
    # BEFORE THE STAMP, and it raises: an env that is stamped installed is an
    # env whose patches are in, or there is no stamp.
    if spec.job_type == "tts":
        try:
            narratorpatches.apply(
                directory, python, recipe_pins(recipe), on_line=on_line
            )
        except narratorpatches.PatchError as exc:
            # Re-raised as this module's error so the CLI refuses by name
            # rather than showing a traceback. No stamp has been written, so
            # the env this leaves behind is one nothing downstream trusts.
            raise EnvError(str(exc)) from exc

    version = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    elapsed = time.monotonic() - started
    stamp.write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": recipe.name,
                # Recorded even on the `--build` path, so `doctor` can say a
                # locally built env no longer matches the recipe bytes it was
                # built from — the same question a pack answers with
                # `pack_recipe_drift`, asked of an env nobody downloaded.
                "recipe_sha256": recipe_sha256(recipe),
                "python_version": version,
                # A venv-built env has NO pack sha, and the key is written as
                # null rather than left out: "built here" is an answer.
                "pack_sha256": None,
                "seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return env_status(home, spec, backend_kind)


#: The one line-ending rule. A recipe is a TEXT declaration, so a CR before a
#: LF is an artefact of the checkout the file arrived in and never a fact about
#: what the env contains.
_CRLF = b"\r\n"
_LF = b"\n"


def recipe_sha256(path: Path) -> str:
    """The recipe's SHA-256 over LINE-ENDING-NORMALISED bytes.

    What ties an env — and a pack — to the declaration that built it. **THE ONE
    IMPLEMENTATION:** `envpack.build_pack`/`check_recipe` (the `recipe_sha256`
    column of `envpacks.json`), `workerenv`'s env stamp and `crucible doctor`'s
    drift line all come here, because a fact with two owners is a fact that
    will eventually disagree with itself, and this one already did.

    MEASURED 2026-09-15, which is why the normalisation is here at all: the
    same commit of `pyproject.toml` hashed to `1ab85cc3…` from the main
    checkout and `cc4fda38…` from a worktree of that SAME commit, while
    `git hash-object` said both were blob `5ef53a3`. The difference was CRLF
    versus LF — this machine has `core.autocrlf=true` and the working file
    predates the repo's `.gitattributes` — and the consequence is a false
    alarm in both directions: `crucible envpack build <name> --check` refusing
    a pack that is perfectly correct, and a Linux CI runner and a Windows desk
    disagreeing about a manifest neither of them is wrong about.

    **NORMALISE, DO NOT HASH THE GIT BLOB.** `git hash-object` would be the
    exact answer for a recipe in a checkout and NO answer at all for the case
    that matters most: `crucible/envs/*.txt` ship inside the installed wheel,
    where there is no repository, no index and no `git` to ask — and
    `check_recipe()` runs on an operator's machine against precisely that copy.
    A digest that needed a checkout would turn `pack_recipe_drift` into
    `git-not-found` on every machine that is not a developer's.

    ONLY CRLF → LF. A lone `\\r` is not a line ending any of these toolchains
    writes, so it stays and counts as content; every real edit — a version
    pinned differently, a package added, a line removed — still changes the
    digest, because normalising a line ENDING cannot erase what is on the line.

    Read whole rather than in chunks: a recipe is a few KB of text (the largest
    is under 4 KB), and a chunked reader would have to carry a CR across every
    boundary to get the same answer.
    """
    return hashlib.sha256(path.read_bytes().replace(_CRLF, _LF)).hexdigest()


def _run(command: list[str], failure: str, on_line: Any) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-40]
        if on_line is not None:
            on_line(line)
    code = process.wait()
    if code != 0:
        raise EnvError(f"{failure}: `{' '.join(command)}` exited {code}\n" + "\n".join(tail))
