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
# THE RULING LANDED, AND THE PROSE WAS RIGHT ALL ALONG. Until 2026-09-15 this
# repo said "SGLang-Omni" in seven places and INSTALLED vllm-omni:
# `docs/PHASE3-TTS.md` section 4, `crucible/residency.py`'s warm-up comment,
# `crucible/voices.py`'s own header and every voice manifest's `estimate_note`
# described narrator as starting SGLang-Omni on `cuda-linux`, while
# `envs/tts/higgs-v3-cuda-linux.txt` installed `vllm==0.28.0` + `vllm-omni==
# 0.28.0` and no SGLang at all. This table stated `vllm-omni`, because the
# RECIPE is what runs and that was the only reading under which this file could
# not lie. The recipe now installs the stack the prose always claimed, so the
# two agree by being made to agree rather than by one of them being softened.
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
# ── OWEN RULED ON 2026-09-15, AND THE ANSWER IS SGLang ──────────────────────
#
# He was asked which one he wanted, having been shown the measurement above:
#
#   "we dont use vllm-omni. we use sglang. vllm-omni doesnt work for higgs."
#
# So this is not a preference between two working stacks. vllm-omni is BROKEN
# for Higgs — its batched talker corrupts the newest batch row, which is the
# truncations, the gibberish and the sustained voice switches all at once — and
# a recipe that serves Higgs on it is a supported way to render a damaged book.
# The recipe is REPLACED rather than kept beside a second one, and this table
# says the stack that recipe installs.
#
# ── WHAT THE OTHER HALF OF THE QUESTION WAS, AND ITS ANSWER ─────────────────
#
# "is wsl crucible using sglang with batching set to exactly what it was before
# we set up crucible?" The two halves had different answers and both are worth
# keeping now that one of them has been fixed.
#
# THE BATCHING: YES, IT ALWAYS WAS. `HIGGS_MAX_NUM_SEQS` is stage 0's admission
# width on vllm-omni, `--tts_engine.factory.max_running_requests` on SGLang, AND
# the width of narrator's own batch on both (`v3_served.serve_concurrency`,
# which `sgl_served.py` deliberately shares rather than naming a second
# variable). BookForge states 16 from its catalog; every voice manifest here
# states 16 from `[voice.serving]`, carrying BookForge's own measurement note
# verbatim. That number was ported, not re-derived, and nothing on this arm ever
# ran at a width nobody chose.
#
# THE STACK: NO, AND THAT IS WHAT THIS CHANGE FIXES. Crucible rendered on
# vllm-omni for the nine days between BookForge's flip and this ruling.
#
# THE THREE `HIGGS_SGL_*` KNOBS ARE STILL UNSET HERE AND STILL INERT, for the
# reason the `HIGGS_*` table in `engines/narrator.py` gives about its own six:
# `serve_higgs_sgl.sh` defaults `HIGGS_SGL_MEM_FRACTION` to 0.60,
# `HIGGS_SGL_MAX_NEW_TOKENS` to 7500 and `HIGGS_SGL_CUDA_GRAPH_MAX_BS` to
# `$HIGGS_MAX_NUM_SEQS` ITSELF — never sglang's own default — and those are the
# catalog's three values. So CUDA graphs are captured at exactly the admitted
# width without Crucible saying a word.
CUDA_LINUX_SERVING_STACK: dict[str, str] = {
    "higgs-v3": "sglang-omni",
}

#: THE INTERPRETER AN ENV MUST BE BUILT WITH, where that is not the server's own.
#:
#: `install_env` builds a venv from `sys.executable` — the interpreter the
#: Crucible server itself runs on, 3.11.16 on owens-pc — and for every env but
#: one that is right. The SGLang-Omni `tts` env is the exception: sglang-omni
#: 0.1.4 pulls torch 2.13.0+cu130 and flashinfer against PYTHON 3.12, and
#: BookForge builds it as a separate conda env for the same reason.
#:
#: A TABLE KEYED BY RECIPE, because the requirement belongs to what is installed
#: rather than to the job type or the backend: `higgs-v3-cuda-linux.txt` needs
#: 3.12 today and a future recipe for the same job type may not.
#:
#: Absent means "the server's own interpreter", which is a real answer and the
#: one every other env gives.
RECIPE_PYTHON: dict[str, str] = {
    "higgs-v3-cuda-linux": "3.12",
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
    #: `major.minor` the env must be BUILT with, or None for the server's own
    #: interpreter. From `RECIPE_PYTHON`, keyed by the recipe — see that table.
    #:
    #: NOT A PREFERENCE. An env built at the wrong version does not install
    #: wrongly, it fails to install at all (there is no torch 2.13.0+cu130 wheel
    #: for 3.11 on this axis), and it fails several GB in. `install_env` refuses
    #: BY NAME before `venv` runs instead.
    python_version: str | None = None


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
        recipe_name = f"{narrator_engine}-{backend_kind}"
        return EnvSpec(
            job_type="tts",
            key=f"tts-{narrator_engine}",
            recipe_name=recipe_name,
            headline=NARRATOR_PACKAGE,
            serving_stack=CUDA_LINUX_SERVING_STACK.get(narrator_engine),
            python_version=RECIPE_PYTHON.get(recipe_name),
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
    #: The recipe's TEXT as installed, line-ending-normalised. None for an env
    #: stamped before this was recorded.
    #:
    #: A hash says THAT a recipe moved and can never say WHAT moved, and those
    #: are different questions with different remedies: a changed comment costs
    #: nothing, a re-pinned package is already checked package-by-package by
    #: `env_status`, and a changed `--index-url` silently swaps the wheel a pin
    #: resolves to. Keeping the bytes is what lets `install_env` tell the three
    #: apart instead of sending every one of them to a multi-GB rebuild.
    recipe_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
            "pack_sha256": self.pack_sha256,
            "recipe_sha256": self.recipe_sha256,
            # The text itself is deliberately NOT in `to_dict`: this feeds
            # `crucible doctor --json`, and several KB of recipe per env would
            # bury the report it is part of. What the text is FOR is the
            # comparison in `install_env`, which reads the stamp directly.
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
    # Absent for every env stamped before the text was recorded, and that
    # absence is load-bearing rather than tidy-uppable: it is exactly the case
    # `install_env` cannot prove anything about and refuses by name.
    recipe_text = record.get("recipe_text")
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
            recipe_text=recipe_text,
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
            recipe_text=recipe_text,
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
        recipe_text=recipe_text,
    )


def require_env(home: Path, spec: EnvSpec, backend_kind: str) -> Path:
    """This venv's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, spec, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return env_python(home, spec)


# ------------------------------------------------------------------ install


def interpreter_for(spec: EnvSpec) -> str:
    """The python that builds this env's venv, or a refusal naming the version.

    TWO SOURCES, BOTH CHECKED, NEITHER A FALLBACK FOR THE OTHER. A venv inherits
    the version of the interpreter that made it, so this decides what the env IS
    and there is no substituting one version for another:

      * the SERVER'S OWN interpreter, when it already is the wanted version (and
        always, for a spec that wants none — every env but the SGLang `tts` one);
      * `python<major.minor>` on PATH, which is how a distro and a `uv python
        install` both present one.

    A spec that names a version this host cannot produce is REFUSED BY NAME,
    before `venv` runs. The alternative is a 3.11 env that pip fails to fill
    several GB in with a wheel-compatibility error naming neither the env nor
    the reason — which is the same shape as every other defect this file
    refuses early.
    """
    wanted = spec.python_version
    if wanted is None:
        return sys.executable
    running = ".".join(map(str, sys.version_info[:2]))
    if running == wanted:
        return sys.executable
    named = shutil.which(f"python{wanted}")
    if named is not None:
        return named
    raise EnvError(
        f"the {spec.key} env must be built with python {wanted} and this host "
        f"offers neither: the Crucible server runs on {running} "
        f"({sys.executable}) and there is no `python{wanted}` on PATH. "
        f"{spec.recipe_name}.txt pins a stack that has no wheels for "
        f"{running} — sglang-omni 0.1.4 and torch 2.13.0+cu130 are built "
        f"against {wanted} — so a venv from this interpreter would install "
        "nothing and say so several GB in. Put a "
        f"python{wanted} on PATH (`uv python install {wanted}`, a distro "
        f"package, or a conda env) and run the install again"
    )


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
            # THE ENV HOLDS WHAT THE RECIPE ASKS. What may still be wrong is
            # the STAMP: it records which recipe bytes built this env, and an
            # edit to the recipe since then leaves it naming bytes that no
            # longer exist. `crucible doctor` calls that `pack_recipe_drift`
            # and has always told the operator to "re-run `crucible install`"
            # — which arrived HERE, returned this line, and did nothing. The
            # advice was unfollowable and the only remedy that worked was
            # `--force`, which deletes several GB of working env to correct a
            # line of JSON.
            here = recipe_sha256(recipe)
            if existing.recipe_sha256 is not None and existing.recipe_sha256 != here:
                _restamp(
                    home, spec, backend_kind,
                    existing=existing, recipe=recipe, here=here, on_line=on_line,
                )
                return env_status(home, spec, backend_kind)
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
        [interpreter_for(spec), "-m", "venv", str(directory)],
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

    # AND THE TWO SYMLINKS pip CANNOT EXPRESS EITHER — cuda-linux only.
    #
    # flashinfer JIT-builds SGLang's attention kernels with the nvcc inside the
    # pip wheel and only does so once that directory looks like a toolkit
    # (`lib64` beside `lib`, an unsuffixed `libcudart.so`). CUDA_HOME points at
    # the same directory and `serve_higgs_sgl.sh` exports it.
    #
    # THE FAILURE IS LATE AND LOOKS LIKE HEALTH, which is why this is here and
    # not in a setup note. Nothing fails at install: pip is happy, the env
    # stamps installed, and this stack has no site-packages patches for
    # `doctor` to report on. It goes wrong at the first render on a card. Both
    # links were created BY HAND on owens-pc on 2026-09-15, and the recipe has
    # claimed ever since that this module "creates and checks them" — a sentence
    # that was true of nothing until now.
    #
    # `cuda-linux` ONLY: `mlx-darwin`'s tts env has no CUDA in it, and asking it
    # for an nvidia directory would call a working Mac env broken.
    if spec.job_type == "tts" and backend_kind == "cuda-linux":
        try:
            narratorpatches.ensure_cuda_toolkit_links(directory, on_line=on_line)
        except narratorpatches.PatchError as exc:
            raise EnvError(str(exc)) from exc

    version = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    elapsed = time.monotonic() - started
    _write_stamp(
        home, spec, backend_kind,
        recipe=recipe,
        python_version=version,
        # A venv-built env has NO pack sha, and the key is written as null
        # rather than left out: "built here" is an answer.
        pack_sha256=None,
        seconds=round(elapsed, 1),
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


def _write_stamp(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    recipe: Path,
    python_version: str | None,
    pack_sha256: str | None,
    seconds: float | None,
) -> None:
    """Write `crucible-env.json`. THE one place that does.

    Both the install and the re-stamp come here so the file has one shape. The
    two used to be one path because only one of them existed; the moment a
    second writer appeared, a key it forgot would be a key `env_status` reads
    as "recorded before this was a thing" rather than as a bug.
    """
    stamp_path(home, spec).write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": recipe.name,
                # Recorded even on the `--build` path, so `doctor` can say a
                # locally built env no longer matches the recipe bytes it was
                # built from - the same question a pack answers with
                # `pack_recipe_drift`, asked of an env nobody downloaded.
                "recipe_sha256": recipe_sha256(recipe),
                # And the bytes themselves, so the NEXT drift can be TOLD APART
                # from a rebuild-worthy one instead of only detected.
                "recipe_text": recipe_text(recipe),
                "python_version": python_version,
                "pack_sha256": pack_sha256,
                "seconds": seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _restamp(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    existing: EnvStatus,
    recipe: Path,
    here: str,
    on_line: Any = None,
) -> None:
    """Correct a stamp whose recipe moved, or refuse and say what a rebuild is for.

    Only ever reached for an env `env_status` has just called INSTALLED, which
    is a stronger statement than it sounds: every `name==version` in the recipe
    is present at that version and every `name @ url` was installed from that
    exact commit. This function decides the one question that check leaves
    open — whether the recipe moved in some way that check cannot see.
    """
    before = existing.recipe_text
    if before is None:
        # An env stamped before the text was recorded. The bytes behind the
        # recorded hash are GONE, so nothing here can distinguish a moved
        # comment from a moved `--index-url`, and guessing which it was is
        # precisely the band-aid this refusal exists to avoid.
        raise EnvError(
            f"{existing.path} holds everything {recipe.name} pins, but its stamp "
            f"records recipe {existing.recipe_sha256[:12]} where this build has "
            f"{here[:12]}, and that stamp predates recording the recipe's text. "
            "Without those bytes there is no telling whether the edit was a "
            "comment or an `--index-url` that silently changes which wheel a pin "
            "resolves to, so this refuses rather than stamping a claim it cannot "
            f"support. `crucible install {spec.job_type} --force` rebuilds it, and "
            "is the only answer that is certainly true."
        )

    after = recipe_text(recipe)
    problems = unverifiable_recipe_changes(before, after, recipe.name)
    if problems:
        raise EnvError(
            f"{existing.path} cannot be re-stamped for {recipe.name}: "
            + "; ".join(problems)
            + ". These are changes no package check can see — `env_status` asks "
            "whether the env holds what the recipe names, and neither an index "
            "URL nor a requirement that was deleted shows up in that answer. "
            f"`crucible install {spec.job_type} --force` rebuilds it."
        )

    moved = sorted(
        name for name in _names_in(after)
        if _pin_of(before, name) != _pin_of(after, name)
    )
    if on_line is not None:
        on_line(
            f"{recipe.name} moved {existing.recipe_sha256[:12]} -> {here[:12]} "
            "with no change to where packages come from"
        )
        for name in moved:
            on_line(f"  {name}: {_pin_of(before, name)} -> {_pin_of(after, name)}")
        on_line(
            "  every one of those is already verified against the installed "
            "package, so the env is what this recipe asks for; correcting the "
            "stamp rather than rebuilding"
        )
    _write_stamp(
        home, spec, backend_kind,
        recipe=recipe,
        python_version=existing.python_version,
        pack_sha256=existing.pack_sha256,
        seconds=None,
    )


def _pin_of(text: str, name: str) -> str | None:
    """What a recipe's text pins `name` at — a version, or a commit."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = _DIRECT_REFERENCE.match(stripped)
        raw = match.group("name") if match else stripped.partition("==")[0]
        if raw.strip().lower().replace("_", "-") != name:
            continue
        if match is None:
            return stripped.partition("==")[2].strip()
        commit = _VCS_COMMIT.search(match.group("url"))
        return commit.group("sha") if commit else stripped
    return None


def recipe_text(path: Path) -> str:
    """The recipe's text, normalised the same way `recipe_sha256` hashes it.

    ONE normalisation, shared, for the same reason the digest has one
    implementation: a stamp whose text says CRLF and whose hash was taken over
    LF is a stamp that disagrees with itself.
    """
    return path.read_bytes().replace(_CRLF, _LF).decode("utf-8")


def _option_lines(text: str) -> list[str]:
    """The `-`-prefixed lines of a recipe, in order.

    These choose WHERE a pin resolves — `--index-url`,
    `--extra-index-url`, `-f` — and `env_status` never looks at them, because
    `_requirement_lines` skips them. That blind spot is the whole reason this
    function exists: `torch==2.5.1` from PyPI and `torch==2.5.1` from
    `download.pytorch.org/whl/cu121` are the same version string and different
    binaries, one of them without CUDA at all, and `pip list` cannot tell them
    apart afterwards.
    """
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("-")
    ]


def unverifiable_recipe_changes(before: str, after: str, recipe_name: str) -> list[str]:
    """What moved between two recipes that `env_status` would NOT have caught.

    Empty means: an env that satisfies `env_status` against `after` genuinely
    satisfies `after`, so its stamp can be corrected without rebuilding it.
    Non-empty means the difference is one no amount of package-checking can
    see, and the env has to be built again to be what the recipe now says.

    THE THREE KINDS OF CHANGE, AND WHY ONLY ONE OF THEM IS FATAL:

      * A COMMENT or a blank line is not installed. Most recipe edits are
        these — this file's own drift was two prose paragraphs and one pin —
        and sending them to a multi-GB rebuild is what made the drift warning
        something to ignore rather than act on.

      * A PIN or a DIRECT REFERENCE that moved is already checked, exactly,
        package by package: `env_status` compares every `name==version` against
        `pip list` and every `name @ url` against the commit in PEP 610's
        `direct_url.json`. Re-checking it here would be a second opinion about
        a question that already has an owner.

      * An OPTION line, or a requirement that VANISHED, is neither. The option
        line case is above. A vanished requirement is fatal for the opposite
        reason to the usual one: `env_status` asks whether everything the
        recipe names is PRESENT, never whether anything else is, so a package
        dropped from the recipe stays in the env and passes every check, and a
        rebuild from this recipe would not have it.
    """
    problems: list[str] = []

    was, now = _option_lines(before), _option_lines(after)
    if was != now:
        for line in [x for x in was if x not in now]:
            problems.append(f"{recipe_name} no longer says {line!r}")
        for line in [x for x in now if x not in was]:
            problems.append(f"{recipe_name} now says {line!r}, and it did not")

    # Parsed, not diffed: a requirement that merely MOVED in the file is not a
    # change to what is installed, and a textual diff would call it one.
    gone = sorted(_names_in(before) - _names_in(after))
    for name in gone:
        problems.append(
            f"{recipe_name} no longer requires {name!r}, which is still installed"
        )
    return problems


def _names_in(text: str) -> set[str]:
    """Every requirement's name in a recipe's text, pins and references alike."""
    found: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = _DIRECT_REFERENCE.match(stripped)
        raw = match.group("name") if match else stripped.partition("==")[0]
        found.add(raw.strip().lower().replace("_", "-"))
    return found


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
