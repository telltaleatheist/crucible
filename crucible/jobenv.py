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

from . import interpreter, narratorpatches
from .errors import CrucibleError

RECIPES_DIR_ENV = "CRUCIBLE_RECIPES_DIR"

#: The file that says an env finished installing. ONE writer (`_write_stamp`)
#: and one reader (`env_status`), because there must not be two answers to "is
#: this env there". It used to have a second writer — a downloaded env pack
#: stamped its own `.partial` tree — and PHASE20 deleted the packs, so an env
#: is now stamped only after pip returned 0 in it.
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
    #: THE ENVIRONMENT HALF of the recipe this env was installed from: every
    #: line but its direct references, hashed. None for an env stamped before
    #: the two halves were told apart.
    #:
    #: TWO HALVES BECAUSE THEY MOVE FOR DIFFERENT REASONS AND COST DIFFERENT
    #: AMOUNTS (PHASE20 section 4). A narrator edit moves one git sha in one
    #: line; the 13 GB of torch and SGLang around it did not move, and a single
    #: hash over the whole file cannot say which happened.
    environment_sha256: str | None = None
    #: The commit each `name @ url` line was installed from, as stamped. None
    #: for an env stamped before the halves were recorded — which is not the
    #: same as `{}`, an env whose recipe has no direct references.
    direct_references: dict[str, str] | None = None
    #: The recipe's TEXT as installed, line-ending-normalised. None for an env
    #: stamped before this was recorded.
    #:
    #: A hash says THAT a recipe moved and can never say WHAT moved, and the
    #: difference decides whether `pip install -r` into the existing venv is
    #: honest: a re-pinned package it will install, a changed `--index-url` it
    #: will NOT act on at all, because the pin it resolves is already satisfied
    #: by the wheel the old index served. Keeping the bytes is what lets
    #: `plan_install` tell those apart instead of stamping a claim pip did not
    #: make true.
    recipe_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "path": str(self.path),
            "detail": self.detail,
            "python_version": self.python_version,
            "packages": dict(self.packages),
            "environment_sha256": self.environment_sha256,
            "direct_references": (
                None if self.direct_references is None
                else dict(self.direct_references)
            ),
            # The text itself is deliberately NOT in `to_dict`: this feeds
            # `crucible doctor --json`, and several KB of recipe per env would
            # bury the report it is part of. What the text is FOR is the
            # comparison in `plan_install`, which reads the stamp directly.
        }


# ------------------------------------------------------------------- layout


def env_dir(home: Path, spec: EnvSpec) -> Path:
    return home / "envs" / spec.key


def env_python(home: Path, spec: EnvSpec) -> Path:
    """The venv interpreter the engines are spawned from."""
    return env_dir(home, spec) / "bin" / "python"


def stamp_path(home: Path, spec: EnvSpec) -> Path:
    """Written only after pip returned 0 in this venv, and at no other moment.

    Public because `plan_env` — the rule `workerenv` shares — is handed the
    path rather than deriving it, so neither module can guess a second one.
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


#: Where a recipe that names no index resolves from: pip's own default
#: `index-url`, which the pip documentation gives as `https://pypi.org/simple`.
#: A recipe with no `--index-url` line is not a recipe that downloads nothing —
#: it is one that downloads from here.
DEFAULT_INDEX_URL = "https://pypi.org/simple"

#: Hugging Face's, as `huggingface_hub` resolves it: `HF_ENDPOINT` when set,
#: and `https://huggingface.co` otherwise. Read rather than hard-coded so a
#: machine behind a mirror is probed at ITS mirror; the constant is that
#: library's documented default and not a guess at one.
HF_ENDPOINT_ENV = "HF_ENDPOINT"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

#: `--index-url https://…`, `--extra-index-url=https://…`, `-f https://…`.
#: The three pip options that choose WHERE a pin resolves.
_INDEX_OPTION = re.compile(
    r"^(?:--index-url|--extra-index-url|-f|--find-links)[=\s]+(?P<url>\S+)$"
)


def recipe_roots() -> list[Path]:
    """Every `envs/<job type>/` this build ships, in name order.

    `recipes_dir` answers for ONE job type because every other caller knows
    which one it wants. This caller does not: PHASE19 2.12's network probe is
    about every index ANY recipe could send pip to, and a list of job types
    written down here would be one more thing to forget when a sixth arrives.
    """
    override = os.environ.get(RECIPES_DIR_ENV)
    if override is not None and override != "":
        root = Path(override).expanduser()
    else:
        root = Path(__file__).resolve().parent / "envs"
    if not root.is_dir():
        raise EnvError(
            f"no env recipes at {root}; they are package data and this install "
            f"has lost them, or ${RECIPES_DIR_ENV} must point at them"
        )
    return sorted(path for path in root.iterdir() if path.is_dir())


def recipe_index_urls() -> list[str]:
    """Every place a `crucible install <type>` downloads from, in order.

    PHASE19-AUTOMATIC-WSL.md 2.12: *"the network probe proves one route and the
    install needs five"*. `guest_no_network` used to fetch the release wheel off
    GitHub, and a VPN or a proxy that passes GitHub and blocks PyPI,
    `download.pytorch.org`, the SGLang index or Hugging Face passed that probe
    and failed minutes later inside pip.

    READ FROM THE RECIPES, never listed. A list spelled here would drift the
    first time a recipe gained an `--extra-index-url`, and the drift would be
    invisible: the probe would go on passing and pip would go on failing. The
    two that no recipe names are added by their own owners — pip's default
    index, and the hub endpoint the weights come from.
    """
    urls = [DEFAULT_INDEX_URL]
    for directory in recipe_roots():
        for recipe in sorted(directory.glob("*.txt")):
            for line in _option_lines(recipe_text(recipe)):
                found = _INDEX_OPTION.match(line)
                if found is not None and found.group("url") not in urls:
                    urls.append(found.group("url"))
    hub = os.environ.get(HF_ENDPOINT_ENV) or DEFAULT_HF_ENDPOINT
    if hub not in urls:
        urls.append(hub)
    return urls


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


#: How a recipe states what installing it costs, in its own header:
#:
#:     # archive-bytes: 5300000000  PHASE20-…md section 0, measured 2026-09-18
#:
#: One integer of BYTES, then a CITATION that the parser requires to be there.
#: An uncited constant is invented, and a number somebody later believes is
#: worse than no number at all — so a line carrying a bare integer is refused
#: exactly as a missing line is.
#:
#: A comment and not a `--option` line, because it is not an instruction to pip:
#: `_requirement_lines` already drops `#` and `-` lines, so this shape is
#: invisible to `recipe_pins`, to `recipe_direct_references` and to pip itself,
#: and it lives in the header beside the pins where PHASE19 2.12 put it.
_ARCHIVE_BYTES = re.compile(
    r"^#\s*archive-bytes:\s*(?P<bytes>\d+)(?P<citation>\s+\S.*?)?\s*$"
)


def recipe_archive_bytes(path: Path) -> int:
    """What this recipe MEASURED, in bytes, or a named refusal.

    Read by the same loader that reads the pins, from the same file, because
    "what this env costs" is a fact about the recipe and belongs beside the
    lines that cost it (PHASE19-AUTOMATIC-WSL.md 2.12, ruling 6).

    It is an ARCHIVE size — what the packs weighed when PHASE20 measured them —
    and an unpacked env is larger, so every sentence built on it says "at
    least". A floor that is too low still beats a pip that dies at 4.9 GB.
    """
    found: list[re.Match[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ARCHIVE_BYTES.match(line.strip())
        if match is not None:
            found.append(match)
    if not found:
        raise EnvError(
            f"recipe_unsized: {path.name} carries no `# archive-bytes:` line, so "
            "there is no telling what installing it needs. The measured sizes "
            "are PHASE20-CODE-NOT-ENVIRONMENTS.md section 0's table and they go "
            "in the recipe's header; guessing one here would be a number "
            "somebody later believes"
        )
    if len(found) > 1:
        raise EnvError(
            f"recipe_size_invalid: {path.name} carries {len(found)} "
            "`# archive-bytes:` lines. One recipe, one size, one owner"
        )
    match = found[0]
    if match.group("citation") is None:
        raise EnvError(
            f"recipe_size_invalid: {path.name}'s `# archive-bytes:` line names "
            "no source. Where a number was measured is part of the number: "
            "write it after the integer"
        )
    size = int(match.group("bytes"))
    if size <= 0:
        raise EnvError(
            f"recipe_size_invalid: {path.name} states an archive size of {size} "
            "bytes, which no env has ever weighed"
        )
    return size


def _filesystem_of(directory: Path) -> Path:
    """The nearest existing ancestor of a path — what `disk_usage` can be asked.

    `~/.crucible/envs/<key>/` does not exist yet on the install this guard is
    for, and neither may `envs/`. The free space that matters is the
    filesystem the venv will land ON, which is the same one its nearest
    existing parent is on.
    """
    path = directory.expanduser().absolute()
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    raise EnvError(
        f"env_disk_unreadable: no existing directory above {directory}, so the "
        "free space where this env would land cannot be measured"
    )


def refuse_without_room(*, job_type: str, recipe: Path, directory: Path) -> None:
    """`env_disk` — PHASE19-AUTOMATIC-WSL.md 2.12 and ruling 6.

    BEFORE pip touches the network. PHASE20 moved the gigabytes off our
    releases and onto the mirrors, which means a first install now downloads
    5.3 GB of tts over minutes and finds out about the disk at the end of it.
    `_guest_ready` deliberately passes no `required_bytes` (the move itself is
    ~31 MB), so `pack_disk` never fires and this is where the disk question
    now lives.

    ONLY ON A FRESH BUILD. The measured number is what a recipe costs to
    install from nothing; what a DRIFT costs is the difference between two
    resolutions and was never measured. Requiring a whole archive's worth of
    free space before reinstalling one narrator line would be a floor invented
    here, and it would refuse on a machine where the env is already sitting in
    most of that space. `install_env` and `install_worker_env` therefore call
    this on `PLAN_BUILD` and on nothing else.
    """
    required = recipe_archive_bytes(recipe)
    filesystem = _filesystem_of(directory)
    free = shutil.disk_usage(filesystem).free
    if free >= required:
        return
    raise EnvError(
        f"env_disk: installing {job_type!r} needs at least "
        f"{required / 1_000_000_000:.1f} GB free and {filesystem} has "
        f"{free / 1_000_000_000:.1f} GB ({free} bytes of the {required} "
        f"{recipe.name} states). That figure is the ARCHIVE size PHASE20 "
        "measured for this recipe and an unpacked env is larger, so it is a "
        "floor and not an estimate. Free space on that drive, or move "
        "$CRUCIBLE_HOME to one that has it, before running this again"
    )


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
    # `.get` and not `[]` for these three alone: a stamp written before PHASE20
    # hashed the recipe whole and recorded no references at all. Absent means
    # "installed before the two halves were told apart", which is exactly what
    # `plan_install` answers — not a default standing in for a fact.
    environment_sha256 = record.get("environment_sha256")
    direct_references = record.get("direct_references")
    # Absent for every env stamped before the text was recorded, and that
    # absence is load-bearing rather than tidy-uppable: it is exactly the case
    # `plan_install` cannot prove anything about and refuses by name.
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
            environment_sha256=environment_sha256,
            direct_references=direct_references,
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
            environment_sha256=environment_sha256,
            direct_references=direct_references,
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
        environment_sha256=environment_sha256,
        direct_references=direct_references,
        recipe_text=recipe_text,
    )


def require_env(home: Path, spec: EnvSpec, backend_kind: str) -> Path:
    """This venv's python, or `env_missing` by name. Never guesses one."""
    status = env_status(home, spec, backend_kind)
    if not status.installed:
        raise EnvError(status.detail)
    return env_python(home, spec)


# ------------------------------------------------------------------ install


def interpreter_for(
    spec: EnvSpec,
    home: Path,
    backend_kind: str,
    *,
    on_line: Any = None,
    on_progress: Any = None,
) -> str:
    """The python that builds this env's venv. ONE SOURCE, and it is a pin.

    A venv inherits the version AND the build of whatever made it, so this
    decides what the env IS:

      * the SERVER'S OWN interpreter, for a spec that wants no particular
        version — every env but the SGLang `tts` one — and for one that wants
        the version the server already runs;
      * otherwise the pinned python-build-standalone of that version,
        DOWNLOADED from the same publisher the server's own came from, into
        `<home>/interpreters/<version>/`, once, verified by digest.

    **THE PATH SEARCH IS GONE** (PHASE20 section 3, item 4). This used to fall
    back to `python<major.minor>` on PATH, which is a distro's, a conda's or
    somebody's `uv python install` — an interpreter of unknown provenance and
    unknown digest with 13 GB of exactly pinned wheels on top of it. There is
    no second place to look now, and a version nobody pinned is
    `interpreter_not_pinned` before `venv` runs rather than a
    wheel-compatibility error several GB in.
    """
    wanted = spec.python_version
    if wanted is None:
        return sys.executable
    running = ".".join(map(str, sys.version_info[:2]))
    if running == wanted:
        return sys.executable
    return str(
        interpreter.ensure_interpreter(
            home, backend_kind, wanted, on_line=on_line, on_progress=on_progress
        )
    )


#: What `install_env` will do to an env that is already on disk, by name. The
#: two drift names are `crucible doctor`'s words too — one fact, one owner, so
#: the doctor cannot report a drift the installer would answer differently.
PLAN_NOTHING = "nothing"
PLAN_REFERENCES = "narrator_sha_drift"
PLAN_RECIPE = "env_recipe_drift"
PLAN_BUILD = "build"


@dataclass(frozen=True)
class EnvPlan:
    """What an install is about to do, decided before it does any of it."""

    action: str
    detail: str
    #: For `PLAN_REFERENCES`: the recipe LINES to reinstall, verbatim. Empty
    #: for every other action.
    lines: tuple[str, ...] = ()


def plan_env(
    *,
    directory: Path,
    stamp: Path,
    recipe: Path,
    backend_kind: str,
    installed: bool,
    force: bool,
    install_command: str,
) -> EnvPlan:
    """THE decision, for `jobenv` and `workerenv` alike (PHASE20 section 4).

    An env is touched only when its recipe moved, and then by pip INTO the
    venv that is there — pip skips what is already satisfied, so the cost is
    the difference rather than the environment. The delete-and-rebuild that
    used to be the answer to every drift is now `--force` and nothing else.

    Shared by the two env modules rather than written twice: they are already
    one module's worth of code twice over (see `workerenv`'s header), and a
    second copy of this rule is a second answer to "does this env need
    anything", which is precisely the shape ARCHITECTURE.md R1 is about.
    """
    if force:
        return EnvPlan(
            PLAN_BUILD,
            f"--force: {directory} is deleted and built again from {recipe.name}",
        )
    if not directory.exists():
        return EnvPlan(PLAN_BUILD, f"there is no {directory}")
    if not stamp.is_file():
        # A half-built venv from an interrupted install: no stamp, so nothing
        # downstream has ever trusted it. Rebuilding it is the only correct move.
        return EnvPlan(
            PLAN_BUILD,
            f"{directory} exists but {stamp.name} does not: the last "
            f"`{install_command}` did not finish",
        )
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["backend"] != backend_kind:
        raise EnvError(
            f"{directory} was installed for backend {record['backend']!r} and "
            f"this host is {backend_kind!r}. A venv full of one backend's "
            "wheels is not re-pointed at another's by pip, so this is the one "
            f"drift that is genuinely a rebuild: `{install_command} --force`"
        )

    before = record.get("recipe_text")
    if before is None:
        # An env stamped before the text was recorded. The bytes behind that
        # stamp are GONE, so nothing here can tell a moved comment from a moved
        # `--index-url` — and `pip install -r` does NOT act on the second, since
        # the pin it resolves is already satisfied by the wheel the old index
        # served. Stamping this recipe over that env would claim bytes nobody
        # installed, so it refuses rather than guessing.
        raise EnvError(
            f"{directory} was stamped before the recipe's text was recorded, so "
            f"there is no telling what moved in {recipe.name} since. Without "
            "those bytes an `--index-url` change — which silently swaps the "
            "wheel a pin resolves to and which pip will not act on, because the "
            "pin is already satisfied — reads exactly like a comment. "
            f"`{install_command} --force` rebuilds it, and is the only answer "
            "that is certainly true."
        )
    after = recipe_text(recipe)
    problems = unverifiable_recipe_changes(before, after, recipe.name)
    if problems:
        raise EnvError(
            f"{directory} cannot be brought to {recipe.name} by pip: "
            + "; ".join(problems)
            + ". Neither of those is a change pip acting on this recipe would "
            "make: an index URL moves which wheel a pin resolves to while the "
            "pin itself stays satisfied, and a requirement that vanished stays "
            f"installed. `{install_command} --force` rebuilds it."
        )

    here_environment = environment_sha256(recipe)
    here_references = recipe_direct_references(recipe)
    stamped_environment = record.get("environment_sha256")
    stamped_references = record.get("direct_references")
    if stamped_environment is None or stamped_references is None:
        # Stamped before the two halves were told apart. The remedy is the
        # ordinary one — `pip install -r`, which is seconds when nothing moved
        # — rather than a re-stamp on its own: a stamp is written after pip
        # returned 0 in this venv and at no other moment.
        return EnvPlan(
            PLAN_RECIPE,
            f"{directory} was stamped before {recipe.name}'s two halves were "
            "recorded apart; pip is run over the recipe so the new stamp "
            "describes bytes that are certainly there",
        )
    if stamped_environment != here_environment:
        return EnvPlan(
            PLAN_RECIPE,
            f"{recipe.name}'s environment half moved "
            f"{stamped_environment[:12]} -> {here_environment[:12]}",
        )
    if stamped_references != here_references:
        moved = sorted(
            name for name in set(stamped_references) | set(here_references)
            if stamped_references.get(name) != here_references.get(name)
        )
        lines = tuple(
            line for line in _requirement_lines(recipe)
            if _reference_name(line) in moved
        )
        if len(lines) != len(moved):
            raise EnvError(
                f"{recipe.name} pins {moved} at commits this env was not built "
                f"from, and only {len(lines)} of those are lines in the file. A "
                "reference that moved must be a line this install can reinstall"
            )
        return EnvPlan(
            PLAN_REFERENCES,
            ", ".join(
                f"{name} {(stamped_references.get(name) or 'absent')[:12]} -> "
                f"{(here_references.get(name) or 'absent')[:12]}"
                for name in moved
            ),
            lines,
        )
    if not installed:
        # The recipe has not moved and the env still does not hold what it
        # pins — a package removed by hand, a half-finished pip. pip over the
        # recipe is what puts it back, and it is the same command either way.
        return EnvPlan(
            PLAN_RECIPE, f"{directory} does not hold what {recipe.name} pins"
        )
    return EnvPlan(PLAN_NOTHING, f"{directory} is what {recipe.name} says")


def _reference_name(line: str) -> str | None:
    """The package a `name @ url` line names, canonical, or None."""
    match = _DIRECT_REFERENCE.match(line)
    if match is None:
        return None
    return match.group("name").strip().lower().replace("_", "-")


def plan_install(
    home: Path, spec: EnvSpec, backend_kind: str, *, force: bool = False
) -> EnvPlan:
    """What `install_env` would do here, without doing any of it.

    `crucible doctor` asks this too, which is why it is separate: the doctor's
    drift line and the installer's remedy have to be the same sentence, and
    two functions computing it is two sentences waiting to disagree.
    """
    return plan_env(
        directory=env_dir(home, spec),
        stamp=stamp_path(home, spec),
        recipe=recipe_for(spec),
        backend_kind=backend_kind,
        installed=env_status(home, spec, backend_kind).installed,
        force=force,
        install_command=f"crucible install {spec.job_type}",
    )


def install_env(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    force: bool = False,
    on_line: Any = None,
) -> EnvStatus:
    """Bring `~/.crucible/envs/<key>/` to this env's recipe, and stamp it.

    `on_line` is called with each line of pip's output so the CLI can show it.
    Returns the resulting status. Raises EnvError naming what went wrong.
    """
    recipe = recipe_for(spec)
    directory = env_dir(home, spec)
    plan = plan_install(home, spec, backend_kind, force=force)
    if plan.action == PLAN_NOTHING:
        return env_status(home, spec, backend_kind)
    if on_line is not None:
        on_line(f"{plan.action}: {plan.detail}")

    started = time.monotonic()
    python_version: str | None = None

    if plan.action == PLAN_BUILD:
        # BEFORE the venv, and long before pip dials a mirror.
        refuse_without_room(
            job_type=spec.job_type, recipe=recipe, directory=directory
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            shutil.rmtree(directory)
        _run(
            [
                interpreter_for(spec, home, backend_kind, on_line=on_line),
                "-m",
                "venv",
                str(directory),
            ],
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
    else:
        python = env_python(home, spec)
        # The venv was not rebuilt, so its interpreter is the one the stamp
        # already names. Asking it again would be a subprocess to learn a fact
        # nothing changed.
        python_version = json.loads(
            stamp_path(home, spec).read_text(encoding="utf-8")
        )["python_version"]

    if plan.action == PLAN_REFERENCES:
        # THE 1b CASE (PHASE20 section 4). One git sha moved in one line; the
        # environment half hashed the same, so nothing else in this venv is out
        # of date and `pip install -r` would re-resolve 13 GB to arrive back
        # where it started. `--no-deps` because the dependencies are the
        # environment half's and were just proved unchanged; `--force-reinstall`
        # because pip otherwise sees the package installed at the same declared
        # version and does nothing, a commit having no version of its own.
        for line in plan.lines:
            _run(
                [
                    str(python), "-m", "pip", "install",
                    "--no-deps", "--force-reinstall", line,
                ],
                f"could not reinstall {line} into {directory}",
                on_line,
            )
    else:
        _run(
            [str(python), "-m", "pip", "install", "-r", str(recipe)],
            f"could not install {recipe} into {directory}",
            on_line,
        )

        # THE TWO SITE-PACKAGES PATCHES pip CANNOT EXPRESS, RE-APPLIED HERE.
        #
        # pip has just written vllm-omni's own `higgs_audio_v3.py` over the edit
        # narrator's sentinel proof reads, which is what made every Higgs load on
        # owens-pc fail from 07:46 on 2026-09-15 after a `--force` at 07:34 — the
        # proof found a 0-byte report because the code that writes records was
        # gone. Before this call the recipe said the patches "must be re-applied"
        # and named nobody to do it; `crucible doctor` then reported them
        # `missing` from a command nobody runs after an install.
        #
        # ONLY WHERE pip TOUCHED SITE-PACKAGES. The `PLAN_REFERENCES` arm above
        # installs narrator alone with `--no-deps`, so the stack these patch is
        # exactly as it was and re-applying them would be work for nothing.
        #
        # ONLY FOR `tts`. Both patches edit the vLLM stack, and the `llm` env
        # pins `vllm` too — patching an LLM server's input processor to admit
        # token -100 is not a thing anyone asked for. `narratorpatches` then
        # selects again by the recipe's own pins, so `mlx-darwin`'s tts env (no
        # vllm, no vllm-omni) runs neither and is not called broken for it.
        #
        # BEFORE THE STAMP, and it raises: an env that is stamped installed is
        # an env whose patches are in, or there is no stamp.
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
        # flashinfer JIT-builds SGLang's attention kernels with the nvcc inside
        # the pip wheel and only does so once that directory looks like a
        # toolkit (`lib64` beside `lib`, an unsuffixed `libcudart.so`).
        # CUDA_HOME points at the same directory and `serve_higgs_sgl.sh`
        # exports it.
        #
        # THE FAILURE IS LATE AND LOOKS LIKE HEALTH, which is why this is here
        # and not in a setup note. Nothing fails at install: pip is happy, the
        # env stamps installed, and this stack has no site-packages patches for
        # `doctor` to report on. It goes wrong at the first render on a card.
        # Both links were created BY HAND on owens-pc on 2026-09-15, and the
        # recipe has claimed ever since that this module "creates and checks
        # them" — a sentence that was true of nothing until now.
        #
        # `cuda-linux` ONLY: `mlx-darwin`'s tts env has no CUDA in it, and
        # asking it for an nvidia directory would call a working Mac env broken.
        if spec.job_type == "tts" and backend_kind == "cuda-linux":
            try:
                narratorpatches.ensure_cuda_toolkit_links(directory, on_line=on_line)
            except narratorpatches.PatchError as exc:
                raise EnvError(str(exc)) from exc

    if python_version is None:
        python_version = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; print('.'.join(map(str, sys.version_info[:3])))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    _write_stamp(
        home, spec, backend_kind,
        recipe=recipe,
        python_version=python_version,
        references=recipe_direct_references(recipe),
        seconds=round(time.monotonic() - started, 1),
    )
    return env_status(home, spec, backend_kind)


#: The one line-ending rule. A recipe is a TEXT declaration, so a CR before a
#: LF is an artefact of the checkout the file arrived in and never a fact about
#: what the env contains.
_CRLF = b"\r\n"
_LF = b"\n"


def recipe_sha256(path: Path) -> str:
    """The recipe's SHA-256 over LINE-ENDING-NORMALISED bytes.

    THE ONE NORMALISATION. `environment_sha256` above, `recipe_text` below and
    `workerenv`'s env stamp all come through here, because a fact with two
    owners is a fact that will eventually disagree with itself, and this one
    already did.

    MEASURED 2026-09-15, which is why the normalisation is here at all: the
    same commit of `pyproject.toml` hashed to `1ab85cc3…` from the main
    checkout and `cc4fda38…` from a worktree of that SAME commit, while
    `git hash-object` said both were blob `5ef53a3`. The difference was CRLF
    versus LF — this machine has `core.autocrlf=true` and the working file
    predates the repo's `.gitattributes` — and the consequence is a false alarm
    in both directions: an env on a Windows desk calling itself drifted from
    the very recipe it was installed from, and a Linux runner disagreeing with
    that desk about a file neither of them is wrong about.

    **NORMALISE, DO NOT HASH THE GIT BLOB.** `git hash-object` would be the
    exact answer for a recipe in a checkout and NO answer at all for the case
    that matters most: `crucible/envs/*.txt` ship inside the installed wheel,
    where there is no repository, no index and no `git` to ask — and
    `plan_install` runs on an operator's machine against precisely that copy.
    A digest that needed a checkout would turn `env_recipe_drift` into
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


def environment_sha256(path: Path) -> str:
    """THE ENVIRONMENT HALF of a recipe: every line but its direct references.

    The two halves of a recipe move for different reasons and cost different
    amounts (PHASE20 section 4). `narrator @ git+…@<sha>` moves whenever
    BookForge's python package is edited — several times a day — and moving it
    changes nothing about the 13 GB of torch, SGLang and CUDA wheels pinned
    above it. A single digest over the whole file cannot say which happened, so
    an env answered from one is an env rebuilt for a one-line edit.

    The references are not ignored, they are stamped SEPARATELY, by name and
    commit (`recipe_direct_references`), and checked against what pip recorded
    in PEP 610's `direct_url.json`. Two facts, two owners, both compared.

    Hashed over `recipe_sha256`'s normalisation, and through the same function,
    so the CRLF rule has exactly one implementation here as everywhere else.
    """
    kept = [
        line for line in recipe_text(path).splitlines()
        if _DIRECT_REFERENCE.match(line.strip()) is None
    ]
    body = "".join(line + "\n" for line in kept)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _write_stamp(
    home: Path,
    spec: EnvSpec,
    backend_kind: str,
    *,
    recipe: Path,
    python_version: str | None,
    references: dict[str, str],
    seconds: float | None,
) -> None:
    """Write `crucible-env.json`. THE one place that does.

    Written after pip returned 0 in this venv and at no other moment, so the
    file never claims bytes that are not installed.
    """
    stamp_path(home, spec).write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": recipe.name,
                # THE TWO HALVES, apart. `environment_sha256` says whether the
                # env's packages moved; `direct_references` says which commit
                # each `name @ url` came from. `plan_install` answers them with
                # two different commands, which is the whole point of stamping
                # them separately.
                "environment_sha256": environment_sha256(recipe),
                "direct_references": dict(references),
                # And the bytes themselves, so a drift pip CANNOT act on — a
                # moved `--index-url`, a requirement that vanished — can be
                # told apart from one it can, instead of only detected.
                "recipe_text": recipe_text(recipe),
                "python_version": python_version,
                "seconds": seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
    """What moved between two recipes that `pip install -r` would NOT fix.

    Empty means: running pip over `after` in the venv that is already there
    brings it to `after`, so `plan_install` can do exactly that and stamp the
    result. Non-empty means pip would return 0 and change nothing relevant, so
    a stamp written afterwards would claim bytes nobody installed — and the env
    has to be built again to be what the recipe now says.

    THE THREE KINDS OF CHANGE, AND WHY ONLY ONE OF THEM IS FATAL:

      * A COMMENT or a blank line is not installed. Most recipe edits are
        these — this file's own drift was two prose paragraphs and one pin —
        and sending them to a multi-GB rebuild is what made the drift warning
        something to ignore rather than act on.

      * A PIN or a DIRECT REFERENCE that moved is exactly what pip acts on:
        `pip install -r` installs the new version, and `env_status` then
        compares every `name==version` against `pip list` and every
        `name @ url` against the commit in PEP 610's `direct_url.json`. Both
        halves have an owner and neither is this function's business.

      * An OPTION line, or a requirement that VANISHED, is neither. An option
        line chooses WHERE a pin resolves, and pip acting on this recipe will
        not re-fetch a pin it already finds satisfied — so `torch==2.5.1` from
        PyPI stays where a recipe now says cu121, with no CUDA in it and no
        check able to see the difference. A vanished requirement is fatal for
        the opposite reason: pip never removes, so the package stays in the env
        and passes every check, and a build from this recipe would not have it.
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
