"""Which site-packages patches each env TYPE carries — the one registry.

`crucible/narratorpatches.py` was written for the `tts` env and is where the
machinery lives: a patch is a distribution, a file, a marker (and optionally a
string that must be gone), an idempotent applier that refuses by name, and a
check. Selection is by the recipe that built the env — a patch whose
distribution is not pinned is `not_applicable`, never skipped silently — so the
same patch table is right on every backend without naming one.

On 2026-09-23 the `llm` env got a patch of its own (below), and the honest shape
for a second env type is a table per type rather than a second copy of the
machinery: this module maps a job type to its table and its appliers'
directory, and `jobenv.install_env`, `crucible doctor` and `crucible env patch`
all ask here. The tts table is `narratorpatches.NARRATOR_PATCHES`, unchanged,
so every tts behaviour and `tests/test_narrator_patches.py` are what they were.

THE `llm` PATCH: mlx-lm's `top_logprobs` ceiling, 11 -> 40
----------------------------------------------------------
PHASE22-DECIDE.md section 2.6. The decide door asks for K = labels + 4 top
logprobs and refuses above the engine's stated cap; stock mlx-lm 0.31.3
validates `top_logprobs` to at most 11, so the Mac could not read a question
with more than seven options (Briefcase needs 11 and 26). The applier is
`envs/llm/patches/patch_mlx_lm_top_logprobs.py`. It edits `mlx-lm`, which only
the `mlx-darwin` llm recipe pins: on `cuda-linux` (vLLM) it is `not_applicable`,
and `llama-windows` has no llm recipe at all (its engine is llama.cpp's own
binary release), which is asked with empty pins and answers the same.

`MlxLmEngine` states `max_logprobs = 40` BECAUSE of this patch, and refuses to
START on an env where `check` does not say `applied` (`engines/mlx_lm.py`), so
the door can never be told 40 by an engine that would answer 400.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import narratorpatches
from .narratorpatches import NarratorPatch, PatchError

#: The `llm` env's appliers. Shipped by `pyproject.toml`'s
#: `crucible = ["envs/**/*"]`, like the tts ones.
LLM_SCRIPTS_DIR = Path(__file__).resolve().parent / "envs" / "llm" / "patches"

MLX_LM_TOP_LOGPROBS = NarratorPatch(
    id="mlx-lm-top-logprobs-40",
    distribution="mlx-lm",
    rel_path="mlx_lm/server.py",
    marker='self._validate("top_logprobs", int, min_val=0, max_val=40, whitelist=[-1])',
    absent_marker=(
        'self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])'
    ),
    stale_marker=None,
    script="patch_mlx_lm_top_logprobs.py",
    why=(
        "stock mlx-lm 0.31.3 answers 400 to top_logprobs above 11, so the decide "
        "door cannot read a question with more than 7 options on the Mac, and "
        "MlxLmEngine (which states 40 because of this patch) refuses to start "
        "without it"
    ),
)

LLM_PATCHES: tuple[NarratorPatch, ...] = (MLX_LM_TOP_LOGPROBS,)


@dataclass(frozen=True)
class PatchSet:
    """One env type's patches and the directory its appliers live in."""

    patches: tuple[NarratorPatch, ...]
    scripts_dir: Path


#: Job type -> its patches. A job type absent here has none, and `check`
#: answers it with no rows rather than inventing any.
REGISTRY: dict[str, PatchSet] = {
    "tts": PatchSet(narratorpatches.NARRATOR_PATCHES, narratorpatches.SCRIPTS_DIR),
    "llm": PatchSet(LLM_PATCHES, LLM_SCRIPTS_DIR),
}


def patched_job_types() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def patches_for(job_type: str) -> PatchSet | None:
    return REGISTRY.get(job_type)


def apply(
    job_type: str,
    env_dir: Path,
    python: Path,
    recipe_pins: dict[str, str],
    *,
    on_line: Any = None,
    runner: Any = None,
) -> list[dict[str, Any]]:
    """Apply and prove this env type's patches; `[]` for a type with none.

    Raises `PatchError` by name, exactly as `narratorpatches.apply` does — it
    IS that function, over this type's table.
    """
    found = REGISTRY.get(job_type)
    if found is None:
        return []
    return narratorpatches.apply(
        env_dir,
        python,
        recipe_pins,
        on_line=on_line,
        runner=runner,
        patches=found.patches,
        scripts_dir=found.scripts_dir,
    )


def check(
    job_type: str, env_dir: Path, recipe_pins: dict[str, str]
) -> list[dict[str, Any]]:
    """One row per patch of this env type; `[]` for a type with none."""
    found = REGISTRY.get(job_type)
    if found is None:
        return []
    return narratorpatches.check(env_dir, recipe_pins, patches=found.patches)


def require_applied(patch: NarratorPatch, env_dir: Path) -> None:
    """Raise `PatchError` naming the status unless `patch` is applied here.

    For an engine that states a number BECAUSE of a patch: it is asked with the
    patch's own distribution as the pins, since the engine running at all means
    that distribution is what it runs.
    """
    [row] = narratorpatches.check(env_dir, {patch.distribution: ""}, patches=(patch,))
    if row["status"] != narratorpatches.APPLIED:
        raise PatchError(
            f"{patch.id} is {row['status']} in {env_dir}: {row['detail']}. "
            f"{row['why']}"
        )


__all__ = [
    "LLM_PATCHES",
    "LLM_SCRIPTS_DIR",
    "MLX_LM_TOP_LOGPROBS",
    "PatchSet",
    "REGISTRY",
    "apply",
    "check",
    "patched_job_types",
    "patches_for",
    "require_applied",
]
