"""Which site-packages patches each env TYPE carries — the one registry.

A patch is a distribution, a file, a marker (and optionally a string that must
be gone), an idempotent applier that refuses by name, and a check. Selection is
by the recipe that built the env — a patch whose distribution is not pinned is
`not_applicable`, never skipped silently — so the same patch table is right on
every backend without naming one. This module maps a job type to its table and
its appliers' directory, and `jobenv.install_env`, `crucible doctor` and
`crucible env patch` all ask here.

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

THE SECOND `llm` PATCH (2026-09-24): the logprobs mlx-lm returns, in float32
------------------------------------------------------------------------
PHASE22-DECIDE.md, the label-mass note. Stock mlx-lm 0.31.3 normalizes
`logits - mx.logsumexp(logits)` in the model's dtype, bf16 for every model
Crucible serves on the Mac, so every returned logprob carries one common
rounding error of up to 0.0625 and a decision's `label_mass` came back 0.94-1.06
(a live qwen3.5-2b triage: 95th percentile 1.055). The applier is
`envs/llm/patches/patch_mlx_lm_fp32_logprobs.py`; it returns float32 logprobs
from all three sites that feed returned logprobs and leaves the sampler reading
the stock ones, so generation is unchanged. `MlxLmEngine.start` refuses
`llm_env_unpatched` unless EVERY patch in `LLM_PATCHES` is `applied`.

THE THIRD `llm` PATCH (2026-09-26): a dead generation thread exits the engine
----------------------------------------------------------------------------
mlx-lm 0.31.3 generates on one thread and lets an exception end only that
thread, so the server stays up answering nothing (upstream ml-explore/mlx-lm
#1672). ContentStudio's 27B died that way twice on 2026-09-25 (`metal::malloc
Resource limit (499000) exceeded`) and its chat sat in flight for 17-20 minutes.
The applier is `envs/llm/patches/patch_mlx_lm_fatal_generation_thread.py`: the
thread's target is wrapped so an exception prints its traceback and calls
`os._exit(70)`. From then on it is an engine that EXITED, which the chat door
and the residency already name.

THE FOURTH `llm` PATCH (2026-09-26): the cache counters are evaluated every step
-------------------------------------------------------------------------------
What killed that thread. mlx-lm 0.31.3's batched caches move `left_padding`,
`lengths`, `offset` and `_idx` with lazy arithmetic nothing forces, so each
decode step extends an unevaluated graph, and the prompt cache stores it with
every finished reply. On a Qwen3.5/3.8 model (ArraysCache layers) the next
request's first step evaluates it and passes Metal's buffer count limit, 499000.
Reproduced on the Mac 27B with ContentStudio's real request bytes: title 2 died
after title 1 with the prompt cache on, and survived alone or with the cache off.
The applier is `envs/llm/patches/patch_mlx_lm_cache_counters.py`: the counters
join the decode step's existing `mx.async_eval`, which stays asynchronous.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Statuses a patch check can report. `stale` is its own answer rather than a
#: kind of `missing`, because an env carrying v1 of a patch looks patched to
#: every marker grep and renders wrongly anyway.
APPLIED = "applied"
MISSING = "missing"
STALE = "stale"
NO_FILE = "no_such_file"
NO_ENV = "no_env"
#: The recipe that built this env does not install the distribution this patch
#: edits, so there is nothing here to patch and nothing wrong. Distinct from
#: `no_such_file`, which means the package SHOULD be here and is not.
NOT_APPLICABLE = "not_applicable"

#: The statuses that are not a problem. `crucible doctor` reads this rather than
#: testing `applied`, because `applied` answers "is the marker in the file" and a
#: patch with no file to be in has no honest answer to that question.
SOUND_STATUSES: frozenset[str] = frozenset({APPLIED, NOT_APPLICABLE})


@dataclass(frozen=True)
class EnvPatch:
    """One edit, where it lands, and how to tell whether it is there."""

    id: str
    #: The pinned distribution this patch edits, as the recipe names it. A patch
    #: whose distribution the recipe does not install is `not_applicable`.
    distribution: str
    #: Relative to the env's `site-packages`.
    rel_path: str
    #: A string the patched file must contain.
    marker: str
    #: A string the patched file must NOT contain, when there is one. A marker
    #: alone answers "did somebody apply something here"; this answers "and is
    #: the thing it replaced actually gone".
    absent_marker: str | None
    #: Present only in the CURRENT version of the patch. Marker present and this
    #: absent is an env carrying an older one, which is reported STALE.
    stale_marker: str | None
    #: The applier in `envs/<job type>/patches/`, run as `<env python> <script> <env>`.
    #: Every script is idempotent and refuses by name (`ANCHOR_NOT_FOUND`) when
    #: upstream has moved the code it edits, rather than skipping quietly.
    script: str
    why: str


class PatchError(RuntimeError):
    """A patch could not be applied, or was not there after applying it."""


def script_path(patch: EnvPatch, scripts_dir: Path) -> Path:
    """The applier for this patch, or `PatchError` naming the missing file."""
    path = scripts_dir / patch.script
    if not path.is_file():
        raise PatchError(
            f"the applier for {patch.id} is not installed: {path} is not there. "
            "A wheel built without `envs/**/*` has the recipes and not the "
            "patches, which is an env that installs and does not render"
        )
    return path


def apply_patches(
    env_dir: Path,
    python: Path,
    recipe_pins: dict[str, str],
    *,
    on_line: Any = None,
    runner: Any = None,
    patches: tuple[EnvPatch, ...],
    scripts_dir: Path,
) -> list[dict[str, Any]]:
    """Re-apply every patch this env's recipe makes applicable, then prove it.

    Called by `jobenv.install_env` AFTER pip and BEFORE the stamp is written, so
    an env that is stamped installed is an env whose patches are in. pip is what
    reverts them — it writes the distribution's own file over the edit — so this
    is the only place the two can be kept in step.

    `recipe_pins` selects the same way `check_patches` does: a patch whose
    distribution the recipe does not install is not run at all.

    THE PROOF IS `check_patches`, NOT THE EXIT CODE. Each script prints `PATCHED` or
    `ALREADY_PATCHED` and exits 0, and its own idea of success is the anchor it
    replaced — not the marker `crucible doctor` will grep for tomorrow. Running
    the checker over the result is what makes those one fact; anything short of
    `applied` raises, because an env that pip built and nobody patched is
    exactly the failure this function exists for.

    `runner` is for tests: a callable taking the argv and returning an object
    with `returncode` and `stdout`. The default runs it.
    """
    run = runner if runner is not None else _run_script
    for patch in patches:
        if patch.distribution not in recipe_pins:
            continue
        argv = [str(python), str(script_path(patch, scripts_dir)), str(env_dir)]
        result = run(argv)
        for line in (result.stdout or "").splitlines():
            if on_line is not None:
                on_line(line)
        if result.returncode != 0:
            raise PatchError(
                f"could not apply {patch.id} to {env_dir}: "
                f"`{' '.join(argv)}` exited {result.returncode}\n"
                + (result.stdout or "").strip()
            )

    rows = check_patches(env_dir, recipe_pins, patches=patches)
    unsound = [row for row in rows if row["status"] not in SOUND_STATUSES]
    if unsound:
        raise PatchError(
            "the patches were applied and the env still does not carry them: "
            + "; ".join(
                f"{row['id']} is {row['status']} — {row['detail']}"
                for row in unsound
            )
        )
    return rows


def _run_script(argv: list[str]) -> Any:
    # stderr JOINED to stdout: the scripts say `NOT_FOUND`, `AMBIGUOUS` and
    # `ANCHOR_NOT_FOUND` on stderr and `PATCHED` on stdout, and the refusal this
    # function raises has to quote whichever one it was.
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )


def site_packages(env_dir: Path) -> Path | None:
    """This venv's `site-packages`, or None when there is no venv.

    The python version is GLOBBED rather than assumed, exactly as BookForge's own
    patch scripts do it: the env is built from whatever interpreter the server is
    running under, which is not necessarily 3.11.
    """
    for candidate in sorted((env_dir / "lib").glob("python*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def check_patches(
    env_dir: Path,
    recipe_pins: dict[str, str],
    *,
    patches: tuple[EnvPatch, ...],
) -> list[dict[str, Any]]:
    """One row per patch: what it is, whether it is in, and what breaks if not.

    `recipe_pins` is `jobenv.recipe_pins(jobenv.recipe_for(spec))` for the env
    being checked — the caller passes it rather than this module reading it,
    because this module must work against a directory in a test with no recipes
    dir at all. A patch whose distribution is absent from those pins is
    `not_applicable`.
    """
    packages = site_packages(env_dir)
    rows: list[dict[str, Any]] = []
    for patch in patches:
        if patch.distribution not in recipe_pins:
            rows.append(
                _row(
                    patch,
                    NOT_APPLICABLE,
                    f"this env's recipe does not install {patch.distribution}, so "
                    f"{patch.rel_path} is not here to patch",
                )
            )
            continue
        if packages is None:
            rows.append(_row(patch, NO_ENV, f"there is no venv at {env_dir}"))
            continue
        target = packages / patch.rel_path
        if not target.is_file():
            rows.append(
                _row(
                    patch,
                    NO_FILE,
                    f"{target} is not there; the env does not hold the package "
                    "this patch edits",
                )
            )
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            rows.append(_row(patch, NO_FILE, f"could not read {target}: {exc}"))
            continue
        if patch.marker not in text:
            rows.append(
                _row(patch, MISSING, f"{patch.marker!r} is not in {patch.rel_path}")
            )
            continue
        if patch.absent_marker is not None and patch.absent_marker in text:
            rows.append(
                _row(
                    patch,
                    STALE,
                    f"{patch.rel_path} carries {patch.marker!r} but still has "
                    f"{patch.absent_marker!r} in it, so what the patch replaces "
                    "is still running",
                )
            )
            continue
        if patch.stale_marker is not None and patch.stale_marker not in text:
            rows.append(
                _row(
                    patch,
                    STALE,
                    f"{patch.rel_path} carries an OLDER version of this patch: "
                    f"{patch.marker!r} is there but {patch.stale_marker!r} is not",
                )
            )
            continue
        rows.append(_row(patch, APPLIED, f"{patch.rel_path} carries {patch.marker!r}"))
    return rows


def _row(patch: EnvPatch, status: str, detail: str) -> dict[str, Any]:
    return {
        "id": patch.id,
        "status": status,
        "applied": status == APPLIED,
        "path": patch.rel_path,
        "detail": detail,
        "why": patch.why,
    }


#: The `llm` env's appliers. Shipped by `pyproject.toml`'s
#: `crucible = ["envs/**/*"]`.
LLM_SCRIPTS_DIR = Path(__file__).resolve().parent / "envs" / "llm" / "patches"

MLX_LM_TOP_LOGPROBS = EnvPatch(
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

MLX_LM_FP32_LOGPROBS = EnvPatch(
    id="mlx-lm-fp32-logprobs",
    distribution="mlx-lm",
    rel_path="mlx_lm/generate.py",
    marker="wide = x.astype(mx.float32)",
    absent_marker="logits - mx.logsumexp(logits",
    stale_marker=None,
    script="patch_mlx_lm_fp32_logprobs.py",
    why=(
        "stock mlx-lm 0.31.3 normalizes the logprobs it returns in bf16, so a "
        "distribution read back sums to 0.94-1.06 and the decide door reports "
        "label_mass above 1; MlxLmEngine refuses to start without it"
    ),
)

MLX_LM_FATAL_GENERATION_THREAD = EnvPatch(
    id="mlx-lm-fatal-generation-thread",
    distribution="mlx-lm",
    rel_path="mlx_lm/server.py",
    marker="target=_crucible_fatal_thread(self._generate)",
    absent_marker="Thread(target=self._generate)",
    stale_marker=None,
    script="patch_mlx_lm_fatal_generation_thread.py",
    why=(
        "stock mlx-lm 0.31.3 lets an exception end its one generation thread "
        "and keeps the process up, so every accepted chat waits forever while "
        "the engine still looks alive (ContentStudio, 2026-09-25: 17-20 minutes "
        "in flight on a dead 27B); patched, the engine exits and the chat door "
        "fails the call by name. MlxLmEngine refuses to start without it"
    ),
)

MLX_LM_CACHE_COUNTERS = EnvPatch(
    id="mlx-lm-cache-counters",
    distribution="mlx-lm",
    rel_path="mlx_lm/generate.py",
    marker="_crucible_cache_counters(self.prompt_cache),",
    absent_marker="mx.async_eval(self._next_tokens, self._next_logprobs, token_context)",
    stale_marker=None,
    script="patch_mlx_lm_cache_counters.py",
    why=(
        "stock mlx-lm 0.31.3 updates its caches' counters lazily and never "
        "forces them, so a Qwen3.5/3.8 engine's graph grows until Metal's buffer "
        "count limit (499000) kills the generation thread; reproduced on the Mac "
        "27B with ContentStudio's real bodies on 2026-09-26 and gone with this "
        "patch. MlxLmEngine refuses to start without it"
    ),
)

LLM_PATCHES: tuple[EnvPatch, ...] = (
    MLX_LM_TOP_LOGPROBS,
    MLX_LM_FP32_LOGPROBS,
    MLX_LM_FATAL_GENERATION_THREAD,
    MLX_LM_CACHE_COUNTERS,
)


#: Inside the env's `site-packages`. The wheel that provides it is
#: `nvidia-cuda-runtime-cu13`, which the cuda-linux tts recipe pins.
CUDA_TOOLKIT_REL = "nvidia/cu13"

#: `(link, target)`, both relative to {@link CUDA_TOOLKIT_REL}. The target is
#: written VERBATIM as a relative symlink, so the pair survives the env being
#: moved or the whole guest being copied — an absolute target would point at the
#: path the env was built at.
CUDA_TOOLKIT_LINKS: tuple[tuple[str, str], ...] = (
    ("lib64", "lib"),
    ("lib/libcudart.so", "libcudart.so.13"),
)

#: Why any of this matters, in one sentence, for the row `doctor` prints.
CUDA_TOOLKIT_WHY = (
    "flashinfer JIT-builds SGLang's attention kernels with the nvcc inside the "
    "pip wheel and only does so when that directory looks like a toolkit; "
    "without these the install still succeeds and the first render on a card is "
    "where it goes wrong"
)


def _toolkit_dir(env_dir: Path) -> Path | None:
    packages = site_packages(env_dir)
    return None if packages is None else packages / CUDA_TOOLKIT_REL


def ensure_cuda_toolkit_links(env_dir: Path, on_line: Any = None) -> None:
    """Create the two symlinks, idempotently. Raises `PatchError` by name.

    Called after pip and BEFORE the stamp, for the same reason `apply` is: an
    env that is stamped installed is one whose links are in, or there is no
    stamp.

    A link that already points where it should is left alone and said so. One
    that exists and points somewhere ELSE is REFUSED rather than replaced —
    somebody or something put it there on purpose, and silently overwriting it
    would destroy the evidence of whatever did.
    """
    toolkit = _toolkit_dir(env_dir)
    if toolkit is None:
        raise PatchError(
            f"{env_dir} has no venv site-packages, so the CUDA toolkit directory "
            f"{CUDA_TOOLKIT_REL} cannot be found. The env was not built."
        )
    if not toolkit.is_dir():
        raise PatchError(
            f"{toolkit} is not there. The cuda-linux tts recipe pins "
            "nvidia-cuda-runtime-cu13, which is what provides it, so an env "
            "without it did not install what the recipe asked for."
        )
    for link_rel, target in CUDA_TOOLKIT_LINKS:
        link = toolkit / link_rel
        if link.is_symlink():
            current = os.readlink(link)
            if current == target:
                if on_line is not None:
                    on_line(f"cuda toolkit: {link_rel} -> {target} already there")
                continue
            raise PatchError(
                f"{link} is a symlink to {current!r}, not {target!r}. Crucible did "
                "not put it there and will not replace it — remove it by hand if "
                "it is wrong, so that whatever created it is not hidden."
            )
        if link.exists():
            raise PatchError(
                f"{link} exists and is not a symlink. It should be a link to "
                f"{target!r}; a real file or directory here is somebody else's "
                "doing and is not overwritten."
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        if on_line is not None:
            on_line(f"cuda toolkit: {link_rel} -> {target} created")


def check_cuda_toolkit_links(env_dir: Path) -> list[dict[str, Any]]:
    """One row per link, in the shape `doctor` already prints for patches."""
    toolkit = _toolkit_dir(env_dir)
    rows: list[dict[str, Any]] = []
    for link_rel, target in CUDA_TOOLKIT_LINKS:
        row_id = f"cuda-toolkit-{link_rel.replace('/', '-')}"
        if toolkit is None:
            rows.append(_link_row(row_id, link_rel, NO_ENV, "no venv site-packages"))
            continue
        if not toolkit.is_dir():
            rows.append(_link_row(
                row_id, link_rel, NO_FILE, f"{CUDA_TOOLKIT_REL} is not in this env"))
            continue
        link = toolkit / link_rel
        if not link.is_symlink():
            rows.append(_link_row(
                row_id, link_rel, MISSING,
                f"{link_rel} is not a symlink to {target!r}"))
            continue
        current = os.readlink(link)
        rows.append(_link_row(
            row_id, link_rel,
            APPLIED if current == target else STALE,
            f"{link_rel} -> {current!r}"
            + ("" if current == target else f", expected {target!r}")))
    return rows


def _link_row(row_id: str, rel: str, status: str, detail: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "status": status,
        "applied": status == APPLIED,
        "path": f"{CUDA_TOOLKIT_REL}/{rel}",
        "detail": detail,
        "why": CUDA_TOOLKIT_WHY,
    }


@dataclass(frozen=True)
class PatchSet:
    """One env type's patches and the directory its appliers live in."""

    patches: tuple[EnvPatch, ...]
    scripts_dir: Path


#: Job type -> its patches. A job type absent here has none, and `check`
#: answers it with no rows rather than inventing any.
REGISTRY: dict[str, PatchSet] = {
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

    Raises `PatchError` by name, exactly as `apply_patches` does — it IS that
    function, over this type's table.
    """
    found = REGISTRY.get(job_type)
    if found is None:
        return []
    return apply_patches(
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
    return check_patches(env_dir, recipe_pins, patches=found.patches)


def require_applied(patch: EnvPatch, env_dir: Path) -> None:
    """Raise `PatchError` naming the status unless `patch` is applied here.

    For an engine that states a number BECAUSE of a patch: it is asked with the
    patch's own distribution as the pins, since the engine running at all means
    that distribution is what it runs.
    """
    [row] = check_patches(env_dir, {patch.distribution: ""}, patches=(patch,))
    if row["status"] != APPLIED:
        raise PatchError(
            f"{patch.id} is {row['status']} in {env_dir}: {row['detail']}. "
            f"{row['why']}"
        )


__all__ = [
    "APPLIED",
    "CUDA_TOOLKIT_LINKS",
    "CUDA_TOOLKIT_REL",
    "EnvPatch",
    "LLM_PATCHES",
    "LLM_SCRIPTS_DIR",
    "MLX_LM_CACHE_COUNTERS",
    "MLX_LM_FATAL_GENERATION_THREAD",
    "MLX_LM_FP32_LOGPROBS",
    "MLX_LM_TOP_LOGPROBS",
    "MISSING",
    "NOT_APPLICABLE",
    "NO_ENV",
    "NO_FILE",
    "PatchError",
    "PatchSet",
    "REGISTRY",
    "SOUND_STATUSES",
    "STALE",
    "apply",
    "apply_patches",
    "check",
    "check_cuda_toolkit_links",
    "check_patches",
    "ensure_cuda_toolkit_links",
    "patched_job_types",
    "patches_for",
    "require_applied",
    "site_packages",
]
