from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APPLIED = "applied"
MISSING = "missing"
STALE = "stale"
NO_FILE = "no_such_file"
NO_ENV = "no_env"
NOT_APPLICABLE = "not_applicable"

SOUND_STATUSES: frozenset[str] = frozenset({APPLIED, NOT_APPLICABLE})


@dataclass(frozen=True)
class EnvPatch:

    id: str
    distribution: str
    rel_path: str
    marker: str
    absent_marker: str | None
    stale_marker: str | None
    script: str
    why: str
    creates: bool = False


class PatchError(RuntimeError):
    ...


def script_path(patch: EnvPatch, scripts_dir: Path) -> Path:
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
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )


def site_packages(env_dir: Path) -> Path | None:
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
        if not target.is_file() and patch.creates:
            rows.append(_row(patch, MISSING, f"{patch.rel_path} has not been placed"))
            continue
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

MLX_LM_DECIDE_ITEMS = EnvPatch(
    id="mlx-lm-decide-items",
    distribution="mlx-lm",
    rel_path="mlx_lm/server.py",
    marker="_crucible_items.mlx_lm_run_job(self.model_provider, request, rqueue)",
    absent_marker=None,
    stale_marker=None,
    script="patch_mlx_lm_decide_items.py",
    why=(
        "stock mlx-lm 0.31.3 answers one prompt per request, so the decide door's "
        "items form (many questions about one state) costs a request each on the "
        "Mac; the patch adds POST /v1/crucible/items, which runs the shared state "
        "once and every item from a copy of its cache on the generation thread. "
        "MlxLmEngine applies it itself at start"
    ),
)

MLX_LM_DECIDE_ITEMS_HELPER = EnvPatch(
    id="mlx-lm-decide-items-helper",
    distribution="mlx-lm",
    rel_path="mlx_lm/_crucible_items.py",
    marker="ITEMS_VERSION = 1",
    absent_marker=None,
    stale_marker=None,
    script="patch_mlx_lm_decide_items_helper.py",
    why=(
        "the items route runs Crucible's engines/items_forward.py, copied into "
        "the env as mlx_lm/_crucible_items.py; an older copy reads the wrong "
        "request. MlxLmEngine applies it itself at start"
    ),
    creates=True,
)

SELF_APPLIED_LLM_PATCHES: tuple[EnvPatch, ...] = (
    MLX_LM_DECIDE_ITEMS,
    MLX_LM_DECIDE_ITEMS_HELPER,
)

LLM_PATCHES: tuple[EnvPatch, ...] = (
    MLX_LM_TOP_LOGPROBS,
    MLX_LM_FP32_LOGPROBS,
    MLX_LM_FATAL_GENERATION_THREAD,
    MLX_LM_CACHE_COUNTERS,
    *SELF_APPLIED_LLM_PATCHES,
)


CUDA_TOOLKIT_REL = "nvidia/cu13"

CUDA_TOOLKIT_LINKS: tuple[tuple[str, str], ...] = (
    ("lib64", "lib"),
    ("lib/libcudart.so", "libcudart.so.13"),
)

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
                "not put it there and will not replace it, so that whatever "
                f"created it is not hidden. If it is wrong, run `rm {link}` and "
                "then `crucible install tts --narrator-engine "
                f"{env_dir.name.removeprefix('tts-')}`."
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

    patches: tuple[EnvPatch, ...]
    scripts_dir: Path


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
    found = REGISTRY.get(job_type)
    if found is None:
        return []
    return check_patches(env_dir, recipe_pins, patches=found.patches)


def ensure_applied(
    patches: tuple[EnvPatch, ...], env_dir: Path, python: Path, *, runner: Any = None
) -> None:
    pins = {patch.distribution: "" for patch in patches}
    rows = check_patches(env_dir, pins, patches=patches)
    if all(row["status"] == APPLIED for row in rows):
        return
    apply_patches(
        env_dir, python, pins, runner=runner, patches=patches, scripts_dir=LLM_SCRIPTS_DIR
    )


def require_applied(patch: EnvPatch, env_dir: Path) -> None:
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
    "MLX_LM_DECIDE_ITEMS",
    "MLX_LM_DECIDE_ITEMS_HELPER",
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
    "SELF_APPLIED_LLM_PATCHES",
    "SOUND_STATUSES",
    "STALE",
    "apply",
    "apply_patches",
    "check",
    "check_cuda_toolkit_links",
    "check_patches",
    "ensure_applied",
    "ensure_cuda_toolkit_links",
    "patched_job_types",
    "patches_for",
    "require_applied",
    "site_packages",
]
