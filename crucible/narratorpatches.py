"""Two edits to somebody else's installed package, which pip cannot express.

PHASE3-TTS.md section 4: the `higgs-v3` tts env needs two site-packages patches
re-applied after any upgrade of its pins, and `crucible doctor` reports them by
name. This module **applies** them (`apply`, run by `jobenv.install_env` after
pip) and **checks** them (`check`, run by `crucible doctor`).

IT ONLY CHECKED UNTIL 2026-09-15, AND THAT IS THE BUG THIS PARAGRAPH IS FOR
---------------------------------------------------------------------------
"Must be re-applied after any upgrade of these pins" — said by the recipe, by
this module's old docstring and by PHASE3-TTS.md — named no one who would do it,
and nobody did. `crucible install tts --narrator-engine higgs-v3 --build --force`
deletes the env and pip-installs the recipe, which restores BOTH files pristine;
the install then reported success and `crucible doctor` reported the patches
`missing` in the same breath, from two commands nobody runs together.

MEASURED on owens-pc, 2026-09-15: a rebuild at 07:34 replaced
`vllm_omni/.../higgs_audio_v3.py`, and from 07:46 every Higgs load failed at
narrator's sentinel proof — `/tmp/narrator-higgs3-<pid>-<hash>.log.sentinel.jsonl
holds no records`, the file 0 bytes, because the code that writes a record per
invocation had just been uninstalled. Renders at 07:23 from the same recipe, the
same pins and the same narrator sha worked. A build that reverts a required patch
and calls itself installed is the defect; the fix is that the build applies it and
refuses to finish if `check` then disagrees.

THE APPLIERS ARE VENDORED, AND WHY THAT IS THE LESSER EVIL
-----------------------------------------------------------
`envs/tts/patches/patch_vllm.py` and `envs/tts/patches/patch_sentinel_filter.py`
are byte-identical copies of BookForge's `electron/scripts/higgs/`, the scripts
every measurement in `electron/data/higgs-models.json` was taken against. They
are copies for the same reason the table below is one: a Crucible server must not
need a BookForge checkout, and narrator's wheel does not carry them (it ships
`engine/higgs/launch/` and nothing else). **The owed move is narrator shipping
its own patches** — the engine that requires a patched server is the honest owner
of the patch — and it is owed on the same ruling as extracting narrator into its
own repo (PHASE3-TTS.md section 4). Until then the copies are pinned by content:
`tests/test_narrator_patches.py` reads `REL`/`MARKER`/`ABSENT_MARKER` out of each
script and asserts they are the table's, so the two halves cannot drift apart in
silence, and `git hash-object` compares a copy to BookForge's blob by hand.

Why the table is duplicated rather than imported
------------------------------------------------
It is a mirror of `electron/tool-paths.ts`'s `HIGGS_PATCHES` in the BookForge
repo, markers and all, and the duplication is deliberate for the reason that file
gives about its own: this is the module every env question goes through, and a
Crucible server must not need a BookForge checkout to answer "is this env
sound". The markers were measured on the certifying box (vllm-omni 0.28.0,
2026-09-05) and are quoted rather than paraphrased, so a `git grep` finds both
copies when one moves.

BOTH PATCHES ARE EDITS TO THE vLLM STACK, WHICH ONLY ONE BACKEND HAS
--------------------------------------------------------------------
`envs/tts/higgs-v3-cuda-linux.txt` pins `vllm` and `vllm-omni`;
`envs/tts/mlx-darwin.txt` pins neither, because Higgs v3 on the Mac is narrator's
in-process MLX backend and there is no served vLLM under it at all. So on
`mlx-darwin` these files are not merely unpatched, they are **not there** — and a
check that reports that as `no_such_file` makes `crucible doctor` call a perfectly
sound Mac unhealthy, which is what it did until 2026-09-13.

The fix is not a backend list in this module. That would be a second copy of a
fact the recipe already states, which is the shape `docs/ARCHITECTURE.md` §1
names. Each patch instead declares the **distribution** it edits, and `check`
takes the pins of the recipe that built the env: a patch whose distribution is
not in that recipe is `not_applicable`, and says so by name rather than being
silently skipped. Add `vllm-omni` to the Mac recipe one day and the check starts
running there on its own, with nothing here to remember to change.

THE STALE MARKER MOVED TO v3 ON 2026-09-13, AND WHY IT IS THAT STRING
---------------------------------------------------------------------
The sentinel patch gained a third generation: it now appends one machine-readable
record per invocation to `$HIGGS_SENTINEL_REPORT`, and narrator reads those
records instead of running three regexes over the server's LOG FILE to decide
whether a 19 GB model may render a book (ARCHITECTURE.md R4 — a log line is never
load-bearing). The `stale_marker` therefore moved from v2's log format string
`"final=%s, window=%d frames"` to `"HIGGS_SENTINEL_REPORT"`.

The choice between that and the helper name `_sentinel_report` is not arbitrary.
Both are unique to v3. The ENV VAR is the interface — the patched file reads it,
narrator exports it and reads the file back — so it is the one string that
genuinely must not drift, and renaming it SHOULD invalidate every doctor's table,
which is exactly what a stale marker is for. A helper could be renamed harmlessly
and this check would then lie.

**Every v1 and v2 env now reports `stale` until the installer re-applies the
patch.** That is intended and loud rather than a regression: those envs really are
carrying a patch whose records narrator can no longer read.

A NAME IN THE SPEC IS STALE, AND THIS IS WHERE IT SHOWS
-------------------------------------------------------
PHASE3-TTS.md section 4 and narrator's own `pyproject.toml` both name
`work/patch_tail_trim.py` as the second patch. It is **retired**: the live script
is `patch_sentinel_filter.py`, and the difference is not cosmetic. The retired
one wrote `_trim_trailing_sentinel_frames`, which is why the check below greps
for `_filter_sentinel_frames` instead — grepping for the helper both scripts
write would certify a band-aided env as patched. The `absent` marker is the other
half of the proof: upstream's one-frame trim `[:, :-1]` occurs twice in the
pristine stage processor and zero times once the filter patch is in.
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
class NarratorPatch:
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
    #: The applier in `envs/tts/patches/`, run as `<env python> <script> <env>`.
    #: Every script is idempotent and refuses by name (`ANCHOR_NOT_FOUND`) when
    #: upstream has moved the code it edits, rather than skipping quietly.
    script: str
    why: str


NARRATOR_PATCHES: tuple[NarratorPatch, ...] = (
    NarratorPatch(
        id="vllm-negative-token-id",
        distribution="vllm",
        rel_path="vllm/v1/engine/input_processor.py",
        marker="min_input_id != -100",
        absent_marker=None,
        stale_marker=None,
        script="patch_vllm.py",
        why=(
            "vLLM 0.28's blanket negative-token-id rejection fires on vllm-omni's "
            "audio placeholder (-100), so every voice-clone request returns HTTP "
            "400 and only the default voice can serve"
        ),
    ),
    NarratorPatch(
        id="higgs-sentinel-filter",
        distribution="vllm-omni",
        rel_path=(
            "vllm_omni/model_executor/stage_input_processors/higgs_audio_v3.py"
        ),
        marker="_filter_sentinel_frames",
        absent_marker="[:, :-1]",
        stale_marker="HIGGS_SENTINEL_REPORT",
        script="patch_sentinel_filter.py",
        why=(
            "without it every rendered chunk ends with ~240 ms of audible garbage "
            "— the ramp-down sentinels are substituted with codec code 0, which is "
            "a VALID code that decodes to real sound, and only one of the seven "
            "frames they smear across is trimmed"
        ),
    ),
)

#: The narrator engine these belong to. They are both edits to the vllm-omni
#: stack Higgs v3 serves through, so an engine on another stack needs neither.
PATCHED_ENGINE = "higgs-v3"

#: Where the vendored appliers live. Shipped by `pyproject.toml`'s
#: `crucible = ["envs/**/*"]`, so a wheel carries them; `tests/test_wheel.py`
#: is what keeps that glob from silently matching nothing.
SCRIPTS_DIR = Path(__file__).resolve().parent / "envs" / "tts" / "patches"


class PatchError(RuntimeError):
    """A patch could not be applied, or was not there after applying it."""


def script_path(patch: NarratorPatch, scripts_dir: Path | None = None) -> Path:
    """The applier for this patch, or `PatchError` naming the missing file.

    `scripts_dir` is the env type's own `envs/<job type>/patches/`, passed by
    `crucible/envpatches.py`; None is the tts directory this module owns.
    """
    path = (SCRIPTS_DIR if scripts_dir is None else scripts_dir) / patch.script
    if not path.is_file():
        raise PatchError(
            f"the applier for {patch.id} is not installed: {path} is not there. "
            "A wheel built without `envs/**/*` has the recipes and not the "
            "patches, which is an env that installs and does not render"
        )
    return path


def apply(
    env_dir: Path,
    python: Path,
    recipe_pins: dict[str, str],
    *,
    on_line: Any = None,
    runner: Any = None,
    patches: tuple[NarratorPatch, ...] | None = None,
    scripts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Re-apply every patch this env's recipe makes applicable, then prove it.

    Called by `jobenv.install_env` AFTER pip and BEFORE the stamp is written, so
    an env that is stamped installed is an env whose patches are in. pip is what
    reverts them — it writes the distribution's own file over the edit — so this
    is the only place the two can be kept in step.

    `recipe_pins` selects the same way `check` does: a patch whose distribution
    the recipe does not install is not run at all, so `mlx-darwin` (no `vllm`,
    no `vllm-omni`) gets neither and is not called broken for it.

    THE PROOF IS `check`, NOT THE EXIT CODE. Each script prints `PATCHED` or
    `ALREADY_PATCHED` and exits 0, and its own idea of success is the anchor it
    replaced — not the marker `crucible doctor` will grep for tomorrow. Running
    the checker over the result is what makes those one fact; anything short of
    `applied` raises, because an env that pip built and nobody patched is
    exactly the failure this function exists for.

    `runner` is for tests: a callable taking the argv and returning an object
    with `returncode` and `stdout`. The default runs it.

    `patches` and `scripts_dir` are another env type's table and appliers
    (`crucible/envpatches.py`, which is how the `llm` env's mlx-lm patch runs
    through this same machinery); None is the tts table this module owns.
    """
    run = runner if runner is not None else _run_script
    table = NARRATOR_PATCHES if patches is None else patches
    for patch in table:
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

    rows = check(env_dir, recipe_pins, patches=table)
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


def check(
    env_dir: Path,
    recipe_pins: dict[str, str],
    *,
    patches: tuple[NarratorPatch, ...] | None = None,
) -> list[dict[str, Any]]:
    """One row per patch: what it is, whether it is in, and what breaks if not.

    `recipe_pins` is `jobenv.recipe_pins(jobenv.recipe_for(spec))` for the env
    being checked — the caller passes it rather than this module reading it,
    because this module must work against a directory in a test with no recipes
    dir at all. A patch whose distribution is absent from those pins is
    `not_applicable`: see the module docstring for why that is read off the
    recipe and not off a backend name.
    """
    packages = site_packages(env_dir)
    rows: list[dict[str, Any]] = []
    for patch in NARRATOR_PATCHES if patches is None else patches:
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


def _row(patch: NarratorPatch, status: str, detail: str) -> dict[str, Any]:
    return {
        "id": patch.id,
        "status": status,
        "applied": status == APPLIED,
        "path": patch.rel_path,
        "detail": detail,
        "why": patch.why,
    }


# ---------------------------------------------------------------------------
# THE OTHER THING pip CANNOT EXPRESS: a directory that must LOOK like a toolkit
# ---------------------------------------------------------------------------
#
# Not a patch, and deliberately not forced into `NarratorPatch`. A patch is an
# edit to a file, identified by a marker inside it; these are two symlinks, and
# `marker`, `absent_marker` and `stale_marker` would all be meaningless fields
# on them. Same discipline, same moment in the install, different shape.
#
# WHAT THEY ARE FOR. flashinfer JIT-builds SGLang's attention kernels using the
# CUDA 13 nvcc that ships inside the pip wheel, and it will only do so once that
# directory looks like a real toolkit install — `lib64` beside `lib`, and an
# unsuffixed `libcudart.so`. `CUDA_HOME` points at the same directory and
# `serve_higgs_sgl.sh` exports it.
#
# WHY THEIR ABSENCE IS DANGEROUS RATHER THAN OBVIOUS. Nothing fails at install
# time; pip is perfectly happy. The env stamps installed, `crucible doctor`
# reports no patch rows for this stack (it has none), and the failure arrives
# later at the first render on a card — which is the knob-whose-absence-looks-
# like-health shape this project has now paid for several times.
#
# THEY WERE CREATED BY HAND on owens-pc on 2026-09-15 to get the first SGLang
# env working, and `envs/tts/higgs-v3-cuda-linux.txt` has claimed since that day
# that "`crucible/narratorpatches.py` creates and checks them". That sentence
# was aspirational: nothing here did. This is that sentence becoming true.

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
