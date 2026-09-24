"""`crucible/engines/narrator.py` — the managed subprocess and its channel.

PHASE3-TTS.md section 4. Everything here drives `tests/fake_narrator.py`, which
speaks narrator's real JSON-lines wire, through the **real** engine: the only
thing the double replaces is the argv (`tests/fake_narrator_engine.py`).

Nothing here needs an accelerator, and nothing here is a mock of the engine. The
pipes, the reader thread, the correlation, the refusals and `stop()` are the code
that will run on the PC.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from crucible import procgroup
from crucible.engines import (
    NARRATOR_ENGINES,
    EngineError,
    NarratorEngine,
    build_voice_engine,
)
from crucible.engines import base as engine_base
from crucible.engines.base import SubprocessEngine as BaseEngine
from crucible.engines.mlx_lm import MlxLmEngine
from crucible.engines.narrator import (
    ENGINE_VARIABLE,
    ENV_PREFIX_VARIABLE,
    STACK_ENV_PREFIX_VARIABLE,
    env_prefix_variable_for,
    MAX_NUM_SEQS_VARIABLE,
    MEM_FRACTION_VARIABLE,
    CONTEXT_LENGTH_VARIABLE,
    HIGGS_V3_MLX_WEIGHTS_GB,
    MLX_BATCH_VARIABLE,
    MLX_CACHE_LIMIT_VARIABLE,
    MLX_MEM_BUDGET_VARIABLE,
    MLX_TIERS,
    MODULE,
    STACK_VARIABLE,
    EngineWouldNotStop,
    higgs_env_prefix,
    mlx_render_profile,
)
from crucible.engines.vllm import VllmEngine
from crucible.errors import JobCancelled
from crucible.ttsstream import STREAM_BATCH_WIDTH
from crucible.narratorvoices import (
    DOCUMENT_VARIABLE,
    MLX_MODEL_VARIABLE,
    VoicesDocument,
    write_document,
)
from crucible.voices import NARRATOR_ENGINE_SAMPLING, parse_voice

from .conftest import end_process_tree
from .fake_narrator_engine import FAKE_NARRATOR, FakeNarratorEngine
from .test_voices import GOOD

BATCH_TERMINAL = frozenset({"batch_done"})

#: A narrator engine id this build does NOT name. Owen's ruling of 2026-09-14
#: left `higgs-v3` as the only entry in `voices.NARRATOR_ENGINE_SAMPLING`, so
#: two of `NarratorEngine`'s refusals — a voices document handed to an engine
#: that reads none, and a serving stack given to an engine that starts none —
#: can no longer be reached THROUGH `build_voice_engine`, which refuses an
#: unnamed engine first. They are the rules the NEXT engine arrives into, they
#: fire in the constructor, and so they are proved by constructing it directly.
#: Deleting them instead would mean the second engine's first render finds out
#: at the spawn what a constructor could have said.
A_FUTURE_ENGINE = "an-engine-with-no-document"

#: owens-mac-studio's unified memory, which is what the in-process arm's tier is
#: chosen by. Stated rather than probed: this suite runs on Windows and Linux
#: too, and a test whose answer depends on the machine it runs on cannot assert
#: a row of a measured table.
A_64_GIB_MAC = 64 * 1024 * 1024 * 1024
#: A Mac under the `moderate` band's 28,000 MiB floor — the row BookForge's
#: table calls `light`.
A_16_GIB_MAC = 16 * 1024 * 1024 * 1024


def a_document(
    home: Path, weights: Path, voice_id: str = "deathstalker"
) -> VoicesDocument:
    """The voices document a load of `voice_id` from `weights` writes — the
    real writer on a real manifest, because what a `higgs-v3` engine is
    constructed with is this and nothing simpler."""
    manifest = parse_voice(
        GOOD.replace('id = "probe"', f'id = "{voice_id}"'),
        Path(f"{voice_id}.toml"),
        voice_id,
    )
    return write_document(home, manifest, manifest.spec("cuda-linux"), weights)


@pytest.fixture(autouse=True)
def brief_quit_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the wait `stop()` gives `quit` before it signals.

    Thirty seconds is right for a loaded SGLang-Omni releasing CUDA from inside
    itself. Several tests here deliberately leave a worker that is not reading
    its stdin at all, and paying that wait for each of them would put minutes on
    the suite for a number none of them is about.
    """
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    # And the readiness poll, which is two seconds because a vLLM load takes
    # minutes and polling it harder buys nothing. Every test here is up in
    # milliseconds, so the interval is the whole of its runtime.
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    return directory


@pytest.fixture
def engine(tmp_path: Path, weights: Path) -> Iterator[FakeNarratorEngine]:
    built = FakeNarratorEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-deathstalker.log",
        # The wire, not the server: this engine's argv is the fake worker and
        # no vllm-omni is started, so there is nothing for the three HIGGS_*
        # variables to configure. They have their own tests below.
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        # The document IS read, by the fake exactly as by narrator: a load
        # names a voice in it or is refused.
        voices=a_document(tmp_path, weights),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    yield built
    try:
        built.stop()
    except EngineError:
        pass


def up(engine: FakeNarratorEngine, weights: Path) -> FakeNarratorEngine:
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0)
    return engine


# --------------------------------------------------------------- the spawn


def test_the_argv_is_narrator_serve_and_nothing_else() -> None:
    """narrator takes no configuration on the command line, so neither does this."""
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path("/opt/env/bin/python"),
        log_path=Path("/tmp/x.log"),
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    assert built.command(Path("/weights"), "owen", 7100, []) == [
        "/opt/env/bin/python",
        "-m",
        MODULE,
    ]
    # The voice, the weights directory and the port are deliberately absent: the
    # first two ride the `load` message, and narrator binds nothing.
    assert "owen" not in built.command(Path("/weights"), "owen", 7100, [])
    assert built.environment()[ENGINE_VARIABLE] == A_FUTURE_ENGINE


def a_venv(tmp_path: Path, name: str = "tts-higgs-v3") -> Path:
    """A directory shaped like the env `crucible install tts` builds: a
    `bin/python` under a root carrying `pyvenv.cfg`. That file is one of the
    three things `higgs_env_prefix` reads to confirm a prefix is a prefix; the
    real env on the PC carries `bin/vllm-omni` as well (see
    `a_venv_on_a_conda_env`)."""
    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.write_text("", encoding="utf-8")
    return python


def test_a_higgs_worker_is_told_the_stack_the_env_and_the_width(
    tmp_path: Path,
) -> None:
    """The three variables narrator refuses by name, and the defect that found
    them: Crucible's first real `tts` render exited 3 before `ready` with
    `HIGGS_STACK is not set`."""
    python = a_venv(tmp_path)
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3",
        python,
        tmp_path / "x.log",
        serving_stack="sglang-omni",
        max_num_seqs=16,
        mem_fraction=None,
        context_length=None,
        voices=document,
        mlx_total_bytes=None,
    )
    environment = built.environment()
    assert environment[ENGINE_VARIABLE] == "higgs-v3"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment[STACK_VARIABLE] == "sglang-omni"
    assert environment[env_prefix_variable_for("sglang-omni")] == str(
        python.parent.parent)
    # AND NOT the other stack's name, which that launcher never reads.
    assert ENV_PREFIX_VARIABLE not in environment
    assert environment[MAX_NUM_SEQS_VARIABLE] == "16"
    # The fourth thing, on both arms: where narrator resolves the voice.
    assert environment[DOCUMENT_VARIABLE] == str(document.path)
    # And NOT the MLX arm's base weights: a checkpoint voice never reads them.
    assert MLX_MODEL_VARIABLE not in environment


def test_a_higgs_worker_without_a_document_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """The keeper's finding on the Mac and the launcher's on the PC, one
    refusal: narrator resolves a Higgs v3 voice by name in the
    NARRATOR_HIGGS_VOICES document, on both arms, so an engine with none can
    load nothing — and says so before a process exists."""
    python = a_venv(tmp_path)
    for stack in ("sglang-omni", None):
        with pytest.raises(EngineError) as caught:
            build_voice_engine(
                "higgs-v3", python, tmp_path / "x.log",
                serving_stack=stack, max_num_seqs=16,
                mem_fraction=None, context_length=None,
                voices=None, mlx_total_bytes=None)
        assert DOCUMENT_VARIABLE in str(caught.value)
        assert "narratorvoices" in str(caught.value)


def test_a_document_for_an_engine_that_reads_none_is_refused(tmp_path: Path) -> None:
    """An engine outside `DOCUMENT_READERS` takes its weights on the load
    message and reads no NARRATOR_HIGGS_* variable; a document handed to it is
    a statement of where the weights are that nothing reads.

    Constructed directly rather than through `build_voice_engine`: see
    `A_FUTURE_ENGINE`."""
    with pytest.raises(EngineError) as caught:
        NarratorEngine(
            narrator_engine=A_FUTURE_ENGINE,
            python=a_venv(tmp_path, "tts-future"),
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None,
        )
    assert "takes its weights on the load message" in str(caught.value)
    assert "'higgs-v3'" in str(caught.value)


def test_the_width_is_a_string_because_an_environment_holds_strings(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    for name, value in built.environment().items():
        assert isinstance(value, str), name


def test_a_higgs_worker_with_no_width_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=None,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert MAX_NUM_SEQS_VARIABLE in str(caught.value)
    assert "[voice.serving]" in str(caught.value)


def test_a_width_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=0,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert "at least 1" in str(caught.value)


def a_venv_on_a_conda_env(tmp_path: Path) -> Path:
    """THE LIVE WSL LAYOUT, which is what broke on 2026-09-14.

    `workerenv.install_worker_env` builds the tts env with `sys.executable -m
    venv`, and on that box the server's own interpreter is a CONDA env — so
    `~/.crucible/envs/tts-higgs-v3/bin/python` is a symlink into
    `~/anaconda3/envs/crucible/bin`, and the conda env at the other end has no
    `pyvenv.cfg` because no conda env ever has one. The stack (`vllm-omni`,
    and the `nvidia/cu13` tree CUDA_HOME is built from) is installed in the
    VENV, so the venv is the prefix and the symlink is a dead end."""
    conda = tmp_path / "anaconda3" / "envs" / "crucible"
    (conda / "bin").mkdir(parents=True)
    (conda / "conda-meta").mkdir()
    base = conda / "bin" / "python3.11"
    base.write_text("", encoding="utf-8")

    venv = tmp_path / ".crucible" / "envs" / "tts-higgs-v3"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {conda / 'bin'}\nexecutable = {base}\n", encoding="utf-8"
    )
    (venv / "bin" / "sgl-omni").write_text("", encoding="utf-8")
    python = venv / "bin" / "python"
    try:
        python.symlink_to(base)
    except OSError as refused:  # a host that cannot make one at all
        pytest.skip(f"this layout is a symlink and this host refused one: {refused}")
    return python


def test_the_prefix_is_the_env_the_stack_is_in_not_the_one_it_symlinks_to(
    tmp_path: Path,
) -> None:
    """The 2026-09-14 regression: every `tts` job on the live WSL server died
    at engine start because the prefix was taken from `Path(python)
    .resolve()`, which follows a venv's `bin/python` symlink out of the env
    and into the base interpreter — then demanded a `pyvenv.cfg` there, which
    a conda env never has.

    `$HIGGS_ENV/bin/vllm-omni` is the file narrator's launcher execs and
    `$HIGGS_ENV/lib/python3.11/site-packages/nvidia/cu13` is its CUDA_HOME;
    both are in the VENV. The base interpreter's prefix has neither."""
    python = a_venv_on_a_conda_env(tmp_path)
    venv = python.parent.parent
    built = build_voice_engine(
        "higgs-v3", python, tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    prefix = built.environment()[env_prefix_variable_for("sglang-omni")]
    assert prefix == str(venv)
    assert higgs_env_prefix(python, "sglang-omni") == venv
    # Said the other way round, because this is the value that was emitted:
    # the conda env the venv was built from is not the prefix.
    assert str(python.resolve().parent.parent) != prefix
    # And what the launcher would exec is under the prefix that was emitted.
    assert (Path(prefix) / "bin" / "sgl-omni").is_file()


def test_a_conda_env_is_its_own_prefix(tmp_path: Path) -> None:
    """A conda env has `conda-meta/` and no `pyvenv.cfg`. Handed one directly
    — the shape `crucible install` produces when the tts env IS a conda env
    rather than a venv on top of one — the prefix is that env itself, not a
    parent walked to in search of a file conda does not write."""
    conda = tmp_path / "anaconda3" / "envs" / "tts-higgs-v3"
    (conda / "bin").mkdir(parents=True)
    (conda / "conda-meta").mkdir()
    python = conda / "bin" / "python"
    python.write_text("", encoding="utf-8")
    built = build_voice_engine(
        "higgs-v3", python, tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None)
    assert built.environment()[env_prefix_variable_for("sglang-omni")] == str(conda)
    assert higgs_env_prefix(python, "sglang-omni") == conda


def test_an_interpreter_that_is_not_in_a_venv_is_refused(tmp_path: Path) -> None:
    """`HIGGS_ENV` is the prefix narrator's launch script builds CUDA_HOME,
    PATH and the vllm-omni binary from. A directory that is no kind of env —
    no `bin/vllm-omni`, no `pyvenv.cfg`, no `conda-meta/` — produces
    `No such file` at the end of a launch instead of here."""
    stray = tmp_path / "not-an-env" / "bin" / "python"
    stray.parent.mkdir(parents=True)
    stray.write_text("", encoding="utf-8")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", stray, tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=16,
            mem_fraction=None, context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
            mlx_total_bytes=None)
    assert env_prefix_variable_for("sglang-omni") in str(caught.value)
    assert "pyvenv.cfg" in str(caught.value)
    assert "conda-meta" in str(caught.value)
    assert "sgl-omni" in str(caught.value)
    # The engine it could not start is named, not just the prefix it read.
    assert "narrator (higgs-v3)" in str(caught.value)


def test_an_arm_that_starts_no_server_is_told_none_of_the_three(
    tmp_path: Path,
) -> None:
    """`mlx-darwin` renders in process — `HiggsV3MlxEngine` reads neither
    HIGGS_STACK nor HIGGS_MAX_NUM_SEQS, and there is no launch script for
    HIGGS_ENV to mean anything to. Three levers read by nothing is how a Mac
    spawn ends up looking served.

    The DOCUMENT is not one of the three: the MLX arm reads it exactly as the
    served arm does, which is what the keeper found on the Mac."""
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC)
    environment = built.environment()
    assert environment[ENGINE_VARIABLE] == "higgs-v3"
    for name in (STACK_VARIABLE, MAX_NUM_SEQS_VARIABLE,
                 *STACK_ENV_PREFIX_VARIABLE.values()):
        assert name not in environment, name
    assert environment[DOCUMENT_VARIABLE] == str(document.path)


def test_the_arm_that_starts_no_server_is_told_its_batch_width(
    tmp_path: Path,
) -> None:
    """THE DEFECT THIS EXISTS FOR (owens-mac-studio, 2026-09-15). "Reads none
    of the three" was read as "reads nothing", and the in-process arm has a
    width of its own: `NARRATOR_HIGGS3_MLX_BATCH`, which narrator defaults to 1
    — one chunk at a time. A `thirdreich` book rendered at 1.99x realtime where
    the same machine measures 13.97x at 64. The width is not optional and its
    absence is silent, which is why it is asserted here rather than left to the
    absence test above."""
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC)
    environment = built.environment()
    assert environment[MLX_BATCH_VARIABLE] == "64"
    # STATED, not derived from the manifest: `max_num_seqs=16` above is the
    # served arm's vLLM admission width and means nothing to a Metal backend,
    # so the two numbers must not be the same number by accident.
    assert environment[MLX_BATCH_VARIABLE] != str(16)


def test_a_stated_mem_fraction_and_context_length_reach_the_served_arm(
    tmp_path: Path,
) -> None:
    """Owen's ruling of 2026-09-19 — `[voice.serving]`'s two new levers.

    `HIGGS_SGL_MEM_FRACTION` is read by narrator's own launcher
    (`engine/higgs/launch/serve_higgs_sgl.sh:59`, which defaults it to 0.60).
    `HIGGS_CONTEXT_LENGTH` is the agreed name for a channel narrator is growing
    on its own branch and NOTHING reads it at the pinned sha — see
    `environment()`, which says so at the line that sets it.
    """
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=4,
        mem_fraction=0.48, context_length=8192,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None).environment()
    assert environment[MEM_FRACTION_VARIABLE] == "0.48"
    assert environment[CONTEXT_LENGTH_VARIABLE] == "8192"
    assert environment[MAX_NUM_SEQS_VARIABLE] == "4"


def test_a_voice_stating_neither_sets_neither_variable(tmp_path: Path) -> None:
    """Absent is narrator's launcher default (0.60, and the Higgs builder's
    4096), which is a number in a file with an owner. Writing it back from here
    would be Crucible restating a value it did not choose — the shape that put
    narrator's own `CHARS_PER_SEC` 15.0 into two voice manifests."""
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=None).environment()
    assert MEM_FRACTION_VARIABLE not in environment
    assert CONTEXT_LENGTH_VARIABLE not in environment


def test_both_levers_reach_the_IN_PROCESS_arm_too(tmp_path: Path) -> None:
    """NOT `cuda-linux` ONLY, and that is the difference from `max_num_seqs`.

    `HIGGS_MAX_NUM_SEQS` is the served arm's vocabulary and the MLX arm takes
    its width from a measured tier table instead. These two are not like that:
    Owen, 2026-09-19 — *"we're going to want to configure darwin to work the
    same way. context limits and such."* So they are stated on either arm and
    narrator answers, by name at load, for a knob its MLX backend lacks.
    """
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=0.48, context_length=8192,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC).environment()
    assert environment[MEM_FRACTION_VARIABLE] == "0.48"
    assert environment[CONTEXT_LENGTH_VARIABLE] == "8192"
    # And the served arm's three are still absent here, unchanged.
    assert MAX_NUM_SEQS_VARIABLE not in environment


def test_the_width_and_the_budget_come_from_one_row(tmp_path: Path) -> None:
    """THE SECOND HALF OF THAT DEFECT. The first fix stated the width alone and
    left the budget at narrator's own 42 GB default — the number BookForge's
    `extreme` tier sends, which is right on a 64 GB Mac and a promise of swap on
    a 16 GB one. The width is a CEILING narrator narrows against the budget
    (`_mlx_width_for_depth`), so a width from this machine and a budget from
    somebody else's is one fact with two owners (docs/ARCHITECTURE.md).

    All three come off one `MlxTier`, and the third is the pinned buffer cache:
    narrator's headroom is `budget - weights - cache`, so a budget taken from
    the row while the cache keeps narrator's 8 GB default is two rows in one
    sum."""
    document = a_document(tmp_path, tmp_path / "weights")
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_64_GIB_MAC).environment()
    # BookForge's `extreme` row, which is what a 64 GB Mac resolves to and what
    # every figure at `MLX_TIERS` was measured at.
    assert environment[MLX_BATCH_VARIABLE] == "64"
    assert environment[MLX_MEM_BUDGET_VARIABLE] == "42"
    assert environment[MLX_CACHE_LIMIT_VARIABLE] == "8"


def test_a_smaller_mac_gets_a_smaller_row(tmp_path: Path) -> None:
    """THE WHOLE REASON THE TABLE IS A TABLE. A hardcoded 64 is the 64 GB
    machine's answer given to every machine; the tier is chosen by BANDS of the
    host's own memory, exactly as BookForge's `orpheusAutoSuggestion` chooses it
    on darwin with no VRAM to read."""
    document = a_document(tmp_path, tmp_path / "weights")
    environment = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack=None, max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=A_16_GIB_MAC).environment()
    # BookForge's `light` row. The cache limit is 3 and not narrator's default
    # 8 for an arithmetic reason rather than a stylistic one: 13 - 8.5 - 8 is
    # negative and narrator refuses that load by name.
    assert environment[MLX_BATCH_VARIABLE] == "24"
    assert environment[MLX_MEM_BUDGET_VARIABLE] == "13"
    assert environment[MLX_CACHE_LIMIT_VARIABLE] == "3"


def test_every_tier_can_hold_its_own_weights_and_cache() -> None:
    """narrator's `_mlx_kv_headroom_gb` raises when `budget - weights - cache`
    is not positive — after it has read 8.5 GB off disk. A row that cannot pass
    that sum is a row nobody can render on, so it is caught here, on the table,
    rather than on a Mac nobody has yet."""
    for engine, rows in MLX_TIERS.items():
        assert rows[-1].min_total_mib == 0, engine
        for row in rows:
            headroom = (
                row.mem_budget_gb - HIGGS_V3_MLX_WEIGHTS_GB - row.cache_limit_gb
            )
            assert headroom > 0, f"{engine} {row.name}"
        # Highest band first, so the walk in `mlx_render_profile` takes the
        # best row a machine clears rather than the first one written down.
        floors = [row.min_total_mib for row in rows]
        assert floors == sorted(floors, reverse=True), engine


def test_the_served_arm_is_not_told_the_mlx_width(tmp_path: Path) -> None:
    """The mirror of the test above, and the rule this file keeps in both
    directions: a variable goes to the arm that reads it. narrator's served arm
    renders through vllm-omni and never constructs `HiggsV3MlxEngine`, so an
    MLX batch ceiling there would be the inert lever `[voice.serving]`'s own
    refusal is written against."""
    document = a_document(tmp_path, tmp_path / "weights")
    built = build_voice_engine(
        "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
        serving_stack="sglang-omni", max_num_seqs=16,
        mem_fraction=None, context_length=None, voices=document,
        mlx_total_bytes=None)
    environment = built.environment()
    for name in (
        MLX_BATCH_VARIABLE, MLX_MEM_BUDGET_VARIABLE, MLX_CACHE_LIMIT_VARIABLE
    ):
        assert name not in environment, name
    assert environment[MAX_NUM_SEQS_VARIABLE] == "16"


def test_the_served_arm_is_refused_a_memory_figure(tmp_path: Path) -> None:
    """The mirror of `test_a_stack_on_an_engine_that_has_none_is_refused`: a
    served narrator's memory is its launcher's two GPU fractions, so a unified
    memory figure there is a second answer to "how much may this have" that
    nothing reads."""
    document = a_document(tmp_path, tmp_path / "weights")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack="sglang-omni", max_num_seqs=16,
            mem_fraction=None, context_length=None, voices=document,
            mlx_total_bytes=A_64_GIB_MAC)
    assert "mlx_total_bytes" in str(caught.value)
    assert "launcher's GPU fractions" in str(caught.value)


def test_the_in_process_arm_without_a_memory_figure_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """No default, and 42 GB is not a safe one — it is a 64 GB machine's
    number. A Mac whose memory Crucible could not read gets a refusal naming
    the probe, not the Mac Studio's row."""
    document = a_document(tmp_path, tmp_path / "weights")
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v3", a_venv(tmp_path), tmp_path / "x.log",
            serving_stack=None, max_num_seqs=16,
            mem_fraction=None, context_length=None, voices=document,
            mlx_total_bytes=None)
    assert MLX_BATCH_VARIABLE in str(caught.value)
    assert MLX_MEM_BUDGET_VARIABLE in str(caught.value)
    assert "probe_unified_memory" in str(caught.value)


def test_an_unreadable_memory_figure_is_refused_rather_than_banded() -> None:
    """A zero or a float is not a small Mac; it is a probe that did not answer,
    and banding it would put the `light` row on a machine nobody measured."""
    for figure in (0, -1, 42.5, None):
        with pytest.raises(EngineError) as caught:
            mlx_render_profile("higgs-v3", figure)  # type: ignore[arg-type]
        assert "probe_unified_memory" in str(caught.value)


def test_an_engine_with_no_measured_mlx_width_is_refused_by_name() -> None:
    """`ttsstream.batch_width_for`'s rule, one arm over: the width must be
    MEASURED and there is no default. 1 is not a safe answer here — it is the
    answer that cost this server a measured 7x — so an unmeasured engine is
    refused rather than quietly rendered one chunk at a time."""
    with pytest.raises(EngineError) as caught:
        mlx_render_profile(A_FUTURE_ENGINE, A_64_GIB_MAC)
    assert A_FUTURE_ENGINE in str(caught.value)
    assert MLX_BATCH_VARIABLE in str(caught.value)


def test_an_engine_that_reads_no_higgs_vocabulary_is_told_no_mlx_width(
    tmp_path: Path,
) -> None:
    """`NARRATOR_HIGGS3_MLX_BATCH` is Higgs v3's name, like `HIGGS_STACK`. An
    engine outside that vocabulary owes its own set (see `environment`) and is
    not refused for lacking a row in a table that is not about it."""
    built = NarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path("/opt/env/bin/python"),
        log_path=Path("/tmp/x.log"),
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    assert MLX_BATCH_VARIABLE not in built.environment()


def test_a_stack_on_an_engine_that_has_none_is_refused(tmp_path: Path) -> None:
    """HIGGS_* is Higgs v3's vocabulary. Dropping the value in silence would
    leave the env recipe and the engine disagreeing about what that env
    starts."""
    with pytest.raises(EngineError) as caught:
        NarratorEngine(
            narrator_engine=A_FUTURE_ENGINE,
            python=a_venv(tmp_path, "tts-future"),
            log_path=tmp_path / "x.log",
            serving_stack="sglang-omni",
            max_num_seqs=16,
            mem_fraction=None,
            context_length=None,
            voices=None,
            mlx_total_bytes=None,
        )
    assert "serving_stack='sglang-omni'" in str(caught.value)


def test_the_four_facts_have_no_defaults(tmp_path: Path) -> None:
    """A default would make FORGETTING to pass one indistinguishable from
    saying `None`, which is exactly how a worker is spawned without
    HIGGS_STACK — or, since 2026-09-14, without a voices document, or, since
    2026-09-15, with a batch width the machine's memory did not choose."""
    python = a_venv(tmp_path)
    with pytest.raises(TypeError):
        NarratorEngine(  # type: ignore[call-arg]
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
        )
    with pytest.raises(TypeError):
        NarratorEngine(  # type: ignore[call-arg]
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
        )
    with pytest.raises(TypeError):
        NarratorEngine(  # type: ignore[call-arg]
            narrator_engine="higgs-v3",
            python=python,
            log_path=tmp_path / "x.log",
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            voices=a_document(tmp_path, tmp_path / "weights"),
        )


def test_the_engine_id_is_in_the_name_so_a_refusal_says_which(
    tmp_path: Path,
) -> None:
    built = build_voice_engine(
        "higgs-v3", Path(sys.executable), Path("/tmp/x.log"),
        serving_stack=None, max_num_seqs=None, mem_fraction=None, context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC)
    assert built.name == "narrator (higgs-v3)"
    assert built.narrator_engine == "higgs-v3"


def test_an_engine_this_build_cannot_start_is_refused_by_name() -> None:
    with pytest.raises(EngineError) as caught:
        build_voice_engine(
            "higgs-v2", Path(sys.executable), Path("/tmp/x.log"),
            serving_stack=None, max_num_seqs=None,
            mem_fraction=None, context_length=None,
            voices=None, mlx_total_bytes=None)
    assert "unknown narrator engine 'higgs-v2'" in str(caught.value)
    assert "['higgs-v3']" in str(caught.value)


def test_every_engine_a_manifest_may_name_is_one_this_build_can_start() -> None:
    """The three tables are written in three files and must not drift.

    `crucible/voices.py` decides what a manifest's `narrator_engine` may say;
    `crucible/engines/__init__.py` decides what `build_voice_engine` will
    start; `crucible/ttsstream.py` decides how wide the streaming door batches
    it. A voice naming an engine the second table lacks would pass every
    manifest check and fail at the spawn; one the third lacks would pass the
    spawn and fail the first `say`. Both are the worst possible places to find
    out, and both are a table somebody added a row to and not the others.

    `tests/test_jobenv.py` compares the same list against the recipes on disk,
    which is the fourth place an engine has to exist before it is real.
    """
    assert set(NARRATOR_ENGINE_SAMPLING) == set(NARRATOR_ENGINES)
    assert set(NARRATOR_ENGINE_SAMPLING) == set(STREAM_BATCH_WIDTH)


def test_there_is_no_base_url_and_saying_so_is_the_point(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.base_url
    assert "has no base url" in str(caught.value)


def test_the_http_engines_kept_both_seams(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    """`stdio()` and `attach()` are override points, not a rewrite.

    The readiness seam has this assertion in `test_engine_readiness.py`; this is
    the same claim for the second seam PHASE3-TTS.md section 4 said `start()`
    would need. vLLM and mlx-lm still get `stdin=DEVNULL` and both output streams
    in the log, byte for byte what they had.
    """
    for cls in (VllmEngine, MlxLmEngine):
        assert cls.stdio is BaseEngine.stdio
        assert cls.attach is BaseEngine.attach
        assert cls.detach is BaseEngine.detach
    plain = BaseEngine(python=Path(sys.executable), log_path=Path("/tmp/x.log"))
    assert BaseEngine.stdio(plain, "handle") == {
        "stdin": engine_base.subprocess.DEVNULL,
        "stdout": "handle",
        "stderr": engine_base.subprocess.STDOUT,
    }
    # And narrator's own wiring keeps both pipes, in UTF-8.
    wiring = engine.stdio("handle")
    assert wiring["stdin"] is engine_base.subprocess.PIPE
    assert wiring["stdout"] is engine_base.subprocess.PIPE
    assert wiring["stderr"] == "handle"
    assert wiring["encoding"] == "utf-8"


# ------------------------------------------------------------- readiness


def test_a_stdout_ready_line_is_readiness(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_DELAY_S", "0")
    said: list[str] = []
    engine.start(weights, "deathstalker", 0, [])
    engine.ready(30.0, on_progress=said.append)
    assert any("is ready on fake" in message for message in said), said
    # No HTTP route was ever polled, and the engine's own description says so.
    assert engine.readiness_description() == "print a ready line on stdout"


def test_a_ready_line_that_never_comes_times_out_by_name(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_READY_NEVER", "1")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(3.0)
    message = str(caught.value)
    assert "did not print a ready line on stdout within 3s" in message
    assert "/v1/models" not in message


def test_an_engine_that_dies_before_it_is_ready_says_so(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commonest real failure on cuda-linux, and the reason the stderr tail
    is quoted into the error rather than pointed at."""
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    engine.start(weights, "deathstalker", 0, [])
    with pytest.raises(EngineError) as caught:
        engine.ready(30.0)
    message = str(caught.value)
    assert "exited 3 before it was ready" in message
    # The worker's own stderr, which is what explains it. This is the second seam
    # paying for itself: stderr goes to the log while stdout stays a pipe.
    assert "told to exit before becoming ready" in message


# ------------------------------------------------------------- the channel


def test_a_load_sends_narrators_own_message_and_returns_its_answer(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    loaded = engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    assert loaded["type"] == "loaded"
    assert loaded["sampleRate"] == 24_000

    sent = transcript.read_text(encoding="utf-8").splitlines()
    assert len(sent) == 1
    message = json.loads(sent[0])
    # THE VOICE ID AND `warm`, NOTHING ELSE. A `higgs-v3` load names a voice
    # in the NARRATOR_HIGGS_VOICES document; narrator refuses `modelDir` by
    # name on both arms (the launcher agent's finding on the PC and the
    # keeper's on the Mac, 2026-09-14), and the weights are the document's
    # `checkpointDir`.
    assert message == {
        "action": "load",
        "voice": "deathstalker",
        "warm": True,
    }
    assert "modelDir" not in message
    # No `caps`. See `NarratorEngine.load` on why that is a decision: narrator's
    # caps channel raises on a key it does not know; a Higgs voice's sampling
    # rides in the document instead.
    assert "caps" not in message


def test_a_load_with_no_document_carries_the_weights_on_the_message(
    tmp_path: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine outside `DOCUMENT_READERS` takes `modelDir` on the load — the
    shape narrator's wire has always had, kept byte for byte for the engine
    after `higgs-v3` (see `A_FUTURE_ENGINE`)."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    built = FakeNarratorEngine(
        narrator_engine=A_FUTURE_ENGINE,
        python=Path(sys.executable),
        log_path=tmp_path / "engine-owen.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=None,
        mlx_total_bytes=None,
    )
    try:
        up(built, weights)
        built.load(voice="owen", weights_dir=weights, warm=True)
    finally:
        built.stop()
    message = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    assert message == {
        "action": "load",
        "voice": "owen",
        "modelDir": str(weights),
        "warm": True,
    }


def test_a_voice_the_document_does_not_carry_is_refused_before_it_is_sent(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """narrator's own refusal, made on this side of the pipe: a load for a voice
    the document does not carry names the file and the voices it does carry."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.load(voice="sigma", weights_dir=weights, warm=True)
    assert "carries no voice 'sigma'" in str(caught.value)
    assert "['deathstalker']" in str(caught.value)
    assert not transcript.exists()


def test_a_directory_the_document_disagrees_with_is_refused(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
) -> None:
    """Two statements of where the weights are, compared rather than trusted."""
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.load(
            voice="deathstalker", weights_dir=tmp_path / "elsewhere", warm=True
        )
    assert "two directories for one voice" in str(caught.value)


def test_the_fake_worker_refuses_a_model_dir_exactly_as_narrator_does(
    tmp_path: Path,
    weights: Path,
) -> None:
    """The test double is only worth having if it fails the way the real thing
    failed. Send it the OLD message shape over the real pipes and it must
    refuse with narrator's words — which is the line the launcher agent read
    off the PC's engine log on 2026-09-14."""
    built = FakeNarratorEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "engine-deathstalker.log",
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, weights),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    try:
        up(built, weights)
        with pytest.raises(EngineError) as caught:
            for _ in built.converse(
                {
                    "action": "load",
                    "voice": "deathstalker",
                    "modelDir": str(weights),
                    "warm": True,
                },
                terminal=frozenset({"loaded"}),
                silence_timeout=10.0,
            ):
                pass
        assert "Higgs v3 load carried modelDir=" in str(caught.value)
    finally:
        built.stop()


def test_rows_retire_out_of_order_and_nothing_reorders_them(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    """The fake retires a batch in reverse on purpose, and so does a real one.

    A consumer that relies on arrival order is wrong, so the engine must not
    quietly make it right: the iterator is arrival order, and `i` is the identity.
    """
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    indices = [
        message["i"]
        for message in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": 7, "text": "one"}, {"i": 8, "text": "two"},
                          {"i": 9, "text": "three"}],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
        )
        if message["type"] == "batch_item"
    ]
    assert indices == [9, 8, 7]


def test_a_failed_row_arrives_beside_its_successful_neighbours(
    engine: FakeNarratorEngine, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_FAIL_ROW", "1")
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    rows = {
        message["i"]: message
        for message in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": i, "text": "a sentence"} for i in (0, 1, 2)],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
        )
        if message["type"] == "batch_item"
    }
    assert sorted(rows) == [0, 1, 2]
    # A failure is `message` and no `data`; that is how the two are told apart.
    assert "data" not in rows[1] and "message" in rows[1]
    assert "data" in rows[0] and "data" in rows[2]


def test_a_whole_request_refusal_ends_the_conversation(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    """A top-level `error` is not a per-row failure and is not reported as one."""
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        list(
            engine.converse(
                {"action": "nonsense"},
                terminal=BATCH_TERMINAL,
                silence_timeout=30.0,
            )
        )
    assert "refused the request" in str(caught.value)
    assert "does not know action" in str(caught.value)


def test_a_cancel_is_sent_and_the_run_is_reported_as_cancelled(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crucible asks narrator to stop rather than hanging up or killing it.

    Hanging up would leave the engine generating into nothing; killing it would
    take the voice off the card for the next job. The fake reads its stdin on the
    main thread, so it finishes the batch it is in before it sees the cancel —
    the real worker has a reader thread and aborts in flight. What this proves is
    the two things that are Crucible's: the cancel is SENT, and the run ends as
    cancelled rather than as a short success.
    """
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    with pytest.raises(JobCancelled):
        for _ in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": 0, "text": "a sentence"}],
            },
            terminal=BATCH_TERMINAL,
            silence_timeout=30.0,
            cancelled=lambda: True,
        ):
            pass
    assert '"action": "cancel"' in transcript.read_text(encoding="utf-8")


def test_an_engine_that_ignores_the_cancel_outlasts_its_grace_and_is_named(
    engine: FakeNarratorEngine,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE BOUND ON THE COOPERATION, and the measurement that put it there.

    The test above proves Crucible's half: the cancel is SENT. It says nothing
    about what happens when narrator does not act on it — and on 2026-09-15 that
    is exactly what a render of `thirdreich` on the Mac did. narrator's stdin
    reader set its flag, as it always has; the arm the render door drives never
    read it; and this method waited, with the job saying `cancelling` and the
    card held, for the eleven minutes it took to render the rest of the book.

    NO SILENCE TIMEOUT COULD HAVE ENDED IT. narrator was talking the whole time —
    a `batch_item` every twenty seconds, each one resetting the silence clock. A
    silence timeout asks "is this process alive"; the only useful question here
    is "did it hear me", and that needs its own clock, started at the cancel and
    never reset by a line.
    """
    monkeypatch.setattr("crucible.engines.narrator.CANCEL_GRACE_SECONDS", 0.5)
    monkeypatch.setenv("CRUCIBLE_FAKE_IGNORE_CANCEL", "1")
    monkeypatch.setenv("CRUCIBLE_FAKE_ROW_DELAY_MS", "80")
    up(engine, weights)
    engine.load(voice="deathstalker", weights_dir=weights, warm=True)
    with pytest.raises(EngineWouldNotStop) as caught:
        for _ in engine.converse(
            {
                "action": "generate_batch",
                "language": "en",
                "items": [{"i": index, "text": "a sentence"} for index in range(40)],
            },
            terminal=BATCH_TERMINAL,
            # Far longer than the grace, so what ends this can only be the
            # cancel's own clock.
            silence_timeout=60.0,
            cancelled=lambda: True,
        ):
            pass
    assert "was sent a cancel" in str(caught.value)
    assert "does not read that flag" in str(caught.value)


def test_a_line_on_stdout_that_is_not_a_message_is_a_refusal_naming_it(
    tmp_path: Path, weights: Path
) -> None:
    """fd 1 carries the wire and nothing else — `crucible/workers.py`'s rule.

    A library that logs to stdout corrupted narrator's aligner on a 401-chunk
    book. Skipping the line would make the next protocol change a silent
    behaviour change, so it is a refusal that quotes what appeared.
    """

    class ChattyEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import sys, time\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "sys.stdin.readline()\n"
                "print('Loading checkpoint shards:  42%', flush=True)\n"
                "time.sleep(30)\n",
            ]

    built = ChattyEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "chatty.log",
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    try:
        built.start(weights, "deathstalker", 0, [])
        built.ready(30.0)
        with pytest.raises(EngineError) as caught:
            list(
                built.converse(
                    {"action": "load", "voice": "deathstalker"},
                    terminal=frozenset({"loaded"}),
                    silence_timeout=30.0,
                )
            )
    finally:
        try:
            built.stop()
        except EngineError:
            pass
    message = str(caught.value)
    assert "not a protocol message" in message
    assert "Loading checkpoint shards:  42%" in message


def test_silence_during_a_request_is_reported_as_silence(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        list(
            engine.converse(
                # `stop` answers `stopped`, which is not the terminal below, so
                # the conversation waits for something that will never come.
                {"action": "stop"},
                terminal=frozenset({"batch_done"}),
                silence_timeout=1.0,
            )
        )
    assert "said nothing at all for 1s" in str(caught.value)


def test_an_engine_that_dies_mid_request_says_so(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    conversation = engine.converse(
        {"action": "quit"},
        terminal=frozenset({"batch_done"}),
        silence_timeout=30.0,
    )
    with pytest.raises(EngineError) as caught:
        list(conversation)
    text = str(caught.value)
    assert "in the middle of a request" in text or "closed its stdout" in text


# ---------------------------------------------------------------- stopping


def test_stop_asks_narrator_to_quit_before_it_signals(
    engine: FakeNarratorEngine,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`quit` is narrator's own primary teardown: it unwinds the stdin loop from
    inside the process and releases the GPU through the atexit hooks."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_TRANSCRIPT", str(transcript))
    up(engine, weights)
    assert engine.pids
    engine.stop()
    assert engine.pids == frozenset()
    assert '{"action": "quit"}' in transcript.read_text(encoding="utf-8")


def test_a_worker_that_will_not_go_is_reported_and_never_sigkilled(
    tmp_path: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owen's hardest rule in this file: a killed CUDA process wedges WSL2 until
    Windows reboots, so `stop()` gives up and says so rather than escalating."""
    monkeypatch.setattr(engine_base, "STOP_TIMEOUT_SECONDS", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            # Ignores SIGTERM AND never reads its stdin, so neither the `quit`
            # nor the signal gets it — which is the state a worker wedged in a
            # WSL dxg GPU wait is actually in.
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                # win32's polite signal is CTRL_BREAK_EVENT, arriving as SIGBREAK.
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf.log",
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    try:
        if procgroup.platform_kind() == procgroup.WIN32:
            # win32 has no WSL2 wedge to protect: the deaf worker's tree is
            # terminated after the wait and `stop()` returns
            # (`crucible/procgroup.py`).
            built.stop()
            assert built.pids == frozenset()
            return
        with pytest.raises(EngineError) as caught:
            built.stop()
        message = str(caught.value)
        assert "did not exit within 1s of SIGTERM" in message
        assert "does not SIGKILL" in message
    finally:
        # This test made the mess, so this test cleans it up. Nothing on this
        # process is holding CUDA, which is the only reason a SIGKILL is allowed
        # anywhere in this repository.
        end_process_tree(pid)


def test_stopping_a_worker_that_ignores_sigterm_does_not_wedge_the_server(
    tmp_path: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal above must be a refusal and not a hang.

    `detach()` used to close stdout unconditionally, and closing a pipe another
    thread is blocked reading deadlocks on the buffered reader's own lock — in
    exactly the case that matters, a worker still alive with its stdout open. So
    `stop()` raised nothing and the server stopped instead. Measured 2026-09-13;
    this test is the one that found it.
    """
    monkeypatch.setattr(engine_base, "STOP_TIMEOUT_SECONDS", 1.0)

    class DeafEngine(NarratorEngine):
        def command(
            self, model_dir: Path, served_name: str, port: int, args: list[str]
        ) -> list[str]:
            return [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                # win32's polite signal is CTRL_BREAK_EVENT, arriving as SIGBREAK.
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                'print(\'{"type": "ready", "device": "fake"}\', flush=True)\n'
                "time.sleep(120)\n",
            ]

    built = DeafEngine(
        narrator_engine="higgs-v3",
        python=Path(sys.executable),
        log_path=tmp_path / "deaf2.log",
        # This engine exercises the WIRE; it starts no server, so the
        # three HIGGS_* variables have nothing to configure.
        serving_stack=None,
        max_num_seqs=None,
        mem_fraction=None,
        context_length=None,
        voices=a_document(tmp_path, tmp_path / "weights"),
        mlx_total_bytes=A_64_GIB_MAC,
    )
    built.start(weights, "deathstalker", 0, [])
    built.ready(30.0)
    pid = next(iter(built.pids))
    started = time.monotonic()
    try:
        if procgroup.platform_kind() == procgroup.WIN32:
            # The tree is terminated on win32. What is asserted is the same:
            # the stop RETURNS, promptly, rather than wedging on the reader.
            built.stop()
        else:
            with pytest.raises(EngineError):
                built.stop()
        # quit grace (1s) + SIGTERM wait (1s) + the reader join, and nothing
        # else (plus, on win32, the tree-kill's own bounded wait).
        assert time.monotonic() - started < 10.0 + procgroup.KILL_WAIT_SECONDS
    finally:
        end_process_tree(pid)


def test_the_reader_thread_is_gone_once_the_engine_is_stopped(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    engine.stop()
    assert engine._reader is None  # noqa: SLF001 — the thread is the thing tested


def test_a_started_engine_refuses_to_be_started_twice(
    engine: FakeNarratorEngine, weights: Path
) -> None:
    up(engine, weights)
    with pytest.raises(EngineError) as caught:
        engine.start(weights, "deathstalker", 0, [])
    assert "is already running" in str(caught.value)


def test_the_fake_worker_is_the_one_the_readiness_tests_use() -> None:
    """One fake, driven by two builders. If this file grew its own, the render
    door and the streaming door would be built against two wires."""
    assert FAKE_NARRATOR.is_file()
    assert FAKE_NARRATOR.name == "fake_narrator.py"
