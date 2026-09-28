from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from crucible import classnames, installonsubmit, jobenv, jobtypes, leases, settle
from crucible.cardkinds import KIND_ALIGN, KIND_DENOISE, KIND_LLM, KIND_TTS
from crucible.config import load_config, write_config
from crucible.engines import EngineError
from crucible.errors import ApiError, JobError
from crucible.jobs import ALL_JOB_TYPES, build_registry
from crucible.jobs.base import Job, JobFailure
from crucible.jobs.queue import (
    SECONDS_PER_DAY,
    JobStore,
    JournalProgress,
    LaneSlot,
    ReapReason,
)
from crucible.jobs.registry_table import REGISTRY_TABLE
from crucible.jobs.template import ManifestCatalog, as_job_error, parse_params
from crucible.jobs.unload import UnloadJobType
from crucible.narratorengines import declared_tts_footprints
from crucible.workers import WorkerError

from .conftest import FAKE_BACKEND

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "crucible" / "jobs"
SESSION_WORKER = Path(__file__).resolve().parent / "fake_workerio_session_worker.py"
ONE_SHOT_WORKER = Path(__file__).resolve().parent / "fake_workerio_one_shot_worker.py"

THE_JOB_TYPES = {
    "echo": "echo",
    "load-model": "llm",
    "unload-model": "llm",
    "load-voice": "tts",
    "unload-voice": "tts",
    "tts": "tts",
    "asr": "asr",
    "align": "align",
    "unload-aligner": "align",
    "align-longform": "align",
    "rvc": "rvc",
    "denoise": "denoise",
    "unload-denoiser": "denoise",
}


def test_the_job_type_names_and_families_are_the_ones_clients_send() -> None:
    assert ALL_JOB_TYPES == THE_JOB_TYPES
    assert [spec.name for spec in jobtypes.JOB_TYPE_SPECS] == list(THE_JOB_TYPES)


def test_the_card_effects_are_derived_from_the_specs_and_unchanged() -> None:
    effect = jobtypes.CardEffect
    assert leases.CARD_EFFECTS == {
        "load-model": effect(makes_resident=KIND_LLM),
        "load-voice": effect(makes_resident=KIND_TTS),
        "tts": effect(makes_resident=KIND_TTS, reuses_what_it_names=True),
        "align": effect(makes_resident=KIND_ALIGN, reuses_what_it_names=True),
        "denoise": effect(makes_resident=KIND_DENOISE, reuses_what_it_names=True),
        "unload-model": effect(takes_off=KIND_LLM),
        "unload-voice": effect(takes_off=KIND_TTS),
        "unload-aligner": effect(takes_off=KIND_ALIGN),
        "unload-denoiser": effect(takes_off=KIND_DENOISE),
        "echo": effect(),
        "asr": effect(),
        "rvc": effect(),
        "align-longform": effect(),
    }


def test_the_settlement_leaves_resident_only_what_a_load_put_there() -> None:
    assert settle.LEAVES_IT_RESIDENT == frozenset({"load-model", "load-voice"})


def test_install_on_submit_tables_are_derived_and_unchanged() -> None:
    assert installonsubmit.BASE_SUBJECTS == {"rvc": (("rvc-base", "base"),)}
    assert installonsubmit.PULLABLE_REFUSALS == frozenset(
        {
            "model_not_installed",
            "voice_not_installed",
            "rvc_base_models_missing",
            "denoise_model_missing",
        }
    )
    assert installonsubmit.CATALOG_IS_COMPLETE == frozenset({"rvc", "denoise", "asr", "align"})
    installable = {
        name for name in ALL_JOB_TYPES if installonsubmit.InstallOnSubmit.installable(name)
    }
    assert installable == {
        "load-model",
        "load-voice",
        "tts",
        "asr",
        "align",
        "align-longform",
        "rvc",
        "denoise",
    }
    assert not installonsubmit.InstallOnSubmit.installable("sorcery")


def test_the_installer_tables_are_derived_and_unchanged() -> None:
    assert jobenv.WORKER_JOB_TYPES == ("align", "asr", "rvc")
    assert jobenv.INSTALLABLE_JOB_TYPES == ("llm", "tts", "align", "asr", "rvc")
    assert jobenv.JOB_TYPES_SERVED_BY_ENV == {
        "align": ("align",),
        "asr": ("asr",),
        "rvc": ("rvc", "denoise"),
    }
    assert jobenv.INSTALLER_FOR == {
        "llm": "llm",
        "tts": "tts",
        "align": "align",
        "asr": "asr",
        "rvc": "rvc",
        "denoise": "rvc",
        "pages": "llm",
    }


def test_every_family_names_capability_classes_the_class_list_knows() -> None:
    for family in jobtypes.FAMILIES:
        assert family.capability_classes
        assert set(family.capability_classes) <= set(classnames.CLASS_NAMES), family.name


def test_the_registry_table_binds_every_spec_once_in_order() -> None:
    assert tuple(binding.spec for binding in REGISTRY_TABLE) == jobtypes.JOB_TYPE_SPECS


def _config(home: Path, **flags: bool) -> Any:
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=flags.get("echo", False),
        enable_llm=flags.get("llm", False),
        enable_asr=flags.get("asr", False),
        enable_tts=flags.get("tts", False),
        enable_align=flags.get("align", False),
        enable_rvc=flags.get("rvc", False),
        enable_denoise=flags.get("denoise", False),
        desktop_allowance_bytes=3 * 1024**3,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
        capability=None,
        tts_engines=declared_tts_footprints(FAKE_BACKEND.kind),
    )
    return load_config(home)


def test_build_registry_takes_up_exactly_the_families_turned_on(
    home: Path,
) -> None:
    everything = build_registry(
        _config(home, **{family.name: True for family in jobtypes.FAMILIES}), FAKE_BACKEND
    )
    assert list(everything) == list(THE_JOB_TYPES)
    for name, plugin in everything.items():
        assert plugin.name == name
    only_denoise = build_registry(_config(home, denoise=True), FAKE_BACKEND)
    assert list(only_denoise) == ["denoise", "unload-denoiser"]


def test_leases_import_neither_the_jobs_package_nor_residency() -> None:
    probe = "import sys, crucible.leases; print(sorted(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    assert "crucible.jobs" not in out
    assert "crucible.residency" not in out


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int


def test_parse_params_names_every_problem_in_one_format() -> None:
    with pytest.raises(ApiError) as raised:
        parse_params(_Params, {"size": "x", "extra": 1}, "rvc")
    assert raised.value.code == "invalid_params"
    assert raised.value.message.startswith("rvc params are not valid: ")
    assert "size: " in raised.value.message and "extra: " in raised.value.message
    with pytest.raises(ApiError) as led:
        parse_params(_Params, {}, "denoise", lead="denoise takes no params")
    assert led.value.message.startswith("denoise takes no params: size: ")


def test_as_job_error_turns_a_refusal_into_a_job_failure_with_the_same_code() -> None:
    def refuse() -> None:
        raise ApiError(409, "env_missing", "run `crucible install asr`")

    with pytest.raises(JobError) as raised:
        as_job_error(refuse)
    assert (raised.value.code, raised.value.message) == (
        "env_missing",
        "run `crucible install asr`",
    )
    assert as_job_error(lambda value: value + 1, 1) == 2


class _Unreadable(Exception):
    pass


def _manifest(model_id: str, revision: str | None) -> Any:
    spec = None if revision is None else SimpleNamespace(
        revision=revision, hf_repo="org/repo", memory_bytes_estimate=7
    )
    return SimpleNamespace(
        id=model_id,
        backends={} if spec is None else {"cuda-linux": spec},
        supports=lambda kind: spec is not None and kind == "cuda-linux",
        spec=lambda kind: spec,
    )


def test_a_manifest_catalog_answers_the_four_questions_every_type_asked() -> None:
    catalog = ManifestCatalog(
        lambda: {"a": _manifest("a", "r1"), "b": _manifest("b", None)},
        _Unreadable,
        unreadable_code="x_manifests_unreadable",
        what="x manifests",
        unknown="x manifest for",
        offer_in_details=True,
    )
    assert catalog.known("a").id == "a"
    with pytest.raises(ApiError) as unknown:
        catalog.known("zzz")
    assert unknown.value.code == "unknown_model"
    assert unknown.value.message == "no x manifest for 'zzz'; this build ships ['a', 'b']"
    assert unknown.value.details == {"model": "zzz", "offered": ["a", "b"]}
    assert catalog.memory_estimate("a", "cuda-linux") == 7
    assert catalog.memory_estimate("b", "cuda-linux") == 0
    assert catalog.provenance("cuda-linux", None) is None
    assert catalog.provenance("cuda-linux", "b") == {
        "id": "b",
        "revision": None,
        "fingerprint": None,
    }
    assert catalog.provenance("cuda-linux", "a")["revision"] == "r1"
    rows = catalog.descriptors(
        "cuda-linux", installed=lambda m, s: True, resident=lambda i: i == "a"
    )
    assert [(row.id, row.resident, row.installed, row.vram_bytes) for row in rows] == [
        ("a", True, True, 7),
        ("b", False, False, 0),
    ]

    def broken() -> dict[str, Any]:
        raise _Unreadable("line 3")

    with pytest.raises(ApiError) as unreadable:
        ManifestCatalog(
            broken, _Unreadable, unreadable_code="c", what="x manifests", unknown="u"
        ).all()
    assert (unreadable.value.status_code, unreadable.value.code) == (500, "c")
    assert unreadable.value.message == "this server cannot read its x manifests: line 3"


class _Residency:
    def __init__(self, resident: Any = None, cleared: bool = False) -> None:
        self.resident = resident
        self.cleared = cleared
        self.calls: list[str] = []
        self.unload_raises: BaseException | None = None

    @property
    def resident_id(self) -> str | None:
        return None if self.resident is None else self.resident.id

    def being_cleared(self, model: str) -> bool:
        return False

    def refuse_if_claimed(self, what: str) -> None:
        self.calls.append(f"claimed? {what}")

    def is_resident(self, kind: str, model: str) -> bool:
        return (
            self.resident is not None
            and self.resident.kind == kind
            and self.resident.id == model
        )

    def await_clearance(self, model: str) -> bool:
        self.calls.append("await_clearance")
        return self.cleared

    def unload(self, model: str) -> None:
        self.calls.append("unload")
        if self.unload_raises is not None:
            raise self.unload_raises
        self.resident = None


class _Ctx:
    def __init__(self, residency: _Residency) -> None:
        self.residency = residency
        self.extra: dict[str, Any] = {}

    def progress(self, fraction: float, message: str) -> None:
        self.residency.calls.append(f"progress {fraction}")

    def done_extra(self, **keys: Any) -> None:
        self.extra.update(keys)


def _job(model: str | None, params: dict[str, Any] | None = None) -> Job:
    return Job(
        id="j", type="unload", model=model, params=params or {}, dir=Path("."), created="t"
    )


def _unloader(spec: jobtypes.JobTypeSpec, residency: _Residency) -> UnloadJobType:
    return UnloadJobType(
        spec, residency, describe=lambda: [], provenance=lambda model: None  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("spec", "kind", "code", "needs"),
    [
        (jobtypes.UNLOAD_MODEL, KIND_LLM, "model_not_resident", "a model"),
        (jobtypes.UNLOAD_VOICE, KIND_TTS, "voice_not_resident", "a voice"),
        (jobtypes.UNLOAD_ALIGNER, KIND_ALIGN, "aligner_not_resident", "an aligner"),
        (jobtypes.UNLOAD_DENOISER, KIND_DENOISE, "separator_not_resident", "a separator"),
    ],
)
def test_the_four_unload_types_keep_their_names_and_codes(
    spec: jobtypes.JobTypeSpec, kind: str, code: str, needs: str
) -> None:
    residency = _Residency()
    plugin = _unloader(spec, residency)
    assert plugin.name == spec.name
    with pytest.raises(ApiError) as missing:
        plugin.preflight(None, {})
    assert (missing.value.code, missing.value.message) == (
        "model_required",
        f"{spec.name} needs {needs}",
    )
    with pytest.raises(ApiError) as bad:
        plugin.preflight("m", {"force": True})
    assert bad.value.code == "invalid_params"
    assert bad.value.message.startswith(f"{spec.name} params are not valid: force: ")
    with pytest.raises(ApiError) as absent:
        plugin.preflight("m", {})
    assert absent.value.code == code
    assert absent.value.details == {"requested": "m", "resident": None}
    with pytest.raises(JobError) as ran:
        plugin.run(_job("m"), _Ctx(residency))  # type: ignore[arg-type]
    assert ran.value.code == code
    assert plugin.check(None).detail == f"no {needs.split(' ', 1)[1]} is resident"
    residency.resident = SimpleNamespace(kind=kind, id="m")
    assert plugin.check(None).detail == "resident: m"


def test_an_unload_reports_progress_only_after_the_clearance_wait() -> None:
    residency = _Residency(resident=SimpleNamespace(kind=KIND_LLM, id="m"))
    ctx = _Ctx(residency)
    _unloader(jobtypes.UNLOAD_MODEL, residency).run(_job("m"), ctx)  # type: ignore[arg-type]
    assert residency.calls == ["await_clearance", "progress 0.0", "unload", "progress 1.0"]
    assert ctx.extra == {"resident": None}

    cleared = _Residency(cleared=True)
    _unloader(jobtypes.UNLOAD_VOICE, cleared).run(_job("v"), _Ctx(cleared))  # type: ignore[arg-type]
    assert cleared.calls == ["await_clearance", "progress 0.0", "progress 1.0"]

    missing = _Residency()
    with pytest.raises(JobError):
        _unloader(jobtypes.UNLOAD_ALIGNER, missing).run(_job("a"), _Ctx(missing))  # type: ignore[arg-type]
    assert missing.calls == ["await_clearance"]


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (KeyError("m"), "model_not_resident"),
        (EngineError("would not stop"), "engine_failed"),
        (WorkerError("would not stop"), "worker_failed"),
    ],
)
def test_every_unload_names_a_failed_stop_by_what_failed(
    raised: BaseException, code: str
) -> None:
    residency = _Residency(resident=SimpleNamespace(kind=KIND_LLM, id="m"))
    residency.unload_raises = raised
    with pytest.raises(JobError) as failed:
        _unloader(jobtypes.UNLOAD_MODEL, residency).run(_job("m"), _Ctx(residency))  # type: ignore[arg-type]
    assert failed.value.code == code


def test_an_unload_type_refuses_a_spec_that_unloads_nothing() -> None:
    with pytest.raises(TypeError):
        _unloader(jobtypes.RVC_JOB, _Residency())


def test_a_job_failure_keeps_the_error_wire_shape() -> None:
    job = _job("m")
    assert job.error is None
    job.failure = JobFailure("queue_failed", "a bug")
    assert job.error == {"code": "queue_failed", "message": "a bug"}
    assert JobFailure.from_dict(job.error) == job.failure
    assert JobFailure.from_dict(None) is None


def test_busy_details_keep_their_keys() -> None:
    details = Job(
        id="j1", type="asr", model="m", params={}, dir=Path("."), created="t0", client="app"
    ).busy_details()
    assert details == {
        "door": "job",
        "holder": "app",
        "job_id": "j1",
        "type": "asr",
        "model": "m",
        "status": "queued",
        "since": "t0",
        "progress": 0.0,
        "message": None,
    }


def test_a_reap_reason_is_still_the_string_on_the_wire() -> None:
    assert ReapReason.FETCHED == "fetched"
    assert json.dumps({"why": ReapReason.AGED.value}) == '{"why": "aged"}'
    assert {reason.value for reason in ReapReason} == {"fetched", "aged", "released"}
    assert SECONDS_PER_DAY == 86_400.0


def test_journal_progress_reads_the_manifest_fields_the_resume_note_uses() -> None:
    progress = JournalProgress.of(
        {
            "units_done": 3,
            "units_total": 9,
            "progress": None,
            "last_saved": "t",
            "jobs": [{"job_id": "a"}, {"job_id": "b"}],
        }
    )
    assert progress.sentence == "3 unit(s) done"
    assert progress.resumed_from == "a"
    assert JournalProgress.of({}).resumed_from is None


def test_the_lane_holds_one_job_and_says_so(tmp_path: Path) -> None:
    slot = LaneSlot()
    slot.admit("a")
    with pytest.raises(RuntimeError):
        slot.admit("b")
    assert slot.take() == "a" and slot.take() is None

    config = SimpleNamespace(jobs_dir=tmp_path / "jobs", retention_days=7, home=None, name="t")
    store = JobStore(config, SimpleNamespace(kind="cuda-linux"), {})
    assert store.queue_depth == 0
    first = store.create("echo", None, {})
    store.enqueue(first)
    assert (store.queue_depth, store.position(first), store.queued()) == (1, 1, [first])
    assert first.events[-1] == {"id": 1, "event": "queued", "data": {"position": 1}}
    second = store.create("echo", None, {})
    with pytest.raises(ApiError) as busy:
        store.enqueue(second)
    assert busy.value.code == "server_busy"
    assert busy.value.details == first.busy_details()
    assert store.position(second) is None
    record = store.hold(first, "app")
    assert set(record) == {
        "job_id",
        "status",
        "held",
        "held_by",
        "held_since",
        "gc_at",
        "artifacts",
    }


def _run(script: Path, stdin: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, str(script)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if "ModuleNotFoundError" in completed.stderr:
        pytest.skip(f"{script.name} needs a module this interpreter lacks")
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_a_session_worker_serves_ops_until_its_input_ends() -> None:
    lines = [
        json.dumps({"op": "echo", "text": "one"}),
        "",
        json.dumps({"op": "echo", "text": "two"}),
    ]
    assert _run(SESSION_WORKER, "\n".join(lines) + "\n") == [
        {"type": "result", "text": "one"},
        {"type": "done"},
        {"type": "result", "text": "two"},
        {"type": "done"},
    ]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "the fake request is not JSON: "),
        ("[1]", "the fake request must be a JSON object, got list"),
        ('{"op": "fly"}', "the fake request's op is 'fly'; this worker takes ['echo', 'explode', 'missing']"),
        ('{"op": "missing"}', "the fake request has no 'text'"),
        ('{"op": "explode"}', "RuntimeError: boom"),
    ],
)
def test_a_session_worker_stops_at_the_first_bad_request(line: str, message: str) -> None:
    replies = _run(SESSION_WORKER, line + "\n" + json.dumps({"op": "echo", "text": "x"}) + "\n")
    assert len(replies) == 1
    assert replies[0]["type"] == "failed"
    assert replies[0]["message"].startswith(message)


@pytest.mark.parametrize(
    ("stdin", "message"),
    [
        ("", "the fake worker was given no request on stdin"),
        ("{bad\n", "the fake request is not JSON: "),
        ('"text"\n', "the fake request must be a JSON object, got str"),
    ],
)
def test_a_one_shot_worker_refuses_a_request_it_cannot_read(stdin: str, message: str) -> None:
    replies = _run(ONE_SHOT_WORKER, stdin)
    assert len(replies) == 1 and replies[0]["type"] == "failed"
    assert replies[0]["message"].startswith(message)


def test_a_one_shot_worker_reads_one_request() -> None:
    assert _run(ONE_SHOT_WORKER, json.dumps({"text": "hi"}) + "\n") == [
        {"type": "result", "text": "hi"}
    ]


@pytest.mark.parametrize(
    ("script", "label"),
    [
        ("asr/worker.py", "asr"),
        ("asr/mlx_worker.py", "asr"),
        ("rvc/worker.py", "rvc"),
    ],
)
def test_each_one_shot_worker_reads_its_request_through_workerio(
    script: str, label: str
) -> None:
    replies = _run(JOBS / script, "[]\n")
    assert replies == [
        {"type": "failed", "message": f"the {label} request must be a JSON object, got list"}
    ]
    empty = _run(JOBS / script, "")
    assert empty == [
        {"type": "failed", "message": f"the {label} worker was given no request on stdin"}
    ]


@pytest.mark.parametrize(
    ("script", "label", "ops"),
    [
        ("align/worker.py", "align", ["align", "load"]),
        ("denoise/worker.py", "denoise", ["load", "separate"]),
        ("asr/qwen_worker.py", "qwen asr", ["load", "split", "transcribe"]),
    ],
)
def test_each_session_worker_serves_through_workerio(
    script: str, label: str, ops: list[str]
) -> None:
    replies = _run(JOBS / script, '{"op": "fly"}\n')
    assert replies == [
        {
            "type": "failed",
            "message": f"the {label} request's op is 'fly'; this worker takes {ops}",
        }
    ]
    assert _run(JOBS / script, "") == []
