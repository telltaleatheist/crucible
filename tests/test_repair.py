from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import catalog, envpatches, jobenv, repair, weights
from crucible.cli import common, doctor, main
from crucible.config import load_config

from .conftest import FAKE_BACKEND, configure_box
from .test_tasks_api import admit, post, watch

SPEC = jobenv.EnvSpec(job_type="rvc", key="rvc", recipe_name="r.txt", headline="pkg")
GOOD = {"pkg/a.py": b"print('a')\n", "pkg/b.py": b"print('b')\n"}


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _build_env(home: Path) -> Path:
    env = home / "envs" / SPEC.key
    packages = env / "lib" / "python3.11" / "site-packages"
    (packages / "pkg").mkdir(parents=True)
    (packages / "pkg-1.0.dist-info").mkdir()
    (env / "bin").mkdir()
    (env / "bin" / "python").write_text("", encoding="utf-8")
    rows = []
    for name, data in GOOD.items():
        (packages / name).write_bytes(data)
        rows.append(f"{name},{_record_digest(data)},{len(data)}")
    rows.append("pkg/__pycache__/a.pyc,sha256=stale,1")
    rows.append("pkg-1.0.dist-info/INSTALLER,,")
    (packages / "pkg-1.0.dist-info" / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return packages


@pytest.fixture
def box(home: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    configure_box(home)
    packages = _build_env(home)
    built: list[jobenv.EnvSpec] = []

    def install_env(_home: Path, spec: jobenv.EnvSpec, _kind: str, *, force: bool, on_line: Any) -> None:
        assert force
        built.append(spec)
        for name, data in GOOD.items():
            (packages / name).write_bytes(data)

    monkeypatch.setattr(jobenv, "every_env", lambda _kind: (SPEC,))
    monkeypatch.setattr(jobenv, "env_status", lambda *a: SimpleNamespace(installed=True, detail="ok"))
    monkeypatch.setattr(jobenv, "install_env", install_env)
    monkeypatch.setattr(catalog, "subjects", lambda *a: [])
    monkeypatch.setattr(jobenv, "SMOKE_IMPORT", {})
    return SimpleNamespace(config=load_config(home), packages=packages, built=built, home=home)


def _run(
    box: SimpleNamespace,
    scope: repair.Scope | None = None,
    doors: repair.Doors = repair.NOBODY,
    **kw: Any,
) -> tuple[repair.Report, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    report = repair.run(
        box.config, FAKE_BACKEND, scope or repair.Scope(), doors,
        emit=lambda kind, data: events.append((kind, data)), **kw,
    )
    return report, events


def _running_out_of_budget() -> dict[str, Any]:
    ticks = iter(range(0, 100_000, 1000))
    return {
        "sleep": lambda _s: None,
        "clock": lambda: float(next(ticks)),
        "wait_budget_s": 2000.0,
    }


def test_a_healthy_env_is_checked_and_nothing_is_rebuilt(box: SimpleNamespace) -> None:
    report, events = _run(box)
    assert report.healthy and report.broken == [] and report.repaired == []
    assert report.checked == [{"kind": "env", "id": "rvc"}]
    assert box.built == []
    plan = next(data for kind, data in events if kind == "repair" and data["step"] == "plan")
    assert plan["bytes_to_check"] > 0, "the first event says how much it will hash"


def test_a_changed_file_is_broken_rebuilt_and_verified_again(box: SimpleNamespace) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    report, _ = _run(box)
    assert box.built == [SPEC]
    assert report.healthy
    assert [row["subject"]["id"] for row in report.repaired] == ["rvc"]
    assert "changed" in report.repaired[0]["was"]


def test_a_missing_file_is_broken(box: SimpleNamespace) -> None:
    (box.packages / "pkg" / "b.py").unlink()
    report, _ = _run(box)
    assert "1 file(s) missing" in report.repaired[0]["was"]


def test_verify_only_reports_and_changes_nothing(box: SimpleNamespace) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert box.built == [] and not report.healthy
    assert report.broken and not report.repaired
    assert (box.packages / "pkg" / "a.py").read_bytes() == b"damaged"


def test_a_file_an_env_patch_rewrites_is_not_held_to_its_record(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch = SimpleNamespace(rel_path="pkg/a.py")
    monkeypatch.setattr(envpatches, "patches_for", lambda _t: SimpleNamespace(patches=(patch,)))
    (box.packages / "pkg" / "a.py").write_bytes(b"patched on purpose")
    report, _ = _run(box)
    assert report.healthy and box.built == []


def _add_record(packages: Path, dist: str, rows: list[str]) -> None:
    (packages / dist).mkdir()
    (packages / dist / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_a_path_two_packages_own_is_healthy_if_it_is_either_of_theirs(box: SimpleNamespace) -> None:
    """flashinfer and torch_c_dlpack_ext both ship build_backend.py; the last installed wins."""
    winner, loser = b"flashinfer's", b"torch_c_dlpack_ext's"
    (box.packages / "build_backend.py").write_bytes(winner)
    _add_record(box.packages, "a-1.dist-info", [f"build_backend.py,{_record_digest(winner)},{len(winner)}"])
    _add_record(box.packages, "b-1.dist-info", [f"build_backend.py,{_record_digest(loser)},{len(loser)}"])
    report, _ = _run(box)
    assert report.healthy and box.built == []


def test_a_shared_path_that_is_neither_of_theirs_is_broken(box: SimpleNamespace) -> None:
    (box.packages / "build_backend.py").write_bytes(b"neither")
    _add_record(box.packages, "a-1.dist-info", [f"build_backend.py,{_record_digest(b'a')},1"])
    _add_record(box.packages, "b-1.dist-info", [f"build_backend.py,{_record_digest(b'b')},1"])
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert report.broken


def test_a_script_two_packages_own_outside_site_packages_is_judged_the_same(box: SimpleNamespace) -> None:
    """modelscope and modelscope_hub both install bin/ms."""
    script = box.packages.parents[2] / "bin" / "ms"
    script.write_bytes(b"#!python\n")
    rel = "../../../bin/ms"
    _add_record(box.packages, "a-1.dist-info", [f"{rel},{_record_digest(b'#!python' + bytes([10]))},9"])
    _add_record(box.packages, "b-1.dist-info", [f"{rel},{_record_digest(b'other')},5"])
    report, _ = _run(box)
    assert report.healthy and box.built == []


def test_a_wheel_whose_record_is_wrong_about_itself_does_not_cost_a_rebuild(box: SimpleNamespace) -> None:
    name = "PyNvVideoCodec/PyNvVideoCodec_121.cpython-311-x86_64-linux-gnu.so"
    (box.packages / "PyNvVideoCodec").mkdir()
    (box.packages / name).write_bytes(b"what the wheel really holds")
    _add_record(box.packages, "pynvvideocodec-2.0.4.dist-info", [f"{name},{_record_digest(b'what its RECORD says')},2402945"])
    report, _ = _run(box)
    assert report.healthy and box.built == []


def test_the_same_file_in_a_later_release_is_judged_again(box: SimpleNamespace) -> None:
    name = "PyNvVideoCodec/PyNvVideoCodec_121.cpython-311-x86_64-linux-gnu.so"
    (box.packages / "PyNvVideoCodec").mkdir()
    (box.packages / name).write_bytes(b"damaged")
    _add_record(box.packages, "pynvvideocodec-2.0.5.dist-info", [f"{name},{_record_digest(b'good')},4"])
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert report.broken


def test_a_rebuild_that_does_not_cure_it_is_failed_by_name(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    monkeypatch.setattr(jobenv, "install_env", lambda *a, **k: None)
    report, _ = _run(box)
    assert not report.healthy and not report.repaired
    assert report.failed[0]["code"] == "env_build_failed"


def test_an_env_in_use_is_rebuilt_only_once_it_is_free(box: SimpleNamespace) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    holders = ["the separator on the card", "the separator on the card", None]
    doors = repair.Doors(
        env_in_use=lambda _s: holders.pop(0), subject_in_use=repair.in_use_by_nothing
    )
    slept: list[float] = []
    report, events = _run(box, doors=doors, sleep=slept.append)
    assert report.healthy and box.built == [SPEC]
    assert slept == [repair.WAIT_POLL_S] * 2
    assert any(d.get("step") == "wait" for k, d in events if k == "repair")


def test_an_env_still_in_use_when_the_budget_ends_is_failed_and_untouched(box: SimpleNamespace) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    doors = repair.Doors(env_in_use=lambda _s: "a job", subject_in_use=repair.in_use_by_nothing)
    report, _ = _run(box, doors=doors, **_running_out_of_budget())
    assert box.built == []
    assert report.failed[0]["code"] == "subject_in_use"
    assert (box.packages / "pkg" / "a.py").read_bytes() == b"damaged"


def test_a_second_run_hashes_nothing_it_already_vouched_for(box: SimpleNamespace) -> None:
    first, _ = _run(box)
    assert first.bytes_hashed > 0 and first.bytes_skipped_verified == 0
    second, _ = _run(box)
    assert second.bytes_hashed == 0 and second.bytes_skipped_verified == first.bytes_hashed


def test_a_file_changed_after_it_was_vouched_for_is_hashed_again(box: SimpleNamespace) -> None:
    _run(box)
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged, and longer than it was")
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert report.broken


def test_cancelling_stops_a_hash_part_way(box: SimpleNamespace) -> None:
    with pytest.raises(repair.RepairCancelled):
        _run(box, cancelled=lambda: True)


# ── weights ──────────────────────────────────────────────────────────────────────────────────


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _subject_on_disk(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]
) -> SimpleNamespace:
    target = box.home / "weights-under-test"
    target.mkdir()
    for name, data in files.items():
        (target / name).write_bytes(data)
    (target / weights.STAMP_NAME).write_text("{}", encoding="utf-8")
    pulls: list[bool] = []

    def installed() -> weights.InstalledWeights | None:
        if not (target / weights.STAMP_NAME).exists():
            return None
        return weights.InstalledWeights(target, "org/repo", "abc123", 10, None, "pulled")

    def pull(*, force: bool, on_line: Any, on_progress: Any) -> None:
        pulls.append(force)
        for name, data in files.items():
            if not (target / name).exists():
                (target / name).write_bytes(data)
        (target / weights.STAMP_NAME).write_text("{}", encoding="utf-8")

    subject = catalog.Subject(
        kind="model", id="m", name=None, job_type="llm", expected_bytes=None, source="s",
        pull_command="x", installed=installed, pull=pull, remove=lambda: target,
    )
    monkeypatch.setattr(jobenv, "every_env", lambda _kind: ())
    monkeypatch.setattr(catalog, "subjects", lambda *a: [subject])
    hub = {name: repair.HubFile(name, _sha(data), None) for name, data in files.items()}
    monkeypatch.setattr(repair, "hub_files", lambda *a: hub)
    return SimpleNamespace(subject=subject, target=target, pulls=pulls, hub=hub)


def test_weights_that_match_the_hub_are_healthy(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"model.safetensors": b"weights"})
    report, _ = _run(box)
    assert report.healthy and not report.broken and s.pulls == []


def test_a_damaged_weight_file_alone_is_deleted_and_fetched(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"good.bin": b"good", "bad.bin": b"bad"})
    (s.target / "bad.bin").write_bytes(b"bit rot")
    good_mtime = (s.target / "good.bin").stat().st_mtime_ns
    report, _ = _run(box)
    assert s.pulls == [False], "only the bad file goes; a whole re-download is the wrong cure"
    assert (s.target / "bad.bin").read_bytes() == b"bad"
    assert (s.target / "good.bin").stat().st_mtime_ns == good_mtime
    assert report.healthy and report.repaired[0]["subject"] == {"kind": "model", "id": "m"}


def test_a_fetch_that_brings_the_same_bad_bytes_is_failed_by_name(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"bad.bin": b"bad"})
    (s.target / "bad.bin").write_bytes(b"bit rot")

    def pull_again(*, force: bool, on_line: Any, on_progress: Any) -> None:
        (s.target / "bad.bin").write_bytes(b"bit rot")
        (s.target / weights.STAMP_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        catalog, "subjects",
        lambda *a: [catalog.Subject(
            kind="model", id="m", name=None, job_type="llm", expected_bytes=None, source="s",
            pull_command="x", installed=s.subject.installed, pull=pull_again,
            remove=lambda: s.target,
        )],
    )
    report, _ = _run(box)
    assert not report.healthy and report.failed[0]["code"] == "weights_sha_mismatch"


def test_the_hub_caches_copy_of_a_bad_file_goes_too(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"bad.bin": b"bad"})
    (s.target / "bad.bin").write_bytes(b"bit rot")
    cache = tmp_path / "hub"
    snapshot = cache / "models--org--repo" / "snapshots" / "abc123"
    blobs = cache / "models--org--repo" / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / ("0" * 64)
    blob.write_bytes(b"bit rot")
    (snapshot / "bad.bin").symlink_to(blob)
    monkeypatch.setattr(weights, "hub_cache_root", lambda: cache)
    report, _ = _run(box)
    assert report.healthy
    assert not blob.exists() and not (snapshot / "bad.bin").is_symlink()


def _with_a_companion(s: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    """A vae/ from another repo, whose path the main repo also has, with other bytes (dreamshaper
    and yue2-3b, read off this PC's real weights, 2026-11-04)."""
    (s.target / "vae").mkdir()
    (s.target / "vae" / "model.bin").write_bytes(data)
    s.hub["vae/model.bin"] = repair.HubFile("vae/model.bin", _sha(b"the main repo's other vae"), None)
    subject = replace(s.subject, pinned_files=lambda: {"vae/model.bin": _sha(b"the companion")})
    monkeypatch.setattr(catalog, "subjects", lambda *a: [subject])


def test_a_companion_is_judged_by_its_pin_and_not_by_the_main_repo(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    _with_a_companion(s, monkeypatch, b"the companion")
    report, _ = _run(box)
    assert report.healthy and not report.broken and s.pulls == []
    assert (s.target / "vae" / "model.bin").read_bytes() == b"the companion"


def test_a_damaged_companion_is_broken_by_its_pin(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    _with_a_companion(s, monkeypatch, b"bit rot")
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert "vae/model.bin" in report.broken[0]["was"]


def test_a_missing_companion_is_broken(box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    _with_a_companion(s, monkeypatch, b"the companion")
    (s.target / "vae" / "model.bin").unlink()
    report, _ = _run(box, repair.Scope(verify_only=True))
    assert "1 missing" in report.broken[0]["was"]


def test_a_subject_nothing_can_be_compared_with_is_unverified(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.pth": b"x"})
    monkeypatch.setattr(repair, "hub_files", lambda *a: {})
    report, _ = _run(box)
    assert report.healthy and "none of its files" in report.unverified[0]["why"]
    assert s.pulls == []


def test_a_clean_result_says_what_it_could_not_compare(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    (s.target / "extra.gguf").write_bytes(b"from elsewhere")
    report, _ = _run(box)
    assert report.healthy
    assert report.not_hash_checked[0]["files"] == ["extra.gguf"]


def test_a_wheel_record_exception_is_listed_not_silently_skipped(box: SimpleNamespace) -> None:
    name = "PyNvVideoCodec/PyNvVideoCodec_121.cpython-311-x86_64-linux-gnu.so"
    (box.packages / "PyNvVideoCodec").mkdir()
    (box.packages / name).write_bytes(b"what the wheel really holds")
    _add_record(box.packages, "pynvvideocodec-2.0.4.dist-info", [f"{name},{_record_digest(b'x')},1"])
    report, _ = _run(box)
    assert report.not_hash_checked[0]["files"] == [name]


def test_a_hub_that_cannot_be_asked_deletes_nothing(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"bad.bin": b"bad"})
    (s.target / "bad.bin").write_bytes(b"bit rot")

    def down(*_a: Any) -> None:
        raise repair.HubUnreachable("no network")

    monkeypatch.setattr(repair, "hub_files", down)
    report, _ = _run(box)
    assert report.healthy, "unverified is not broken"
    assert report.unverified[0]["why"] == "no network"
    assert (s.target / "bad.bin").read_bytes() == b"bit rot" and s.pulls == []


def test_a_loaded_models_weights_are_read_but_never_replaced(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"bad.bin": b"bad"})
    (s.target / "bad.bin").write_bytes(b"bit rot")
    doors = repair.Doors(
        env_in_use=repair.in_use_by_nothing, subject_in_use=lambda _s: "it is on the card"
    )
    report, _ = _run(box, doors=doors, **_running_out_of_budget())
    assert report.failed[0]["code"] == "subject_in_use"
    assert (s.target / "bad.bin").read_bytes() == b"bit rot" and s.pulls == []


def test_a_subject_with_no_hub_pin_is_unverified(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    local = weights.InstalledWeights(s.target, None, None, 1, None, "local")
    monkeypatch.setattr(
        catalog, "subjects",
        lambda *a: [SimpleNamespace(
            kind="model", id="m", installed=lambda: local, shares_weights_of=None
        )],
    )
    report, _ = _run(box)
    assert report.healthy and "local" in report.unverified[0]["why"]


def test_a_voice_sharing_its_bases_weights_is_not_hashed_twice(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _subject_on_disk(box, monkeypatch, {"x.bin": b"x"})
    alias = SimpleNamespace(
        kind="voice", id="zeroshot", installed=s.subject.installed, shares_weights_of="m"
    )
    monkeypatch.setattr(catalog, "subjects", lambda *a: [s.subject, alias])
    report, _ = _run(box)
    assert report.checked == [{"kind": "model", "id": "m"}]


# ── the request ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("scope", "says"),
    [
        ("nope", "must be an object"),
        ({"stray": 1}, "unknown key"),
        ({"envs": 3}, "scope.envs"),
        ({"weights": "some"}, "scope.weights"),
        ({"verify_only": "yes"}, "scope.verify_only"),
    ],
)
def test_a_bad_scope_is_refused(scope: Any, says: str) -> None:
    assert any(says in problem for problem in repair.scope_problems(scope))


def test_a_good_scope_reads_back() -> None:
    assert repair.scope_problems({"envs": ["rvc"], "weights": "none", "verify_only": True}) == []
    scope = repair.Scope.from_request({"envs": ["rvc"], "weights": "none", "verify_only": True})
    assert scope == repair.Scope(envs=("rvc",), weights=False, verify_only=True)
    assert repair.Scope.from_request(None) == repair.Scope()


def test_the_task_door_runs_it_and_keeps_the_result(
    box: SimpleNamespace, client: Any, auth: Any
) -> None:
    task_id = admit(client, auth, {"type": "repair", "scope": {"verify_only": True}})
    events = watch(client, auth, task_id)
    assert "done" in [event["event"] for event in events], events
    assert "repair" in [event["event"] for event in events]
    task = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
    assert task["state"] == "done", task
    assert task["result"]["checked"] == [{"kind": "env", "id": "rvc"}]
    assert task["result"]["healthy"] is True


def test_a_task_door_scope_it_cannot_run_is_refused(
    box: SimpleNamespace, client: Any, auth: Any
) -> None:
    posted = post(client, auth, {"type": "repair", "scope": {"stray": 1}})
    assert posted.status_code == 400
    assert posted.json()["error"]["code"] == "invalid_request"


def test_no_other_task_takes_a_scope(box: SimpleNamespace, client: Any, auth: Any) -> None:
    posted = post(client, auth, {"type": "engine-restart", "scope": {}})
    assert posted.status_code in (400, 422)


# ── the CLI door ─────────────────────────────────────────────────────────────────────────────


def _cli(monkeypatch: pytest.MonkeyPatch, box: SimpleNamespace, state: str, *flags: str) -> int:
    monkeypatch.setattr(common, "here", lambda: (box.config, FAKE_BACKEND))
    monkeypatch.setattr(doctor.local, "status", lambda: {"state": state})
    return main(["doctor", *flags])


def test_the_cli_repairs_when_no_server_is_up(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    assert _cli(monkeypatch, box, "unreachable", "--repair") == 0
    assert box.built == [SPEC]
    assert "repaired env rvc" in capsys.readouterr().out


def test_the_cli_leaves_replacing_to_a_running_server(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    assert _cli(monkeypatch, box, "running", "--repair") != 0
    assert box.built == []
    assert "server_running" in capsys.readouterr().err


def test_the_cli_verifies_beside_a_running_server_and_answers_json(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    assert _cli(monkeypatch, box, "running", "--verify-only", "--json") != 0
    assert box.built == []
    assert json.loads(capsys.readouterr().out)["broken"][0]["subject"]["id"] == "rvc"


def test_the_cli_can_look_at_only_weights(
    box: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (box.packages / "pkg" / "a.py").write_bytes(b"damaged")
    assert _cli(monkeypatch, box, "unreachable", "--verify-only", "--only", "weights") == 0
    assert box.built == []
