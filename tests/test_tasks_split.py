from __future__ import annotations

import json
from typing import Any

import pytest

from crucible import tasks
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.jobs import base as job_states
from crucible.platform import paths
from crucible.tasks import hostdoor, runner, states, validate

from .conftest import FAKE_BACKEND

PUBLIC_NAMES = {
    "TASK_TYPES", "Task", "TaskStore", "env_installed",
    "install_command", "searched_note", "which",
}


def test_the_package_answers_its_public_surface_and_no_more() -> None:
    assert set(tasks.__all__) == PUBLIC_NAMES
    assert all(hasattr(tasks, name) for name in PUBLIC_NAMES)


def test_task_states_are_the_job_states_and_a_task_is_never_interrupted() -> None:
    assert (states.RUNNING, states.DONE, states.FAILED, states.CANCELLED) == (
        job_states.RUNNING, job_states.DONE, job_states.FAILED, job_states.CANCELLED,
    )
    # A task is never interrupted, and never waits in the job queue, so it is never removed.
    assert states.TERMINAL_STATES == job_states.TERMINAL_STATES - {
        job_states.INTERRUPTED, job_states.REMOVED,
    }


def test_the_host_door_variable_has_one_owner() -> None:
    assert hostdoor.HOST_DOOR_ENV is paths.HOST_DOOR_ENV


def test_a_patched_install_command_reaches_the_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: "/opt/crucible")
    assert runner.install_argv("tts", "higgs-v3") == [
        "/opt/crucible", "install", "tts", "--verbose", "--narrator-engine", "higgs-v3",
    ]


def test_a_patched_which_is_what_install_command_asks(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.executable", str(tmp_path / "python"))
    monkeypatch.setattr(tasks, "which", lambda _name: "/usr/local/bin/crucible")
    assert tasks.install_command() == "/usr/local/bin/crucible"


def test_each_module_table_has_its_own_validator(home: Any) -> None:
    config = None
    assert validate.job_type_entry(config, FAKE_BACKEND, "job_types[0]", "llm") == (
        "job_types[0]: must be an object with a `type`"
    )
    assert validate.need_entry(config, FAKE_BACKEND, "needs[0]", {"class": "no-such"}).startswith(
        "needs[0]: 'no-such' is not a capability class"
    )
    assert validate.subject_entry(config, FAKE_BACKEND, "subjects[0]", {"kind": 1, "id": 2}) == (
        "subjects[0]: `kind` and `id` must both be strings"
    )


def test_a_module_reports_every_tables_problems_in_table_order(
    home: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible import cli

    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init"]) == 0
    config = load_config(home)
    with pytest.raises(ApiError) as refused:
        validate.validate_module(config, FAKE_BACKEND, {
            "name": "app", "version": "1",
            "subjects": "not a list",
            "needs": [{"class": "x", "why": "y"}],
            "job_types": [7],
        })
    assert refused.value.details["problems"] == [
        "job_types[0]: must be an object with a `type`",
        "needs[0]: unknown key(s) ['why']. A need is a CLASS and nothing else; an app "
        "that wants one specific model names it under `subjects`, which is a choice "
        "and says so",
        "subjects: must be a list",
    ]


def test_an_unknown_task_type_is_refused_by_the_validator_table(home: Any) -> None:
    with pytest.raises(ApiError) as refused:
        validate.validate_request(None, FAKE_BACKEND, {"type": "reboot"})
    assert refused.value.code == "invalid_request"
    assert set(validate.VALIDATORS) == set(tasks.TASK_TYPES)


def _task() -> tasks.Task:
    return tasks.Task(id="t1", type="engine", request={}, created="now", started="now")


def test_the_relay_passes_host_events_through_and_names_the_terminal_one() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    lines = [
        b"\n",
        json.dumps({"event": "step", "data": {"name": "wsl"}}).encode(),
        b"not json",
        json.dumps({"event": "done", "data": "not a dict"}).encode(),
    ]
    terminal = hostdoor.relay_lines(_task(), lines, lambda kind, data: seen.append((kind, data)))
    assert terminal == "done"
    assert seen == [("step", {"name": "wsl"}), ("progress", {"line": "not json"}), ("done", {})]


def test_the_relay_stops_at_a_cancel() -> None:
    task = _task()
    task.cancel_requested = True
    with pytest.raises(states.TaskCancelled):
        hostdoor.relay_lines(task, [b"{}"], lambda kind, data: None)


def test_a_host_refusal_code_falls_back_to_the_move_code() -> None:
    assert hostdoor.host_refusal_code("<html>") == "engine_move_needs_host"
    assert hostdoor.host_refusal_code('{"error": {"code": "wsl_off"}}') == "wsl_off"


def test_install_lines_carry_the_reason_and_throttle_only_measured_progress() -> None:
    task = _task()
    seen: list[tuple[str, dict[str, Any]]] = []
    throttle = runner.Throttle(3600.0)
    for line in ("crucible: the wheel is missing", "plain line", "plain line"):
        runner.relay_install_line(task, line, throttle, lambda k, d: seen.append((k, d)))
    assert task.reason == "the wheel is missing"
    assert [data["line"] for _kind, data in seen] == [
        "crucible: the wheel is missing", "plain line", "plain line",
    ]
