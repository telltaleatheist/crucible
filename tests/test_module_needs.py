"""A module names CLASSES and the SERVER resolves them. PHASE15-HOST.md 5.3a.

MEASURED, by Foundry against the Mac, 2026-09-14. `foundry.module.json`
carried `qwen3.8-27b-4bit` and `dots-ocr` as RESOLVED ids, because
`gen-modules.py` resolved a `[[needs]] class` at generation time — the
cuda-linux answer, because the generator runs on a PC. Posted to the Mac,
`validate_module` refused the WHOLE module `unknown_subject` (dots-ocr has no
mlx-darwin block), and `qwen3.8-27b-4bit` is not what that machine's
capability selected anyway (`qwen3.8-27b-8bit`). The generator was a second owner
of a decision that is the server's: PHASE9 says the capability record is the
one place a class is resolved, and the record is per machine.

The two halves of the ruling are tested against each other here: a module
with `pages` in it posted to a fake mlx-darwin server comes back **done with
`unmet`**, not refused — because a Mac with no page reader is still a Mac
Foundry can use for text.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.config import CapabilityRecord, CapabilityRow

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

GIB = 1024 ** 3


def a_module(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "foundry",
        "version": "0.6.0+test",
        "job_types": [],
        "needs": [],
        "subjects": [],
    }
    body.update(overrides)
    return body


def record(*rows: CapabilityRow, backend: str = FAKE_BACKEND.kind) -> CapabilityRecord:
    return CapabilityRecord(
        backend_kind=backend,
        total_bytes=24 * GIB,
        desktop_allowance_bytes=3 * GIB,
        rows=rows,
    )


def row(
    capability: str,
    *,
    enabled: bool = True,
    selected: str = "",
    reason: str = "",
    shortfall: int = 0,
) -> CapabilityRow:
    return CapabilityRow(
        capability=capability,
        enabled=enabled,
        selected=selected,
        reason=reason or f"{capability}: whatever this card decided",
        shortfall_bytes=shortfall,
    )


def wait_for(client: TestClient, auth: dict[str, str], task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        body = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never finished: {body}")


def post_module(
    client: TestClient, auth: dict[str, str], module: dict[str, Any]
) -> Any:
    return client.post(
        "/v1/tasks", headers=auth, json={"type": "module", "module": module}
    )


# ------------------------------------------------------------- what is valid


def test_a_class_this_build_does_not_have_is_refused_whole(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A word this build has never heard of is the same defect everywhere."""
    response = post_module(
        client, auth, a_module(needs=[{"class": "transcribe-but-nicer"}])
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_module"
    assert "transcribe-but-nicer" in error["message"]


def test_a_need_carrying_anything_but_a_class_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    """An app that wants ONE model names it under `subjects`, which says so."""
    response = post_module(
        client,
        auth,
        a_module(needs=[{"class": "clean", "model": "qwen3.5-9b"}]),
    )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "a need is a CLASS" in message.replace("A need", "a need")


def test_needs_are_validated_with_everything_else_in_one_refusal(
    client: TestClient, auth: dict[str, str]
) -> None:
    """*"A module is validated WHOLE before anything starts."*"""
    response = post_module(
        client,
        auth,
        a_module(
            needs=[{"class": "nope"}],
            subjects=[{"kind": "model", "id": "also-nope"}],
        ),
    )
    problems = response.json()["error"]["details"]["problems"]
    assert len(problems) == 2


# --------------------------------------------------------- what is resolved


def test_a_class_resolves_to_what_THIS_card_selected_and_pulls_it(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []

    async def fake_pull(self: Any, task: Any, subject: Any) -> None:
        # The DOWNLOAD is faked and nothing else: the resolution, the step
        # event and the skip-if-installed are the shipping code's.
        pulled.append(subject.id)

    monkeypatch.setattr("crucible.tasks.TaskStore._pull", fake_pull)
    with make_client(
        capability=record(row("clean", selected="qwen3.5-9b"))
    ) as server:
        accepted = post_module(server, auth, a_module(needs=[{"class": "clean"}]))
        assert accepted.status_code == 202, accepted.text
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "done"
    assert pulled == ["qwen3.5-9b"]
    assert finished["unmet"] == []



def test_a_class_this_ENGINE_does_not_serve_is_unmet_and_the_module_is_DONE(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """The whole of 5.3a's ruling, in one assertion.

    A Mac with no page reader is still a Mac Foundry can use for text, so
    `pages` is `unmet` beside the pulls that did happen — never a refusal of
    the module.
    """
    with make_client(
        backend=FAKE_MAC_BACKEND,
        capability=record(
            row(
                "pages",
                enabled=False,
                reason=(
                    "disabled: mlx-vlm's own server does not put the image "
                    "into the prompt for dots.ocr"
                ),
            ),
            backend=FAKE_MAC_BACKEND.kind,
        ),
    ) as mac:
        accepted = post_module(mac, auth, a_module(needs=[{"class": "pages"}]))
        assert accepted.status_code == 202, accepted.text
        finished = wait_for(mac, auth, accepted.json()["task_id"])
    assert finished["state"] == "done"
    assert finished["unmet"] == [
        {
            "class": "pages",
            "reason": (
                "disabled: mlx-vlm's own server does not put the image into "
                "the prompt for dots.ocr"
            ),
        }
    ]


def test_the_unmet_reason_is_the_capability_row_s_OWN_sentence(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """Never a sentence written by the task: the row said why."""
    said = "disabled: the smallest candidate needs 24.0 GiB and there is 21.0"
    with make_client(
        capability=record(row("analysis", enabled=False, reason=said))
    ) as server:
        accepted = post_module(server, auth, a_module(needs=[{"class": "analysis"}]))
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["unmet"][0]["reason"] == said


def test_a_server_with_NO_capability_record_says_so_by_name(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """"Nothing has probed this card" and "this card cannot" differ.

    They are different things for an operator to fix, so they are different
    sentences — and neither of them refuses the module.
    """
    with make_client(capability=None) as server:
        accepted = post_module(server, auth, a_module(needs=[{"class": "clean"}]))
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "done"
    assert "no capability record" in finished["unmet"][0]["reason"]
    assert "crucible capability --write" in finished["unmet"][0]["reason"]


def test_a_stale_record_naming_a_model_this_backend_lacks_is_unmet_not_a_crash(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(
        capability=record(row("pages", selected="a-model-that-left"))
    ) as server:
        accepted = post_module(server, auth, a_module(needs=[{"class": "pages"}]))
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "done"
    assert "stale" in finished["unmet"][0]["reason"]


def test_an_explicit_subject_this_backend_cannot_hold_is_STILL_unknown_subject(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """5.3a keeps this: an explicit id is a CHOICE, and a wrong one is wrong.

    The asymmetry is the point. A class is "give me whatever serves this",
    which a machine can answer with "nothing here does". An id is "give me
    this one", which it cannot.
    """
    # `qwen3.5-9b-vl` has no mlx-darwin block (crucible/catalog.py
    # `backends_declaring`). It replaced `faster-whisper-large-v3` here on
    # 2026-09-24, the day Owen's asr lineup made every transcriber one id on
    # both backends — which is how `faster-whisper-large-v3` had replaced
    # `dots-ocr` on 2026-09-21, the day dots-ocr grew its mlx-darwin block.
    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        response = post_module(
            mac, auth, a_module(subjects=[{"kind": "model", "id": "qwen3.5-9b-vl"}])
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_module"
    assert "qwen3.5-9b-vl" in response.json()["error"]["message"]


def test_unmet_is_an_EMPTY_LIST_on_every_other_task(
    client: TestClient, auth: dict[str, str], fake_weights
) -> None:
    """Empty and not absent: "nothing was unmet" and "this server predates
    the field" must not be one reading."""
    fake_weights("qwen3.5-9b")
    listed = client.get("/v1/tasks", headers=auth).json()["tasks"]
    assert listed == []
    accepted = post_module(client, auth, a_module(needs=[]))
    # A module that asks for nothing is refused, so the empty-list check goes
    # through a task that runs: a pull of something already installed is
    # refused too, so an install-free module with one subject it has.
    assert accepted.status_code == 400

    with_subject = post_module(
        client,
        auth,
        a_module(subjects=[{"kind": "model", "id": "qwen3.5-9b"}]),
    )
    assert with_subject.status_code == 202
    finished = wait_for(client, auth, with_subject.json()["task_id"])
    assert finished["unmet"] == []


def test_the_done_event_carries_unmet_so_a_stream_reader_need_not_re_read(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(
        capability=record(row("pages", enabled=False, reason="no card"))
    ) as server:
        accepted = post_module(server, auth, a_module(needs=[{"class": "pages"}]))
        task_id = accepted.json()["task_id"]
        wait_for(server, auth, task_id)
        stream = server.get(f"/v1/tasks/{task_id}/events", headers=auth).text
    lines = [line for line in stream.splitlines() if line.startswith("data: ")]
    done = json.loads(lines[-1][len("data: ") :])
    assert done["unmet"] == [{"class": "pages", "reason": "no card"}]
