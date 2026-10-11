from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from crucible.api import responses

SDK_SRC = Path(__file__).resolve().parent.parent / "sdk" / "ts" / "src"
CLIENT_TS = SDK_SRC / "client.ts"
SHAPE_TS = SDK_SRC / "shape.ts"


def accessors() -> set[str]:
    source = SHAPE_TS.read_text(encoding="utf-8")
    return set(re.findall(r"export function (\w+)\(object: Json, key: string", source))


def body_at(source: str, start: int) -> str:
    opening = source.index("{\n", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unbalanced braces after offset {start}")


def function_body(source: str, name: str) -> tuple[str, str]:
    found = re.search(r"\bfunction " + re.escape(name) + r"\(\s*(\w+)\s*:", source)
    assert found, f"client.ts has no function {name}"
    return found.group(1), body_at(source, found.start())


def method_body(source: str, name: str) -> str:
    found = re.search(r"^  (?:async )?" + re.escape(name) + r"\(", source, re.MULTILINE)
    assert found, f"client.ts has no method {name}"
    return body_at(source, found.start())


def keys_read(source: str, body: str, variable: str, seen: frozenset[str] = frozenset()) -> set[str]:
    var = re.escape(variable)
    keys = set(
        re.findall(
            r"\b(?:" + "|".join(sorted(accessors())) + r")\(\s*" + var + r"\s*,\s*'([a-z_]+)'",
            body,
        )
    )
    keys |= set(re.findall(r"\b" + var + r"\['([a-z_]+)'\]", body))
    keys |= set(re.findall(r"'([a-z_]+)' in " + var + r"\b", body))
    for helper in re.findall(r"\b(read\w+)\(\s*" + var + r"\s*[,)]", body):
        if helper in seen:
            continue
        parameter, inner = function_body(source, helper)
        keys |= keys_read(source, inner, parameter, seen | {helper})
    return keys


def fields_of(model: type[BaseModel]) -> set[str]:
    return set(model.model_fields)


# shape.ts accessors that throw on a null value, and those that throw on an absent key.
NULL_REFUSING = frozenset({"str", "num", "bool", "objectField", "arrayField", "strArray"})
KEY_REQUIRING = NULL_REFUSING | frozenset({
    "field", "nullableStr", "nullableNum", "nullableBool", "nullableObject", "nullableArray",
    "nullableStrArray",
})


def accessors_used(
    source: str, body: str, variable: str, seen: frozenset[str] = frozenset()
) -> dict[str, set[str]]:
    """For every key read off `variable`, the shape.ts accessors that read it, plus `in`
    for a presence test and `[]` for a raw index (neither refuses anything)."""
    var = re.escape(variable)
    used: dict[str, set[str]] = {}
    pattern = r"\b(" + "|".join(sorted(accessors())) + r")\(\s*" + var + r"\s*,\s*'([a-z_]+)'"
    for accessor, key in re.findall(pattern, body):
        used.setdefault(key, set()).add(accessor)
    for key in re.findall(r"\b" + var + r"\['([a-z_]+)'\]", body):
        used.setdefault(key, set()).add("[]")
    for key in re.findall(r"'([a-z_]+)' in " + var + r"\b", body):
        used.setdefault(key, set()).add("in")
    for helper in re.findall(r"\b(read\w+)\(\s*" + var + r"\s*[,)]", body):
        if helper in seen:
            continue
        parameter, inner = function_body(source, helper)
        for key, found in accessors_used(source, inner, parameter, seen | {helper}).items():
            used.setdefault(key, set()).update(found)
    return used


def may_be_null(model: type[BaseModel], key: str) -> bool:
    annotation = model.model_fields[key].annotation
    if annotation is Any:
        return True
    return type(None) in get_args(annotation) or annotation is type(None)


def may_be_absent(model: type[BaseModel], key: str) -> bool:
    # A field with a default can be left out: every route here that serves one of these models
    # with `response_model_exclude_unset`, and the event-stream snapshots, which are dicts that
    # never pass through the model at all.
    return not model.model_fields[key].is_required()


SOURCE = CLIENT_TS.read_text(encoding="utf-8")

CASES: list[tuple[str, str, str, type[BaseModel], dict[str, str]]] = [
    ("method", "ping", "body", responses.Ping, {
        "pairing_version": "the SDK checks api_version; pairing_version is for the pairing flow",
    }),
    ("method", "job", "body", responses.JobStatus, {
        "sampling": "a tts job's done_extra key; the SDK reads it off the done event",
    }),
    ("function", "readVoiceInfo", "entry", responses.VoiceInfo, {
        "source": "provenance of the manifest; the SDK does not surface it",
        "identity_basis": "provenance of the weights identity",
        "max_chars_basis": "the basis sentence for max_chars",
        "pace_basis": "the basis sentence for pace",
        "inherited_from": "which packaged voice an override shadows",
        "manifest": "packaged, local or repo",
    }),
    ("function", "readActivity", "body", responses.Activity, {
        "settings": "the operator page's audit rows",
        "catalog": "the operator page's audit rows",
        "accelerator": "present only with ?accelerator_probe=true; the SDK exposes accelerator()",
    }),
    ("function", "readActivity", "server", responses.ActivityServer, {}),
    ("function", "readActivity", "resident", responses.ActivityResident, {
        "reference": "the reference clip a zero-shot voice was loaded with",
    }),
    ("function", "readActivity", "slot", responses.ActivitySlot, {}),
    ("function", "readHeldBy", "held", responses.ActivityHeld, {}),
    ("function", "readActivityChat", "chat", responses.ActivityChat, {}),
    ("function", "readActivityChat", "row", responses.ActivityChatRow, {}),
    ("function", "readActivityJob", "data", responses.ActivityJob, {}),
    ("function", "readQueueSession", "data", responses.SessionState, {}),
    ("method", "queue", "body", responses.QueueList, {}),
    ("function", "readQueueItem", "row", responses.QueueItem, {}),
    ("function", "readWaitingFor", "found", responses.QueueWaitingFor, {}),
    ("method", "removeFromQueue", "body", responses.QueueRemoved, {}),
    ("function", "readRemoval", "entry", responses.JobRemoval, {}),
    ("method", "#failure", "envelope", responses.ErrorBody, {}),
    ("function", "busyRefusal", "body", responses.JobBusyDetails, {
        "door": "read by heldAtTheOperatorDoor to choose the error class",
    }),
    ("function", "heldRefusal", "body", responses.CardHeldDetails, {
        "door": "read by heldAtTheOperatorDoor to choose the error class",
    }),
]


@pytest.mark.parametrize(
    ("kind", "name", "variable", "model", "unread"),
    CASES,
    ids=[f"{case[3].__name__}-{case[1]}-{case[2]}" for case in CASES],
)
def test_the_sdk_reads_exactly_what_the_response_model_declares(
    kind: str, name: str, variable: str, model: type[BaseModel], unread: dict[str, str]
) -> None:
    body = method_body(SOURCE, name) if kind == "method" else function_body(SOURCE, name)[1]
    read = keys_read(SOURCE, body, variable)
    declared = fields_of(model)
    assert set(unread) <= declared, (
        f"{model.__name__} no longer declares {sorted(set(unread) - declared)}; drop them "
        "from this test's unread list"
    )
    assert read == declared - set(unread), (
        f"sdk/ts/src/client.ts {name} reads {sorted(read)} off `{variable}` and "
        f"crucible/api/responses.py {model.__name__} declares "
        f"{sorted(declared - set(unread))} for it. The pydantic model is what the server "
        "validates on the way out; the SDK's reader is what apps are typed against. "
        f"Only in the SDK: {sorted(read - declared)}. Only on the server: "
        f"{sorted(declared - set(unread) - read)}."
    )


# Fields the model declares nullable that the server never sends null, so the SDK reading them
# strictly is right. Each is a model annotation wider than the value, not an SDK bug.
NULL_ONLY_ON_PAPER: dict[tuple[str, str], str] = {
    ("ActivityResident", "memory_bytes_estimate"): (
        "every Resident* class in crucible/residency.py carries an int, and "
        "activity.py _resident_brief copies it"
    ),
    ("VoiceInfo", "orphan"): (
        "null only from voice_rows(store=None), which the job type's own readiness check "
        "calls; both routes that serve the rows (GET /v1/voices, /v1/info) pass the store"
    ),
    ("ErrorBody", "details"): (
        "read strictly only on a 426, and every 426 (crucible/api/deps.py) carries "
        "server_api_version; every other status reads it behind an `in` test"
    ),
}

# Fields the model gives a default (so a route could leave them out) that the server always
# writes: reading the key as required is right.
ABSENT_ONLY_ON_PAPER: dict[tuple[str, str], str] = {
    ("Activity", "updating"): (
        "activity.py activity_body writes it on every read and every events snapshot; the "
        "fleet is kept current, so the SDK requires it on purpose"
    ),
}


@pytest.mark.parametrize(
    ("kind", "name", "variable", "model", "unread"),
    CASES,
    ids=[f"{case[3].__name__}-{case[1]}-{case[2]}" for case in CASES],
)
def test_the_sdk_tolerates_every_null_and_absence_the_model_allows(
    kind: str, name: str, variable: str, model: type[BaseModel], unread: dict[str, str]
) -> None:
    # The key sets matching (above) says nothing about values: B-Sides on 1.0.140 refused a
    # whole activity() read on `waiting_for.next_check_at`, which the model declares
    # `str | None` and KeepWait sends null, while the SDK read it with `str`.
    body = method_body(SOURCE, name) if kind == "method" else function_body(SOURCE, name)[1]
    used = accessors_used(SOURCE, body, variable)
    wrong: list[str] = []
    for key, found in sorted(used.items()):
        if key not in model.model_fields:
            continue
        named = (model.__name__, key)
        if named not in NULL_ONLY_ON_PAPER and may_be_null(model, key) and found & NULL_REFUSING:
            wrong.append(f"{key} may be null but is read with {sorted(found & NULL_REFUSING)}")
        if named not in ABSENT_ONLY_ON_PAPER and may_be_absent(model, key) and found <= KEY_REQUIRING:
            wrong.append(f"{key} may be absent but is read with {sorted(found)}")
    assert not wrong, (
        f"sdk/ts/src/client.ts {name} refuses what crucible/api/responses.py "
        f"{model.__name__} allows: " + "; ".join(wrong)
    )


def test_the_door_discriminator_is_the_one_the_sdk_branches_on() -> None:
    _, branch = function_body(SOURCE, "heldAtTheOperatorDoor")
    assert "['door']" in branch
    constant = re.search(r"const OPERATOR_DOOR = '(\w+)';", SOURCE)
    assert constant is not None
    door = responses.CardHeldDetails.model_fields["door"].annotation
    assert constant.group(1) in door.__args__
    assert "job" in responses.JobBusyDetails.model_fields["door"].annotation.__args__


def test_the_accessor_list_comes_from_the_sdk_itself() -> None:
    assert {"str", "num", "bool", "nullableStr", "objectField", "optStr"} <= accessors()
