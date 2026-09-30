from __future__ import annotations

import re
from pathlib import Path

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
    ("method", "activity", "body", responses.Activity, {
        "settings": "the operator page's audit rows",
        "catalog": "the operator page's audit rows",
        "accelerator": "present only with ?accelerator_probe=true; the SDK exposes accelerator()",
    }),
    ("method", "activity", "server", responses.ActivityServer, {}),
    ("method", "activity", "resident", responses.ActivityResident, {
        "reference": "the reference clip a zero-shot voice was loaded with",
    }),
    ("method", "activity", "slot", responses.ActivitySlot, {}),
    ("function", "readHeldBy", "held", responses.ActivityHeld, {}),
    ("function", "readActivityChat", "chat", responses.ActivityChat, {}),
    ("function", "readActivityChat", "row", responses.ActivityChatRow, {}),
    ("function", "readActivityJob", "data", responses.ActivityJob, {}),
    ("function", "readLease", "data", responses.ActivityLease, {}),
    ("method", "queue", "body", responses.QueueList, {}),
    ("function", "readQueueItem", "row", responses.QueueItem, {}),
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
