from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.protocol import CLIENT_HEADER

CHROME = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
)


def _submit(client: TestClient, auth: dict[str, str], **headers: str) -> str:
    response = client.post(
        "/v1/jobs",
        headers={**auth, **headers},
        json={
            "type": "echo",
            "params": {},
            "inputs": {"x.bin": {"inline_base64": "YWN0aXZpdHk="}},
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _client_of(client: TestClient, auth: dict[str, str], job_id: str) -> Any:
    rows = client.get("/v1/activity", headers=auth).json()
    for row in rows["running"] + rows["queued"]:
        if row["job_id"] == job_id:
            return row["client"]
    return client.get(f"/v1/jobs/{job_id}", headers=auth).json().get("client")


def test_a_stated_name_is_what_the_job_records(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = _submit(
        client,
        auth,
        **{CLIENT_HEADER: "bookforge-reader", "User-Agent": CHROME},
    )
    assert _client_of(client, auth, job_id) == "bookforge-reader"


def test_without_one_the_user_agent_is_still_the_answer(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = _submit(client, auth, **{"User-Agent": "curl/8.5.0"})
    assert _client_of(client, auth, job_id) == "curl/8.5.0"


@pytest.mark.parametrize(
    "stated,why",
    [
        ("", "empty is not a name"),
        ("x" * 81, "longer than a pairing client_name may be"),
        ("bad\nname", "a newline would forge a second line in a log"),
        ("bad\x00name", "a NUL"),
        ("bad\x1bname", "an escape, which a terminal would ACT on"),
    ],
)
def test_an_invalid_name_falls_back_rather_than_refusing(
    client: TestClient, auth: dict[str, str], stated: str, why: str
) -> None:
    job_id = _submit(
        client, auth, **{CLIENT_HEADER: stated, "User-Agent": "curl/8.5.0"}
    )
    assert _client_of(client, auth, job_id) == "curl/8.5.0", why


def test_a_name_with_no_user_agent_to_fall_back_to_still_works(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = _submit(client, auth, **{CLIENT_HEADER: "bookforge-reader"})
    assert _client_of(client, auth, job_id) == "bookforge-reader"


def test_the_header_is_spelled_the_same_as_the_sdk_sends_it() -> None:
    assert CLIENT_HEADER == "X-Crucible-Client"
