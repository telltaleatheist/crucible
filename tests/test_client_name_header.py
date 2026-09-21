"""A client gets to name itself, because a browser cannot set a User-Agent.

THE SYMPTOM, 2026-09-20. The BookForge Reader extension's popup showed:

    Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 …
    Chrome/154.0.0.0 Safari/537.36 is running a load-voice job here (0%)

Owen: *"it looks like an error. is it necessary info or can it be removed?"* It
was the extension describing its OWN job. `User-Agent` is a forbidden header
name in a browser — `fetch` silently drops any attempt to set it — so the SDK's
`clientName` never reached the server, and `Job.client` recorded the browser's
UA instead. Every consumer of that field then had 120 characters of Chrome where
a name belongs, the bench's "held by" column included.

So the SDK also sends `X-Crucible-Client`, which a browser CAN set, and
`_client_agent` prefers it. The User-Agent rule is unchanged and is still the
answer for curl, for the CLI and for anything that sends no such header.

WHY AN INVALID ONE IS IGNORED RATHER THAN REFUSED. This is a LABEL for a bench,
not an authorization: nothing is decided by it, so nothing is worth refusing a
render over. It is validated to the shape `connect.py` gives a pairing
`client_name` — 1-80 characters, no control characters — because a name is
printed into a terminal, a log line and a popup, and a control character in any
of those is the client's choice of what your screen does.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import CLIENT_NAME_HEADER

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
    # The job finished before the read; its record still carries the holder.
    return client.get(f"/v1/jobs/{job_id}", headers=auth).json().get("client")


def test_a_stated_name_is_what_the_job_records(
    client: TestClient, auth: dict[str, str]
) -> None:
    """THE FIX. The extension sends both; the name is what a bench shows."""
    job_id = _submit(
        client,
        auth,
        **{CLIENT_NAME_HEADER: "bookforge-reader", "User-Agent": CHROME},
    )
    assert _client_of(client, auth, job_id) == "bookforge-reader"


def test_without_one_the_user_agent_is_still_the_answer(
    client: TestClient, auth: dict[str, str]
) -> None:
    """curl, the CLI, and every client that predates the header."""
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
    """IGNORED, NOT REFUSED. Nothing is decided by this field, so a bad one must
    never cost somebody a render — but it must not reach a screen either."""
    job_id = _submit(
        client, auth, **{CLIENT_NAME_HEADER: stated, "User-Agent": "curl/8.5.0"}
    )
    assert _client_of(client, auth, job_id) == "curl/8.5.0", why


def test_a_name_with_no_user_agent_to_fall_back_to_still_works(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A browser sends a UA it did not choose; something else may send none."""
    job_id = _submit(client, auth, **{CLIENT_NAME_HEADER: "bookforge-reader"})
    assert _client_of(client, auth, job_id) == "bookforge-reader"


def test_the_header_is_spelled_the_same_as_the_sdk_sends_it() -> None:
    """One spelling, one home. A header named in two places is a header with two
    spellings the day one of them is edited — which is why `API_HEADER` lives
    beside it in `crucible/__init__.py` and not in `api.py`."""
    assert CLIENT_NAME_HEADER == "X-Crucible-Client"
