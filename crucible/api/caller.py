from __future__ import annotations

import re

from fastapi import Request

from .. import CLIENT_NAME_HEADER

#: `X-Crucible-Client`'s value, validated exactly as `connect.py` validates a
#: pairing `client_name`: 1-80 characters, no control characters. One shape
#: for "a name a person will read", so a client cannot be called one thing at
#: the pairing door and another on a job row.
_CLIENT_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")


def client_agent(request: Request) -> str | None:
    """Who is speaking to this server, or None because they did not say.

    The SDK sends `<clientName> crucible-client/<version>`; anything else may
    send whatever it likes, or nothing. Truncated because it is a header, and
    a header is attacker-controlled length even inside one trust domain.

    One function rather than one expression per door, because there are now
    two doors that record a holder — `POST /v1/jobs` and `POST /v1/tts/stream`
    — and a bench puts both names in the same column. Two copies of "how we
    read the User-Agent" would be two truncation limits and two spellings of
    "did not say" the day one of them was edited.

    `X-Crucible-Client` FIRST, BECAUSE A BROWSER CANNOT SET ITS UA
    (2026-09-20). The BookForge Reader extension's popup read *"Mozilla/5.0
    (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 … Chrome/154.0.0.0
    Safari/537.36 is running a load-voice job here"* — the extension
    describing its OWN job, because `User-Agent` is a forbidden header name
    in a browser: the SDK sets `clientName` and Chrome silently drops it.
    Every consumer of `Job.client` then had a 120-character browser string
    where a name belongs, and a bench's "held by" column is the worst place
    for one.

    So the SDK sends the name in a header a browser CAN set, and this reads
    that first. The User-Agent rule is unchanged and is still the answer for
    curl, for the CLI, and for anything that sends no such header.

    AN INVALID HEADER IS IGNORED, NOT REFUSED. It is the same shape
    `connect.py` validates a pairing `client_name` with — 1-80 characters,
    no control characters — and failing it means falling back to the
    User-Agent, because this is a LABEL for a bench and not an
    authorization: nothing is decided by it, so nothing is worth refusing a
    render over. Control characters are the part that matters; a name is
    printed in a terminal, a log line and a popup.

    ONE TRUNCATION, and it is still this line's: the 80-character limit
    rejects a header outright rather than trimming it, and the 200 below is
    what a User-Agent gets. Two truncations would be two answers to "how
    long is a client name".
    """
    stated = (request.headers.get(CLIENT_NAME_HEADER) or "").strip()
    if stated and _CLIENT_NAME.fullmatch(stated):
        return stated
    return (request.headers.get("user-agent") or "").strip()[:200] or None
