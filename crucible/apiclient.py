"""`crucible api …` — the CLIENT half of the command line.

ONE BINARY, TWO KINDS OF VERB, AND THE DIFFERENCE IS THE ADDRESS
----------------------------------------------------------------
Everything already in `crucible/cli.py` acts on **this machine's installation**:
`init` writes config.toml, `install` unpacks an env pack, `service` writes a
unit, `models pull` puts weights on this disk. None of them takes a `--url`,
because there is nothing to point them at — the subject is the filesystem they
are running on.

The verbs in this module are a **client**. They speak HTTP to a server at a URL
with a bearer token, and that server may be this machine's, the one inside WSL,
or the Mac across the tailnet. `--url` and `--token` are what make them
different in kind rather than merely different in subject, and a reader coming
from the operator half should expect exactly that asymmetry.

They are not a second program. `crucible/launcher.py` writes ONE shim onto PATH
and records it in launcher.json, and `crucible/uninstall.py` removes that one; a
second binary would be a second launcher, a second PATH entry and a second thing
an upgrade can leave stale — which is what `cli_launcher_conflict` already cost
twice this week. It is also what Ollama does: `ollama serve` and `ollama pull`
are operator verbs, `ollama run` is inference, one binary, and nobody finds it
confusing. docs/INTENT.md is explicit that Crucible should be like Ollama.

WHY THE NAMESPACE IS SPELLED `api`
----------------------------------
Because `crucible models` and `crucible voices` are ALREADY TAKEN, by verbs that
list the manifests this BUILD carries. A client's "models" is a different fact
with a different owner — what one particular server holds right now — and two
verbs with one word is the shape ARCHITECTURE.md R1 exists to stop. Behind
`api` the two can never be confused: `crucible models list` reads this build,
`crucible api models` asks a server. The word also tells a reader the one thing
they must know before typing it, which is that a request is about to leave this
process.

OUTPUT IS DATA, ALWAYS, AND THERE IS NO `--json` FLAG
------------------------------------------------------
Owen's stated purpose for this surface is a fine-tuning script driving it, so
every command writes JSON to stdout and nothing else:

  * a command that makes one request prints one indented JSON document;
  * a command that FOLLOWS a stream prints one compact JSON object per line
    (JSONL) as each event arrives, then the subject's final state as a last
    line.

A flag would mean two output shapes to keep working and a script that forgot it
would get prose. Human-readable and machine-readable are the same thing here.

REFUSALS ARE PRINTED VERBATIM, WITH THE CODE
---------------------------------------------
The server answers a refusal as `{"error": {"code", "message", "details"?}}`
(crucible/errors.py). This module prints that body to stderr unchanged and exits
1. It never paraphrases: a code renamed on the way past is a code nobody can
grep for, in this repo or in the server's own source. When the body is not that
shape — a proxy's HTML, say — the raw bytes are printed, also unchanged.

WHAT IS DELIBERATELY NOT HERE
------------------------------
`POST /v1/pairing/start` and `/poll` are the REQUESTING app's half of the
connect dance; a CLI that already holds the token has nothing to do with them,
and `crucible token --url` is how a credential is handed out. `/v1/peer*` is the
orchestrator's relation to its engine (PHASE17), authenticated under its own
names for its own reason, and a client borrowing that door would be a second
claimant. Everything else the API serves has a verb below.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from . import API_HEADER, API_VERSION, VERSION
from .config import crucible_home
from .errors import ConfigError, CrucibleError
from .pairing import parse_pairing_line

EXIT_OK = 0
EXIT_REFUSED = 1

#: Long enough for a 27B to answer a non-streamed completion, which is the
#: slowest single response this client ever waits on. The event streams set no
#: read timeout at all — see `follow`.
REQUEST_TIMEOUT_SECONDS = 900.0

#: The connect is a separate budget from the read, for the reason
#: `crucible/api.py`'s proxy gives: a server that is not there should say so in
#: seconds, and a server that is thinking may take fifteen minutes.
CONNECT_NOTE = (
    "Crucible answers /v1/ping without a token; try that first if this is the "
    "wrong address"
)


class ClientRefusal(CrucibleError):
    """Something this CLI will not do, named. Never a server's refusal."""


# ------------------------------------------------------------------ connection


@dataclass(frozen=True)
class Connection:
    """Where the request goes and what it carries. Resolved once per command."""

    url: str
    token: str
    #: The server's name when the source knew it (the pairing file and
    #: config.toml both do), None when the operator only gave --url/--token.
    #: Reported, never checked: /v1/info is where identity is asserted, and a
    #: second comparison here would be a second owner of "is this the right
    #: server".
    name: str | None
    source: str


def resolve(args: argparse.Namespace) -> Connection:
    """The address and token, from exactly one of three places, named.

    THE ORDER IS NOT A FALLBACK CHAIN. Each source is chosen explicitly and the
    combinations that do not make sense are refused rather than silently
    resolved:

      --url + --token   a server anywhere. Both or neither.
      --pairing <line>  a `crucible://name@host:port/#token` line, which is what
                        `crucible token --url` prints and what an app's connect
                        door takes. One string carries all three facts.
      (nothing)         THIS machine's installed engine, through
                        `crucible/local.py:connection`, which is already the one
                        owner of "where is the local engine and what is its
                        token" and which knows that on Windows the answer is the
                        WSL guest's pairing file rather than config.toml.

    **`--url` without `--token` is refused by name and never falls back to the
    local token.** Sending this machine's bearer to an address the operator
    typed is a credential leak with a one-character typo as its cause, and
    silently doing it would be the worst kind of helpfulness.
    """
    given_url = getattr(args, "url", None)
    given_token = getattr(args, "token", None)
    given_pairing = getattr(args, "pairing", None)

    if given_pairing is not None and (given_url is not None or given_token is not None):
        raise ClientRefusal(
            "connection_overspecified: --pairing already carries the address and "
            "the token, so it cannot be combined with --url or --token. Pass one "
            "of the two forms"
        )
    if given_pairing is not None:
        try:
            pair = parse_pairing_line(given_pairing)
        except ValueError as exc:
            raise ClientRefusal(f"pairing_line_invalid: {exc}") from None
        return Connection(
            url=pair.url.rstrip("/"), token=pair.token, name=pair.name,
            source="--pairing",
        )
    if given_url is not None and given_token is None:
        raise ClientRefusal(
            "token_required: --url names a server this machine may not be, so it "
            "must come with --token. This command will not send the local "
            "engine's bearer to an address that was typed on the command line"
        )
    if given_token is not None and given_url is None:
        raise ClientRefusal(
            "url_required: --token is for a server elsewhere, so it must come "
            "with --url. To use the local engine's own token, pass neither"
        )
    if given_url is not None and given_token is not None:
        return Connection(
            url=given_url.rstrip("/"), token=given_token, name=None,
            source="--url/--token",
        )

    # Nothing stated: the installed engine. Imported here rather than at module
    # scope so that `crucible api --url ... ` on a machine with no installation
    # works — `local` reads config.toml on import of nothing, but `connection`
    # does, and a remote-only user should never be refused for a local file.
    from .local import LocalError, connection as local_connection

    try:
        url, name, token = local_connection(crucible_home())
    except (LocalError, ConfigError, OSError) as exc:
        raise ClientRefusal(
            f"no_local_engine: {exc}. Pass --url and --token, or --pairing, to "
            "reach a server that is not this machine's"
        ) from None
    return Connection(url=url.rstrip("/"), token=token, name=name, source="local")


# ------------------------------------------------------------------- transport


def _headers(connection: Connection) -> dict[str, str]:
    """Every request's three headers. `require_auth` and `require_api_version`.

    The User-Agent is what `_client_agent` records against a job and shows on
    the bench, so a render started from a terminal is distinguishable from one
    BookForge started. It is not a claim of an ACT — that is `X-Crucible-Act`,
    which only the two doors that take one send, and only when told to.
    """
    return {
        "Authorization": f"Bearer {connection.token}",
        API_HEADER: str(API_VERSION),
        "User-Agent": f"crucible-cli/{VERSION}",
    }


def _opener() -> urllib.request.OpenerDirector:
    """The default opener, proxies and all.

    Deliberately NOT `local.py`'s `ProxyHandler({})`. That module talks only to
    127.0.0.1, where a shell's proxy variables are noise; this one is built to
    reach the Mac across a tailnet, which is precisely the case a proxy exists
    for. `no_proxy` is the operator's lever for the loopback case and urllib
    already honours it.
    """
    return urllib.request.build_opener()


def _open(
    connection: Connection,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """One HTTP call, returning the open response. The caller reads it.

    Open rather than read, because an SSE follow and a 40 MB artifact download
    both need the body as a stream and neither should go through memory.
    """
    headers = _headers(connection)
    if content_type is not None:
        headers["Content-Type"] = content_type
    if extra_headers is not None:
        headers.update(extra_headers)
    request = urllib.request.Request(
        connection.url + path, data=body, headers=headers, method=method
    )
    return _opener().open(request, timeout=timeout)


def call(
    connection: Connection,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    """One JSON request, one parsed answer. `None` for a 204.

    A 204 is a real answer here — `DELETE /v1/leases/{id}` and
    `DELETE /v1/catalog/{kind}/{id}` both succeed with no body — so it returns
    None rather than an empty dict, and the caller prints what it means.
    """
    body = None if json_body is None else json.dumps(json_body).encode("utf-8")
    with _open(
        connection,
        method,
        path,
        body=body,
        content_type=None if body is None else "application/json",
        extra_headers=extra_headers,
    ) as response:
        if response.status == 204:
            return None
        raw = response.read()
    if raw == b"":
        return None
    return json.loads(raw.decode("utf-8"))


def follow(
    connection: Connection, path: str, *, last_event_id: int = 0
) -> Iterator[dict[str, Any]]:
    """An SSE stream, yielded event by event as it arrives.

    `Last-Event-ID` is the server's own resume (PHASE3-TTS.md section 7): a
    stream reattached with it is replayed what it missed and follows live from
    there. It is sent whenever the caller names an id, and the caller names one
    by passing `--since <the last id it printed>`.

    **No automatic reconnect.** A follow that dies has had something happen to
    it — the tunnel, the server, the process — and quietly starting again would
    turn an outage into a pause nobody sees. The command prints the ids it
    delivers, so resuming is re-running with `--since`, which is a decision a
    person or a script makes with the failure in front of them.

    `timeout=None`: a job stream is silent between a `progress` and the next
    one, and a 25-minute render legitimately says nothing for minutes at a
    time. The server sends a keepalive comment every 15 s
    (`KEEPALIVE_SECONDS`), which is what actually detects a dead socket here.
    """
    extra = {"Accept": "text/event-stream"}
    if last_event_id:
        extra["Last-Event-ID"] = str(last_event_id)
    with _open(connection, "GET", path, extra_headers=extra, timeout=None) as response:
        current: dict[str, Any] = {}
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
            if line == "":
                if current:
                    yield current
                    current = {}
                continue
            if line.startswith(":"):  # the server's keepalive comment
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "id":
                current["id"] = int(value)
            elif field == "event":
                current["event"] = value
            elif field == "data":
                current["data"] = json.loads(value)
        if current:
            yield current


def download(connection: Connection, path: str, destination: Path | None) -> int:
    """An artifact's bytes to a file, or to stdout when `destination` is None.

    Returns the byte count. Copied in chunks because a long-form align's
    artifacts are tens of megabytes and a render's FLACs arrive by the hundred.
    """
    with _open(connection, "GET", path) as response:
        if destination is None:
            written = 0
            while chunk := response.read(64 * 1024):
                sys.stdout.buffer.write(chunk)
                written += len(chunk)
            sys.stdout.buffer.flush()
            return written
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with destination.open("wb") as handle:
            while chunk := response.read(64 * 1024):
                handle.write(chunk)
                written += len(chunk)
        return written


def upload(connection: Connection, source: Path) -> dict[str, Any]:
    """`POST /v1/uploads` — one file, as multipart, returning its blob record.

    The body is assembled here rather than with `requests` or `httpx` because
    this is the only multipart request the whole client makes and neither
    library is worth carrying for it; `python-multipart` is the SERVER's
    dependency and parses rather than writes. The field name is `file`, which
    is what `api.py:upload`'s `UploadFile` parameter is called.

    The file is read whole. Uploads here are a job's inputs — a chunk of audio,
    a page image — and the server writes them to disk in one pass anyway.
    """
    if not source.is_file():
        raise ClientRefusal(f"input_missing: {source} is not a file")
    boundary = uuid.uuid4().hex
    guessed, _ = mimetypes.guess_type(source.name)
    part_type = guessed if guessed is not None else "application/octet-stream"
    buffer = io.BytesIO()
    buffer.write(f"--{boundary}\r\n".encode("ascii"))
    buffer.write(
        f'Content-Disposition: form-data; name="file"; filename="{source.name}"\r\n'
        f"Content-Type: {part_type}\r\n\r\n".encode("utf-8")
    )
    buffer.write(source.read_bytes())
    buffer.write(f"\r\n--{boundary}--\r\n".encode("ascii"))
    with _open(
        connection,
        "POST",
        "/v1/uploads",
        body=buffer.getvalue(),
        content_type=f"multipart/form-data; boundary={boundary}",
    ) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------- output


def emit(value: Any) -> None:
    """One JSON document on stdout. The answer to a command that made a request."""
    json.dump(value, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_line(value: Any) -> None:
    """One compact JSON object on stdout. A line of a followed stream."""
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def report_http_error(exc: urllib.error.HTTPError) -> int:
    """Print the server's refusal to stderr, unchanged, and return exit 1.

    Unchanged is the whole point. `crucible/errors.py` gives every refusal a
    `code`, and several carry a `details.field` naming exactly what was wrong;
    a client that turned `chunk_too_long` into "that sentence is too long"
    would have deleted the one string a person can search this repo for.
    """
    raw = exc.read()
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace").strip()
        print(f"crucible: HTTP {exc.code} from the server, and its body is not "
              f"Crucible's error shape:", file=sys.stderr)
        print(text if text else "(empty)", file=sys.stderr)
        return EXIT_REFUSED
    print(f"crucible: HTTP {exc.code}", file=sys.stderr)
    json.dump(body, sys.stderr, indent=2)
    sys.stderr.write("\n")
    return EXIT_REFUSED


# ------------------------------------------------------------ argument helpers


def read_json_argument(raw: str, flag: str) -> Any:
    """`{"a": 1}` or `@path/to.json`. One spelling for every JSON-taking flag.

    `@` because that is what curl, gh and this repo's own tools use, and a
    fine-tuning script's `chunks` list is thousands of sentences long — far past
    what a shell will take as one argument on Windows.
    """
    if raw.startswith("@"):
        path = Path(raw[1:])
        if not path.is_file():
            raise ClientRefusal(f"{flag}_file_missing: {path} is not a file")
        text = path.read_text(encoding="utf-8")
        where = str(path)
    else:
        text, where = raw, f"the {flag} argument"
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ClientRefusal(f"{flag}_invalid_json: {where}: {exc}") from None


def split_assignment(raw: str, flag: str) -> tuple[str, str]:
    """`name=value`, refused by name when it is not."""
    name, separator, value = raw.partition("=")
    if separator != "=" or name == "" or value == "":
        raise ClientRefusal(
            f"{flag}_malformed: {raw!r} is not `name=value`"
        )
    return name, value


# --------------------------------------------------------------- job following
#
# A followed job and a followed task are the same three steps — print the
# events, decide on the terminal one, print the final state — over two stores
# whose routes differ only in their path. The difference that matters is which
# words end them, so that is the parameter and the rest is shared.

#: The one terminal status that is not a failure.
#:
#: There is no copy of `api.py`'s TERMINAL_EVENTS here, and there does not need
#: to be: the server's stream ENDS at a terminal event, so a `follow` that runs
#: out of events has already reached one and nothing has to recognise which. The
#: status is then read off the job document, and this is the only word that
#: means it went well. Not imported from `crucible.api` — that module drags
#: FastAPI and uvicorn in behind it, and a CLI whose `--help` takes two seconds
#: is a CLI people stop using. tests/test_api_client.py proves the word by
#: running a real job to completion rather than by restating it.
SUCCEEDED = "done"


def follow_to_the_end(
    connection: Connection,
    events_path: str,
    state_path: str,
    *,
    since: int,
) -> tuple[int, dict[str, Any]]:
    """Print every event as a line, then the final state. Returns (exit, state).

    The final state is fetched rather than inferred from the terminal event,
    because the state document carries `artifacts` and `error` and the event
    carries neither in full — and a `--artifacts-dir` run has to know the names
    before it can ask for them.
    """
    for event in follow(connection, events_path, last_event_id=since):
        emit_line(event)
    state = call(connection, "GET", state_path)
    emit_line(state)
    status = state.get("status")
    return (EXIT_OK if status == SUCCEEDED else EXIT_REFUSED), state


# -------------------------------------------------------------------- commands
#
# Every command below is `(connection, args) -> int`. The wrapper in `command`
# resolves the connection, catches the two error kinds this surface has, and
# owns the exit code, so nothing here has to repeat a try/except.


def cmd_get(path: str) -> Callable[[Connection, argparse.Namespace], int]:
    """A read with no arguments. Fourteen of the routes are exactly this."""

    def run(connection: Connection, args: argparse.Namespace) -> int:
        emit(call(connection, "GET", path))
        return EXIT_OK

    return run


def cmd_capability(connection: Connection, args: argparse.Namespace) -> int:
    """`GET /v1/capability`, optionally paying for a live accelerator probe.

    The query parameter is the server's (`?accelerator_probe=true`), and it
    costs an `nvidia-smi` — which is why it is a flag and not the default.
    """
    query = "?accelerator_probe=true" if args.accelerator_probe else ""
    emit(call(connection, "GET", "/v1/capability" + query))
    return EXIT_OK


def cmd_catalog_remove(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/catalog/{args.kind}/{args.subject_id}")
    emit({"removed": {"kind": args.kind, "id": args.subject_id}})
    return EXIT_OK


def cmd_settings(connection: Connection, args: argparse.Namespace) -> int:
    """`GET /v1/settings`, or `PUT` when `--patch` is given.

    One command for both because the answer is the same document either way —
    `put_settings` returns the whole settings document AFTER the write, so a
    caller never has to guess what took.
    """
    if args.patch is None:
        emit(call(connection, "GET", "/v1/settings"))
        return EXIT_OK
    patch = read_json_argument(args.patch, "--patch")
    extra = None if args.act is None else {"X-Crucible-Act": args.act}
    emit(call(connection, "PUT", "/v1/settings", json_body=patch, extra_headers=extra))
    return EXIT_OK


def cmd_upstream_test(connection: Connection, args: argparse.Namespace) -> int:
    body = None if args.body is None else read_json_argument(args.body, "--body")
    emit(call(
        connection, "POST", f"/v1/settings/upstreams/{args.name}/test",
        # The route's body is optional and means "use the stored record"; an
        # empty object is NOT the same request as no body, so absence is
        # forwarded as absence.
        json_body=body,
    ))
    return EXIT_OK


def cmd_pairing_requests(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "GET", "/v1/pairing/requests"))
    return EXIT_OK


def cmd_pairing_decide(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", "/v1/pairing/decision", json_body={
        "id": args.id, "user_code": args.code, "allow": args.allow,
    }))
    return EXIT_OK


def cmd_chat(connection: Connection, args: argparse.Namespace) -> int:
    """`POST /v1/openai/chat/completions` — the cleanup and translation door.

    THE BODY IS THE CLIENT'S, VERBATIM. This proxy forwards what it is given in
    both directions except `model` (api.py's `_forward_body`), and the fields
    that matter to Owen's passes — `thinking`, `chat_template_kwargs`,
    `max_tokens` — are the ENGINE's vocabulary, not this CLI's. Modelling them
    as flags would put a second owner on every one of them and would be wrong
    the first time vLLM added a field. So the whole body is `--body`, and
    `--model` and `--message` exist only as the two-line shorthand for a
    one-shot question.

    `--stream` is the request's own `stream: true` plus an SSE reader: the
    server relays the engine's frames untouched, so what is printed is what the
    engine sent, one `data:` payload per line.
    """
    if args.body is not None and (args.model is not None or args.message is not None):
        raise ClientRefusal(
            "chat_overspecified: --body is the whole request, so it cannot be "
            "combined with --model or --message"
        )
    if args.body is not None:
        body = read_json_argument(args.body, "--body")
        if not isinstance(body, dict):
            raise ClientRefusal(
                f"body_not_an_object: a chat request is a JSON object, got "
                f"{type(body).__name__}"
            )
    else:
        if args.model is None or args.message is None:
            raise ClientRefusal(
                "chat_underspecified: pass --body with the whole request, or "
                "both --model and --message for a one-shot question"
            )
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.message}],
        }
    if args.stream:
        body = dict(body, stream=True)
    extra = None if args.act is None else {"X-Crucible-Act": args.act}

    if not args.stream:
        emit(call(
            connection, "POST", "/v1/openai/chat/completions",
            json_body=body, extra_headers=extra,
        ))
        return EXIT_OK

    # A streamed completion is OpenAI's SSE, not Crucible's: the frames carry no
    # `event:` and no `id:`, and the stream ends with the literal `[DONE]`. So
    # it is read here rather than through `follow`, which parses Crucible's
    # shape and would drop every one of these on the floor.
    headers = {"Accept": "text/event-stream"}
    if extra is not None:
        headers.update(extra)
    with _open(
        connection, "POST", "/v1/openai/chat/completions",
        body=json.dumps(body).encode("utf-8"),
        content_type="application/json",
        extra_headers=headers,
        timeout=None,
    ) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                break
            emit_line(json.loads(payload))
    return EXIT_OK


def cmd_upload(connection: Connection, args: argparse.Namespace) -> int:
    emit(upload(connection, Path(args.path)))
    return EXIT_OK


# ------------------------------------------------------------------------ jobs


def cmd_job_submit(connection: Connection, args: argparse.Namespace) -> int:
    """`POST /v1/jobs` — the one door every job type goes through.

    **THE PARAMS ARE NOT MODELLED HERE.** Each job type owns a pydantic model
    with `extra="forbid"` (`TtsParams`, `AsrParams`, `RvcParams`, …) and those
    models are the contract; a set of argparse flags mirroring them would be a
    second owner of every field, wrong the day a field moves, and would refuse
    valid requests the server would have taken. So `--params` is JSON and the
    server is the validator — which also means a new job type needs no change
    to this file. docs/API-CLI.md carries a worked body per type.

    Inputs go up through `POST /v1/uploads` and are referenced by `blob_id`
    rather than inlined. `inline_base64` exists on the wire and is not offered:
    it puts the bytes through the request body and through this process's
    memory to save one round trip, and every real input here is audio.
    """
    body: dict[str, Any] = {"type": args.type}
    if args.model is not None:
        body["model"] = args.model
    if args.params is not None:
        params = read_json_argument(args.params, "--params")
        if not isinstance(params, dict):
            raise ClientRefusal(
                f"params_not_an_object: --params is the job's params object, got "
                f"{type(params).__name__}"
            )
        body["params"] = params

    inputs: dict[str, dict[str, str]] = {}
    for raw in args.input:
        name, path = split_assignment(raw, "--input")
        blob = upload(connection, Path(path))
        inputs[name] = {"blob_id": blob["blob_id"]}
    for raw in args.input_blob:
        name, blob_id = split_assignment(raw, "--input-blob")
        inputs[name] = {"blob_id": blob_id}
    if inputs:
        body["inputs"] = inputs

    if args.artifacts_dir is not None and not args.follow:
        # NOT an implicit --follow. Downloading artifacts means waiting for the
        # job, and a flag that silently decides whether a command blocks for
        # twenty minutes is the kind of surprise this repo refuses by name.
        raise ClientRefusal(
            "artifacts_need_follow: --artifacts-dir waits for the job to finish, "
            "so it must be asked for with --follow. Without it this command "
            "returns as soon as the job is admitted"
        )

    accepted = call(connection, "POST", "/v1/jobs", json_body=body)
    job_id = accepted["job_id"]
    if not args.follow:
        emit(accepted)
        return EXIT_OK

    emit_line(accepted)
    code, state = follow_to_the_end(
        connection,
        f"/v1/jobs/{job_id}/events",
        f"/v1/jobs/{job_id}",
        since=0,
    )
    if args.artifacts_dir is not None and code == EXIT_OK:
        directory = Path(args.artifacts_dir)
        saved = []
        for name in state["artifacts"]:
            target = directory / name
            written = download(
                connection, f"/v1/jobs/{job_id}/artifacts/{name}", target
            )
            saved.append({"name": name, "path": str(target), "bytes": written})
        emit_line({"artifacts_saved": saved})
    return code


def cmd_job_get(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "GET", f"/v1/jobs/{args.job_id}"))
    return EXIT_OK


def cmd_job_events(connection: Connection, args: argparse.Namespace) -> int:
    code, _ = follow_to_the_end(
        connection,
        f"/v1/jobs/{args.job_id}/events",
        f"/v1/jobs/{args.job_id}",
        since=args.since,
    )
    return code


def cmd_job_cancel(connection: Connection, args: argparse.Namespace) -> int:
    """`DELETE /v1/jobs/{id}` — which answers `cancelling`, not `cancelled`.

    The server sets a flag and the runner ends when it sees it, so the answer is
    the request's outcome and not the job's. Printed as it arrives; watch the
    stream for the `cancelled` event.
    """
    emit(call(connection, "DELETE", f"/v1/jobs/{args.job_id}"))
    return EXIT_OK


def cmd_job_artifact(connection: Connection, args: argparse.Namespace) -> int:
    """One artifact's bytes. To a file with `--out`, else to stdout.

    This is the ONE command that does not print JSON, and it cannot: the bytes
    are a FLAC or a WAV. `--out -` is spelled explicitly so a pipe is something
    the caller asked for rather than something they got by leaving a flag off.
    """
    destination = None if args.out == "-" else Path(args.out)
    written = download(
        connection, f"/v1/jobs/{args.job_id}/artifacts/{args.name}", destination
    )
    if destination is not None:
        emit({"name": args.name, "path": str(destination), "bytes": written})
    return EXIT_OK


# ----------------------------------------------------------------------- tasks


def cmd_task_submit(connection: Connection, args: argparse.Namespace) -> int:
    """`POST /v1/tasks` — work done TO the server, not with its card.

    `TaskCreate`'s validator refuses a field that belongs to another task type,
    so the fields are passed through exactly as given and absent ones are left
    absent. Sending `{"kind": null}` with an `install` would be this CLI
    inventing a request the operator did not make.
    """
    body: dict[str, Any] = {"type": args.type}
    for flag, field in (
        ("kind", "kind"), ("id", "id"), ("job_type", "job_type"),
        ("narrator_engine", "narrator_engine"), ("target", "target"),
    ):
        value = getattr(args, flag)
        if value is not None:
            body[field] = value
    if args.module is not None:
        body["module"] = read_json_argument(args.module, "--module")

    accepted = call(connection, "POST", "/v1/tasks", json_body=body)
    task_id = accepted["task_id"]
    if not args.follow:
        emit(accepted)
        return EXIT_OK
    emit_line(accepted)
    code, _ = follow_to_the_end(
        connection, f"/v1/tasks/{task_id}/events", f"/v1/tasks/{task_id}", since=0
    )
    return code


def cmd_task_events(connection: Connection, args: argparse.Namespace) -> int:
    code, _ = follow_to_the_end(
        connection,
        f"/v1/tasks/{args.task_id}/events",
        f"/v1/tasks/{args.task_id}",
        since=args.since,
    )
    return code


def cmd_task_get(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "GET", f"/v1/tasks/{args.task_id}"))
    return EXIT_OK


def cmd_task_cancel(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "DELETE", f"/v1/tasks/{args.task_id}"))
    return EXIT_OK


# ------------------------------------------------------- tts, the serial door
#
# PHASE3-TTS.md sections 6 and 7 are two doors and this CLI keeps them two.
#
#   BATCHED   `crucible api job submit --type tts --model <voice> --params @…`
#             One job, N chunks, one `generate_batch`, `<index>.flac` artifacts.
#             That is the render door and it needs no verb of its own.
#
#   SERIAL    `crucible api stream …` — a session, one `say` at a time, audio
#             arriving on an SSE stream while the row is still generating.
#
# The session id survives the process, which is what makes four separate
# commands a usable shape: a ladder script opens once, says a row, reads until
# that row is done, decides, says the next.


def cmd_stream_open(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", "/v1/tts/stream", json_body={
        "voice": args.voice, "language": args.language,
    }))
    return EXIT_OK


def cmd_stream_say(connection: Connection, args: argparse.Namespace) -> int:
    """One row. `--take` has no default here, exactly as the wire has none.

    PHASE3-TTS.md section 7: a default take on the wire would be the server
    choosing, and now that five fine-tunes declare a second rung it would be a
    render at a take nobody asked for. The same argument makes it `required` in
    argparse rather than `default=0`.
    """
    text = args.text
    if args.text_file is not None:
        if text is not None:
            raise ClientRefusal(
                "say_overspecified: pass --text or --text-file, not both"
            )
        path = Path(args.text_file)
        if not path.is_file():
            raise ClientRefusal(f"text_file_missing: {path} is not a file")
        text = path.read_text(encoding="utf-8")
    if text is None:
        raise ClientRefusal("say_needs_text: pass --text or --text-file")
    emit(call(connection, "POST", f"/v1/tts/stream/{args.session_id}", json_body={
        "op": "say", "id": args.row, "text": text, "take": args.take,
    }))
    return EXIT_OK


def cmd_stream_cancel(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", f"/v1/tts/stream/{args.session_id}", json_body={
        "op": "cancel", "id": args.row,
    }))
    return EXIT_OK


def cmd_stream_cancel_all(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", f"/v1/tts/stream/{args.session_id}",
              json_body={"op": "cancel_all"}))
    return EXIT_OK


def cmd_stream_close(connection: Connection, args: argparse.Namespace) -> int:
    """`DELETE`, not `{"op": "close"}`. The two are the same to the server."""
    emit(call(connection, "DELETE", f"/v1/tts/stream/{args.session_id}"))
    return EXIT_OK


def cmd_stream_events(connection: Connection, args: argparse.Namespace) -> int:
    """Follow a session, optionally writing its audio out as WAVs.

    `--until <row>` is what makes a serial ladder scriptable. Without it this
    follows to `closed`, which for a session a script is still feeding never
    comes; with it the command returns as soon as that row is `done` or `error`,
    which is the moment the script has something to judge.

    `--audio-dir` writes `<row>.wav`. The rate is the one on the session's
    `ready` frame — the engine's own, which the load already checked against the
    manifest — and never a constant: PHASE3-TTS.md section 6 refuses a row whose
    rate is not the loaded one rather than resampling it, and a header this
    client invented would undo that. A session followed with `--audio-dir` but
    `--since` past the `ready` frame is refused, because there is then no rate
    to write and guessing one would put a wrong header on real audio.
    """
    audio_dir = None if args.audio_dir is None else Path(args.audio_dir)
    if audio_dir is not None and args.since:
        raise ClientRefusal(
            "audio_needs_the_ready_frame: --audio-dir writes the sample rate the "
            f"session reported on its `ready` event, and --since {args.since} "
            "starts after it. Resume without --audio-dir and decode the "
            "pcm_base64 yourself, or follow from the start"
        )
    pcm: dict[str, bytearray] = {}
    sample_rate: int | None = None
    exit_code = EXIT_OK

    for event in follow(connection, f"/v1/tts/stream/{args.session_id}/events",
                        last_event_id=args.since):
        emit_line(event)
        name, data = event.get("event"), event.get("data", {})
        if name == "ready":
            sample_rate = data["sample_rate"]
        elif name == "audio" and audio_dir is not None:
            pcm.setdefault(data["id"], bytearray()).extend(
                base64.b64decode(data["pcm_base64"])
            )
        elif name == "error":
            exit_code = EXIT_REFUSED
            if args.until is not None and data.get("id") == args.until:
                break
        elif name == "done" and args.until is not None and data["id"] == args.until:
            break
        elif name == "closed":
            break

    if audio_dir is not None and pcm:
        if sample_rate is None:
            raise ClientRefusal(
                "no_ready_frame: the session sent audio without a `ready` event, "
                "so there is no sample rate to write a WAV header with"
            )
        written = []
        audio_dir.mkdir(parents=True, exist_ok=True)
        for row, samples in pcm.items():
            target = audio_dir / f"{row}.wav"
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(bytes(samples))
            written.append({"id": row, "path": str(target), "bytes": len(samples)})
        emit_line({"audio_saved": written, "sample_rate": sample_rate})
    return exit_code


# ---------------------------------------------------------------------- leases


def cmd_lease_open(connection: Connection, args: argparse.Namespace) -> int:
    """`POST /v1/models/{id}/lease` — "I am mid-run on what is resident".

    The path says `models` and the id may name a model, a voice or an aligner:
    the server reads the kind off `Residency.resident`, so there is no `kind` to
    send and nothing here to disambiguate.
    """
    emit(call(connection, "POST", f"/v1/models/{args.subject_id}/lease", json_body={
        "act": args.act, "ttl_seconds": args.ttl,
    }))
    return EXIT_OK


def cmd_lease_heartbeat(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", f"/v1/leases/{args.lease_id}/heartbeat"))
    return EXIT_OK


def cmd_lease_release(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/leases/{args.lease_id}")
    emit({"released": args.lease_id})
    return EXIT_OK


# ------------------------------------------------------------------- the verb


def command(args: argparse.Namespace) -> int:
    """`crucible api …`'s one entry point: resolve, run, name what went wrong.

    Three failure kinds and three sentences. A server refusal is printed
    verbatim by `report_http_error`. A client refusal — a malformed flag, a
    missing file, a combination this CLI will not make a request out of — is
    this module's own named code. A transport failure is the address being
    wrong or the server being down, which is not a refusal at all and says so.
    """
    try:
        connection = resolve(args)
        return int(args.api_func(connection, args))
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except urllib.error.HTTPError as exc:
        return report_http_error(exc)
    except BrokenPipeError:
        # `crucible api job events … | head` is a legitimate thing to type, and
        # it is not the server going away. CAUGHT BEFORE OSError, which it is a
        # subclass of: written the other way round (measured 2026-09-16, first
        # run against the live engine) `crucible api info | head -12` printed
        # `server_unreachable: the local engine did not answer: [Errno 32]
        # Broken pipe` about a server that had answered perfectly.
        return EXIT_OK
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        target = getattr(args, "url", None) or "the local engine"
        print(f"crucible: server_unreachable: {target} did not answer: {exc}. "
              f"{CONNECT_NOTE}", file=sys.stderr)
        return EXIT_REFUSED


def _connection_flags(parser: argparse.ArgumentParser) -> None:
    """The three flags that make these verbs a client. See the module preamble."""
    group = parser.add_argument_group("server")
    group.add_argument(
        "--url",
        default=None,
        help="the server's base URL, e.g. http://192.168.68.20:7100. Requires "
             "--token. Omit both to use this machine's installed engine",
    )
    group.add_argument(
        "--token",
        default=None,
        help="the bearer token for --url. Never taken from the local "
             "installation for a URL typed here",
    )
    group.add_argument(
        "--pairing",
        default=None,
        help="a `crucible://name@host:port/#token` line, as `crucible token "
             "--url` prints it. Carries the address and the token together",
    )


def add_parser(subparsers: Any) -> None:
    """`crucible api …`. One namespace, one binary — see the module preamble."""
    parser = subparsers.add_parser(
        "api",
        help="talk to a Crucible server over HTTP: submit jobs, stream tts, read state",
        description=(
            "The CLIENT half of the command line. Every verb here makes an HTTP "
            "request to a server that may be this machine's, the one in WSL, or "
            "one across the network; the operator verbs (init, install, service, "
            "models, …) act on this machine's installation instead and take no "
            "address. Output is JSON on stdout, one document per command, or one "
            "compact object per line while following a stream. A refusal is the "
            "server's own {\"error\": {...}} on stderr with exit 1."
        ),
    )
    _connection_flags(parser)
    verbs = parser.add_subparsers(dest="api_command", required=True)

    def verb(name: str, help_text: str) -> argparse.ArgumentParser:
        return verbs.add_parser(name, help=help_text)

    # ---- the plain reads. `GET`, no arguments, print the document.
    for name, path, help_text in (
        ("ping", "/v1/ping", "is this a Crucible, and which one (needs no token)"),
        ("health", "/v1/health", "the lane, the queue and every job type's readiness"),
        ("info", "/v1/info", "identity, host, job types and every capability row"),
        ("setup", "/v1/setup", "the pairing lines and job types an app's connect door takes"),
        ("accelerator", "/v1/accelerator", "what is on the card right now, probed"),
        ("activity", "/v1/activity", "what this server is doing and who asked"),
        ("models", "/v1/models", "the llm models this SERVER holds (`crucible models list` reads this BUILD)"),
        ("voices", "/v1/voices", "the tts voices this SERVER holds"),
        ("catalog", "/v1/catalog", "every subject this backend can hold, installed or not"),
        ("openai-models", "/v1/openai/models", "the resident model plus routed upstreams, OpenAI-shaped"),
    ):
        reader = verb(name, help_text)
        reader.set_defaults(api_func=cmd_get(path))

    capability = verb("capability", "what this host can hold, and why")
    capability.add_argument(
        "--accelerator-probe", action="store_true",
        help="pay for a live nvidia-smi probe rather than reading the record",
    )
    capability.set_defaults(api_func=cmd_capability)

    catalog_remove = verb("catalog-remove", "delete an installed subject's files")
    catalog_remove.add_argument("kind", help="model, voice, rvc, denoise, asr, align")
    catalog_remove.add_argument("subject_id")
    catalog_remove.set_defaults(api_func=cmd_catalog_remove)

    settings = verb("settings", "read the settings document, or PUT a patch")
    settings.add_argument(
        "--patch", default=None,
        help="a partial settings document, as JSON or @file. Applied whole or "
             "not at all; the answer is the document after the write",
    )
    settings.add_argument(
        "--act", default=None,
        help="the capability class this write is for, recorded on the bench",
    )
    settings.set_defaults(api_func=cmd_settings)

    upstream_test = verb("upstream-test", "ask a configured upstream what it serves")
    upstream_test.add_argument("name")
    upstream_test.add_argument(
        "--body", default=None,
        help='{"key": "…"} or {"url": "…"} to test before saving, as JSON or '
             "@file. Omit to use the stored record",
    )
    upstream_test.set_defaults(api_func=cmd_upstream_test)

    pairing_requests = verb("pairing-requests", "connect requests waiting for a decision")
    pairing_requests.set_defaults(api_func=cmd_pairing_requests)

    pairing_decide = verb("pairing-decide", "approve or deny one connect request")
    pairing_decide.add_argument("--id", required=True)
    pairing_decide.add_argument("--code", required=True, help="the displayed XXXX-XXXX code")
    decision = pairing_decide.add_mutually_exclusive_group(required=True)
    decision.add_argument("--allow", dest="allow", action="store_true")
    decision.add_argument("--deny", dest="allow", action="store_false")
    pairing_decide.set_defaults(api_func=cmd_pairing_decide)

    chat = verb("chat", "a chat completion — the cleanup and translation door")
    chat.add_argument(
        "--body", default=None,
        help="the whole OpenAI-shaped request, as JSON or @file. Forwarded "
             "verbatim except `model`",
    )
    chat.add_argument("--model", default=None, help="shorthand: the model for a one-shot question")
    chat.add_argument("--message", default=None, help="shorthand: the user message")
    chat.add_argument("--stream", action="store_true", help="send stream:true and print each frame")
    chat.add_argument("--act", default=None, help="the capability class, for the bench")
    chat.set_defaults(api_func=cmd_chat)

    upload_verb = verb("upload", "put a file on the server and get its blob_id")
    upload_verb.add_argument("path")
    upload_verb.set_defaults(api_func=cmd_upload)

    # ---- jobs
    job = verb("job", "submit, watch, cancel and read back one unit of work")
    job_verbs = job.add_subparsers(dest="job_command", required=True)

    submit = job_verbs.add_parser("submit", help="POST /v1/jobs")
    submit.add_argument(
        "--type", required=True,
        # NOT `llm` — there is no such job type, and writing one here would send
        # somebody looking for it. LLM work is the chat proxy; the llm-class job
        # types are the two that move a model on and off the card. `api info`'s
        # `job_types` is the list to ASK rather than the one to remember, which
        # is why this help says so instead of pretending to be complete.
        help="tts, asr, align, align-longform, rvc, denoise, echo, load-model, "
             "unload-model, load-voice, unload-voice, unload-aligner, "
             "unload-denoiser — `crucible api info` says which this server offers",
    )
    submit.add_argument("--model", default=None, help="the model, voice or aligner id this type serves")
    submit.add_argument("--params", default=None, help="the type's params object, as JSON or @file")
    submit.add_argument(
        "--input", action="append", default=[], metavar="NAME=PATH",
        help="upload PATH and give the job that blob as input NAME; repeatable",
    )
    submit.add_argument(
        "--input-blob", action="append", default=[], metavar="NAME=BLOB_ID",
        help="give the job an already-uploaded blob as input NAME; repeatable",
    )
    submit.add_argument("--follow", action="store_true", help="watch the event stream until the job ends")
    submit.add_argument(
        "--artifacts-dir", default=None,
        help="save every artifact here once the job is done. Requires --follow",
    )
    submit.set_defaults(api_func=cmd_job_submit)

    job_get = job_verbs.add_parser("get", help="GET /v1/jobs/{id}")
    job_get.add_argument("job_id")
    job_get.set_defaults(api_func=cmd_job_get)

    job_events = job_verbs.add_parser("events", help="follow GET /v1/jobs/{id}/events")
    job_events.add_argument("job_id")
    job_events.add_argument(
        "--since", type=int, default=0, metavar="EVENT_ID",
        help="resume after this event id (sent as Last-Event-ID)",
    )
    job_events.set_defaults(api_func=cmd_job_events)

    job_cancel = job_verbs.add_parser("cancel", help="DELETE /v1/jobs/{id}")
    job_cancel.add_argument("job_id")
    job_cancel.set_defaults(api_func=cmd_job_cancel)

    job_artifact = job_verbs.add_parser("artifact", help="GET one artifact's bytes")
    job_artifact.add_argument("job_id")
    job_artifact.add_argument("name")
    job_artifact.add_argument(
        "--out", default="-",
        help="where to write it; `-` (the default) means stdout",
    )
    job_artifact.set_defaults(api_func=cmd_job_artifact)

    # ---- tasks
    task = verb("task", "work done TO the server: pulls, installs, engine restarts")
    task_verbs = task.add_subparsers(dest="task_command", required=True)

    task_submit = task_verbs.add_parser("submit", help="POST /v1/tasks")
    task_submit.add_argument("--type", required=True, choices=["pull", "install", "module", "engine", "engine-restart"])
    task_submit.add_argument("--kind", default=None, help="pull: the subject kind")
    task_submit.add_argument("--id", default=None, help="pull: the subject id")
    task_submit.add_argument("--job-type", default=None, help="install: the job type to install")
    task_submit.add_argument("--narrator-engine", default=None, help="install: which tts engine")
    task_submit.add_argument("--module", default=None, help="module: the module document, as JSON or @file")
    task_submit.add_argument("--target", default=None, help="engine: what to move it to")
    task_submit.add_argument("--follow", action="store_true", help="watch until the task ends")
    task_submit.set_defaults(api_func=cmd_task_submit)

    task_list = task_verbs.add_parser("list", help="GET /v1/tasks — the last few, newest first")
    task_list.set_defaults(api_func=cmd_get("/v1/tasks"))

    task_get = task_verbs.add_parser("get", help="GET /v1/tasks/{id}")
    task_get.add_argument("task_id")
    task_get.set_defaults(api_func=cmd_task_get)

    task_events = task_verbs.add_parser("events", help="follow GET /v1/tasks/{id}/events")
    task_events.add_argument("task_id")
    task_events.add_argument("--since", type=int, default=0, metavar="EVENT_ID")
    task_events.set_defaults(api_func=cmd_task_events)

    task_cancel = task_verbs.add_parser("cancel", help="DELETE /v1/tasks/{id}")
    task_cancel.add_argument("task_id")
    task_cancel.set_defaults(api_func=cmd_task_cancel)

    # ---- the serial tts door
    stream = verb("stream", "serialized tts: one session, one row at a time")
    stream_verbs = stream.add_subparsers(dest="stream_command", required=True)

    stream_open = stream_verbs.add_parser("open", help="POST /v1/tts/stream")
    stream_open.add_argument("--voice", required=True)
    stream_open.add_argument("--language", required=True)
    stream_open.set_defaults(api_func=cmd_stream_open)

    stream_say = stream_verbs.add_parser("say", help="one row into an open session")
    stream_say.add_argument("session_id")
    stream_say.add_argument("--row", required=True, help="the client's id for this row")
    stream_say.add_argument("--text", default=None)
    stream_say.add_argument("--text-file", default=None, help="read the text from a file instead")
    stream_say.add_argument(
        "--take", type=int, required=True,
        help="which rung of the voice's ladder. No default: the wire has none",
    )
    stream_say.set_defaults(api_func=cmd_stream_say)

    stream_events = stream_verbs.add_parser("events", help="follow the session's SSE stream")
    stream_events.add_argument("session_id")
    stream_events.add_argument("--since", type=int, default=0, metavar="EVENT_ID")
    stream_events.add_argument(
        "--until", default=None, metavar="ROW",
        help="stop once this row is done or errors, instead of waiting for close",
    )
    stream_events.add_argument(
        "--audio-dir", default=None,
        help="write each row's audio here as <row>.wav, at the rate the session "
             "reported on its ready event",
    )
    stream_events.set_defaults(api_func=cmd_stream_events)

    stream_cancel = stream_verbs.add_parser("cancel", help="cancel one row in flight")
    stream_cancel.add_argument("session_id")
    stream_cancel.add_argument("--row", required=True)
    stream_cancel.set_defaults(api_func=cmd_stream_cancel)

    stream_cancel_all = stream_verbs.add_parser("cancel-all", help="cancel every row in flight")
    stream_cancel_all.add_argument("session_id")
    stream_cancel_all.set_defaults(api_func=cmd_stream_cancel_all)

    stream_close = stream_verbs.add_parser("close", help="DELETE /v1/tts/stream/{id}")
    stream_close.add_argument("session_id")
    stream_close.set_defaults(api_func=cmd_stream_close)

    # ---- leases
    lease = verb("lease", "say you are mid-run on what is resident")
    lease_verbs = lease.add_subparsers(dest="lease_command", required=True)

    lease_open = lease_verbs.add_parser("open", help="POST /v1/models/{id}/lease")
    lease_open.add_argument("subject_id", help="the resident model, voice or aligner")
    lease_open.add_argument("--act", required=True, help="the capability class this run is")
    lease_open.add_argument("--ttl", type=int, required=True, metavar="SECONDS")
    lease_open.set_defaults(api_func=cmd_lease_open)

    lease_heartbeat = lease_verbs.add_parser("heartbeat", help="push the deadline out")
    lease_heartbeat.add_argument("lease_id")
    lease_heartbeat.set_defaults(api_func=cmd_lease_heartbeat)

    lease_release = lease_verbs.add_parser("release", help="give the card back")
    lease_release.add_argument("lease_id")
    lease_release.set_defaults(api_func=cmd_lease_release)

    parser.set_defaults(func=command)
