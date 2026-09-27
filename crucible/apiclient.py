from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
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

PAIRING_ENV = "CRUCIBLE_PAIRING"

EXIT_OK = 0
EXIT_REFUSED = 1

REQUEST_TIMEOUT_SECONDS = 900.0

CONNECT_NOTE = (
    "Crucible answers /v1/ping without a token; try that first if this is the "
    "wrong address"
)


class ClientRefusal(CrucibleError):
    ...


@dataclass(frozen=True)
class Connection:

    url: str
    token: str
    name: str | None
    source: str


def resolve(args: argparse.Namespace) -> Connection:
    given_url = getattr(args, "url", None)
    given_token = getattr(args, "token", None)
    given_pairing = getattr(args, "pairing", None)
    pairing_file = getattr(args, "pairing_file", None)
    pairing_env = os.environ.get(PAIRING_ENV) or None
    given_server = getattr(args, "server", None)

    sources = [
        name
        for name, value in (
            ("--pairing", given_pairing),
            ("--pairing-file", pairing_file),
            ("--server", given_server),
            (f"${PAIRING_ENV}", pairing_env),
            ("--url/--token", given_url or given_token),
        )
        if value is not None
    ]
    if len(sources) > 1:
        raise ClientRefusal(
            f"connection_overspecified: {', '.join(sources)} each name a server; "
            "pass exactly one. (An environment variable counts: unset "
            f"{PAIRING_ENV} to use a flag.)"
        )
    if given_server is not None:
        pairing_file = str(saved_pairing_path(given_server))
        if not Path(pairing_file).is_file():
            raise ClientRefusal(
                f"server_not_paired: this computer has not paired with "
                f"{given_server!r}. Run `crucible pair <its address>` once first"
            )
    if pairing_file is not None:
        try:
            lines = [
                line.strip()
                for line in Path(pairing_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise ClientRefusal(f"pairing_file_unreadable: {pairing_file}: {exc}") from None
        if len(lines) != 1:
            raise ClientRefusal(
                f"pairing_file_invalid: {pairing_file} holds {len(lines)} non-empty "
                "lines; it must hold exactly one pairing line"
            )
        given_pairing = lines[0]
        source = f"--pairing-file {pairing_file}"
    elif pairing_env is not None:
        given_pairing = pairing_env.strip()
        source = f"${PAIRING_ENV}"
    else:
        source = "--pairing"

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
            source=source,
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

    from .local import LocalError, connection as local_connection

    try:
        url, name, token = local_connection(crucible_home())
    except (LocalError, ConfigError, OSError) as exc:
        raise ClientRefusal(
            f"no_local_engine: {exc}. Pass --url and --token, or --pairing, to "
            "reach a server that is not this machine's"
        ) from None
    return Connection(url=url.rstrip("/"), token=token, name=name, source="local")


def _headers(connection: Connection) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {connection.token}",
        API_HEADER: str(API_VERSION),
        "User-Agent": f"crucible-cli/{VERSION}",
    }


def _opener() -> urllib.request.OpenerDirector:
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
            if line.startswith(":"):
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


def emit(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_line(value: Any) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def report_http_error(exc: urllib.error.HTTPError) -> int:
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


def read_text_argument(raw: str, flag: str) -> str:
    if raw.startswith("@"):
        path = Path(raw[1:])
        if not path.is_file():
            raise ClientRefusal(f"{flag}_file_missing: {path} is not a file")
        return path.read_text(encoding="utf-8")
    return raw


def read_json_argument(raw: str, flag: str) -> Any:
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
    name, separator, value = raw.partition("=")
    if separator != "=" or name == "" or value == "":
        raise ClientRefusal(
            f"{flag}_malformed: {raw!r} is not `name=value`"
        )
    return name, value


SUCCEEDED = "done"


def follow_to_the_end(
    connection: Connection,
    events_path: str,
    state_path: str,
    *,
    since: int,
) -> tuple[int, dict[str, Any]]:
    for event in follow(connection, events_path, last_event_id=since):
        emit_line(event)
    state = call(connection, "GET", state_path)
    emit_line(state)
    status = state.get("status")
    return (EXIT_OK if status == SUCCEEDED else EXIT_REFUSED), state


def cmd_get(path: str) -> Callable[[Connection, argparse.Namespace], int]:

    def run(connection: Connection, args: argparse.Namespace) -> int:
        emit(call(connection, "GET", path))
        return EXIT_OK

    return run


def cmd_capability(connection: Connection, args: argparse.Namespace) -> int:
    params: list[tuple[str, str]] = []
    if args.accelerator_probe:
        params.append(("accelerator_probe", "true"))
    if args.capability_class is not None:
        params.append(("class", args.capability_class))
    if args.context_tokens is not None:
        params.append(("context_tokens", args.context_tokens))
    if args.concurrency is not None:
        params.append(("concurrency", args.concurrency))
    query = "" if not params else "?" + urllib.parse.urlencode(params)
    emit(call(connection, "GET", "/v1/capability" + query))
    return EXIT_OK


def cmd_catalog_remove(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/catalog/{args.kind}/{args.subject_id}")
    emit({"removed": {"kind": args.kind, "id": args.subject_id}})
    return EXIT_OK


def cmd_voice_write(connection: Connection, args: argparse.Namespace) -> int:
    document = read_json_argument(args.manifest, "--manifest")
    emit(call(
        connection, "PUT", f"/v1/voices/{args.voice_id}", json_body=document,
    ))
    return EXIT_OK


def cmd_voice_remove(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/voices/{args.voice_id}")
    emit({"removed": {"manifest": args.voice_id}})
    return EXIT_OK


def cmd_settings(connection: Connection, args: argparse.Namespace) -> int:
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


class _Question(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        questions = getattr(namespace, self.dest)
        if questions is None:
            questions = []
            setattr(namespace, self.dest, questions)
        questions.append((option_string, list(values)))


def decide_questions(given: list[tuple[str, list[str]]] | None) -> dict[str, Any]:
    if not given:
        raise ClientRefusal(
            "decide_needs_a_question: pass at least one --choice, --score or --yesno"
        )
    questions: dict[str, Any] = {}
    for flag, values in given:
        name = values[0]
        if name in questions:
            raise ClientRefusal(f"decide_question_repeated: question {name!r} is given twice")
        if flag == "--yesno":
            questions[name] = {"type": "yesno", "instructions": values[1]}
        elif flag == "--score":
            questions[name] = {
                "type": "score", "instructions": values[1],
                "levels": [level.strip() for level in values[2].split(",")],
            }
        elif flag == "--choice":
            if len(values) < 2:
                raise ClientRefusal(
                    f"decide_choice_malformed: --choice {name} needs NAME "
                    '"instructions" opt=description …'
                )
            options: dict[str, str] = {}
            for word in values[2:]:
                option, description = split_assignment(word, "--choice")
                if option in options:
                    raise ClientRefusal(
                        f"decide_option_repeated: --choice {name}: option {option!r} "
                        "is given twice"
                    )
                options[option] = description
            questions[name] = {"type": "choice", "instructions": values[1], "options": options}
        else:
            raise AssertionError(flag)
    return questions


def cmd_decide(connection: Connection, args: argparse.Namespace) -> int:
    images = []
    for raw in args.image:
        path = Path(raw)
        if not path.is_file():
            raise ClientRefusal(f"image_missing: {path} is not a file")
        data = path.read_bytes()
        if not data:
            raise ClientRefusal(f"image_empty: {path} has no bytes")
        images.append(base64.b64encode(data).decode("ascii"))
    if args.state is None:
        if not images:
            raise ClientRefusal(
                "decide_needs_state: pass --state <text|@file>, or at least one --image"
            )
        state = ""
    else:
        state = read_text_argument(args.state, "--state")
    body: dict[str, Any] = {"model": args.model, "state": state}
    if images:
        body["images"] = images
    body["questions"] = decide_questions(args.questions)
    if args.missing is not None:
        body["missing"] = args.missing
    extra = None if args.act is None else {"X-Crucible-Act": args.act}
    emit(call(connection, "POST", "/v1/decide", json_body=body, extra_headers=extra))
    return EXIT_OK


def cmd_align(connection: Connection, args: argparse.Namespace) -> int:
    if not args.window:
        raise ClientRefusal("windows_required: give at least one --window INDEX TEXT AUDIO")
    chunks: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, str]] = {}
    for raw_index, raw_text, raw_audio in args.window:
        try:
            index = int(raw_index)
        except ValueError:
            raise ClientRefusal(
                f"window_index_invalid: --window index {raw_index!r} is not an integer"
            ) from None
        audio = Path(raw_audio)
        name = f"{index}{audio.suffix}"
        if not audio.suffix:
            raise ClientRefusal(
                f"window_audio_unnamed: {raw_audio!r} has no extension; ffmpeg reads "
                "the container from it"
            )
        if name in inputs:
            raise ClientRefusal(f"window_index_repeated: index {index} appears more than once")
        text = read_text_argument(raw_text, "--window TEXT")
        chunks.append({"index": index, "text": text})
        inputs[name] = {"blob_id": upload(connection, audio)["blob_id"]}
    if args.out is not None and not args.follow:
        raise ClientRefusal(
            "out_needs_follow: --out waits for the job to finish, so it must be "
            "asked for with --follow"
        )
    body = {
        "type": "align",
        "model": args.model,
        "params": {"language": args.language, "chunks": chunks},
        "inputs": inputs,
    }
    accepted = call(connection, "POST", "/v1/jobs", json_body=body)
    if not args.follow:
        emit(accepted)
        return EXIT_OK
    emit_line(accepted)
    job_id = accepted["job_id"]
    code, _ = follow_to_the_end(
        connection, f"/v1/jobs/{job_id}/events", f"/v1/jobs/{job_id}", since=0
    )
    if args.out is not None and code == EXIT_OK:
        target = Path(args.out)
        written = download(connection, f"/v1/jobs/{job_id}/artifacts/alignment.json", target)
        emit_line({"alignment": str(target), "bytes": written})
    return code


def cmd_upload(connection: Connection, args: argparse.Namespace) -> int:
    emit(upload(connection, Path(args.path)))
    return EXIT_OK


def cmd_job_submit(connection: Connection, args: argparse.Namespace) -> int:
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
    if args.resume is not None:
        params = body.setdefault("params", {})
        if params.get("resume") not in (None, args.resume):
            raise ClientRefusal(
                f"resume_given_twice: --resume {args.resume} and --params "
                f"resume {params['resume']!r} name two journals; send one"
            )
        params["resume"] = args.resume

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
    emit(call(connection, "DELETE", f"/v1/jobs/{args.job_id}"))
    return EXIT_OK


def cmd_job_artifact(connection: Connection, args: argparse.Namespace) -> int:
    destination = None if args.out == "-" else Path(args.out)
    written = download(
        connection, f"/v1/jobs/{args.job_id}/artifacts/{args.name}", destination
    )
    if destination is not None:
        emit({"name": args.name, "path": str(destination), "bytes": written})
    return EXIT_OK


def cmd_resumable_get(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "GET", f"/v1/resumable/{args.resume_id}"))
    return EXIT_OK


def cmd_resumable_discard(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "DELETE", f"/v1/resumable/{args.resume_id}"))
    return EXIT_OK


def cmd_task_submit(connection: Connection, args: argparse.Namespace) -> int:
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


def cmd_stream_open(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", "/v1/tts/stream", json_body={
        "voice": args.voice, "language": args.language,
    }))
    return EXIT_OK


def cmd_stream_say(connection: Connection, args: argparse.Namespace) -> int:
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
    emit(call(connection, "DELETE", f"/v1/tts/stream/{args.session_id}"))
    return EXIT_OK


def cmd_stream_events(connection: Connection, args: argparse.Namespace) -> int:
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


def cmd_lease_open(connection: Connection, args: argparse.Namespace) -> int:
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


SERVERS_DIR = "servers"

PAIR_TIMEOUT_SECONDS = 10.0

PAIR_DEFAULT_PORT = 7100


def server_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name.strip())
    slug = slug.strip(".-")
    if not slug:
        raise ClientRefusal(f"server_name_invalid: {name!r} names no server")
    return slug


def saved_pairing_path(name: str) -> Path:
    return crucible_home() / SERVERS_DIR / f"{server_slug(name)}.pairing"


def _pair_origin(address: str) -> str:
    raw = address.strip()
    if not raw or any(ch in raw for ch in " @?#"):
        raise ClientRefusal(
            "pair_bad_address: give the other computer's address, like "
            "192.168.68.88 or kylies-pc, with no token or path in it"
        )
    if "://" not in raw:
        if raw.count(":") > 1 and not raw.startswith("["):
            raw = f"[{raw}]"
        raw = "http://" + raw
    parts = urllib.parse.urlsplit(raw)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.path not in ("", "/") or port == -1):
        raise ClientRefusal(
            f"pair_bad_address: {address!r} is not an address this can dial; "
            "give an IP address or a computer name, e.g. 192.168.68.88"
        )
    if port is None:
        port = 443 if parts.scheme == "https" else PAIR_DEFAULT_PORT
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}:{port}"


def _pair_call(origin: str, path: str, body: dict[str, Any] | None = None,
               token: str | None = None) -> dict[str, Any]:
    headers = {API_HEADER: str(API_VERSION), "User-Agent": f"crucible-cli/{VERSION}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        origin + path, method="GET" if body is None else "POST", headers=headers,
        data=None if body is None else json.dumps(body).encode("utf-8"),
    )
    try:
        with _opener().open(request, timeout=PAIR_TIMEOUT_SECONDS) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            error = json.load(exc).get("error", {})
            code, message = error.get("code"), error.get("message")
        except (ValueError, AttributeError):
            code = message = None
        if isinstance(code, str) and isinstance(message, str):
            raise ClientRefusal(f"{code}: {message}") from None
        raise ClientRefusal(
            f"pair_not_crucible: {origin} answered HTTP {exc.code}, not as a Crucible"
        ) from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ClientRefusal(
            f"pair_unreachable: nothing answered at {origin} from this computer "
            f"({getattr(exc, 'reason', exc)}). On that computer, Crucible has to be "
            "running and shared with the network: on Windows that is "
            "`crucible lan enable` there, which also says if its network is "
            "marked Public. Check the address, then try again"
        ) from None
    except ValueError:
        raise ClientRefusal(f"pair_not_crucible: {origin} did not answer with JSON") from None
    if not isinstance(value, dict):
        raise ClientRefusal(f"pair_not_crucible: {origin} did not answer as a Crucible")
    return value


def pair(address: str, *, client_name: str, sleep: Callable[[float], None] | None = None,
         notify: Callable[[str], None] | None = None) -> tuple[str, str, str]:
    import time

    sleep = time.sleep if sleep is None else sleep
    origin = _pair_origin(address)
    ping = _pair_call(origin, "/v1/ping")
    name = ping.get("name")
    if ping.get("crucible") is not True or not isinstance(name, str) or not name:
        raise ClientRefusal(f"pair_not_crucible: {origin} is not a Crucible")
    if ping.get("api_version") != API_VERSION:
        raise ClientRefusal(
            f"api_version_mismatch: {name} speaks API version "
            f"{ping.get('api_version')} and this computer speaks {API_VERSION}. "
            "Update whichever of the two is older"
        )
    if ping.get("pairing_version") != 1:
        raise ClientRefusal(
            f"pairing_unavailable: {name} is too old to connect by address; "
            "update Crucible on that computer"
        )
    start = _pair_call(origin, "/v1/pairing/start", {"client_name": client_name[:80]})
    if start.get("name") != name or not isinstance(start.get("id"), str) \
            or not isinstance(start.get("device_code"), str):
        raise ClientRefusal(f"pair_not_crucible: {origin} returned an incompatible pairing request")
    if start.get("approval_required") is not False and notify is not None:
        notify(
            f"{name} asks for approval: on that computer, open Crucible's console "
            f"and approve the code {start.get('user_code')}. Waiting..."
        )
    interval = max(2.0, float(start.get("interval") or 2))
    remaining = float(start.get("expires_in") or 300)
    first = True
    while remaining > 0:
        if not first:
            sleep(interval)
            remaining -= interval
        first = False
        try:
            answer = _pair_call(origin, "/v1/pairing/poll",
                                {"id": start["id"], "device_code": start["device_code"]})
        except ClientRefusal as exc:
            if str(exc).startswith("pairing_slow_down"):
                continue
            raise
        status = answer.get("status")
        if status == "approved":
            token = answer.get("token")
            if answer.get("name") != name or not isinstance(token, str) or not token:
                raise ClientRefusal(f"pair_not_crucible: {origin} returned an incompatible approval")
            _pair_call(origin, "/v1/info", token=token)
            return name, origin, token
        if status == "denied":
            raise ClientRefusal(f"pair_denied: {name} turned this computer's request down")
        if status == "expired":
            break
    raise ClientRefusal(
        f"pair_expired: nobody approved the request on {name} in time; run this again"
    )


def cmd_pair(args: argparse.Namespace) -> int:
    import socket

    try:
        name, url, token = pair(
            args.address,
            client_name=f"crucible on {socket.gethostname()}",
            notify=lambda line: print(f"crucible: {line}", file=sys.stderr, flush=True),
        )
        target = Path(args.save) if args.save else saved_pairing_path(name)
        from .pairing import pairing_line, write_pairing_line

        write_pairing_line(target, pairing_line(name, url, token),
                           private_directory=not args.save)
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (CrucibleError, OSError) as exc:
        print(f"crucible: pair_save_failed: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    emit({
        "name": name,
        "url": url,
        "pairing_file": str(target),
        "use": (f"crucible api --pairing-file \"{target}\" <verb>" if args.save
                else f"crucible api --server {server_slug(name)} <verb>"),
    })
    return EXIT_OK


def add_pair_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "pair",
        help="connect to the Crucible on another computer by its address; no token to copy",
    )
    parser.add_argument("address", help="that computer's address, e.g. 192.168.68.88")
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="where to keep the pairing line (default: this machine's Crucible "
             f"home, {SERVERS_DIR}/<name>.pairing, used by `crucible api --server`)",
    )
    parser.set_defaults(func=cmd_pair)


def command(args: argparse.Namespace) -> int:
    try:
        connection = resolve(args)
        return int(args.api_func(connection, args))
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except urllib.error.HTTPError as exc:
        return report_http_error(exc)
    except BrokenPipeError:
        return EXIT_OK
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        target = getattr(args, "url", None) or "the local engine"
        print(f"crucible: server_unreachable: {target} did not answer: {exc}. "
              f"{CONNECT_NOTE}", file=sys.stderr)
        return EXIT_REFUSED


def _connection_flags(parser: argparse.ArgumentParser) -> None:
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
             "--url` prints it. Carries the address and the token together. It "
             "is visible in any process listing: a runner should use "
             f"--pairing-file or ${PAIRING_ENV}",
    )
    group.add_argument(
        "--pairing-file",
        default=None,
        metavar="PATH",
        help="read the pairing line from this file (exactly one line), so the "
             "token never appears in argv",
    )
    group.add_argument(
        "--server",
        default=None,
        metavar="NAME",
        help="a server this computer paired with by `crucible pair <address>`, "
             "by its name (e.g. crucible@kylies-pc or crucible-kylies-pc)",
    )


def add_parser(subparsers: Any) -> None:
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
    capability.add_argument(
        "--class", dest="capability_class", default=None,
        help="the client-sized class to size (generate)",
    )
    capability.add_argument(
        "--context-tokens", default=None,
        help="tokens per request for --class; above this host's ceiling is refused",
    )
    capability.add_argument(
        "--concurrency", default=None,
        help="requests in flight at once for --class",
    )
    capability.set_defaults(api_func=cmd_capability)

    catalog_remove = verb("catalog-remove", "delete an installed subject's files")
    catalog_remove.add_argument("kind", help="model, voice, rvc, denoise, asr, align")
    catalog_remove.add_argument("subject_id")
    catalog_remove.set_defaults(api_func=cmd_catalog_remove)

    voice_write = verb("voice-write", "add or replace a voice this server owns")
    voice_write.add_argument("voice_id")
    voice_write.add_argument(
        "--manifest", required=True,
        help="the whole manifest document, as JSON or @file — the table a "
             "voices/<id>.toml holds. An overlay wins over a packaged voice of "
             "the same id, and `api voice-remove` is the undo",
    )
    voice_write.set_defaults(api_func=cmd_voice_write)

    voice_remove = verb("voice-remove", "delete this server's own manifest for a voice")
    voice_remove.add_argument("voice_id")
    voice_remove.set_defaults(api_func=cmd_voice_remove)

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

    decide = verb("decide", "a distribution over each question's fixed answers — snap's grammar")
    decide.add_argument("--model", required=True, help="the resident model to read the decision from")
    decide.add_argument(
        "--state", default=None,
        help="what the questions are about: text, or @file. May be left out "
             "when at least one --image is given; the state is then \"\"",
    )
    decide.add_argument(
        "--image", action="append", default=[], metavar="PATH",
        help="an image file, sent base64-encoded; repeatable",
    )
    decide.add_argument(
        "--choice", dest="questions", action=_Question, nargs="+",
        metavar=("NAME", "INSTRUCTIONS OPT=DESC"),
        help='NAME "instructions" opt=description [opt=description …]; repeatable',
    )
    decide.add_argument(
        "--score", dest="questions", action=_Question, nargs=3,
        metavar=("NAME", "INSTRUCTIONS", "LEVELS"),
        help='NAME "instructions" "level1,level2,…" (ordered, low to high); repeatable',
    )
    decide.add_argument(
        "--yesno", dest="questions", action=_Question, nargs=2,
        metavar=("NAME", "INSTRUCTIONS"),
        help='NAME "a statement the state may make true"; repeatable',
    )
    decide.add_argument(
        "--missing", default=None, metavar="refuse|report",
        help="a label outside the engine's top-K: refuse the decision "
             "(label_not_in_probs, the server's default) or report it — null "
             "probability, named in missing_labels, the rest renormalised over "
             "the letters returned",
    )
    decide.add_argument("--act", default=None, help="the capability class, for the bench")
    decide.set_defaults(api_func=cmd_decide, questions=None)

    align = verb("align", "place known words in time: windows of audio + their text, one job")
    align.add_argument("--model", required=True, help="the aligner, e.g. qwen3-aligner")
    align.add_argument(
        "--language", required=True,
        help="ISO code, one of the aligner's eleven: en de fr es it pt ru ja ko zh yue",
    )
    align.add_argument(
        "--window", action="append", nargs=3, default=[],
        metavar=("INDEX", "TEXT", "AUDIO"),
        help="one window: its index, the words spoken in it (text or @file), and "
             "its audio file (at most 300 s); repeatable, all in one job",
    )
    align.add_argument("--follow", action="store_true", help="watch the job until it ends")
    align.add_argument(
        "--out", default=None,
        help="save alignment.json here once the job is done. Requires --follow",
    )
    align.set_defaults(api_func=cmd_align)

    upload_verb = verb("upload", "put a file on the server and get its blob_id")
    upload_verb.add_argument("path")
    upload_verb.set_defaults(api_func=cmd_upload)

    job = verb("job", "submit, watch, cancel and read back one unit of work")
    job_verbs = job.add_subparsers(dest="job_command", required=True)

    submit = job_verbs.add_parser("submit", help="POST /v1/jobs")
    submit.add_argument(
        "--type", required=True,
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
    submit.add_argument(
        "--resume", default=None, metavar="RESUME_ID",
        help="continue the journal a job answered with (params.resume); without "
             "it the job starts fresh. `crucible api resumable list` shows them",
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

    resumable = verb("resumable", "the resume journals: what can be resumed, and discarding one")
    resumable_verbs = resumable.add_subparsers(dest="resumable_command", required=True)
    resumable_list = resumable_verbs.add_parser(
        "list", help="GET /v1/resumable — every journal, newest first"
    )
    resumable_list.set_defaults(api_func=cmd_get("/v1/resumable"))
    resumable_get = resumable_verbs.add_parser("get", help="GET /v1/resumable/{id}")
    resumable_get.add_argument("resume_id")
    resumable_get.set_defaults(api_func=cmd_resumable_get)
    resumable_discard = resumable_verbs.add_parser(
        "discard", help="DELETE /v1/resumable/{id} — refused while a job writes it"
    )
    resumable_discard.add_argument("resume_id")
    resumable_discard.set_defaults(api_func=cmd_resumable_discard)

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
