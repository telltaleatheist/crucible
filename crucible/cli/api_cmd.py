from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import urllib.error
import urllib.parse
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from ..client import transport
from ..client.connection import PAIRING_ENV, Connection
from ..client.connection import resolve as resolve_connection
from ..client.errors import ClientRefusal, error_in, next_step, unreachable
from ..client.transport import call, follow, upload
from ..protocol import ACT_HEADER
from .common import EXIT_OK, EXIT_REFUSED

SUCCEEDED = "done"

Handler = Callable[[Connection, argparse.Namespace], int]


def resolve(args: argparse.Namespace) -> Connection:
    return resolve_connection(
        url=getattr(args, "url", None),
        token=getattr(args, "token", None),
        pairing=getattr(args, "pairing", None),
        pairing_file=getattr(args, "pairing_file", None),
        server=getattr(args, "server", None),
    )


def emit(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_line(value: Any) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _file_sink(destination: Path) -> BinaryIO:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.open("wb")


def download(connection: Connection, path: str, destination: Path | None) -> int:
    if destination is not None:
        return transport.download(connection, path, lambda: _file_sink(destination))
    written = transport.download(
        connection, path, lambda: contextlib.nullcontext(sys.stdout.buffer)
    )
    sys.stdout.buffer.flush()
    return written


def report_http_error(
    exc: urllib.error.HTTPError, connection: Connection | None = None
) -> int:
    raw = exc.read()
    body, error = error_in(raw)
    if body is None:
        text = raw.decode("utf-8", errors="replace").strip()
        print(f"crucible: HTTP {exc.code} from the server, and its body is not "
              f"Crucible's error shape:", file=sys.stderr)
        print(text if text else "(empty)", file=sys.stderr)
        return EXIT_REFUSED
    step = None if error is None else next_step(error, connection)
    if step is not None:
        print(f"crucible: HTTP {exc.code} {step}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"crucible: HTTP {exc.code}", file=sys.stderr)
    json.dump(body, sys.stderr, indent=2)
    sys.stderr.write("\n")
    return EXIT_REFUSED


def _file_named(raw: str, flag: str) -> Path:
    path = Path(raw[1:])
    if not path.is_file():
        raise ClientRefusal(f"{flag}_file_missing: {path} is not a file")
    return path


def read_text_argument(raw: str, flag: str) -> str:
    if raw.startswith("@"):
        return _file_named(raw, flag).read_text(encoding="utf-8")
    return raw


def read_json_argument(raw: str, flag: str) -> Any:
    if raw.startswith("@"):
        path = _file_named(raw, flag)
        text, where = path.read_text(encoding="utf-8"), str(path)
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


def act_header(act: str | None) -> dict[str, str] | None:
    return None if act is None else {ACT_HEADER: act}


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


def _submitted(
    connection: Connection, accepted: dict[str, Any], kind: str, key: str, follow_it: bool
) -> tuple[int, dict[str, Any] | None]:
    if not follow_it:
        emit(accepted)
        return EXIT_OK, None
    emit_line(accepted)
    unit_id = accepted[key]
    return follow_to_the_end(
        connection, f"/v1/{kind}/{unit_id}/events", f"/v1/{kind}/{unit_id}", since=0
    )


def cmd_get(path: str) -> Handler:

    def run(connection: Connection, args: argparse.Namespace) -> int:
        emit(call(connection, "GET", path))
        return EXIT_OK

    return run


def cmd_emit(method: str, template: str) -> Handler:

    def run(connection: Connection, args: argparse.Namespace) -> int:
        emit(call(connection, method, template.format(**vars(args))))
        return EXIT_OK

    return run


cmd_job_get = cmd_emit("GET", "/v1/jobs/{job_id}")
cmd_job_cancel = cmd_emit("DELETE", "/v1/jobs/{job_id}")
cmd_job_hold = cmd_emit("POST", "/v1/jobs/{job_id}/hold")
cmd_resumable_get = cmd_emit("GET", "/v1/resumable/{resume_id}")
cmd_resumable_discard = cmd_emit("DELETE", "/v1/resumable/{resume_id}")
cmd_task_get = cmd_emit("GET", "/v1/tasks/{task_id}")
cmd_task_cancel = cmd_emit("DELETE", "/v1/tasks/{task_id}")
cmd_stream_close = cmd_emit("DELETE", "/v1/tts/stream/{session_id}")
cmd_lease_heartbeat = cmd_emit("POST", "/v1/leases/{lease_id}/heartbeat")
cmd_pairing_requests = cmd_get("/v1/pairing/requests")


def cmd_capability(connection: Connection, args: argparse.Namespace) -> int:
    params: list[tuple[str, str]] = []
    if args.accelerator_probe:
        params.append(("accelerator_probe", "true"))
    for name, value in (
        ("class", args.capability_class),
        ("context_tokens", args.context_tokens),
        ("concurrency", args.concurrency),
    ):
        if value is not None:
            params.append((name, value))
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
    emit(call(
        connection, "PUT", "/v1/settings", json_body=patch, extra_headers=act_header(args.act)
    ))
    return EXIT_OK


def cmd_upstream_test(connection: Connection, args: argparse.Namespace) -> int:
    body = None if args.body is None else read_json_argument(args.body, "--body")
    emit(call(
        connection, "POST", f"/v1/settings/upstreams/{args.name}/test",
        json_body=body,
    ))
    return EXIT_OK


def cmd_pairing_decide(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", "/v1/pairing/decision", json_body={
        "id": args.id, "user_code": args.code, "allow": args.allow,
    }))
    return EXIT_OK


def chat_body(args: argparse.Namespace) -> dict[str, Any]:
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
        return body
    if args.model is None or args.message is None:
        raise ClientRefusal(
            "chat_underspecified: pass --body with the whole request, or "
            "both --model and --message for a one-shot question"
        )
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
    }


CHAT_PATH = "/v1/openai/chat/completions"


def cmd_chat(connection: Connection, args: argparse.Namespace) -> int:
    body = chat_body(args)
    extra = act_header(args.act)
    if not args.stream:
        emit(call(connection, "POST", CHAT_PATH, json_body=body, extra_headers=extra))
        return EXIT_OK
    for frame in transport.chat_frames(connection, CHAT_PATH, dict(body, stream=True), extra):
        emit_line(frame)
    return EXIT_OK


class _Question(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        questions = getattr(namespace, self.dest)
        if questions is None:
            questions = []
            setattr(namespace, self.dest, questions)
        questions.append((option_string, list(values)))


def _choice_question(name: str, values: list[str]) -> dict[str, Any]:
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
    return {"type": "choice", "instructions": values[1], "options": options}


QUESTION_SHAPES: dict[str, Callable[[str, list[str]], dict[str, Any]]] = {
    "--yesno": lambda name, values: {"type": "yesno", "instructions": values[1]},
    "--score": lambda name, values: {
        "type": "score", "instructions": values[1],
        "levels": [level.strip() for level in values[2].split(",")],
    },
    "--choice": _choice_question,
}


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
        questions[name] = QUESTION_SHAPES[flag](name, values)
    return questions


def _images(raw_paths: list[str]) -> list[str]:
    images = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_file():
            raise ClientRefusal(f"image_missing: {path} is not a file")
        data = path.read_bytes()
        if not data:
            raise ClientRefusal(f"image_empty: {path} has no bytes")
        images.append(base64.b64encode(data).decode("ascii"))
    return images


def cmd_decide(connection: Connection, args: argparse.Namespace) -> int:
    images = _images(args.image)
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
    emit(call(connection, "POST", "/v1/decide", json_body=body,
              extra_headers=act_header(args.act)))
    return EXIT_OK


def _window(
    connection: Connection, raw: list[str], inputs: dict[str, dict[str, str]]
) -> dict[str, Any]:
    raw_index, raw_text, raw_audio = raw
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
    inputs[name] = {"blob_id": upload(connection, audio)["blob_id"]}
    return {"index": index, "text": text}


def cmd_align(connection: Connection, args: argparse.Namespace) -> int:
    if not args.window:
        raise ClientRefusal("windows_required: give at least one --window INDEX TEXT AUDIO")
    inputs: dict[str, dict[str, str]] = {}
    chunks = [_window(connection, raw, inputs) for raw in args.window]
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
    code, _ = _submitted(connection, accepted, "jobs", "job_id", args.follow)
    if args.follow and args.out is not None and code == EXIT_OK:
        target = Path(args.out)
        written = download(
            connection, f"/v1/jobs/{accepted['job_id']}/artifacts/alignment.json", target
        )
        emit_line({"alignment": str(target), "bytes": written})
    return code


def cmd_upload(connection: Connection, args: argparse.Namespace) -> int:
    emit(upload(connection, Path(args.path)))
    return EXIT_OK


def job_body(args: argparse.Namespace) -> dict[str, Any]:
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
    return body


def job_inputs(connection: Connection, args: argparse.Namespace) -> dict[str, dict[str, str]]:
    inputs: dict[str, dict[str, str]] = {}
    for raw in args.input:
        name, path = split_assignment(raw, "--input")
        blob = upload(connection, Path(path))
        inputs[name] = {"blob_id": blob["blob_id"]}
    for raw in args.input_blob:
        name, blob_id = split_assignment(raw, "--input-blob")
        inputs[name] = {"blob_id": blob_id}
    return inputs


def save_artifacts(
    connection: Connection, job_id: str, names: list[str], directory: Path
) -> None:
    saved = []
    for name in names:
        target = directory / name
        written = download(connection, f"/v1/jobs/{job_id}/artifacts/{name}", target)
        saved.append({"name": name, "path": str(target), "bytes": written})
    emit_line({"artifacts_saved": saved})


def cmd_job_submit(connection: Connection, args: argparse.Namespace) -> int:
    body = job_body(args)
    inputs = job_inputs(connection, args)
    if inputs:
        body["inputs"] = inputs
    if args.artifacts_dir is not None and not args.follow:
        raise ClientRefusal(
            "artifacts_need_follow: --artifacts-dir waits for the job to finish, "
            "so it must be asked for with --follow. Without it this command "
            "returns as soon as the job is admitted"
        )
    accepted = call(connection, "POST", "/v1/jobs", json_body=body)
    code, state = _submitted(connection, accepted, "jobs", "job_id", args.follow)
    if state is not None and args.artifacts_dir is not None and code == EXIT_OK:
        save_artifacts(
            connection, accepted["job_id"], state["artifacts"], Path(args.artifacts_dir)
        )
    return code


def cmd_follow(kind: str, key: str) -> Handler:

    def run(connection: Connection, args: argparse.Namespace) -> int:
        unit_id = getattr(args, key)
        code, _ = follow_to_the_end(
            connection, f"/v1/{kind}/{unit_id}/events", f"/v1/{kind}/{unit_id}",
            since=args.since,
        )
        return code

    return run


cmd_job_events = cmd_follow("jobs", "job_id")
cmd_task_events = cmd_follow("tasks", "task_id")


def cmd_job_artifact(connection: Connection, args: argparse.Namespace) -> int:
    destination = None if args.out == "-" else Path(args.out)
    written = download(
        connection, f"/v1/jobs/{args.job_id}/artifacts/{args.name}", destination
    )
    if destination is not None:
        emit({"name": args.name, "path": str(destination), "bytes": written})
    return EXIT_OK


TASK_FIELDS = ("kind", "id", "job_type", "narrator_engine", "target")


def cmd_task_submit(connection: Connection, args: argparse.Namespace) -> int:
    body: dict[str, Any] = {"type": args.type}
    for name in TASK_FIELDS:
        value = getattr(args, name)
        if value is not None:
            body[name] = value
    if args.module is not None:
        body["module"] = read_json_argument(args.module, "--module")
    accepted = call(connection, "POST", "/v1/tasks", json_body=body)
    code, _ = _submitted(connection, accepted, "tasks", "task_id", args.follow)
    return code


def cmd_stream_open(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", "/v1/tts/stream", json_body={
        "voice": args.voice, "language": args.language,
    }))
    return EXIT_OK


def _say_text(args: argparse.Namespace) -> str:
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
    return text


def _stream_op(op: str, *with_row: str) -> Handler:

    def run(connection: Connection, args: argparse.Namespace) -> int:
        body: dict[str, Any] = {"op": op}
        if with_row:
            body["id"] = args.row
        emit(call(connection, "POST", f"/v1/tts/stream/{args.session_id}", json_body=body))
        return EXIT_OK

    return run


def cmd_stream_say(connection: Connection, args: argparse.Namespace) -> int:
    text = _say_text(args)
    emit(call(connection, "POST", f"/v1/tts/stream/{args.session_id}", json_body={
        "op": "say", "id": args.row, "text": text, "take": args.take,
    }))
    return EXIT_OK


cmd_stream_cancel = _stream_op("cancel", "row")
cmd_stream_cancel_all = _stream_op("cancel_all")


@dataclass
class _Heard:
    sample_rate: int | None = None
    pcm: dict[str, bytearray] = field(default_factory=dict)
    exit_code: int = EXIT_OK


def _hear(event: dict[str, Any], heard: _Heard, args: argparse.Namespace, keep_audio: bool) -> bool:
    name, data = event.get("event"), event.get("data", {})
    if name == "ready":
        heard.sample_rate = data["sample_rate"]
    elif name == "audio" and keep_audio:
        heard.pcm.setdefault(data["id"], bytearray()).extend(
            base64.b64decode(data["pcm_base64"])
        )
    elif name == "error":
        heard.exit_code = EXIT_REFUSED
        return args.until is not None and data.get("id") == args.until
    elif name == "done":
        return args.until is not None and data["id"] == args.until
    elif name == "closed":
        return True
    return False


def _write_wavs(audio_dir: Path, heard: _Heard) -> None:
    if heard.sample_rate is None:
        raise ClientRefusal(
            "no_ready_frame: the session sent audio without a `ready` event, "
            "so there is no sample rate to write a WAV header with"
        )
    written = []
    audio_dir.mkdir(parents=True, exist_ok=True)
    for row, samples in heard.pcm.items():
        target = audio_dir / f"{row}.wav"
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(heard.sample_rate)
            handle.writeframes(bytes(samples))
        written.append({"id": row, "path": str(target), "bytes": len(samples)})
    emit_line({"audio_saved": written, "sample_rate": heard.sample_rate})


def cmd_stream_events(connection: Connection, args: argparse.Namespace) -> int:
    audio_dir = None if args.audio_dir is None else Path(args.audio_dir)
    if audio_dir is not None and args.since:
        raise ClientRefusal(
            "audio_needs_the_ready_frame: --audio-dir writes the sample rate the "
            f"session reported on its `ready` event, and --since {args.since} "
            "starts after it. Resume without --audio-dir and decode the "
            "pcm_base64 yourself, or follow from the start"
        )
    heard = _Heard()
    for event in follow(connection, f"/v1/tts/stream/{args.session_id}/events",
                        last_event_id=args.since):
        emit_line(event)
        if _hear(event, heard, args, audio_dir is not None):
            break
    if audio_dir is not None and heard.pcm:
        _write_wavs(audio_dir, heard)
    return heard.exit_code


def cmd_lease_open(connection: Connection, args: argparse.Namespace) -> int:
    emit(call(connection, "POST", f"/v1/models/{args.subject_id}/lease", json_body={
        "act": args.act, "ttl_seconds": args.ttl,
    }))
    return EXIT_OK


def cmd_job_release(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/jobs/{args.job_id}/hold")
    emit({"released": args.job_id})
    return EXIT_OK


def cmd_lease_release(connection: Connection, args: argparse.Namespace) -> int:
    call(connection, "DELETE", f"/v1/leases/{args.lease_id}")
    emit({"released": args.lease_id})
    return EXIT_OK


def command(args: argparse.Namespace) -> int:
    try:
        connection = resolve(args)
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return int(args.api_func(connection, args))
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except urllib.error.HTTPError as exc:
        return report_http_error(exc, connection)
    except BrokenPipeError:
        return EXIT_OK
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"crucible: {unreachable(connection, exc)}", file=sys.stderr)
        return EXIT_REFUSED


@dataclass(frozen=True)
class Arg:
    names: tuple[str, ...]
    options: Mapping[str, Any]


def arg(*names: str, **options: Any) -> Arg:
    return Arg(names, options)


@dataclass(frozen=True)
class OneOf:
    args: tuple[Arg, ...]


@dataclass(frozen=True)
class Verb:
    name: str
    help: str
    run: Handler | None = None
    args: tuple[Arg | OneOf, ...] = ()
    verbs: tuple["Verb", ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)


CONNECTION_ARGS = (
    arg(
        "--url",
        default=None,
        help="the server's base URL, e.g. http://192.168.68.20:7100. Requires "
             "--token. Omit both to use this machine's installed engine",
    ),
    arg(
        "--token",
        default=None,
        help="the bearer token for --url. Never taken from the local "
             "installation for a URL typed here",
    ),
    arg(
        "--pairing",
        default=None,
        help="a `crucible://name@host:port/#token` line, as `crucible token "
             "--url` prints it. Carries the address and the token together. It "
             "is visible in any process listing: a runner should use "
             f"--pairing-file or ${PAIRING_ENV}",
    ),
    arg(
        "--pairing-file",
        default=None,
        metavar="PATH",
        help="read the pairing line from this file (exactly one line), so the "
             "token never appears in argv",
    ),
    arg(
        "--server",
        default=None,
        metavar="NAME",
        help="a server this computer paired with by `crucible pair <address>`, "
             "by its name (e.g. crucible@kylies-pc or crucible-kylies-pc)",
    ),
)

READERS = (
    ("ping", "/v1/ping", "is this a Crucible, and which one (needs no token)"),
    ("health", "/v1/health", "the lane, the queue and every job type's readiness"),
    ("info", "/v1/info", "identity, host, job types and every capability row"),
    ("setup", "/v1/setup", "the pairing lines and job types an app's connect door takes"),
    ("accelerator", "/v1/accelerator", "what is on the card right now, probed"),
    ("activity", "/v1/activity", "what this server is doing and who asked"),
    ("models", "/v1/models",
     "the llm models this SERVER holds (`crucible models list` reads this BUILD)"),
    ("voices", "/v1/voices", "the tts voices this SERVER holds"),
    ("catalog", "/v1/catalog", "every subject this backend can hold, installed or not"),
    ("openai-models", "/v1/openai/models",
     "the resident model plus routed upstreams, OpenAI-shaped"),
)

SINCE = arg("--since", type=int, default=0, metavar="EVENT_ID")
BENCH_ACT = arg("--act", default=None, help="the capability class, for the bench")

JOB_VERBS = (
    Verb("submit", "POST /v1/jobs", cmd_job_submit, (
        arg(
            "--type", required=True,
            help="tts, asr, align, align-longform, rvc, denoise, image, audio, segment, echo, "
                 "load-model, unload-model, load-voice, unload-voice, unload-aligner, "
                 "unload-denoiser, load-image, unload-image, load-audio, unload-audio, "
                 "load-segment, unload-segment — "
                 "`crucible api info` says which this server offers",
        ),
        arg(
            "--model", default=None,
            help="the model, voice, aligner, image, audio or segment model id this type serves",
        ),
        arg("--params", default=None, help="the type's params object, as JSON or @file"),
        arg(
            "--input", action="append", default=[], metavar="NAME=PATH",
            help="upload PATH and give the job that blob as input NAME; repeatable",
        ),
        arg(
            "--input-blob", action="append", default=[], metavar="NAME=BLOB_ID",
            help="give the job an already-uploaded blob as input NAME; repeatable",
        ),
        arg(
            "--resume", default=None, metavar="RESUME_ID",
            help="continue the journal a job answered with (params.resume); without "
                 "it the job starts fresh. `crucible api resumable list` shows them",
        ),
        arg("--follow", action="store_true", help="watch the event stream until the job ends"),
        arg(
            "--artifacts-dir", default=None,
            help="save every artifact here once the job is done. Requires --follow",
        ),
    )),
    Verb("get", "GET /v1/jobs/{id}", cmd_job_get, (arg("job_id"),)),
    Verb("events", "follow GET /v1/jobs/{id}/events", cmd_job_events, (
        arg("job_id"),
        arg(
            "--since", type=int, default=0, metavar="EVENT_ID",
            help="resume after this event id (sent as Last-Event-ID)",
        ),
    )),
    Verb("cancel", "DELETE /v1/jobs/{id}", cmd_job_cancel, (arg("job_id"),)),
    Verb(
        "hold", "POST /v1/jobs/{id}/hold — keep its artifacts for a later job's inputs",
        cmd_job_hold, (arg("job_id"),),
    ),
    Verb(
        "release", "DELETE /v1/jobs/{id}/hold — the chain is done; remove the job now",
        cmd_job_release, (arg("job_id"),),
    ),
    Verb("artifact", "GET one artifact's bytes", cmd_job_artifact, (
        arg("job_id"),
        arg("name"),
        arg("--out", default="-", help="where to write it; `-` (the default) means stdout"),
    )),
)

RESUMABLE_VERBS = (
    Verb("list", "GET /v1/resumable — every journal, newest first", cmd_get("/v1/resumable")),
    Verb("get", "GET /v1/resumable/{id}", cmd_resumable_get, (arg("resume_id"),)),
    Verb(
        "discard", "DELETE /v1/resumable/{id} — refused while a job writes it",
        cmd_resumable_discard, (arg("resume_id"),),
    ),
)

TASK_VERBS = (
    Verb("submit", "POST /v1/tasks", cmd_task_submit, (
        arg(
            "--type", required=True,
            choices=["pull", "install", "module", "engine", "engine-restart"],
        ),
        arg("--kind", default=None, help="pull: the subject kind"),
        arg("--id", default=None, help="pull: the subject id"),
        arg("--job-type", default=None, help="install: the job type to install"),
        arg("--narrator-engine", default=None, help="install: which tts engine"),
        arg("--module", default=None, help="module: the module document, as JSON or @file"),
        arg("--target", default=None, help="engine: what to move it to"),
        arg("--follow", action="store_true", help="watch until the task ends"),
    )),
    Verb("list", "GET /v1/tasks — the last few, newest first", cmd_get("/v1/tasks")),
    Verb("get", "GET /v1/tasks/{id}", cmd_task_get, (arg("task_id"),)),
    Verb("events", "follow GET /v1/tasks/{id}/events", cmd_task_events, (arg("task_id"), SINCE)),
    Verb("cancel", "DELETE /v1/tasks/{id}", cmd_task_cancel, (arg("task_id"),)),
)

STREAM_VERBS = (
    Verb("open", "POST /v1/tts/stream", cmd_stream_open, (
        arg("--voice", required=True),
        arg("--language", required=True),
    )),
    Verb("say", "one row into an open session", cmd_stream_say, (
        arg("session_id"),
        arg("--row", required=True, help="the client's id for this row"),
        arg("--text", default=None),
        arg("--text-file", default=None, help="read the text from a file instead"),
        arg(
            "--take", type=int, required=True,
            help="which rung of the voice's ladder. No default: the wire has none",
        ),
    )),
    Verb("events", "follow the session's SSE stream", cmd_stream_events, (
        arg("session_id"),
        SINCE,
        arg(
            "--until", default=None, metavar="ROW",
            help="stop once this row is done or errors, instead of waiting for close",
        ),
        arg(
            "--audio-dir", default=None,
            help="write each row's audio here as <row>.wav, at the rate the session "
                 "reported on its ready event",
        ),
    )),
    Verb("cancel", "cancel one row in flight", cmd_stream_cancel, (
        arg("session_id"),
        arg("--row", required=True),
    )),
    Verb("cancel-all", "cancel every row in flight", cmd_stream_cancel_all, (arg("session_id"),)),
    Verb("close", "DELETE /v1/tts/stream/{id}", cmd_stream_close, (arg("session_id"),)),
)

LEASE_VERBS = (
    Verb("open", "POST /v1/models/{id}/lease", cmd_lease_open, (
        arg("subject_id", help="the resident model, voice or aligner"),
        arg("--act", required=True, help="the capability class this run is"),
        arg("--ttl", type=int, required=True, metavar="SECONDS"),
    )),
    Verb("heartbeat", "push the deadline out", cmd_lease_heartbeat, (arg("lease_id"),)),
    Verb("release", "give the card back", cmd_lease_release, (arg("lease_id"),)),
)

DECIDE_ARGS = (
    arg("--model", required=True, help="the resident model to read the decision from"),
    arg(
        "--state", default=None,
        help="what the questions are about: text, or @file. May be left out "
             "when at least one --image is given; the state is then \"\"",
    ),
    arg(
        "--image", action="append", default=[], metavar="PATH",
        help="an image file, sent base64-encoded; repeatable",
    ),
    arg(
        "--choice", dest="questions", action=_Question, nargs="+",
        metavar=("NAME", "INSTRUCTIONS OPT=DESC"),
        help='NAME "instructions" opt=description [opt=description …]; repeatable',
    ),
    arg(
        "--score", dest="questions", action=_Question, nargs=3,
        metavar=("NAME", "INSTRUCTIONS", "LEVELS"),
        help='NAME "instructions" "level1,level2,…" (ordered, low to high); repeatable',
    ),
    arg(
        "--yesno", dest="questions", action=_Question, nargs=2,
        metavar=("NAME", "INSTRUCTIONS"),
        help='NAME "a statement the state may make true"; repeatable',
    ),
    arg(
        "--missing", default=None, metavar="refuse|report",
        help="a label outside the engine's top-K: refuse the decision "
             "(label_not_in_probs, the server's default) or report it — null "
             "probability, named in missing_labels, the rest renormalised over "
             "the letters returned",
    ),
    BENCH_ACT,
)

API_VERBS = (
    *(Verb(name, help_text, cmd_get(path)) for name, path, help_text in READERS),
    Verb("capability", "what this host can hold, and why", cmd_capability, (
        arg(
            "--accelerator-probe", action="store_true",
            help="pay for a live nvidia-smi probe rather than reading the record",
        ),
        arg(
            "--class", dest="capability_class", default=None,
            help="the client-sized class to size (generate)",
        ),
        arg(
            "--context-tokens", default=None,
            help="tokens per request for --class; above this host's ceiling is refused",
        ),
        arg("--concurrency", default=None, help="requests in flight at once for --class"),
    )),
    Verb("catalog-remove", "delete an installed subject's files", cmd_catalog_remove, (
        arg("kind", help="model, voice, rvc, denoise, asr, align"),
        arg("subject_id"),
    )),
    Verb("voice-write", "add or replace a voice this server owns", cmd_voice_write, (
        arg("voice_id"),
        arg(
            "--manifest", required=True,
            help="the whole manifest document, as JSON or @file — the table a "
                 "voices/<id>.toml holds. An overlay wins over a packaged voice of "
                 "the same id, and `api voice-remove` is the undo",
        ),
    )),
    Verb(
        "voice-remove", "delete this server's own manifest for a voice", cmd_voice_remove,
        (arg("voice_id"),),
    ),
    Verb("settings", "read the settings document, or PUT a patch", cmd_settings, (
        arg(
            "--patch", default=None,
            help="a partial settings document, as JSON or @file. Applied whole or "
                 "not at all; the answer is the document after the write",
        ),
        arg(
            "--act", default=None,
            help="the capability class this write is for, recorded on the bench",
        ),
    )),
    Verb("upstream-test", "ask a configured upstream what it serves", cmd_upstream_test, (
        arg("name"),
        arg(
            "--body", default=None,
            help='{"key": "…"} or {"url": "…"} to test before saving, as JSON or '
                 "@file. Omit to use the stored record",
        ),
    )),
    Verb("pairing-requests", "connect requests waiting for a decision", cmd_pairing_requests),
    Verb("pairing-decide", "approve or deny one connect request", cmd_pairing_decide, (
        arg("--id", required=True),
        arg("--code", required=True, help="the displayed XXXX-XXXX code"),
        OneOf((
            arg("--allow", dest="allow", action="store_true"),
            arg("--deny", dest="allow", action="store_false"),
        )),
    )),
    Verb("chat", "a chat completion — the cleanup and translation door", cmd_chat, (
        arg(
            "--body", default=None,
            help="the whole OpenAI-shaped request, as JSON or @file. Forwarded "
                 "verbatim except `model`",
        ),
        arg("--model", default=None, help="shorthand: the model for a one-shot question"),
        arg("--message", default=None, help="shorthand: the user message"),
        arg("--stream", action="store_true", help="send stream:true and print each frame"),
        BENCH_ACT,
    )),
    Verb(
        "decide", "a distribution over each question's fixed answers — snap's grammar",
        cmd_decide, DECIDE_ARGS, defaults={"questions": None},
    ),
    Verb(
        "align", "place known words in time: windows of audio + their text, one job",
        cmd_align, (
            arg("--model", required=True, help="the aligner, e.g. qwen3-aligner"),
            arg(
                "--language", required=True,
                help="ISO code, one of the aligner's eleven: en de fr es it pt ru ja ko zh yue",
            ),
            arg(
                "--window", action="append", nargs=3, default=[],
                metavar=("INDEX", "TEXT", "AUDIO"),
                help="one window: its index, the words spoken in it (text or @file), and "
                     "its audio file (at most 300 s); repeatable, all in one job",
            ),
            arg("--follow", action="store_true", help="watch the job until it ends"),
            arg(
                "--out", default=None,
                help="save alignment.json here once the job is done. Requires --follow",
            ),
        ),
    ),
    Verb("upload", "put a file on the server and get its blob_id", cmd_upload, (arg("path"),)),
    Verb("job", "submit, watch, cancel and read back one unit of work", verbs=JOB_VERBS),
    Verb(
        "resumable", "the resume journals: what can be resumed, and discarding one",
        verbs=RESUMABLE_VERBS,
    ),
    Verb("task", "work done TO the server: pulls, installs, engine restarts", verbs=TASK_VERBS),
    Verb("stream", "serialized tts: one session, one row at a time", verbs=STREAM_VERBS),
    Verb("lease", "say you are mid-run on what is resident", verbs=LEASE_VERBS),
)

API_DESCRIPTION = (
    "The CLIENT half of the command line. Every verb here makes an HTTP "
    "request to a server that may be this machine's, the one in WSL, or "
    "one across the network; the operator verbs (init, install, service, "
    "models, …) act on this machine's installation instead and take no "
    "address. Output is JSON on stdout, one document per command, or one "
    "compact object per line while following a stream. A refusal is the "
    "server's own {\"error\": {...}} on stderr with exit 1."
)


def add_args(parser: Any, args: tuple[Arg | OneOf, ...]) -> None:
    for item in args:
        if isinstance(item, OneOf):
            add_args(parser.add_mutually_exclusive_group(required=True), item.args)
        else:
            parser.add_argument(*item.names, **item.options)


def add_verbs(subparsers: Any, verbs: tuple[Verb, ...]) -> None:
    for verb in verbs:
        parser = subparsers.add_parser(verb.name, help=verb.help)
        add_args(parser, verb.args)
        if verb.verbs:
            nested = parser.add_subparsers(dest=f"{verb.name}_command", required=True)
            add_verbs(nested, verb.verbs)
        else:
            parser.set_defaults(api_func=verb.run, **verb.defaults)


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "api",
        help="talk to a Crucible server over HTTP: submit jobs, stream tts, read state",
        description=API_DESCRIPTION,
    )
    add_args(parser.add_argument_group("server"), CONNECTION_ARGS)
    add_verbs(parser.add_subparsers(dest="api_command", required=True), API_VERBS)
    parser.set_defaults(func=command)

