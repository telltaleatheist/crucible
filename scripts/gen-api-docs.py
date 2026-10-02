#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "docs" / "API.md"

GROUPS: tuple[tuple[str, str, str], ...] = (
    (
        "/ping",
        "Discovery",
        "Answered without a token. How a client finds a server and learns its api version.",
    ),
    (
        "/pair",
        "Pairing",
        "The token exchange. Start and poll need only the version header; approval is "
        "authenticated, because approving is the act that grants access.",
    ),
    (
        "/setup",
        "Setup",
        "The operator page's own door: it hands out a token and the pairing line.",
    ),
    (
        "/info",
        "What this server is",
        "Identity, backend, and every model this build ships with — including the ones "
        "this card cannot hold, and why.",
    ),
    (
        "/accelerator",
        "The card",
        "What the accelerator is and what is free on it right now.",
    ),
    (
        "/capability",
        "What fits",
        "Crucible's own verdict about this host: which classes are enabled, which model "
        "each runs, and the arithmetic behind a refusal.",
    ),
    (
        "/settings",
        "Settings",
        "The one door apps configure Crucible through. Crucible is set-and-forget; "
        "everything an app wants changed is written here.",
    ),
    (
        "/models",
        "Models",
        "What is installed, what is resident, and what a pull would cost.",
    ),
    (
        "/jobs",
        "Jobs",
        "The work. Every job type is created, polled and cancelled through the same routes. "
        "The `image` job's params, its result and how to prompt it are in docs/IMAGE.md; "
        "the `audio` job's (sound effects, music and songs) are in docs/AUDIO.md; "
        "the `segment` job's (subject cutouts and point-and-box selections, as masks) "
        "are in docs/SEGMENT.md; the `video` job's (clips with sound from words or a "
        "start picture) are in docs/VIDEO.md.",
    ),
    (
        "/queue",
        "Queue",
        "Jobs, calls and queue sessions that find the server busy wait here (waiting is "
        "the default; `\"queue\": false` refuses instead), in order: list them, "
        "remove one, keep one alive, or follow every change. How an app should use it is "
        "docs/QUEUE.md.",
    ),
    (
        "/resumable",
        "Resumable jobs",
        "The resume journals: every job type that keeps one writes its finished work to "
        "disk as it lands, and a job sent `params.resume` continues it "
        "(docs/RESUMABLE-JOBS.md).",
    ),
    (
        "/tasks",
        "Tasks",
        "Long host-side work — installs, pulls, env packs — that is not a job because no "
        "model runs.",
    ),
    ("/activity", "Activity", "What the server is doing right now, in one read."),
    (
        "/events",
        "Events",
        "Every change on the server as one SSE stream, so an app follows it instead of "
        "polling /v1/activity, /v1/tasks and /v1/health. The event names and payloads, "
        "resuming, and what a slow reader is told are in docs/EVENTS.md.",
    ),
    ("/health", "Health", "Is this process alive. Cheaper than /v1/activity and says less."),
    (
        "/playground",
        "Playground",
        "The pages the operator page's playground draws: one per image, video and audio "
        "model, with the params its form shows and whether this server can run it now.",
    ),
    (
        "/catalog",
        "Catalog",
        "Everything this build can serve, of every kind, and what is on disk. Removing "
        "a subject here is how weights are reclaimed.",
    ),
    (
        "/voices",
        "Voices",
        "The narration voices this build ships, and what each one costs.",
    ),
    (
        "/tts/stream",
        "Streaming narration",
        "A long-lived session that takes text and gives audio back over SSE, instead of "
        "one render per request.",
    ),
    (
        "/uploads",
        "Uploads",
        "Bytes too big for a request body. An upload answers with a `blob_id` a job "
        "input then names.",
    ),
    (
        "/queue/sessions",
        "Queue sessions",
        "One client holding the server for a run of requests it cannot know in advance: "
        "it waits in the line, opens, runs its items back to back with nothing from anyone "
        "else in between, and closes. Not a TTS stream session. docs/QUEUE.md says how an "
        "app uses one.",
    ),
    (
        "/openai",
        "OpenAI-compatible",
        "A chat surface shaped like OpenAI's, for clients that already speak it.",
    ),
    (
        "/peer",
        "Peers",
        "Orchestrator and engine talking to each other (docs/internals/host-and-platform.md). Not an "
        "app-facing surface.",
    ),
    ("", "Everything else", ""),
)

METHOD_ORDER = {"get": 0, "post": 1, "patch": 2, "put": 3, "delete": 4}

AUTH_HEADERS = {"authorization", "x-crucible-api"}


def build_app() -> Any:
    home = Path(tempfile.mkdtemp(prefix="crucible-apidoc-"))
    os.environ["CRUCIBLE_HOME"] = str(home)
    from crucible.api import create_app
    from crucible.backend import Backend, Gpu
    from crucible.config import load_config, write_config

    backend = Backend(
        kind="cuda-linux",
        platform="linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="documentation", vram_bytes=25_757_220_864),
        detail="generated docs",
    )
    write_config(
        home,
        name="crucible@docs",
        host="127.0.0.1",
        port=7100,
        token="documentation-only",
        backend_kind=backend.kind,
        enable_echo=True,
        enable_llm=True,
        enable_asr=True,
        enable_tts=True,
        enable_align=True,
        enable_rvc=True,
        enable_denoise=True,
        enable_image=True,
        enable_audio=True,
        enable_segment=True,
        enable_video=True,
        desktop_allowance_bytes=3 * 1024 ** 3,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    return create_app(load_config(home), backend)


def auth_scopes(app: Any) -> dict[tuple[str, str], tuple[bool, bool]]:
    from crucible.api import require_api_version, require_auth

    scopes: dict[tuple[str, str], tuple[bool, bool]] = {}

    def walk(routes: Any, prefix: str, inherited: tuple[Any, ...]) -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                context = route.include_context
                walk(
                    included.routes,
                    prefix + (context.prefix or ""),
                    inherited + tuple(context.dependencies or ()),
                )
                continue
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            calls = {dependant.call}
            stack = list(dependant.dependencies) + [
                one.dependency for one in inherited if one.dependency is not None
            ]
            while stack:
                found = stack.pop()
                calls.add(getattr(found, "call", found))
                stack.extend(getattr(found, "dependencies", ()))
            spelling = getattr(route, "path_format", route.path)
            for method in getattr(route, "methods", ()) or ():
                scopes[(prefix + spelling, method.upper())] = (
                    require_auth in calls,
                    require_api_version in calls,
                )

    walk(app.routes, "", ())
    return scopes


def deref(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    seen = 0
    while "$ref" in schema and seen < 10:
        name = schema["$ref"].rsplit("/", 1)[-1]
        schema = components.get(name, {})
        seen += 1
    return schema


def type_words(schema: dict[str, Any], components: dict[str, Any]) -> str:
    schema = deref(schema, components)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            parts = [type_words(one, components) for one in schema[key]]
            kept = [part for part in parts if part != "null"]
            suffix = " or null" if "null" in parts else ""
            return " or ".join(dict.fromkeys(kept)) + suffix
    if "const" in schema:
        return "`" + repr(schema["const"]) + "`"
    if "enum" in schema:
        return " or ".join("`" + repr(value) + "`" for value in schema["enum"])
    kind = schema.get("type")
    if kind == "array":
        return "array of " + type_words(schema.get("items", {}), components)
    if kind == "object" and schema.get("title") and "properties" in schema:
        return str(schema["title"])
    if kind == "object":
        values = schema.get("additionalProperties")
        if isinstance(values, dict) and values:
            return "object of " + type_words(values, components)
        return "object"
    if kind is None:
        return str(schema.get("title") or "any")
    return str(kind)


def cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def render_fields(
    schema: dict[str, Any], components: dict[str, Any]
) -> list[str]:
    schema = deref(schema, components)
    properties: dict[str, Any] = schema.get("properties", {})
    if not properties:
        return []
    required = set(schema.get("required", ()))
    lines = [
        "| field | type | required | default | what it is |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, prop in properties.items():
        resolved = deref(prop, components)
        if "default" in prop:
            default = prop["default"]
        elif "default" in resolved:
            default = resolved["default"]
        else:
            default = _NO_DEFAULT
        if default is _NO_DEFAULT:
            shown = "—"
        elif default is None:
            shown = "`null`"
        else:
            shown = "`" + repr(default) + "`"
        note = cell(prop.get("description") or resolved.get("description"))
        lines.append(
            "| `"
            + name
            + "` | "
            + type_words(prop, components)
            + " | "
            + ("yes" if name in required else "no")
            + " | "
            + shown
            + " | "
            + note
            + " |"
        )
    return lines


_NO_DEFAULT = object()


def answer_schema(response: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for media, entry in response.get("content", {}).items():
        if entry.get("schema"):
            return media, entry["schema"]
    return None


def model_name(schema: dict[str, Any]) -> str | None:
    ref = schema.get("$ref")
    return None if ref is None else ref.rsplit("/", 1)[-1]


def answer_label(code: str, response: dict[str, Any]) -> str:
    found = answer_schema(response)
    label = "`" + code + "`"
    if found is None:
        return label
    schema = found[1]
    name = model_name(schema) or model_name(schema.get("items", {}))
    if name is None:
        return label
    return label + " " + ("array of " if schema.get("type") == "array" else "") + name


def answer_fields(
    code: str, response: dict[str, Any], components: dict[str, Any]
) -> list[str]:
    found = answer_schema(response)
    if found is None:
        return []
    media, schema = found
    each = schema.get("type") == "array"
    fields = render_fields(schema.get("items", {}) if each else schema, components)
    if not fields:
        return []
    shape = "an array; each item" if each else "the body"
    return ["**Answer `" + code + "`** (`" + media + "`), " + shape + ":", "", *fields, ""]


def render_answers(responses: dict[str, Any], components: dict[str, Any]) -> list[str]:
    codes = sorted(responses)
    if not codes:
        return []
    out = [
        "*Answers:* " + ", ".join(answer_label(code, responses[code]) for code in codes),
        "",
    ]
    for code in codes:
        if code.startswith("2"):
            out += answer_fields(code, responses[code], components)
    return out


def group_for(path: str) -> int:
    stripped = path[len("/v1") :] if path.startswith("/v1") else path
    best = len(GROUPS) - 1
    for index, (prefix, _, _) in enumerate(GROUPS):
        if not prefix:
            continue
        if stripped.startswith(prefix) and (
            not GROUPS[best][0] or len(prefix) > len(GROUPS[best][0])
        ):
            best = index
    return best


def render_features() -> list[str]:
    from crucible.features import FEATURES

    out = [
        "## Features",
        "",
        "`GET /v1/info` answers `features`, the names below, so an app checks for what it "
        "needs instead of comparing versions. A name says the routes and fields exist in "
        "this build; whether a job type is enabled on this host is `job_types`. Defined "
        "in `crucible/features.py`.",
        "",
        "| feature | what it is |",
        "| --- | --- |",
    ]
    for name in sorted(FEATURES):
        out.append("| `" + name + "` | " + cell(FEATURES[name]) + " |")
    return out + [""]


def render(app: Any) -> str:
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    scopes = auth_scopes(app)

    out: list[str] = [
        "# The Crucible API",
        "",
        "**GENERATED — do not edit.** `python scripts/gen-api-docs.py` writes this file",
        "from the FastAPI app the server actually runs, and `scripts/release.sh` refuses a",
        "cut when it is stale. Change a request or answer model and regenerate; never edit here.",
        "",
        "Every route is under `/v1` unless it says otherwise. Protected routes need",
        "`Authorization: Bearer <token>` **and** `X-Crucible-Api: 1`, checked in that order.",
        "An error is always a JSON body under an `error` key holding `code`, `message` and",
        "sometimes `details`. Crucible refuses by name and with numbers: branch on `code`,",
        "show a person the `message`.",
        "",
        "The prose for WHY a field exists lives in the internals docs (`docs/internals/*.md`); what",
        "is here is what you may send and what comes back.",
        "",
    ]

    by_group: dict[int, list[tuple[str, str, dict[str, Any]]]] = {}
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() not in METHOD_ORDER:
                continue
            by_group.setdefault(group_for(path), []).append(
                (path, method.upper(), operation)
            )

    for index, (_, title, blurb) in enumerate(GROUPS):
        rows = by_group.get(index)
        if not rows:
            continue
        rows.sort(key=lambda row: (row[0], METHOD_ORDER[row[1].lower()]))
        out += ["## " + title, ""]
        if blurb:
            out += [blurb, ""]
        for path, method, operation in rows:
            if (path, method) not in scopes:
                raise SystemExit(
                    f"{method} {path} is in the OpenAPI document but auth_scopes "
                    "never reached it, so its door is unknown. The route walk "
                    "does not match how this FastAPI nests routers"
                )
            needs_token, needs_version = scopes[(path, method)]
            if needs_token:
                door = "token + `X-Crucible-Api: 1`"
            elif needs_version:
                door = "`X-Crucible-Api: 1` only"
            else:
                door = "open"
            out += ["### `" + method + " " + path + "`", ""]
            summary = cell(operation.get("summary"))
            description = cell(operation.get("description"))
            if description:
                out += [description, ""]
            elif summary:
                out += [summary, ""]
            out += ["*Door:* " + door, ""]

            wanted = [
                param
                for param in operation.get("parameters", [])
                if str(param.get("name", "")).lower() not in AUTH_HEADERS
            ]
            if wanted:
                out += [
                    "| parameter | in | required | type | what it is |",
                    "| --- | --- | --- | --- | --- |",
                ]
                for param in wanted:
                    out.append(
                        "| `"
                        + str(param.get("name"))
                        + "` | "
                        + str(param.get("in"))
                        + " | "
                        + ("yes" if param.get("required") else "no")
                        + " | "
                        + type_words(param.get("schema", {}), components)
                        + " | "
                        + cell(param.get("description"))
                        + " |"
                    )
                out.append("")

            body = operation.get("requestBody", {}).get("content", {})
            for media, entry in body.items():
                fields = render_fields(entry.get("schema", {}), components)
                if fields:
                    out += ["**Body** (`" + media + "`)", "", *fields, ""]
                else:
                    out += ["**Body**: `" + media + "`", ""]

            out += render_answers(operation.get("responses", {}), components)

    out += render_features()

    out += [
        "## Models in full",
        "",
        "Every request and answer schema the routes above refer to, for a reader following "
        "a nested field.",
        "",
    ]
    for name in sorted(components):
        fields = render_fields(components[name], components)
        if not fields:
            continue
        out += ["### `" + name + "`", ""]
        note = cell(components[name].get("description"))
        if note:
            out += [note, ""]
        out += [*fields, ""]

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="generate docs/API.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="refuse if docs/API.md is not what this run would write",
    )
    args = parser.parse_args()
    text = render(build_app())
    if not args.check:
        with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print("wrote " + str(OUTPUT.relative_to(ROOT)) + f" ({len(text.splitlines())} lines)")
        return 0
    if not OUTPUT.is_file():
        print(str(OUTPUT.relative_to(ROOT)) + " does not exist; run scripts/gen-api-docs.py")
        return 1
    current = OUTPUT.read_text(encoding="utf-8")
    if current == text:
        print(str(OUTPUT.relative_to(ROOT)) + " is current")
        return 0
    print(str(OUTPUT.relative_to(ROOT)) + " is STALE. Run scripts/gen-api-docs.py. Diff:")
    diff = difflib.unified_diff(
        current.splitlines(), text.splitlines(), "on disk", "generated", lineterm=""
    )
    for line in list(diff)[:60]:
        print(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
