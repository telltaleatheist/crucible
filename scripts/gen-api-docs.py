#!/usr/bin/env python3
"""Generate `docs/API.md` from the running app's own OpenAPI schema.

Owen, 2026-09-16: *"we should probably be writing documentation for how to use
crucible, with all the variables you can pass, as we go. api documentation or
something like it"*.

GENERATED, and that is the whole point. Seventeen phase docs describe the routes
each phase ADDED, which means there is no document anyone can read to answer
"what can I send to a chat request" — and a hand-written one would be wrong
within a release. This reads the same FastAPI app the server runs, so a field
added to a request model appears here on the next run, and a field removed
disappears. `scripts/release.sh` runs `--check` before the cut, the same way it
does for `modules/*.module.json` — the manifests that shipped stale in v0.6.3
because nothing compared them.

What it CANNOT know is why a field exists. Prose lives in the phase docs and in
the models' own docstrings, which are carried across into the descriptions here.
A field with no description is a field whose model says nothing about it.
"""

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

#: Route groups, in the order a reader meets them: find the server, pair with it,
#: ask what it can do, then make it work. The key is matched as a path prefix
#: after `/v1`, longest first; the empty prefix catches whatever is left.
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
        "The work. Every job type is created, polled and cancelled through the same routes.",
    ),
    (
        "/tasks",
        "Tasks",
        "Long host-side work — installs, pulls, env packs — that is not a job because no "
        "model runs.",
    ),
    ("/activity", "Activity", "What the server is doing right now, in one read."),
    ("/health", "Health", "Is this process alive. Cheaper than /v1/activity and says less."),
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
        "one render per request (PHASE3-TTS.md section 7).",
    ),
    (
        "/uploads",
        "Uploads",
        "Bytes too big for a request body. An upload answers with a `blob_id` a job "
        "input then names.",
    ),
    (
        "/leases",
        "Leases",
        "A client saying it intends a run, so the card is not taken out from under it "
        "mid-chapter.",
    ),
    (
        "/openai",
        "OpenAI-compatible",
        "A chat surface shaped like OpenAI's, for clients that already speak it.",
    ),
    (
        "/peer",
        "Peers",
        "Orchestrator and engine talking to each other (PHASE17-ORCHESTRATOR.md). Not an "
        "app-facing surface.",
    ),
    ("", "Everything else", ""),
)

METHOD_ORDER = {"get": 0, "post": 1, "patch": 2, "put": 3, "delete": 4}

AUTH_HEADERS = {"authorization", "x-crucible-api"}


def build_app() -> Any:
    """The same app the server runs, with every job type switched on.

    Every job type, because a route registered only when its class is enabled
    would otherwise be missing from the reference on the say-so of a fixture's
    defaults.
    """
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
        desktop_allowance_bytes=3 * 1024 ** 3,
    )
    return create_app(load_config(home), backend)


def auth_scopes(app: Any) -> dict[tuple[str, str], tuple[bool, bool]]:
    """(path, METHOD) -> (needs a token, needs the version header).

    Read off each route's resolved dependency tree rather than off the router it
    was registered on, because the routers are built inside `create_app` and a
    route moved between them must not go on claiming the old scope here.

    THE WALK DESCENDS, and it did not. A flat pass over `app.routes` finds ONE
    route — `GET /`, the operator page — so every other door in this reference
    was written "open", which is how `docs/API.md` shipped in 1.0.1 and 1.0.2
    saying that `/v1/info` and `/v1/capability` need no token. Measured here on
    fastapi 0.141.1 / starlette 1.6.0: `include_router` no longer splices the
    routes in, it appends a `_IncludedRouter` whose real ones hang off
    `original_router`, with the prefix and the router-level dependencies in a
    separate `include_context`. `tests/test_api_client.py` met the same wrapper
    from the other side on 2026-09-16 and answered it by asking `app.openapi()`;
    that document says nothing about auth, so this asks the tree and follows it
    down.
    """
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
            # The router-level `dependencies=[…]` a route was included under are
            # folded into its own dependant by FastAPI, but they are carried down
            # here as well rather than trusted to be. A door is stated in two
            # places and reading only one of them is what this function was
            # already doing wrong.
            stack = list(dependant.dependencies) + [
                one.dependency for one in inherited if one.dependency is not None
            ]
            while stack:
                found = stack.pop()
                calls.add(getattr(found, "call", found))
                stack.extend(getattr(found, "dependencies", ()))
            # `path_format`, not `path`: one route is registered with a
            # converter — `/v1/models/{subject_id:path}/lease`, so that a model
            # id with a slash in it survives — and the OpenAPI document this is
            # joined to keys on the bare `{subject_id}`. `path_format` is the
            # spelling FastAPI's own generator uses, which is what makes the two
            # halves the same list.
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
    """One readable type for a property, refs resolved and unions flattened."""
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
        return " | ".join("`" + repr(value) + "`" for value in schema["enum"])
    kind = schema.get("type")
    if kind == "array":
        return "array of " + type_words(schema.get("items", {}), components)
    if kind == "object" and schema.get("title"):
        return str(schema["title"])
    if kind is None:
        return str(schema.get("title") or "any")
    return str(kind)


def cell(text: str | None) -> str:
    """One table cell: whitespace collapsed, pipes escaped."""
    return " ".join((text or "").split()).replace("|", "\\|")


def render_fields(
    schema: dict[str, Any], components: dict[str, Any]
) -> list[str]:
    """One table of every variable a caller may pass into this body."""
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


def render(app: Any) -> str:
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    scopes = auth_scopes(app)

    out: list[str] = [
        "# The Crucible API",
        "",
        "**GENERATED — do not edit.** `python scripts/gen-api-docs.py` writes this file",
        "from the FastAPI app the server actually runs, and `scripts/release.sh` refuses a",
        "cut when it is stale. Change a request model and regenerate; never edit here.",
        "",
        "Every route is under `/v1` unless it says otherwise. Protected routes need",
        "`Authorization: Bearer <token>` **and** `X-Crucible-Api: 1`, checked in that order.",
        "An error is always a JSON body under an `error` key holding `code`, `message` and",
        "sometimes `details`. Crucible refuses by name and with numbers: branch on `code`,",
        "show a person the `message`.",
        "",
        "The prose for WHY a field exists lives in the phase docs (`docs/PHASE*.md`); what",
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
            # NOT `.get(…, (False, False))`. A path the walk did not reach is a
            # walk that is wrong, and defaulting it prints "open" — the one
            # answer that reads like somebody decided it. That default is what
            # kept the flat walk above invisible for two releases.
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

            codes = sorted(operation.get("responses", {}))
            if codes:
                out += ["*Answers:* " + ", ".join("`" + code + "`" for code in codes), ""]

    out += [
        "## Request models in full",
        "",
        "Every schema the routes above refer to, for a reader following a nested field.",
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
