"""The API reference, rendered from the FastAPI app the server actually runs.

One renderer, two readers: `scripts/gen-api-docs.py` writes it to docs/API.md (and
release.sh refuses a cut when that is stale), and the running server serves it at
`GET /docs` (a page) and `GET /v1/docs.md`, so a client author reads the server they
are talking to instead of asking. The job types' section comes from crucible/jobdocs.py.
"""
from __future__ import annotations

from typing import Any

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


def auth_scopes(app: Any) -> dict[tuple[str, str], tuple[bool, bool]]:
    from .api.deps import require_api_version, require_auth

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
    from .features import FEATURES

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


def first_sentence(text: str | None) -> str:
    flat = cell(text)
    end = flat.find(". ")
    return flat if end < 0 else flat[: end + 1]


def door_of(scope: tuple[bool, bool]) -> str:
    needs_token, needs_version = scope
    if needs_token:
        return "token + `X-Crucible-Api: 1`"
    if needs_version:
        return "`X-Crucible-Api: 1` only"
    return "open"


def render_index(spec: dict[str, Any]) -> list[str]:
    """Every verb on one screen: each route and each job type, one line each."""
    from .jobdocs import JOB_DOCS

    out = [
        "## Every command",
        "",
        "Every route this server answers and every job type it runs, one line each. The "
        "sections below give each one in full.",
        "",
        "| route | what it does |",
        "| --- | --- |",
    ]
    rows = [
        (path, method.upper(), operation)
        for path, methods in spec.get("paths", {}).items()
        for method, operation in methods.items()
        if method.lower() in METHOD_ORDER
    ]
    rows.sort(key=lambda row: (row[0], METHOD_ORDER[row[1].lower()]))
    for path, method, operation in rows:
        what = first_sentence(operation.get("description") or operation.get("summary"))
        out.append("| `" + method + " " + path + "` | " + what + " |")
    out += [
        "",
        "Job types are submitted with `POST /v1/jobs` (`{\"type\", \"model\", \"params\", "
        "\"inputs\"}`); each is in full under **Job types** below.",
        "",
        "| job type | what it does |",
        "| --- | --- |",
    ]
    for name, doc in JOB_DOCS.items():
        out.append("| `" + name + "` | " + first_sentence(doc.summary) + " |")
    return out + [""]


def render_job_types(enabled: frozenset[str] | None) -> list[str]:
    """Every job type in full, from crucible/jobdocs.py and its params model."""
    import json

    from .jobdocs import JOB_DOCS

    out = [
        "## Job types",
        "",
        "Each is a `POST /v1/jobs` body: `type`, `model` when it takes one, `params` "
        "(validated against the table below; unknown keys are refused), and `inputs`, a "
        "map of file name to `{\"blob_id\"}` (from `POST /v1/uploads`), "
        "`{\"inline_base64\"}` or `{\"artifact\": {\"job_id\", \"name\"}}`. Follow "
        "`GET /v1/jobs/{id}/events` to the `done` event and download each artifact it "
        "names from `GET /v1/jobs/{id}/artifacts/{name}`.",
        "",
    ]
    if enabled is not None:
        out += [
            "Enabled on this server: " + (", ".join("`" + name + "`" for name in sorted(enabled))
                                          or "none") + ".",
            "",
        ]
    for name, doc in JOB_DOCS.items():
        title = "### `" + name + "`"
        if enabled is not None and name not in enabled:
            title += " (not enabled on this server)"
        out += [title, "", cell(doc.summary), ""]
        out += ["*Model:* " + (cell(doc.model) if doc.model else "none"), ""]
        if doc.params is None:
            out += ["*Params:* none (`{}`).", ""]
        else:
            schema = doc.params.model_json_schema()
            fields = render_fields(schema, schema.get("$defs", {}))
            out += (["**Params**", "", *fields, ""] if fields
                    else ["*Params:* none (`{}`).", ""])
        out += ["*Inputs:* " + cell(doc.inputs), "", "*Returns:* " + cell(doc.returns), ""]
        for note in doc.notes:
            out.append("- " + cell(note))
        if doc.notes:
            out.append("")
        out += ["```json", json.dumps(doc.example, indent=2), "```", ""]
    return out


def render(app: Any, enabled: frozenset[str] | None = None) -> str:
    """The whole reference. `enabled` (the job types this server runs) marks the rest;
    None, for docs/API.md, which describes the build rather than one host."""
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
        "Every running server serves this same reference, for itself: `GET /docs` (a page),",
        "`GET /v1/docs.md` (this markdown), `GET /v1/docs` (the index as JSON, with each job",
        "type's params schema) and `GET /v1/openapi.json`. None of them needs a token.",
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

    out += render_index(spec)

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
            door = door_of(scopes[(path, method)])
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

    out += render_job_types(enabled)
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


# The page the running server serves at /docs: the same markdown, as HTML. Written here
# rather than taken from a library so the server ships no new dependency; it reads only
# what render() writes (headings, paragraphs, pipe tables, bullet lists, fenced code,
# `code`, **bold**, *italic*).

_PAGE_STYLE = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --code: #f2f2f5;
        --accent: #b4441f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161617; --fg: #f5f5f7; --muted: #a1a1a6; --line: #3a3a3c; --code: #232325;
          --accent: #ff8a5c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 28px; margin: 8px 0 16px; }
h2 { font-size: 22px; margin: 40px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 17px; margin: 28px 0 8px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background: var(--code); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--code); padding: 12px 14px; border-radius: 8px; overflow-x: auto; }
pre code { padding: 0; background: none; }
.table { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left;
         vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td:first-child { white-space: nowrap; }
input#filter { width: 100%; padding: 8px 12px; margin: 4px 0 12px; font-size: 15px;
               border: 1px solid var(--line); border-radius: 8px; background: var(--bg);
               color: var(--fg); }
"""

_PAGE_SCRIPT = """
document.getElementById('filter').addEventListener('input', function (event) {
  var words = event.target.value.toLowerCase();
  for (var node = document.getElementById('every-command').nextElementSibling;
       node && node.tagName !== 'H2'; node = node.nextElementSibling) {
    node.querySelectorAll('tr').forEach(function (row) {
      if (row.querySelector('th')) return;
      row.style.display = row.textContent.toLowerCase().indexOf(words) < 0 ? 'none' : '';
    });
  }
});
"""


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _inline(text: str) -> str:
    import html
    import re

    pieces = re.split(r"(`[^`]*`)", text)
    out = []
    for piece in pieces:
        if len(piece) >= 2 and piece.startswith("`") and piece.endswith("`"):
            out.append("<code>" + html.escape(piece[1:-1]) + "</code>")
            continue
        piece = html.escape(piece)
        piece = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", piece)
        piece = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", piece)
        out.append(piece)
    return "".join(out)


def _cells(row: str) -> list[str]:
    import re

    inner = row.strip()[1:-1]
    escaped_pipe = "\\" + "|"
    return [cell.strip().replace(escaped_pipe, "|")
            for cell in re.split(r"(?<!\\)\|", inner)]


def to_html(markdown: str, *, title: str) -> str:
    """The reference as one self-contained page; route and job names in the index link to
    their sections."""
    import html

    lines = markdown.split("\n")
    anchors = {
        _slug(line.lstrip("#").strip())
        for line in lines if line.startswith("#")
    }
    body: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush() -> None:
        if paragraph:
            body.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            flush()
            block = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            body.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            body.append(f'<h{level} id="{_slug(text)}">{_inline(text)}</h{level}>')
            if _slug(text) == "every-command":
                body.append('<input id="filter" placeholder="Filter commands…" '
                            'autocomplete="off">')
        elif line.startswith("|"):
            flush()
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(lines[index])
                index += 1
            index -= 1
            head, rest = _cells(rows[0]), [r for r in rows[2:]]
            table = ["<div class=\"table\"><table><tr>"
                     + "".join("<th>" + _inline(c) + "</th>" for c in head) + "</tr>"]
            for row in rest:
                cells = _cells(row)
                first = cells[0]
                linked = _inline(first)
                if first.startswith("`") and first.endswith("`") and _slug(first) in anchors:
                    linked = f'<a href="#{_slug(first)}">{linked}</a>'
                table.append("<tr><td>" + linked + "</td>"
                             + "".join("<td>" + _inline(c) + "</td>" for c in cells[1:])
                             + "</tr>")
            body.append("".join(table) + "</table></div>")
        elif line.startswith("- "):
            flush()
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append("<li>" + _inline(lines[index][2:]) + "</li>")
                index += 1
            index -= 1
            body.append("<ul>" + "".join(items) + "</ul>")
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
        index += 1
    flush()
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_PAGE_STYLE}</style></head>"
        f"<body><main>{''.join(body)}</main><script>{_PAGE_SCRIPT}</script></body></html>"
    )
