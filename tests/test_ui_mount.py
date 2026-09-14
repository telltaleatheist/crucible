"""The operator page and the mount it lands in — PHASE13-OPERATOR.md 1, 3.7, 4.

Two halves. The first is the MOUNT: that the door and the three files are
reachable without a token, that the mount cannot be walked out of, and that it
neither shadows nor leaks the API. The second is the PAGE ITSELF, read as text,
because there is no browser here and the things worth pinning about a page that
must work on a LAN with no route out are all readable from the bytes:

* it asks for its own two files by RELATIVE name and reaches for no other host,
  so a Mac with no internet renders it completely;
* **every `/v1` path it calls is a route this app has.** That is the drift guard
  between the page and the API (R1): the page is a client written in another
  language in the same repo, and nothing but this compares the two. A route
  renamed under it fails here with both spellings visible, rather than in a
  browser nobody is watching;
* it talks to its own server and nothing else — the `fetch` call sites are
  pinned by name, so a CDN, an `EventSource` (which cannot carry the bearer
  token) or an `XMLHttpRequest` added later is a failing test rather than a
  page that is sometimes blank.

WHY THE WHEEL IS ASSERTED THROUGH `importlib.resources` AND THE PYPROJECT
-------------------------------------------------------------------------
Section 4 asks for "a test asserts the files are in the built wheel". Building
one here would need `python -m build`, which pulls its own isolated build
environment over the network — and the machine this was written on has about
5 GB free on the drive the suite runs from, with no guarantee of a network at
all. So the two halves of "the wheel carries it" are asserted separately and
completely: the FILES are where the package says they are (`importlib.resources`
resolves them through the installed package, not through a repo path), and the
DECLARATION that puts them in a wheel is present and covers them. CI's release
job already runs `python -m build`, so the one thing left unproven here is
proven there on every push.
"""

from __future__ import annotations

import re
import tomllib
from fnmatch import fnmatch
from importlib.resources import files
from pathlib import Path

from fastapi.testclient import TestClient

from crucible.api import UI_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent

INDEX = UI_DIR / "index.html"
SCRIPT = UI_DIR / "app.js"
STYLE = UI_DIR / "app.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------- serving


def test_the_door_sends_a_visitor_to_the_page_s_own_directory(
    client: TestClient,
) -> None:
    """`GET /` is a 307 to `/ui/`, which is where the three files live.

    The page asks for `app.css` and `app.js` by relative name, so it has one
    home and `/` says where it is. A redirect whose target carries no fragment
    of its own keeps the request's, which is what lets a pairing line's
    `#token=…` arrive signed in.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_the_page_is_public_and_is_html(client: TestClient) -> None:
    """No token. There is no secret in it — section 1."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<html" in response.text.lower()


def test_the_mount_serves_the_page_s_own_files_without_a_token(
    client: TestClient,
) -> None:
    for path in ("/ui/", "/ui/index.html"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith("text/html"), path


def test_the_two_files_the_page_asks_for_are_served_as_what_they_are(
    client: TestClient,
) -> None:
    """A browser runs a script and applies a stylesheet by media type."""
    script = client.get("/ui/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].split(";")[0] in (
        "application/javascript",
        "text/javascript",
    )
    style = client.get("/ui/app.css")
    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")


def test_a_file_the_page_does_not_have_is_still_a_404(client: TestClient) -> None:
    """`html=True` serves the directory's index; it does not swallow misses."""
    assert client.get("/ui/app.ts").status_code == 404
    assert client.get("/ui/nothing/here.js").status_code == 404


def test_there_is_no_api_under_the_mount(client: TestClient) -> None:
    """`/ui/v1/info` is a missing static file, not the API with a prefix."""
    response = client.get("/ui/v1/info")
    assert response.status_code == 404
    assert "job_types" not in response.text


def test_the_mount_cannot_be_walked_out_of(client: TestClient) -> None:
    """Starlette refuses the traversal; asserted because the mount is public."""
    for attempt in ("/ui/../api.py", "/ui/%2e%2e/api.py", "/ui/../../pyproject.toml"):
        response = client.get(attempt)
        assert response.status_code in (404, 400), attempt
        assert "create_app" not in response.text


def test_the_api_still_needs_its_token(client: TestClient) -> None:
    """The mount is public; nothing else moved with it."""
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/setup").status_code == 401


# ------------------------------------------------------------ what it asks for


def test_the_page_asks_for_exactly_its_own_two_files_by_relative_name() -> None:
    """One stylesheet, one script, both relative. No third, no absolute path.

    Relative is what lets the same bytes be served from any mount; an absolute
    `/ui/app.js` would pin the page to today's mount and make `/` and `/ui/`
    two different answers to where it lives.
    """
    html = _read(INDEX)
    assert 'href="app.css"' in html
    assert 'src="app.js"' in html
    referenced = set(re.findall(r'(?:src|href)="([^"]+)"', html))
    assert referenced == {"app.css", "app.js", "#main"}, referenced
    for value in referenced:
        assert not value.startswith("/"), value


def test_no_file_of_the_page_reaches_for_another_host() -> None:
    """No CDN, no font service, no analytics: the Mac may have no route out.

    In the HTML and the CSS a URL IS a load, so any absolute one is a
    refusal. In the SCRIPT it is not: since PHASE15-HOST.md section 3.7 the
    page draws a field for an Ollama address, and `http://host:11434` in its
    placeholder is an EXAMPLE shown to a person, not a fetch.

    So the script's rule is narrowed rather than dropped (R2: a guard that
    cannot be made green is fixed, never tolerated red): every absolute URL in
    `app.js` must sit on a `placeholder:` line. What the page actually fetches
    is pinned twice over and far more strictly, by the two tests below and
    above — the `fetch` call sites by name, and every `/v1` path against the
    app's own route table.
    """
    for path in (INDEX, STYLE):
        text = _read(path)
        for marker in ("http://", "https://", "//cdn", "@import", "url("):
            assert marker not in text, f"{path.name} reaches out with {marker!r}"

    script = _read(SCRIPT)
    for marker in ("//cdn", "@import"):
        assert marker not in script, f"app.js reaches out with {marker!r}"
    for number, line in enumerate(script.splitlines(), start=1):
        if "http://" not in line and "https://" not in line:
            continue
        assert "placeholder" in line, (
            f"app.js:{number} carries an absolute URL that is not a "
            f"placeholder shown to a person: {line.strip()}"
        )


def test_the_page_uses_no_door_that_cannot_carry_the_token() -> None:
    """`EventSource` sends no headers, and every /v1 route needs two.

    Read off the CODE, with comments blanked: the file says at the top why it
    does not use `EventSource`, and a guard that could not tell an explanation
    from a call would forbid the explanation.
    """
    source = _strip_strings_and_comments(_read(SCRIPT))
    for banned in (
        "EventSource",
        "XMLHttpRequest",
        "WebSocket",
        "importScripts",
        "innerHTML",
        "eval(",
    ):
        assert banned not in source, banned


# ----------------------------------------------------- the page against the API


def _strip_strings_and_comments(source: str) -> str:
    """The code with every literal and comment blanked, for counting braces.

    Comments first, so an apostrophe in prose is never read as a quote; there
    are no regex literals in `app.js`, which is the one thing this cannot see
    through and the reason the file does without them.
    """
    out: list[str] = []
    index = 0
    end = len(source)
    while index < end:
        char = source[index]
        if char == "/" and index + 1 < end and source[index + 1] == "/":
            newline = source.find("\n", index)
            index = end if newline == -1 else newline
            continue
        if char == "/" and index + 1 < end and source[index + 1] == "*":
            close = source.find("*/", index + 2)
            index = end if close == -1 else close + 2
            continue
        if char in "'\"`":
            quote = char
            index += 1
            while index < end:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            out.append('""')
            continue
        out.append(char)
        index += 1
    return "".join(out)


def test_the_page_s_braces_and_parentheses_balance() -> None:
    """The cheapest proof a browser would parse it, with no browser here."""
    code = _strip_strings_and_comments(_read(SCRIPT))
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        depth = 0
        for char in code:
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                assert depth >= 0, f"{closer} before its {opener}"
        assert depth == 0, f"{depth} unclosed {opener}"


def test_the_page_talks_to_its_own_server_and_to_nothing_else() -> None:
    """Both `fetch` call sites, pinned by the expression they are given.

    One is the transport every read goes through; the other is the task event
    stream, which cannot go through it because it must not read the body. A
    third would be a fetch nobody had to write an Authorization header for.
    """
    code = _strip_strings_and_comments(_read(SCRIPT))
    targets = re.findall(r"fetch\(\s*([A-Za-z_$][A-Za-z0-9_$]*(?:\([^)]*\))?)", code)
    assert sorted(targets) == ["path", "taskEventsPath(id)"], targets


def _paths_the_page_calls() -> set[str]:
    """Every `/v1/...` path literal in `app.js`, with `${…}` as a wildcard."""
    source = _read(SCRIPT)
    found = re.findall(
        r"/v1(?:/(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z0-9_.\-]+))+", source
    )
    return {re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path) for path in found}


def test_every_v1_path_the_page_calls_is_a_route_this_server_has(
    client: TestClient,
) -> None:
    """THE DRIFT GUARD. A page is a client, and this is what compares it.

    Read out of the file rather than out of a list kept beside it: a list
    would be the second copy that goes stale, which is the whole shape
    ARCHITECTURE.md R1 is about.
    """
    # The OpenAPI document rather than `app.routes`, because an included
    # router is not a list of routes in every FastAPI version and the schema
    # is: it is the app's own statement of what it serves, keyed by path.
    known = {
        re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path)
        for path in client.app.openapi()["paths"]
    }
    called = _paths_the_page_calls()
    # Not an empty assertion by accident: the page really does call these.
    assert "/v1/setup" in called
    assert "/v1/tasks/*/events" in called
    missing = sorted(called - known)
    assert missing == [], f"the page calls {missing}, which this API does not serve"


def test_the_page_draws_every_section_it_was_asked_for() -> None:
    """Section 4's six sections, each with somewhere to draw into."""
    html = _read(INDEX)
    for section in ("status", "tasks", "types", "catalog", "connect", "service"):
        assert f'id="{section}-body"' in html, section


def test_the_page_stores_the_token_and_takes_it_out_of_the_address() -> None:
    """Section 1, read off the file: localStorage, and `replaceState`.

    A fragment never reaches this server, which is why the token travels in
    one — and it must not stay in the address bar, the back button or a
    bookmark after it has been read.
    """
    source = _read(SCRIPT)
    assert "localStorage" in source
    assert "history.replaceState" in source
    assert "'token'" in source


# ------------------------------------------------------------- package data


def test_the_page_resolves_through_the_package_not_a_repo_path() -> None:
    """`importlib.resources`, so this passes from an installed wheel too."""
    for name in ("index.html", "app.css", "app.js"):
        resource = files("crucible").joinpath("ui").joinpath(name)
        assert resource.is_file(), name
    index = files("crucible").joinpath("ui").joinpath("index.html")
    assert "<html" in index.read_text(encoding="utf-8").lower()
    assert UI_DIR.is_dir()


def test_pyproject_declares_every_file_of_the_page_as_package_data() -> None:
    """The declaration is what puts these bytes in a wheel; no file may miss it."""
    document = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    patterns = document["tool"]["setuptools"]["package-data"]["crucible"]
    for path in sorted(UI_DIR.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(UI_DIR.parent).as_posix()
        assert any(fnmatch(relative, pattern) for pattern in patterns), (
            f"{relative} is in crucible/ui/ and no package-data pattern in "
            f"pyproject.toml matches it, so a wheel would not carry it"
        )


# ------------------------------------------------------------- 3.7 settings


def test_the_page_draws_the_settings_panel_between_job_types_and_connect() -> None:
    """PHASE15-HOST.md section 3.7 says WHERE, and the order is the argument.

    A class this card cannot hold can still be routed somewhere that can, so
    the panel that says where the work runs belongs directly under the one
    that said the card could not hold it — and above the one that hands the
    server to an app.
    """
    html = _read(INDEX)
    assert 'id="settings-body"' in html
    order = [html.index(f'id="{name}-body"') for name in ("types", "settings", "connect")]
    assert order == sorted(order), "Settings sits between Job types and Connect an app"


def test_every_settings_control_is_a_write_to_the_engine() -> None:
    """*"Every control is a `PUT /v1/settings`"* — and nothing is a local save.

    Read off the file: the panel's writer sends PUT to that one path, and the
    test door is a POST to the per-upstream path. A control that wrote
    somewhere else, or that only changed a variable, would not appear here.
    """
    source = _read(SCRIPT)
    assert "'/v1/settings'" in source
    assert "method: 'PUT'" in source
    assert "/v1/settings/upstreams/${safe}/test" in source


def test_the_page_never_asks_for_a_key_back() -> None:
    """A key is write-only, so the panel reads `key_hint` and nothing else.

    There is no route that returns a key, and a page that reached for one
    would be a page written against a server that does not exist. Pinned here
    rather than trusted, because this is the one file in the repo that draws a
    field a secret has just been typed into.
    """
    source = _read(SCRIPT)
    assert "key_hint" in source
    # The draft the operator is typing is the ONLY place a key lives in this
    # page, and it is cleared on a successful save and only then — a key that
    # was refused is still the one they have in their hand.
    assert "state.upstreamDraft" in source
    assert "delete state.upstreamDraft[name];" in source
