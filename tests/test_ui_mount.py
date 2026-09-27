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


def test_the_door_sends_a_visitor_to_the_page_s_own_directory(
    client: TestClient,
) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_the_page_is_public_and_is_html(client: TestClient) -> None:
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
    assert client.get("/ui/app.ts").status_code == 404
    assert client.get("/ui/nothing/here.js").status_code == 404


def test_there_is_no_api_under_the_mount(client: TestClient) -> None:
    response = client.get("/ui/v1/info")
    assert response.status_code == 404
    assert "job_types" not in response.text


def test_the_mount_cannot_be_walked_out_of(client: TestClient) -> None:
    for attempt in ("/ui/../api.py", "/ui/%2e%2e/api.py", "/ui/../../pyproject.toml"):
        response = client.get(attempt)
        assert response.status_code in (404, 400), attempt
        assert "create_app" not in response.text


def test_the_api_still_needs_its_token(client: TestClient) -> None:
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/setup").status_code == 401


def test_the_page_asks_for_exactly_its_own_two_files_by_relative_name() -> None:
    html = _read(INDEX)
    assert 'href="app.css"' in html
    assert 'src="app.js"' in html
    referenced = set(re.findall(r'(?:src|href)="([^"]+)"', html))
    assert referenced == {"app.css", "app.js", "#main"}, referenced
    for value in referenced:
        assert not value.startswith("/"), value


def test_no_file_of_the_page_reaches_for_another_host() -> None:
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


def _strip_strings_and_comments(source: str) -> str:
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
    code = _strip_strings_and_comments(_read(SCRIPT))
    targets = re.findall(r"fetch\(\s*([A-Za-z_$][A-Za-z0-9_$]*(?:\([^)]*\))?)", code)
    assert sorted(targets) == ["path", "taskEventsPath(id)"], targets


def _paths_the_page_calls() -> set[str]:
    source = _read(SCRIPT)
    found = re.findall(
        r"/v1(?:/(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z0-9_.\-]+))+", source
    )
    return {re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path) for path in found}


def test_every_v1_path_the_page_calls_is_a_route_this_server_has(
    client: TestClient,
) -> None:
    known = {
        re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path)
        for path in client.app.openapi()["paths"]
    }
    called = _paths_the_page_calls()
    assert "/v1/setup" in called
    assert "/v1/tasks/*/events" in called
    missing = sorted(called - known)
    assert missing == [], f"the page calls {missing}, which this API does not serve"


def test_the_page_draws_every_section_it_was_asked_for() -> None:
    html = _read(INDEX)
    for section in ("status", "tasks", "types", "catalog", "connect", "service"):
        assert f'id="{section}-body"' in html, section


def test_the_page_stores_the_token_and_takes_it_out_of_the_address() -> None:
    source = _read(SCRIPT)
    assert "localStorage" in source
    assert "history.replaceState" in source
    assert "'token'" in source


def test_the_page_resolves_through_the_package_not_a_repo_path() -> None:
    for name in ("index.html", "app.css", "app.js"):
        resource = files("crucible").joinpath("ui").joinpath(name)
        assert resource.is_file(), name
    index = files("crucible").joinpath("ui").joinpath("index.html")
    assert "<html" in index.read_text(encoding="utf-8").lower()
    assert UI_DIR.is_dir()


def test_pyproject_declares_every_file_of_the_page_as_package_data() -> None:
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


def test_the_page_draws_the_settings_panel_between_job_types_and_connect() -> None:
    html = _read(INDEX)
    assert 'id="settings-body"' in html
    order = [html.index(f'id="{name}-body"') for name in ("types", "settings", "connect")]
    assert order == sorted(order), "Settings sits between Job types and Connect an app"


def test_every_settings_control_is_a_write_to_the_engine() -> None:
    source = _read(SCRIPT)
    assert "'/v1/settings'" in source
    assert "method: 'PUT'" in source
    assert "/v1/settings/upstreams/${safe}/test" in source


def test_the_page_never_asks_for_a_key_back() -> None:
    source = _read(SCRIPT)
    assert "key_hint" in source
    assert "state.upstreamDraft" in source
    assert "delete state.upstreamDraft[name];" in source
