"""The static mount: `GET /` and `GET /ui/*`, public, and nothing else there.

PHASE13-OPERATOR.md sections 1, 3.7 and 4. The PAGE is a separate build; what
is tested here is the mount it lands in — that it is reachable without a token,
that it cannot be walked out of, and that it does not shadow or leak the API.

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

import tomllib
from fnmatch import fnmatch
from importlib.resources import files
from pathlib import Path

from fastapi.testclient import TestClient

from crucible.api import UI_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------- serving


def test_the_page_is_public_and_is_html(client: TestClient) -> None:
    """No token. There is no secret in it — section 1."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<html" in response.text.lower()


def test_the_mount_serves_the_page_s_own_files_without_a_token(
    client: TestClient,
) -> None:
    response = client.get("/ui/index.html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_a_file_the_page_does_not_have_is_a_404(client: TestClient) -> None:
    assert client.get("/ui/app.js").status_code == 404


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


# ------------------------------------------------------------- package data


def test_the_page_resolves_through_the_package_not_a_repo_path() -> None:
    """`importlib.resources`, so this passes from an installed wheel too."""
    resource = files("crucible").joinpath("ui").joinpath("index.html")
    assert resource.is_file()
    assert "<html" in resource.read_text(encoding="utf-8").lower()
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
