"""The running server documents every command it has (Owen, 2026-10-04: "crucible should
have a list of every verb/command so nobody has to ask you how to do something, they can
just check the documentation on the running server").

These refuse a route without a description and a job type without an entry in
crucible/jobdocs.py, so a new command cannot ship undocumented.
"""
from __future__ import annotations

from typing import Callable

from fastapi.testclient import TestClient

from crucible.jobdocs import JOB_DOCS
from crucible.jobtypes import JOB_TYPE_SPECS


def test_every_route_says_what_it_does(client: TestClient) -> None:
    spec = client.get("/v1/openapi.json").json()
    bare = [
        f"{method.upper()} {path}"
        for path, methods in spec["paths"].items()
        for method, operation in methods.items()
        if not (operation.get("description") or "").strip()
    ]
    assert not bare, (
        "these routes have no docstring, so /docs shows them with nothing to say: "
        f"{bare}"
    )


def test_every_job_type_is_documented_in_declared_order() -> None:
    assert list(JOB_DOCS) == [spec.name for spec in JOB_TYPE_SPECS]
    for name, doc in JOB_DOCS.items():
        assert doc.summary.strip() and doc.inputs.strip() and doc.returns.strip(), name
        assert doc.example.get("type") == name, name


def test_every_param_says_what_it_is() -> None:
    bare = sorted(
        f"{name}: {doc.params.__name__}.{field}"
        for name, doc in JOB_DOCS.items()
        if doc.params is not None
        for field, info in doc.params.model_fields.items()
        if not (info.description or "").strip()
    )
    assert not bare, (
        "these params have no Field(description=...), so the served params table "
        f"says nothing about them: {bare}"
    )


def test_the_docs_are_served_without_a_token(make_client: Callable[..., TestClient]) -> None:
    with make_client(enable_echo=True) as client:
        page = client.get("/docs")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert 'id="every-command"' in page.text
        assert "POST /v1/jobs" in page.text

        markdown = client.get("/v1/docs.md")
        assert markdown.status_code == 200
        assert "## Job types" in markdown.text

        index = client.get("/v1/docs").json()
        routes = {f"{row['method']} {row['path']}" for row in index["routes"]}
        assert {"POST /v1/jobs", "GET /v1/docs", "POST /v1/server/updating"} <= routes
        types = {row["name"]: row for row in index["job_types"]}
        assert list(types) == list(JOB_DOCS)
        assert types["echo"]["enabled"] is True
        assert types["echo"]["params"]["properties"]["delay_ms"]["default"] == 25
