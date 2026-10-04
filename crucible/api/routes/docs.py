"""The API reference, served by the server it describes.

`GET /docs` is the page, `GET /v1/docs.md` the same text as markdown, `GET /v1/docs` the
index as JSON (every route and every job type, with its params schema), and
`GET /v1/openapi.json` the OpenAPI document. All are open: reading how to call a server
grants nothing, and the token is still what every working route needs.
"""
from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from ... import VERSION
from ...apidocs import auth_scopes, door_of, first_sentence, render, to_html
from ...jobdocs import JOB_DOCS
from ..context import AppContext, Routers

MARKDOWN = "text/markdown; charset=utf-8"


def _enabled(ctx: AppContext) -> frozenset[str]:
    return frozenset(ctx.store.registry)


def register(routers: Routers, ctx: AppContext) -> None:
    public = routers.public
    app = ctx.app

    @public.get("/docs")
    async def docs_index() -> dict[str, Any]:
        """Every command this server answers, as data: each route with its door and one
        line, and each job type with its model, params schema, inputs, returns, notes, an
        example body and whether it is enabled here. The page is `GET /docs`."""
        spec = app.openapi()
        scopes = auth_scopes(app)
        enabled = _enabled(ctx)
        routes = [
            {
                "method": method.upper(),
                "path": path,
                "door": door_of(scopes[(path, method.upper())]),
                "summary": first_sentence(operation.get("description")),
            }
            for path, methods in spec.get("paths", {}).items()
            for method, operation in methods.items()
        ]
        job_types = [
            {
                "name": name,
                "enabled": name in enabled,
                "summary": doc.summary,
                "model": doc.model,
                "params": None if doc.params is None else doc.params.model_json_schema(),
                "inputs": doc.inputs,
                "returns": doc.returns,
                "notes": list(doc.notes),
                "example": doc.example,
            }
            for name, doc in JOB_DOCS.items()
        ]
        return {
            "version": VERSION,
            "page": "/docs",
            "markdown": "/v1/docs.md",
            "openapi": "/v1/openapi.json",
            "routes": routes,
            "job_types": job_types,
        }

    @public.get("/docs.md", response_class=PlainTextResponse)
    async def docs_markdown() -> PlainTextResponse:
        """The whole API reference as markdown: every route, every job type, every model."""
        return PlainTextResponse(render(app, _enabled(ctx)), media_type=MARKDOWN)

    @app.get("/docs", include_in_schema=False)
    async def docs_page(request: Request) -> HTMLResponse:
        text = render(app, _enabled(ctx))
        return HTMLResponse(to_html(text, title=f"Crucible {VERSION} API — {ctx.config.name}"))
