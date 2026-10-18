"""The per-model prompting guides are served by the server they describe, and every guide
names models this build actually ships."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from crucible import prompting

ROOT = Path(prompting.__file__).parents[1]


def _shipped_model_ids() -> set[str]:
    ids: set[str] = set()
    for manifest in ROOT.rglob("*.toml"):
        ids.update(re.findall(r'^id = "([^"]+)"', manifest.read_text(encoding="utf-8"), re.M))
    return ids


def test_every_guide_has_its_file_and_every_file_a_guide() -> None:
    files = {path.stem for path in prompting.HERE.glob("*.md")}
    assert files == set(prompting.GUIDES)


def test_every_covered_model_ships_in_this_build() -> None:
    shipped = _shipped_model_ids()
    named = {model for models in prompting.GUIDES.values() for model in models}
    assert named <= shipped, sorted(named - shipped)


def test_every_guide_names_the_models_it_covers() -> None:
    for guide, models in prompting.GUIDES.items():
        body = prompting.text(guide)
        assert body.startswith("# "), guide
        for model in models:
            assert model in body, (guide, model)


def test_the_index_lists_every_guide(client: TestClient) -> None:
    body = client.get("/v1/prompting").json()
    assert [entry["name"] for entry in body["guides"]] == list(prompting.GUIDES)
    qwen = next(entry for entry in body["guides"] if entry["name"] == "qwen3.5")
    assert "qwen3.5-9b-vl" in qwen["models"]
    assert qwen["markdown"] == "/v1/prompting/qwen3.5.md"


def test_a_guide_is_reached_by_its_name_or_any_model_it_covers(client: TestClient) -> None:
    by_name = client.get("/v1/prompting/qwen3.5.md")
    by_model = client.get("/v1/prompting/qwen3.5-2b.md")
    assert by_name.status_code == 200
    assert by_name.headers["content-type"].startswith("text/markdown")
    assert by_model.text == by_name.text == prompting.text("qwen3.5")


def test_an_unknown_name_is_refused_by_name(client: TestClient) -> None:
    response = client.get("/v1/prompting/no-such-model.md")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "prompting_guide_not_found"


def test_every_guide_is_linked_where_its_models_are_called(client: TestClient) -> None:
    from crucible.jobdocs import JOB_DOCS

    assert set(prompting.JOB_TYPE_GUIDES) <= set(JOB_DOCS)
    paths = client.get("/v1/openapi.json").json()["paths"]
    for path in prompting.ROUTE_GUIDES:
        assert "post" in paths.get(path, {}), path
    placed = {
        guide
        for guides in (*prompting.JOB_TYPE_GUIDES.values(), *prompting.ROUTE_GUIDES.values())
        for guide in guides
    }
    assert placed == set(prompting.GUIDES)


def test_the_reference_points_at_every_guide(client: TestClient) -> None:
    text = client.get("/v1/docs.md").text
    assert "## How to prompt each model" in text
    for guide in prompting.GUIDES:
        assert f"[/v1/prompting/{guide}.md](/v1/prompting/{guide}.md)" in text, guide
    image = text.split("### `image`", 1)[1].split("\n### ", 1)[0]
    assert "**How to prompt it:**" in image and "/v1/prompting/qwen-image-2.1.md" in image
    chat = text.split("### `POST /v1/openai/chat/completions`", 1)[1].split("\n### ", 1)[0]
    assert "/v1/prompting/qwen3.5.md" in chat


def test_the_docs_page_follows_the_links(client: TestClient) -> None:
    page = client.get("/docs").text
    assert '<a href="/v1/prompting/qwen-image-2.1.md">' in page
