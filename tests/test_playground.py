from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import desktop, installonsubmit, jobenv, local, playground, weights
from crucible.api import UI_DIR
from crucible.jobs import audio as audio_job
from crucible.jobs import image as image_job
from crucible.jobs import video as video_job
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take
from crucible.jobs.image import ImageParams
from crucible.jobs.video.params import VideoParams, settle

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, stamp_env
from .test_audio_api import MUSIC, SFX, SONG, _weights
from .test_desktop_lifecycle import tray_icon
from .test_ui_mount import _strip_strings_and_comments

PAGE = UI_DIR / "playground.html"
SCRIPT = UI_DIR / "playground.js"
BACKENDS = (FAKE_BACKEND.kind, FAKE_MAC_BACKEND.kind)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _audio_envs(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in jobenv.audio_envs(backend_kind):
        stamp_env(home, spec, backend_kind, monkeypatch)


def _pages(client: TestClient, auth: dict[str, str]) -> dict[str, dict[str, Any]]:
    response = client.get("/v1/playground", headers=auth)
    assert response.status_code == 200, response.text
    return {page["id"]: page for page in response.json()["pages"]}


def _fields(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in page["fields"]}


def _defaults(fields: list[dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in fields:
        if field["required"] or (field["kind"] == "text" and field.get("placeholder")):
            params[field["name"]] = field["placeholder"]
        elif field["default"] is not None:
            params[field["name"]] = field["default"]
    return params


def test_the_playground_route_is_behind_the_token(client: TestClient) -> None:
    assert client.get("/v1/playground").status_code == 401


@pytest.fixture
def hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(weights.HF_TOKEN_ENV, "hf_test_token_not_real")


@pytest.fixture
def no_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(weights.HF_TOKEN_ENV, raising=False)


def test_every_declared_model_has_a_page_and_one_not_installed_downloads_on_first_use(
    client: TestClient, auth: dict[str, str], hf_token: None
) -> None:
    pages = _pages(client, auth)
    declared = {
        *image_job.MANIFESTS.all(),
        *video_job.MANIFESTS.all(),
        *audio_job.MANIFESTS.all(),
    }
    assert set(pages) == declared
    for page in pages.values():
        assert page["standing"] == playground.DOWNLOAD and page["available"], page["id"]
        assert "first Generate downloads" in page["reason"]
        assert "engine" in page["reason"] and "weights" in page["reason"]
    assert pages[SFX]["makes"] == "sound effects" and pages[SFX]["media"] == "audio"
    assert pages["qwen-image-2.1"]["media"] == "image"
    assert pages["ltx-2.5-distilled"]["media"] == "video"


def test_a_gated_model_without_a_hugging_face_token_says_how_to_get_one(
    client: TestClient, auth: dict[str, str], no_hf_token: None
) -> None:
    pages = _pages(client, auth)
    assert pages[SFX]["standing"] == playground.UNAVAILABLE and not pages[SFX]["available"]
    assert "gated" in pages[SFX]["reason"] and weights.HF_TOKEN_ENV in pages[SFX]["reason"]
    assert pages[SONG]["standing"] == playground.DOWNLOAD
    assert pages["qwen-image-2.1"]["standing"] == playground.DOWNLOAD


def test_an_installed_model_is_ready_and_a_missing_one_downloads_its_weights(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, hf_token: None,
) -> None:
    _audio_envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SFX, FAKE_BACKEND.kind)
    _weights(home, SONG, FAKE_BACKEND.kind)
    with make_client(enable_audio=True) as client:
        pages = _pages(client, auth)
    assert pages[SFX]["standing"] == playground.READY and pages[SFX]["reason"] is None
    assert pages[SONG]["standing"] == playground.READY
    assert pages[MUSIC]["standing"] == playground.DOWNLOAD and pages[MUSIC]["available"]
    assert "downloads its weights" in pages[MUSIC]["reason"]
    assert "engine" not in pages[MUSIC]["reason"] and "Catalog" not in pages[MUSIC]["reason"]

    sfx = _fields(pages[SFX])
    assert list(sfx) == ["prompt", "duration_s", "steps", "format", "seed"]
    assert sfx["prompt"]["required"] and sfx["prompt"]["kind"] == "text"
    assert sfx["duration_s"]["default"] == 10 and sfx["duration_s"]["max"] == 120
    assert sfx["steps"]["default"] == 8 and sfx["steps"]["max"] == 50
    assert sfx["format"]["options"] == ["flac", "wav", "mp3"]
    assert sfx["seed"]["default"] is None and not sfx["seed"]["required"]

    song = _fields(pages[SONG])
    assert list(song) == ["tags", "lyrics", "instrumental", "planning_set", "min_duration_s",
                          "max_duration_s", "cfg", "format", "seed"]
    assert not song["lyrics"]["required"], "an instrumental needs none; the server refuses a sung song without"
    assert song["instrumental"]["kind"] == "boolean" and song["instrumental"]["default"] is False


def test_a_model_with_weights_but_no_engine_downloads_the_engine_and_says_its_size(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str]
) -> None:
    _weights(home, SFX, FAKE_BACKEND.kind)
    with make_client(enable_audio=True) as client:
        page = _pages(client, auth)[SFX]
    assert page["standing"] == playground.DOWNLOAD
    assert "the audio engine" in page["reason"] and "weights" not in page["reason"]
    expected = installonsubmit._env_bytes("audio", None, FAKE_BACKEND.kind)
    assert expected and page["download_bytes"] == expected


def test_a_model_with_no_build_for_this_backend_cannot_be_generated(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every audio model has a Mac build now (YuE2 since 2026-10-03), so the song's is hidden.
    from crucible.audiomodels import AudioManifest

    monkeypatch.setattr(
        AudioManifest, "supports",
        lambda self, kind: kind in self.backends and not (self.id == SONG and kind == FAKE_MAC_BACKEND.kind),
    )
    _audio_envs(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    with make_client(enable_audio=True, backend=FAKE_MAC_BACKEND) as client:
        page = _pages(client, auth)[SONG]
    assert page["standing"] == playground.UNAVAILABLE and not page["available"]
    assert "mlx-darwin" in page["reason"] and page["fields"] == []


def test_without_install_on_submit_nothing_missing_is_offered(
    make_client: Callable[..., TestClient], home: Path, monkeypatch: pytest.MonkeyPatch,
    hf_token: None,
) -> None:
    _audio_envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SFX, FAKE_BACKEND.kind)
    with make_client(enable_audio=True) as client:
        state = client.app.state
        config = replace(state.config, install_on_submit=False)
        pages = {page["id"]: page for page in playground.pages(
            config, state.backend, state.store.registry)}
    assert pages[SFX]["standing"] == playground.READY
    assert pages[MUSIC]["standing"] == playground.UNAVAILABLE
    assert "install_on_submit" in pages[MUSIC]["reason"]
    assert pages["qwen-image-2.1"]["standing"] == playground.UNAVAILABLE
    assert "not enabled" in pages["qwen-image-2.1"]["reason"]


def test_a_type_turned_off_with_its_engine_installed_is_not_offered(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, hf_token: None,
) -> None:
    _audio_envs(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client() as client:
        page = _pages(client, auth)[SONG]
    assert page["standing"] == playground.UNAVAILABLE
    assert "audio" in page["reason"]


@pytest.mark.parametrize("backend_kind", BACKENDS)
def test_the_image_form_s_defaults_and_limits_are_what_the_job_accepts(backend_kind: str) -> None:
    for manifest in image_job.MANIFESTS.all().values():
        if not manifest.supports(backend_kind):
            continue
        spec = manifest.spec(backend_kind)
        fields = playground.image_fields(manifest, spec)
        params = ImageParams(**_defaults(fields))
        assert params.width % spec.size_multiple == 0 and params.height % spec.size_multiple == 0
        side = {field["name"]: field for field in fields}["width"]
        assert side["max"] % spec.size_multiple == 0 and side["min"] % spec.size_multiple == 0
        assert side["max"] <= spec.max_side


@pytest.mark.parametrize("backend_kind", BACKENDS)
def test_the_video_form_s_defaults_and_longest_clip_settle_at_every_rate(backend_kind: str) -> None:
    for manifest in video_job.MANIFESTS.all().values():
        if not manifest.supports(backend_kind):
            continue
        spec = manifest.spec(backend_kind)
        fields = playground.video_fields(manifest, spec)
        settle(VideoParams(**_defaults(fields)), manifest.id, spec, 0)
        longest = {field["name"]: field for field in fields}["duration_s"]["max"]
        for fps in spec.fps:
            params = {**_defaults(fields), "duration_s": longest, "fps": fps}
            settle(VideoParams(**params), manifest.id, spec, 0)


@pytest.mark.parametrize("backend_kind", BACKENDS)
def test_the_audio_form_offers_only_what_each_model_takes(backend_kind: str) -> None:
    for manifest in audio_job.MANIFESTS.all().values():
        if not manifest.supports(backend_kind):
            continue
        spec = manifest.spec(backend_kind)
        fields = playground.audio_fields(manifest, spec)
        refuse_what_the_model_cannot_take(AudioParams(**_defaults(fields)), manifest, spec)
        ceilings = {field["name"]: field.get("max") for field in fields}
        params = {**_defaults(fields)}
        params.update({name: ceilings[name] for name in ("duration_s", "steps", "cfg")
                       if name in ceilings})
        refuse_what_the_model_cannot_take(AudioParams(**params), manifest, spec)


def test_the_playground_page_and_script_are_served_without_a_token(client: TestClient) -> None:
    page = client.get("/ui/playground.html")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    script = client.get("/ui/playground.js")
    assert script.status_code == 200
    assert script.headers["content-type"].split(";")[0] in (
        "application/javascript",
        "text/javascript",
    )


def test_the_console_links_the_playground() -> None:
    assert 'href="playground.html"' in _read(UI_DIR / "index.html")


def test_the_playground_page_asks_only_for_its_own_files() -> None:
    referenced = set(re.findall(r'(?:src|href)="([^"]+)"', _read(PAGE)))
    assert referenced == {"app.css", "playground.js", "playground.html", "./", "#main"}


def test_the_playground_script_uses_only_doors_that_carry_the_token() -> None:
    code = _strip_strings_and_comments(_read(SCRIPT))
    for banned in ("EventSource", "XMLHttpRequest", "WebSocket", "innerHTML", "eval("):
        assert banned not in code, banned
    assert re.findall(r"fetch\(\s*([A-Za-z_$][A-Za-z0-9_$]*)", code) == ["path"]
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert code.count(opener) == code.count(closer), opener
    for marker in ("http://", "https://", "//cdn", "@import"):
        assert marker not in _read(SCRIPT), marker


def test_every_v1_path_the_playground_calls_is_a_route(client: TestClient) -> None:
    known = {
        re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path)
        for path in client.app.openapi()["paths"]
    }
    found = re.findall(
        r"/v1(?:/(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z0-9_.\-]+))+", _read(SCRIPT)
    )
    called = {re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", "*", path) for path in found}
    assert {"/v1/playground", "/v1/jobs", "/v1/jobs/*/events", "/v1/jobs/*/artifacts/*",
            "/v1/tasks/*/events"} <= called
    assert sorted(called - known) == []


def test_the_script_can_show_every_result_a_family_makes() -> None:
    script = _read(SCRIPT)
    made = {
        "image": [image_job.ARTIFACT_NAME],
        "video": [video_job.ARTIFACT_NAME],
        "audio": [audio_job.artifact_name(form) for form in ("flac", "wav")],
    }
    for family in playground.FAMILIES:
        for name in made[family.media]:
            extension = name.rsplit(".", 1)[1]
            assert re.search(rf"\b{extension}: '{family.media}/", script), name


def test_the_playground_submits_jobs_that_wait_by_default() -> None:
    script = _read(SCRIPT)
    assert "queue:" not in script
    assert "'crucible.token'" in script and "history.replaceState" in script


def test_the_playground_follows_an_install_and_then_sends_the_job_again() -> None:
    code = _strip_strings_and_comments(_read(SCRIPT))
    script = _read(SCRIPT)
    assert "refusal.code === 'installing'" in script
    assert "details.task_id" in script
    assert "followInstall(job, job.install)" in code
    assert "taskEventsPath(install.taskId)" in code
    assert "for (var round = 0; receipt === null; round += 1)" in code
    assert "Catalog" not in script


def test_the_door_keeps_the_section_it_was_asked_for(client: TestClient) -> None:
    response = client.get("/?section=connect", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/?section=connect"


@pytest.mark.parametrize(
    ("verb", "page"),
    [("open-console", "/"), ("playground", "/ui/playground.html")],
)
def test_the_browser_verbs_open_their_page_signed_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verb: str, page: str
) -> None:
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "a+b"))
    opened: list[str] = []
    monkeypatch.setattr(local.webbrowser, "open", opened.append)
    local.run_engine_verb(verb, tmp_path)
    assert opened == [f"http://127.0.0.1:7100{page}#token=a%2Bb"]


@pytest.mark.parametrize(
    ("label", "verb"),
    [(desktop.CONSOLE_LABEL, "open-console"), (desktop.PLAYGROUND_LABEL, "playground")],
)
def test_the_tray_opens_the_web_console_and_the_playground(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, label: str, verb: str
) -> None:
    tray = tray_icon(monkeypatch, tmp_path)
    tray.state.update(state="running", detail="The paired engine is answering")
    asked: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        local, "run_engine_verb",
        lambda said, home: asked.append((said, home)) or {"state": "opened"},
    )
    item = next(item for item in tray.icon.menu if item.text == label)
    item.action()
    assert asked == [(verb, tmp_path)]
    assert tray.state["state"] == "running"
