from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import hosttools

from .conftest import TOOL_BYTES


def test_a_504_from_the_release_host_is_waited_on_in_a_sentence(
    flaky_host: SimpleNamespace, tmp_path: Path, real_tool_download: Callable[..., str]
) -> None:
    flaky_host.answers[:] = [504, 503, 200]
    target = tmp_path / "tool"
    said: list[str] = []
    digest = real_tool_download(flaky_host.url, target, pause_s=0, on_wait=said.append)
    assert digest == hashlib.sha256(TOOL_BYTES).hexdigest()
    assert target.read_bytes() == TOOL_BYTES
    assert flaky_host.answers == []
    assert len(said) == 2
    assert "HTTP 504" in said[0] and "try 1 of 30" in said[0]
    assert "HTTP 503" in said[1] and "try 2 of 30" in said[1]


def test_a_404_is_refused_at_once_as_the_tools_own_error(
    flaky_host: SimpleNamespace, tmp_path: Path, real_tool_download: Callable[..., str]
) -> None:
    flaky_host.answers[:] = [404, 200]
    said: list[str] = []
    with pytest.raises(hosttools.HostToolError) as refused:
        real_tool_download(flaky_host.url, tmp_path / "tool", pause_s=0, on_wait=said.append)
    assert refused.value.code == "tool_download_failed"
    assert "404" in refused.value.message
    assert flaky_host.answers == [200], "a 404 is not weather: it is not asked twice"
    assert said == []
