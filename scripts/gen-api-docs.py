#!/usr/bin/env python3

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

from crucible.apidocs import render  # noqa: E402


def build_app() -> Any:
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
        enable_image=True,
        enable_audio=True,
        enable_segment=True,
        enable_video=True,
        desktop_allowance_bytes=3 * 1024 ** 3,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    return create_app(load_config(home), backend)


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
