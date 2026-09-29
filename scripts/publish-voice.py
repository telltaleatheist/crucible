#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from crucible.backend import CUDA_LINUX
from crucible.narratorengines import declared_tts_footprints
from crucible.tomltable import REVISION_PATTERN
from crucible.voicerefs import DEFAULT_REF
from crucible.voicerepo import REPO_MANIFEST_NAME, Pin, merge, packaged_pins, parse_repo_manifest
from crucible.voices import VoiceError

SEED_REVISIONS: dict[str, str] = {
    "deathstalker": "28f6c4d3733be9746bb26ddd7b09c9d2a4950198",
    "mistborn": "8a8d1bf375c7c4aa097a481d332b94e835a8c6f4",
    "owen": "5885e74691be8c39d137dd8dd5b03813f822d1c4",
    "sigma": "8f6b3ca274cbc7b83a537ce25e05d223eb35c987",
    "thirdreich": "f83ce206e82ca639fa8a16cddd7ace713f9027c6",
}

TOKEN_HELP = (
    "set $HF_TOKEN to a write token for the repo, or run `huggingface-cli login` "
    "once on this machine"
)


class Refused(Exception):
    ...


class Hub:

    def __init__(self, token: str) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi(token=token)
        self._token = token

    def commit_of(self, repo: str, revision: str) -> str | None:
        return getattr(self._api.model_info(repo, revision=revision), "sha", None)

    def manifest_at(self, repo: str, revision: str, into: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo,
                filename=REPO_MANIFEST_NAME,
                revision=revision,
                local_dir=str(into),
                token=self._token,
            )
        )

    def tag_commit(self, repo: str, tag: str) -> str | None:
        for found in self._api.list_repo_refs(repo).tags:
            if found.name == tag:
                return found.target_commit
        return None

    def delete_tag(self, repo: str, tag: str) -> None:
        self._api.delete_tag(repo, tag=tag)

    def create_tag(self, repo: str, tag: str, revision: str) -> None:
        self._api.create_tag(repo, tag=tag, revision=revision)


def hub_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def _pin_of(voice_id: str) -> Pin:
    pins = packaged_pins()
    pin = pins.get(voice_id)
    if pin is None:
        raise Refused(
            f"{voice_id!r} is not a voice this build ships; crucible/voices/pins.toml "
            f"names {sorted(pins)}"
        )
    return pin


def _verified_commit(hub: Any, repo: str, sha: str) -> str:
    if not REVISION_PATTERN.match(sha):
        raise Refused(
            f"{sha!r} is not a full 40-character commit sha. Copy it from the repo's "
            f"commit list on https://huggingface.co/{repo}/commits/main"
        )
    try:
        found = hub.commit_of(repo, sha)
    except Exception as exc:
        raise Refused(
            f"{repo} has no commit {sha} ({type(exc).__name__}: {exc}); check the sha "
            f"against https://huggingface.co/{repo}/commits/main"
        ) from None
    if found != sha:
        raise Refused(f"{repo}@{sha} answered as commit {found!r}, not that sha")
    return sha


def _check_manifest(hub: Any, pin: Pin, sha: str) -> str:
    with tempfile.TemporaryDirectory() as scratch:
        try:
            path = hub.manifest_at(pin.hf_repo, sha, Path(scratch))
        except Exception as exc:
            raise Refused(
                f"{pin.hf_repo}@{sha[:12]} has no readable {REPO_MANIFEST_NAME} "
                f"({type(exc).__name__}: {exc}). Commit one beside the weights "
                "(`crucible voices export` writes it), then publish that commit"
            ) from None
        try:
            repo = parse_repo_manifest(path.read_text(encoding="utf-8"), path)
            engine = repo.voice["narrator_engine"]
            footprints = [f for f in declared_tts_footprints(CUDA_LINUX) if f.engine == engine]
            if not footprints:
                raise VoiceError(f"narrator_engine {engine!r} is not an engine this build serves")
            merge(repo, pin.at(sha), footprints[0])
        except VoiceError as exc:
            raise Refused(
                f"{pin.hf_repo}@{sha[:12]}'s {REPO_MANIFEST_NAME} does not parse: {exc}. "
                f"Fix it, commit, and publish the new commit; check it first with "
                f"`crucible voices check {pin.hf_repo}@<sha>`"
            ) from None
        return repo.voice["display"]


def publish(hub: Any, voice_id: str, sha: str | None, *, create: bool) -> str:
    pin = _pin_of(voice_id)
    tag = pin.ref or DEFAULT_REF
    current = hub.tag_commit(pin.hf_repo, tag)
    if create:
        if current is not None:
            raise Refused(
                f"{pin.hf_repo} already has a {tag!r} tag at {current[:12]}; move it "
                f"with `python scripts/publish-voice.py {voice_id} <sha>`"
            )
        sha = sha or SEED_REVISIONS.get(voice_id)
    if sha is None:
        raise Refused(
            f"name the commit to publish: `python scripts/publish-voice.py {voice_id} <sha>`"
        )
    sha = _verified_commit(hub, pin.hf_repo, sha)
    display = _check_manifest(hub, pin, sha)
    if current == sha:
        return f"{pin.hf_repo}: {tag!r} already names {sha[:12]} ({display}); nothing moved"
    if current is not None:
        hub.delete_tag(pin.hf_repo, tag)
    hub.create_tag(pin.hf_repo, tag, sha)
    moved = "created at" if current is None else f"moved {current[:12]} ->"
    return (
        f"{pin.hf_repo}: {tag!r} {moved} {sha[:12]} ({display}). Machines move with "
        f"`crucible voices pull {voice_id}` (`crucible voices check-updates` shows it first)"
    )


def main(argv: list[str] | None = None, hub: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description="Point a voice repo's `crucible` tag at a commit whose "
        f"{REPO_MANIFEST_NAME} Crucible reads.",
    )
    parser.add_argument("voice", help="the Crucible voice id, e.g. mistborn")
    parser.add_argument("sha", nargs="?", default=None, help="the 40-character commit to publish")
    parser.add_argument(
        "--create",
        action="store_true",
        help="create the tag on a repo that has none, at <sha> or at the revision "
        "this build shipped before tags",
    )
    args = parser.parse_args(argv)
    try:
        if hub is None:
            token = hub_token()
            if not token:
                raise Refused(f"no HuggingFace token on this machine: {TOKEN_HELP}")
            hub = Hub(token)
        print(publish(hub, args.voice, args.sha, create=args.create))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
