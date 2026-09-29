from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

from .. import weights
from ..client import transport
from ..client.connection import Connection
from ..config import config_path, crucible_home
from ..errors import ConfigError
from ..voicecatalog import (
    check_updates,
    following_pin,
    load_all_voices,
    load_voice,
    pull_target,
    refresh_ref,
)
from ..tomltable import REVISION_PATTERN
from ..voicerefs import VoiceRefError, resolve_ref
from ..voices import VoiceError
from . import common
from .api_cmd import report_http_error
from .common import EXIT_OK, _fail


def cmd_voices_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
    try:
        manifests = load_all_voices()
    except VoiceError as exc:
        return _fail(str(exc))
    rows = []
    for manifest in manifests.values():
        if not manifest.supports(backend.kind):
            rows.append(
                {
                    "id": manifest.id,
                    "backend_supported": False,
                    "installed": False,
                    "detail": f"no {backend.kind} block; declares "
                    f"{sorted(manifest.backends)}",
                }
            )
            continue
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "kind": manifest.kind,
                "narrator_engine": manifest.narrator_engine,
                "backend_supported": True,
                "installed": found is not None,
                "hf_repo": spec.hf_repo,
                "revision": spec.weights_identity,
                "source": spec.source,
                "identity_basis": spec.identity_basis,
                "memory_bytes_estimate": spec.memory_bytes_estimate,
                "estimate_basis": spec.estimate_basis,
                "max_chars": spec.max_chars,
                "max_chars_basis": spec.max_chars_basis,
                "pace_basis": manifest.pace_basis,
                "inherited_from": manifest.inherited_from,
                "manifest": manifest.manifest_source,
                "detail": (
                    f"{found.bytes / 1e9:.2f} GB at {found.path}"
                    if found is not None
                    else f"no weights at {spec.path} — this voice names a "
                    "directory on this server, which Crucible does not fetch "
                    "and cannot replace"
                    if spec.source == weights.LOCAL
                    else f"not pulled — `crucible voices pull {manifest.id}`"
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        mark = "installed" if row["installed"] else (
            "unsupported" if not row["backend_supported"] else "not pulled"
        )
        print(f"{row['id']:<22} {mark:<12} {row['detail']}")
    return EXIT_OK


def cmd_voices_pull(args: argparse.Namespace) -> int:
    config, backend = common.here()
    if args.all == (args.voice is not None):
        return _fail(
            "name one voice (`crucible voices pull mistborn`) or pass --all to move "
            "every voice installed here to the revision its tag names now"
        )
    try:
        voice_ids = [args.voice] if args.voice is not None else _installed_following(
            config, backend
        )
    except VoiceError as exc:
        return _fail(str(exc))
    if not voice_ids:
        print("no installed voice here follows a tag; nothing to move")
    for voice_id in voice_ids:
        code = _pull_one(config, backend, voice_id, force=args.force)
        if code != EXIT_OK:
            return code
    return EXIT_OK


def _installed_following(config, backend) -> list[str]:
    return [
        voice.id
        for voice in load_all_voices().values()
        if voice.supports(backend.kind)
        and voice.spec(backend.kind).source == weights.PINNED
        and following_pin(voice.id) is not None
        and weights.installed(config, voice, voice.spec(backend.kind)) is not None
    ]


def _pull_one(config, backend, voice_id: str, *, force: bool) -> int:
    looked = refresh_ref(config.home, voice_id)
    if looked is not None and looked.error is not None:
        print(f"{voice_id}: {looked.error}")
    try:
        manifest = load_voice(voice_id)
        target = pull_target(config.home, manifest)
    except VoiceError as exc:
        return _fail(str(exc))
    if not target.supports(backend.kind):
        return _fail(
            f"voice {voice_id!r} has no {backend.kind} block; "
            f"{target.path.name} declares {sorted(target.backends)}"
        )
    if target is not manifest:
        refused = _refuse_if_loaded(config, backend, voice_id)
        if refused is not None:
            return refused
    spec = target.spec(backend.kind)
    if spec.source == weights.LOCAL:
        print(f"{target.id}: {spec.path} for {backend.kind}")
    else:
        print(f"{target.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    try:
        result = weights.pull(
            config, target, spec, force=force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{target.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def _refuse_if_loaded(config, backend, voice_id: str) -> int | None:
    server = common.server_here(config, backend)
    if server is None:
        return None
    try:
        activity = transport.call(server, "GET", "/v1/activity")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _fail(
            f"the server at {server.url} did not say what it has loaded ({exc}), so "
            f"voice {voice_id!r} is not moved underneath it. Run `crucible voices "
            f"pull {voice_id}` again"
        )
    resident = activity.get("resident") if isinstance(activity, dict) else None
    if isinstance(resident, dict) and resident.get("id") == voice_id:
        return _fail(
            f"voice {voice_id!r} is loaded on the server at {server.url}, and its "
            "weights are not swapped underneath it; this machine stays on the "
            "revision it has. Unload it (`crucible api job submit --type "
            f"unload-voice --follow`), then run `crucible voices pull {voice_id}`"
        )
    return None


def cmd_voices_check_updates(args: argparse.Namespace) -> int:
    config, backend = common.here()
    server = common.server_here(config, backend)
    try:
        rows = (
            check_updates(config.home)
            if server is None
            else transport.call(server, "POST", "/v1/voices/updates")["voices"]
        )
    except urllib.error.HTTPError as exc:
        return report_http_error(exc, server)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(
            f"server_unreachable: the server here stopped answering ({exc}). Run "
            "`crucible voices check-updates` again"
        )
    except VoiceError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        print(_update_line(row))
    return EXIT_OK


def _update_line(row: dict) -> str:
    here = (row.get("revision") or "nothing")[:12]
    latest = (row.get("latest_revision") or "unknown")[:12]
    if row.get("update_error"):
        state = f"hub did not answer; staying on {here}: {row['update_error']}"
    elif row.get("update_available"):
        state = f"{here} -> {latest}: run `crucible voices pull {row['id']}`"
    else:
        state = f"current at {here}"
    return f"{row['id']:<22} {row['hf_repo']}@{row['ref']}  {state}"


def _repo_at(reference: str) -> tuple[str, str] | None:
    repo, sep, revision = reference.partition("@")
    if not sep or "/" not in repo:
        return None
    return repo, revision


def _repo_manifest_for(reference: str):
    from ..voicerepo import Pin, fetch_repo_manifest, parse_repo_manifest

    named = _repo_at(reference)
    if named is None:
        path = Path(reference)
        if not path.is_file():
            raise VoiceError(
                f"{reference!r} is neither an <owner>/<name>@<sha> reference nor "
                "a file on this machine"
            )
        return parse_repo_manifest(path.read_text(encoding="utf-8"), path), None
    repo, revision = named
    if not REVISION_PATTERN.match(revision):
        try:
            revision = resolve_ref(crucible_home(), repo, revision)
        except VoiceRefError as exc:
            raise VoiceError(str(exc)) from None
    pin = Pin(id="probe", hf_repo=repo, revision=revision, path=Path(reference))
    text, path = fetch_repo_manifest(crucible_home(), pin)
    return parse_repo_manifest(text, path), pin


def _pin_through_the_server(
    server: Connection, voice_id: str, repo: str, revision: str
) -> Path | int:
    try:
        answer = transport.call(
            server,
            "PUT",
            f"/v1/voices/{voice_id}",
            json_body={"pin": {"hf_repo": repo, "revision": revision}},
        )
    except urllib.error.HTTPError as exc:
        return report_http_error(exc, server)
    except (urllib.error.URLError, OSError) as exc:
        return _fail(
            f"server_unreachable: the server at {server.url} answered a moment "
            f"ago and not now ({exc}); nothing was pinned. Run `crucible voices "
            f"pin {voice_id} {repo}@{revision}` again"
        )
    return Path(answer["path"])


def cmd_voices_pin(args: argparse.Namespace) -> int:
    from ..voicerepo import Pin, home_pins_path, voice_for_pin, write_home_pin

    named = _repo_at(args.reference)
    if named is None:
        return _fail(
            f"{args.reference!r} is not an <owner>/<name>@<sha> reference. A pin "
            "is a repo AND a commit: the manifest at that sha and the weights at "
            "that sha are the same commit, which is the whole point"
        )
    repo, revision = named
    config, backend = common.here()
    server = common.server_here(config, backend)
    try:
        voice = voice_for_pin(
            Pin(id=args.voice, hf_repo=repo, revision=revision, path=home_pins_path())
        )
        if server is None:
            pin = write_home_pin(args.voice, repo, revision)
        else:
            written = _pin_through_the_server(server, args.voice, repo, revision)
            if isinstance(written, int):
                return written
            pin = Pin(id=args.voice, hf_repo=repo, revision=revision, path=written)
    except VoiceError as exc:
        return _fail(str(exc))
    print(f"{args.voice}: {pin.hf_repo}@{pin.revision[:12]} -> {pin.path}")
    if server is not None:
        print(f"  written by the server at {server.url}, which checked the voice is not loaded")
    print(f"  {voice.display} ({voice.kind}, {voice.narrator_engine}), arms "
          f"{sorted(voice.backends)}")
    print(f"  pull the weights with `crucible voices pull {args.voice}`")
    return EXIT_OK


def cmd_voices_check(args: argparse.Namespace) -> int:
    from ..config import tts_engine_footprints
    from ..voicerepo import Pin, footprint_unset, merge

    try:
        repo, pin = _repo_manifest_for(args.reference)
    except VoiceError as exc:
        return _fail(str(exc))
    engine = repo.voice["narrator_engine"]
    try:
        footprint = tts_engine_footprints(crucible_home()).get(engine)
    except ConfigError as exc:
        return _fail(f"config_unreadable: {exc}")
    if footprint is None:
        return _fail(
            f"engine_footprint_unset: this manifest is served by {engine!r} and "
            f"this machine's config states no [tts.{engine}] table, so the "
            "checks that depend on what the engine costs here cannot run and a "
            f"server here would refuse the voice. {footprint_unset(engine)}"
        )
    identity = (
        Pin(id=args.id, hf_repo=pin.hf_repo, revision=pin.revision, path=pin.path)
        if pin is not None
        else Pin(
            id=args.id,
            hf_repo="checked/locally",
            revision="0" * 40,
            path=Path(args.reference),
        )
    )
    try:
        voice = merge(repo, identity, footprint)
    except VoiceError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(voice.to_dict(), indent=2))
        return EXIT_OK
    print(f"{voice.display} ({voice.kind}, {voice.narrator_engine}, "
          f"{voice.language}, {voice.sample_rate} Hz)")
    pace = voice.pace
    if pace.pace_chars_per_sec is None:
        print("pace:     not measured — an uncertified voice (docs/internals/voices.md, \"The voice schema\")")
    else:
        print(
            f"pace:     {pace.pace_chars_per_sec} chars/s ({voice.pace_basis}), "
            f"band {pace.min_chars_per_sec}-{pace.max_chars_per_sec}"
        )
        if voice.inherited_from is not None:
            print(f"          inherited from {voice.inherited_from}")
    if pace.safe_min_chars is not None:
        print(f"packs:    {pace.safe_min_chars}-{pace.safe_max_chars} chars")
    elif pace.target_chars is not None:
        print(f"packs:    {pace.target_chars} chars")
    for arm in sorted(voice.backends):
        spec = voice.backends[arm]
        cap = (
            "not measured"
            if spec.max_chars is None
            else f"{spec.max_chars} ({spec.max_chars_basis})"
        )
        print(
            f"{arm}: cap {cap}, sampling "
            + ", ".join(f"{k} {v}" for k, v in sorted(spec.sampling.items()))
        )
    print(f"takes:    {len(voice.takes)} rung(s)")
    return EXIT_OK


def cmd_voices_card(args: argparse.Namespace) -> int:
    from .. import voicecard
    from ..voicerepo import REPO_MANIFEST_NAME
    from ..weights import hf_token_at

    try:
        repo, pin = _repo_manifest_for(args.reference)
    except VoiceError as exc:
        return _fail(str(exc))
    if pin is None:
        if args.upload:
            return _fail(
                "--upload needs an <owner>/<name>@<sha> reference: a local file "
                "names no repo to commit to"
            )
        print("---")
        print(voicecard.render_frontmatter(repo, ""))
        print("---")
        print()
        print(voicecard.render_limits(repo), end="")
        return EXIT_OK

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        return _fail(f"huggingface_hub is not importable: {exc}")
    token = hf_token_at(config_path(crucible_home()))
    try:
        card_path = hf_hub_download(
            repo_id=pin.hf_repo,
            filename="README.md",
            revision=pin.revision,
            local_dir=str(
                crucible_home()
                / "voice-manifests"
                / pin.hf_repo.replace("/", "--")
                / pin.revision
            ),
            token=token,
        )
    except Exception as exc:
        return _fail(
            f"could not read README.md from {pin.hf_repo}@{pin.revision[:12]}: "
            f"{type(exc).__name__}: {exc}"
        )
    existing = Path(card_path).read_text(encoding="utf-8")
    try:
        rendered, added = voicecard.render_card(repo, existing)
    except VoiceError as exc:
        return _fail(str(exc))
    if added:
        print(
            "note: this card had no limits section; one was ADDED after the "
            "frontmatter"
        )
    if not args.upload:
        print(rendered, end="")
        print(
            f"\n--- not uploaded. Pass --upload to commit this README.md to "
            f"{pin.hf_repo}, rendered from its own {REPO_MANIFEST_NAME}.",
        )
        return EXIT_OK
    if rendered == existing:
        print(f"{pin.hf_repo}: the card already says what the manifest says")
        return EXIT_OK
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=rendered.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=pin.hf_repo,
            commit_message=(
                f"Render README.md from {REPO_MANIFEST_NAME} "
                f"(crucible voices card)"
            ),
        )
    except Exception as exc:
        return _fail(f"could not upload README.md to {pin.hf_repo}: {exc}")
    print(f"{pin.hf_repo}: README.md rendered from {REPO_MANIFEST_NAME} and pushed")
    return EXIT_OK


def cmd_voices_export(args: argparse.Namespace) -> int:
    from .. import voicecard
    from ..voices import MANIFEST_REPO

    try:
        manifest = load_voice(args.voice)
    except VoiceError as exc:
        return _fail(str(exc))
    if manifest.manifest_source == MANIFEST_REPO:
        return _fail(
            f"voice {args.voice!r} already comes out of a repo manifest at its "
            "pin, so there is nothing to convert. Read it with `crucible voices "
            "check`"
        )
    try:
        text, dropped = voicecard.export_manifest(
            manifest,
            pace_basis=args.pace_basis,
            measured_from=args.measured_from,
            inherited_from=args.inherited_from,
            max_chars_basis=args.max_chars_basis,
            uncertified=args.uncertified,
            edges=args.edges,
        )
    except VoiceError as exc:
        return _fail(str(exc))
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.voice}: wrote {args.out}")
    else:
        print(text, end="")
    print(f"\n# {len(dropped)} row(s) this file does NOT carry:", file=sys.stderr)
    for line in dropped:
        print(f"#   {line}", file=sys.stderr)
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    voices = subparsers.add_parser("voices", help="list and pull voice weights")
    voice_commands = voices.add_subparsers(dest="voices_command", required=True)

    voices_list = voice_commands.add_parser(
        "list", help="every voice manifest this build ships and where it stands here"
    )
    voices_list.add_argument("--json", action="store_true", help="machine-readable")
    voices_list.set_defaults(func=cmd_voices_list)

    voices_pull = voice_commands.add_parser(
        "pull",
        help="fetch a voice's weights at its pin: the tag's current commit, or the exact sha",
    )
    voices_pull.add_argument(
        "voice", nargs="?", default=None, help="the Crucible voice id, e.g. deathstalker"
    )
    voices_pull.add_argument(
        "--all",
        action="store_true",
        help="move every voice installed here to the revision its tag names now",
    )
    voices_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    voices_pull.set_defaults(func=cmd_voices_pull)

    voices_updates = voice_commands.add_parser(
        "check-updates",
        help="look up each voice's tag on the Hub and say which a pull would move",
    )
    voices_updates.add_argument("--json", action="store_true", help="machine-readable")
    voices_updates.set_defaults(func=cmd_voices_check_updates)

    voices_pin = voice_commands.add_parser(
        "pin", help="point a voice id at a repo and commit (docs/internals/voices.md, \"Pins\")"
    )
    voices_pin.add_argument("voice", help="the Crucible voice id, e.g. mistborn")
    voices_pin.add_argument(
        "reference", help="<owner>/<name>@<40-character sha>"
    )
    voices_pin.set_defaults(func=cmd_voices_pin)

    voices_check = voice_commands.add_parser(
        "check",
        help="parse a crucible-voice.toml exactly as the loader would",
    )
    voices_check.add_argument(
        "reference", help="<owner>/<name>@<sha>, or a path to a local file"
    )
    voices_check.add_argument(
        "--id",
        default="probe",
        help="the id to check it under; only the refusal messages see it",
    )
    voices_check.add_argument("--json", action="store_true", help="machine-readable")
    voices_check.set_defaults(func=cmd_voices_check)

    voices_card = voice_commands.add_parser(
        "card", help="render a repo's README.md from its crucible-voice.toml"
    )
    voices_card.add_argument(
        "reference", help="<owner>/<name>@<sha>, or a path to a local file"
    )
    voices_card.add_argument(
        "--upload",
        action="store_true",
        help="commit the rendered README.md to the repo (needs an HF token)",
    )
    voices_card.set_defaults(func=cmd_voices_card)

    voices_export = voice_commands.add_parser(
        "export",
        help="a packaged manifest as a crucible-voice.toml, machine rows dropped",
    )
    voices_export.add_argument("voice", help="the Crucible voice id")
    voices_export.add_argument("--out", help="write here instead of to stdout")
    voices_export.add_argument(
        "--pace-basis",
        choices=("measured", "inherited"),
        help="how this voice's pace was got; the packaged schema cannot say",
    )
    voices_export.add_argument(
        "--measured-from",
        help="what a measured pace was measured on; required with "
        "--pace-basis measured",
    )
    voices_export.add_argument(
        "--inherited-from",
        help="which run and checkpoint an inherited pace came from, and why "
        "these weights have no ladder; required with --pace-basis inherited",
    )
    voices_export.add_argument(
        "--max-chars-basis",
        choices=("measured", "placeholder"),
        help="how the per-arm caps were got; the packaged schema cannot say",
    )
    voices_export.add_argument(
        "--uncertified",
        action="store_true",
        help="say on purpose that this voice has no measured pace",
    )
    voices_export.add_argument(
        "--edges",
        choices=("percentile",),
        help="the band's two edges came off a distribution, not off the pace",
    )
    voices_export.set_defaults(func=cmd_voices_export)
