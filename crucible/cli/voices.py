from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import weights
from ..config import config_path, crucible_home
from ..voices import VoiceError, load_all_voices, load_voice
from .common import EXIT_OK, _fail, _models_config


def cmd_voices_list(args: argparse.Namespace) -> int:
    """Every voice manifest this build ships and where it stands on this host.

    The same shape as `crucible models list`, and deliberately not the
    `/v1/voices` row: this command answers "what is on this disk", which a person
    runs before a load, while the row answers "what can this server be asked for",
    which a client reads.
    """
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
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
                # A LOCAL VOICE IS NOT PULLABLE, so its line must not offer a
                # pull (PHASE18-UNCERTIFIED.md section 3): `crucible voices
                # pull` on one is refused by name, and a screening merge that
                # is gone is the expected end of its life rather than a broken
                # install. The `/v1/voices` row's `reason` says the same thing
                # for a client; this is the same question asked at a terminal.
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
    resolved = _models_config()
    if isinstance(resolved, int):
        return resolved
    config, backend = resolved
    try:
        manifest = load_voice(args.voice)
    except VoiceError as exc:
        return _fail(str(exc))
    if not manifest.supports(backend.kind):
        return _fail(
            f"voice {args.voice!r} has no {backend.kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    spec = manifest.spec(backend.kind)
    # WHAT THE BLOCK NAMES, in whichever of its two shapes (PHASE18-
    # UNCERTIFIED.md section 3). `spec.revision[:12]` subscripted None on a
    # local block, so this line raised a TypeError BEFORE `weights.pull` could
    # refuse it by name — the operator got a traceback where there is a
    # sentence saying the bytes are somebody else's.
    if spec.source == weights.LOCAL:
        print(f"{manifest.id}: {spec.path} for {backend.kind}")
    else:
        print(f"{manifest.id}: {spec.hf_repo}@{spec.revision[:12]} for {backend.kind}")
    try:
        result = weights.pull(
            config, manifest, spec, force=args.force,
            on_line=lambda line: print(f"  {line}"),
        )
    except weights.WeightsError as exc:
        return _fail(str(exc))
    print(f"{manifest.id}: {result.bytes / 1e9:.2f} GB at {result.path}")
    return EXIT_OK


def _repo_at(reference: str) -> tuple[str, str] | None:
    """`<owner>/<name>@<sha>` split, or None because this is not one."""
    repo, sep, revision = reference.partition("@")
    if not sep or "/" not in repo:
        return None
    return repo, revision


def _repo_manifest_for(reference: str):
    """A `RepoManifest` from `<repo>@<sha>` or from a local file.

    One reader for both `check` and `card`, because "what does this manifest
    say" must not have two answers depending on which command asked.
    """
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
    pin = Pin(id="probe", hf_repo=repo, revision=revision, path=Path(reference))
    text, path = fetch_repo_manifest(crucible_home(), pin)
    return parse_repo_manifest(text, path), pin


def cmd_voices_pin(args: argparse.Namespace) -> int:
    """`crucible voices pin <id> <repo>@<sha>` — what a deploy does per machine.

    THE PIN IS LOADED BEFORE IT IS WRITTEN, exactly as `PUT /v1/voices/{id}`
    loads it: a revision that carries no `crucible-voice.toml`, a schema this
    build cannot read, and a machine with no `[tts.<engine>]` table are each
    refused here with nothing written.
    """
    from ..voicerepo import Pin, home_pins_path, voice_for_pin, write_home_pin

    named = _repo_at(args.reference)
    if named is None:
        return _fail(
            f"{args.reference!r} is not an <owner>/<name>@<sha> reference. A pin "
            "is a repo AND a commit: the manifest at that sha and the weights at "
            "that sha are the same commit, which is the whole point"
        )
    repo, revision = named
    try:
        voice = voice_for_pin(
            Pin(id=args.voice, hf_repo=repo, revision=revision, path=home_pins_path())
        )
        pin = write_home_pin(args.voice, repo, revision)
    except VoiceError as exc:
        return _fail(str(exc))
    print(f"{args.voice}: {pin.hf_repo}@{pin.revision[:12]} -> {pin.path}")
    print(f"  {voice.display} ({voice.kind}, {voice.narrator_engine}), arms "
          f"{sorted(voice.backends)}")
    print(f"  pull the weights with `crucible voices pull {args.voice}`")
    return EXIT_OK


def cmd_voices_check(args: argparse.Namespace) -> int:
    """Parse a manifest exactly as the loader would, and print it or the refusal.

    What the training side runs before it pushes. It merges with THIS machine's
    `[tts.<engine>]` table rather than with an invented one, because the
    question the command answers is "would a server load this", and a server
    that has not been told what the engine costs would not.
    """
    from ..config import tts_engine_footprints
    from ..voicerepo import Pin, merge

    try:
        repo, pin = _repo_manifest_for(args.reference)
    except VoiceError as exc:
        return _fail(str(exc))
    engine = repo.voice["narrator_engine"]
    footprint = tts_engine_footprints(crucible_home()).get(engine)
    if footprint is None:
        return _fail(
            f"engine_footprint_unset: this manifest is served by {engine!r} and "
            f"this machine's config states no [tts.{engine}] table, so the "
            "checks that depend on what the engine costs here cannot run and a "
            "server here would refuse the voice. Run `crucible init`, or add the "
            "table to config.toml"
        )
    # A LOCAL FILE HAS NO PIN, and the merge needs one to fill `hf_repo` and
    # `revision` in. A stand-in is used and SAID: what this command answers is
    # whether the manifest's own numbers survive the loader, and the repo and
    # sha play no part in that.
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
        print("pace:     not measured — an uncertified voice (PHASE18 4.1)")
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
        # "not measured" rather than a blank or a zero: a voice may state no
        # cap since 2026-09-19 (PHASE18-UNCERTIFIED.md section 4), and an
        # operator reading `cap None` would not know whether the number is
        # missing or the field is broken.
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
    """Render the repo's README from its manifest, and with --upload commit it."""
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
        # A LOCAL FILE HAS NO CARD TO MERGE INTO, so what is printed is the two
        # blocks this command owns. That is what the training side wants before
        # it pushes: see the frontmatter and the limits section that the repo's
        # README will get, without a repo yet existing.
        print("---")
        print(voicecard.render_frontmatter(repo, ""))
        print("---")
        print()
        print(voicecard.render_limits(repo), end="")
        return EXIT_OK

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
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
    """A packaged manifest as a `crucible-voice.toml`, plus the rows it drops."""
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
        )
    except VoiceError as exc:
        return _fail(str(exc))
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.voice}: wrote {args.out}")
    else:
        print(text, end="")
    # PRINTED, NEVER DROPPED SILENTLY (section 4). Every line is a machine fact
    # that moved rather than vanished, and the person converting the file is the
    # one who has to put it where it now lives.
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
        "pull", help="fetch a voice's weights at the manifest's pinned revision"
    )
    voices_pull.add_argument("voice", help="the Crucible voice id, e.g. deathstalker")
    voices_pull.add_argument(
        "--force", action="store_true", help="re-pull even if it is already installed"
    )
    voices_pull.set_defaults(func=cmd_voices_pull)

    voices_pin = voice_commands.add_parser(
        "pin", help="point a voice id at a repo and commit (PHASE21 section 2.4)"
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
    voices_export.set_defaults(func=cmd_voices_export)
