"""The model card is RENDERED from the manifest, and the manifest from a file.

PHASE21-VOICES-FROM-HF.md sections 2.5 and 4. Two directions, one pair of
facts:

    `card`    `crucible-voice.toml` -> the repo's `README.md` frontmatter and
              its `## Measured limits` section. One writer, in the repo with the
              loader, so "what the loader reads" and "what the card says" cannot
              be two things. Until 2026-09-19 the card was written by a regex in
              a campaign script generated fresh per deploy, and thirdreich's
              card carried `higgs_target_chars` ten days after that field was
              retired because nothing ever read the card back.

    `export`  a packaged `voices/<id>.toml` -> a `crucible-voice.toml`, plus the
              MACHINE rows it drops, printed. The bridge from today's files, and
              section 8.2's cross-check against what the training side writes
              out of the measurement database. A disagreement between the two is
              a finding, not something to average.

── WHAT THIS MODULE WILL NOT INVENT ─────────────────────────────────────────

`max_chars_basis` and `[voice.pace] basis` do not exist in the packaged schema,
and the whole reason section 2.1 adds them is that an inherited pace and a
placeholder cap are real states that shipped as measured facts — thirdreich's
`higgs_max_chars_mlx: 900` was never measured, and deathstalker's 16.64 survived
onto weights that measured 15.91. So `export` REFUSES to guess them: it asks the
person running it, by name, and writes what they say. A default here would
launder exactly the two defects the field exists to expose.
"""

from __future__ import annotations

import re
from typing import Any

from .voicerepo import REPO_MANIFEST_NAME, REPO_SCHEMA, RepoManifest
from .voices import VoiceError

#: The frontmatter keys the card writer owns. Everything else in a card's
#: frontmatter is left exactly as it was — this list is the whole of what one
#: run of `crucible voices card` may change, and it is a TABLE here rather than
#: a regex in a campaign script so that adding one is an edit to a list a test
#: reads.
FM_OWNED: tuple[str, ...] = (
    "higgs_max_chars_served",
    "higgs_max_chars_mlx",
    "higgs_pace_chars_per_sec",
    "higgs_safe_min_chars",
    "higgs_safe_max_chars",
    "higgs_pace_basis",
    "higgs_max_chars_served_basis",
    "higgs_max_chars_mlx_basis",
)

#: KEYS THAT ARE RETIRED, and why, refused BY NAME when a card still carries
#: one. Not dropped quietly: a retired key in a card is a number an audit script
#: may still be reading, and the person who has to know is the one deploying.
RETIRED_FM_KEYS: dict[str, str] = {
    "higgs_target_chars": (
        "retired 2026-09-09 when Owen's 800-character ruling replaced the "
        "single packing target for fine-tunes; thirdreich's card still carried "
        "it ten days later. A voice packs to a measured safe band or to a "
        "target, never to both, and the manifest says which"
    ),
}

#: Which arm each cap key states. The two are not always equal — thirdreich
#: carried 1623 served against 900 on mlx — which is why the card has two keys
#: and the manifest has two arms.
_CAP_KEYS: dict[str, str] = {
    "higgs_max_chars_served": "cuda-linux",
    "higgs_max_chars_mlx": "mlx-darwin",
}

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
_LIMITS_SECTION = re.compile(r"## (?:Measured|Recorded) limits.*?(?=\n## |\Z)", re.S)


class CardError(VoiceError):
    """A card cannot be rendered, or would lose something if it were."""


# ------------------------------------------------------------------- the card


def _pace_lines(repo: RepoManifest) -> list[str]:
    pace = repo.pace or {}
    lines: list[str] = []
    rate = pace.get("pace_chars_per_sec")
    if rate is not None:
        lines.append(f"higgs_pace_chars_per_sec: {rate}")
        lines.append(f"higgs_pace_basis: {repo.pace_basis}")
    for key in ("safe_min_chars", "safe_max_chars"):
        value = pace.get(key)
        if value is not None:
            lines.append(f"higgs_{key}: {value}")
    return lines


def render_frontmatter(repo: RepoManifest, existing: str) -> str:
    """The card's frontmatter with this manifest's facts in it.

    EVERY LINE THIS FILE DOES NOT OWN SURVIVES, in its original order. The owned
    keys are removed wherever they were and re-added at the end, which is what
    the campaign script did and is what keeps a hand-written `license:` or
    `tags:` block untouched.

    A KEY WITH NOTHING TO SAY IS NOT WRITTEN AT ALL. An uncertified voice has no
    pace, and a card carrying `higgs_pace_chars_per_sec:` with nothing after it
    is a field an audit script reads as zero.
    """
    for line in existing.split("\n"):
        name = line.split(":", 1)[0].strip()
        if name in RETIRED_FM_KEYS:
            raise CardError(
                f"this card carries {name}, which is retired: "
                f"{RETIRED_FM_KEYS[name]}. Remove the line and run this again — "
                "it is not dropped silently, because something may still be "
                "reading it"
            )
    keep = [
        line
        for line in existing.split("\n")
        if line.split(":", 1)[0].strip() not in FM_OWNED
    ]
    while keep and keep[-1].strip() == "":
        keep.pop()
    owned = list(_pace_lines(repo))
    for key, arm in _CAP_KEYS.items():
        block = repo.arms.get(arm)
        if block is None:
            continue
        owned.append(f"{key}: {block['max_chars']}")
        owned.append(f"{key}_basis: {repo.max_chars_basis[arm]}")
    return "\n".join(keep + owned)


def render_limits(repo: RepoManifest) -> str:
    """The `## Measured limits` section, out of the manifest and nothing else.

    Every claim here has a field behind it. The campaign template carried five
    values the manifest does not hold (`pace_n`, `clean`, `pace_extra`,
    `pace_on`, the run and checkpoint), and that is precisely why the card and
    the manifest drifted: half the section was typed per deploy. The manifest's
    own `measured_from` is the one prose field, and it is reproduced verbatim.
    """
    pace = repo.pace or {}
    lines = ["## Measured limits (from this repo's crucible-voice.toml)", ""]
    rate = pace.get("pace_chars_per_sec")
    if rate is None:
        lines.append(
            "- **Pace:** not measured on these weights. This voice states no "
            "pace band, so a client packs to the per-chunk cap below and "
            "narrator guards against its engine's own default band rather than "
            "against a number measured here."
        )
    else:
        lines.append(
            f"- **Pace:** {rate} characters of text per second of audio "
            f"({repo.pace_basis}), with the band running "
            f"{pace['min_chars_per_sec']} to {pace['max_chars_per_sec']}."
        )
        # THE BASIS'S OWN SENTENCE, whichever basis it is. An inherited pace
        # is served exactly like a measured one — it is a certificate the voice
        # states — and the card is where a person finds out WHICH other weights
        # it came from, which is the whole of whether to trust it.
        if repo.measured_from:
            lines.append(f"  {repo.measured_from}")
        if repo.inherited_from:
            lines.append(f"  Inherited from {repo.inherited_from}")
    if pace.get("safe_min_chars") is not None:
        lines.append(
            f"- **Safe chunk band:** {pace['safe_min_chars']}-"
            f"{pace['safe_max_chars']} characters. A client packs between these "
            "two edges."
        )
    elif pace.get("target_chars") is not None:
        lines.append(
            f"- **Packing target:** {pace['target_chars']} characters. A client "
            "packs to this single target rather than to a band."
        )
    for arm in sorted(repo.arms):
        block = repo.arms[arm]
        lines.append(
            f"- **Per-chunk cap, {arm}:** {block['max_chars']} characters "
            f"({repo.max_chars_basis[arm]})."
        )
    lines.append(
        f"- **Sampling:** "
        + ", ".join(
            f"{key} {value}"
            for key, value in sorted(
                repo.arms[sorted(repo.arms)[0]]["sampling"].items()
            )
        )
        + "."
    )
    rungs = len(repo.takes) if repo.takes else 1
    lines.append(
        f"- **Retake ladder:** {rungs} rung(s). A client asks for take N; what "
        "take N is belongs to the server."
    )
    # ONE trailing newline, not two. The section is spliced in where another one
    # was, and the blank line before the next `##` heading is that heading's own
    # — a second one here widens the gap every time the card is re-rendered.
    return "\n".join(lines) + "\n"


def render_card(repo: RepoManifest, existing: str) -> tuple[str, bool]:
    """The whole card, and whether the limits section had to be ADDED.

    Added rather than refused, because thirdreich's and sigma's cards have no
    limits section at all today (section 1) and the point of this command is to
    give them one. It is reported so nobody discovers it in a diff.
    """
    matched = _FRONTMATTER.match(existing)
    if matched is None:
        raise CardError(
            "this card has no YAML frontmatter, so there is nothing to rewrite "
            "and this command refuses to guess at where the facts go. Add a "
            "`---` block at the top of README.md and run it again"
        )
    frontmatter, body = matched.group(1), matched.group(2)
    limits = render_limits(repo)
    body, replaced = _LIMITS_SECTION.subn(lambda _m: limits, body, count=1)
    if replaced == 0:
        # BEFORE THE FIRST OTHER SECTION, so a card that has no limits section
        # today — thirdreich's and sigma's, which carry no safe band at all
        # (section 1) — gets one where a reader expects it rather than above the
        # title. A card with no `##` heading at all gets it at the end.
        at = body.find("\n## ")
        if at == -1:
            body = body.rstrip("\n") + "\n\n" + limits
        else:
            body = body[: at + 1] + limits + "\n" + body[at + 1 :]
    return (
        "---\n" + render_frontmatter(repo, frontmatter) + "\n---\n" + body,
        replaced == 0,
    )


def read_frontmatter(card: str) -> dict[str, str]:
    """The card's frontmatter as flat key -> text. What a test parses back.

    Deliberately not a YAML parser: the keys this module writes are scalars on
    one line each, and a dependency that could read a nested block would invite
    one to be written.
    """
    matched = _FRONTMATTER.match(card)
    if matched is None:
        raise CardError("this card has no YAML frontmatter")
    found: dict[str, str] = {}
    for line in matched.group(1).split("\n"):
        if ":" not in line or line.startswith((" ", "-", "#")):
            continue
        key, _, value = line.partition(":")
        found[key.strip()] = value.strip()
    return found


# ----------------------------------------------------------------- the export


def _toml_number(value: float) -> str:
    """A rate or a sampling value, written the way the catalog writes it.

    A WHOLE NUMBER IS WRITTEN WHOLE. Every value that reaches here has been
    through `crucible/voices.py`'s `_number()`, which returns a float, so
    `top_k` 50 would otherwise be written `50.0` — read back identically by that
    same `_number()`, and read by a PERSON as a different number from the 50 the
    catalog states everywhere else. The loader takes either; this is about the
    file being readable by whoever has to check it before it is pushed.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _toml_string(value: str) -> str:
    """A TOML string — multi-line for prose, so a note stays readable."""
    if "\n" in value or len(value) > 90:
        escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'"""{escaped}"""'
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def export_manifest(
    manifest: Any,
    *,
    pace_basis: str | None,
    measured_from: str | None,
    inherited_from: str | None,
    max_chars_basis: str | None,
    uncertified: bool,
) -> tuple[str, list[str]]:
    """A packaged manifest as a `crucible-voice.toml`, and the rows it drops.

    Section 4. The dropped rows are RETURNED rather than logged, so the caller
    prints them and nothing goes silently: every one of them is a machine fact
    that now lives in that machine's `config.toml` (`[tts.<engine>]`) or in the
    pin, and a person converting a file has to be able to see that they moved
    rather than vanished.
    """
    from .voices import MAX_CHARS_BASES, PACE_BASES

    pace = manifest.pace
    has_band = pace.pace_chars_per_sec is not None
    packs = pace.target_chars is not None or pace.safe_min_chars is not None
    if not has_band and not uncertified:
        raise CardError(
            f"voice {manifest.id!r} states no pace band, so the file this would "
            "write is an UNCERTIFIED voice (PHASE18 section 4.1) — a real state, "
            "and one worth stating on purpose. Pass --uncertified to say so"
        )
    # A `[voice.pace]` TABLE THAT EXISTS OWES A `basis`, band or no band: the key
    # certifies the table, and a packing target taken from a predecessor is the
    # same hazard as a pace taken from one. A voice with nothing to say here
    # omits the table in whole, which is what `--uncertified` alone produces.
    if has_band or packs:
        if pace_basis not in PACE_BASES:
            raise CardError(
                f"voice {manifest.id!r} has a pace band and the packaged schema "
                "cannot say how it was got. Pass --pace-basis "
                f"{'|'.join(sorted(PACE_BASES))}: an inherited pace is "
                "indistinguishable from a measured one at the point of use, "
                "which is exactly how deathstalker's 16.64 survived onto weights "
                "that measured 15.91"
            )
        # EACH BASIS OWES EXACTLY ITS OWN SENTENCE, and this writer refuses to
        # invent either (ruled 2026-09-19; `voicerepo._repo_pace` is the reader
        # of the same rule). The packaged schema states neither, so there is
        # nothing here to carry across — only a person who knows.
        owed, given, refused_name, refused_value = (
            ("--measured-from", measured_from, "--inherited-from", inherited_from)
            if pace_basis == "measured"
            else ("--inherited-from", inherited_from, "--measured-from", measured_from)
        )
        if given is None or given.strip() == "":
            raise CardError(
                f"--pace-basis {pace_basis} owes {owed}: "
                + (
                    "what was measured, on which checkpoint, over how many "
                    "renders. The number is only worth what the next reader can "
                    "find out about it"
                    if pace_basis == "measured"
                    else "which run and checkpoint the number came from, and why "
                    "these weights have no ladder yet. A sibling checkpoint of "
                    "the same corpus is near enough; a different corpus two "
                    "versions back is deathstalker's 16.64 onto weights that "
                    "measured 15.91"
                )
            )
        if refused_value is not None and refused_value.strip() != "":
            raise CardError(
                f"--pace-basis {pace_basis} was given {refused_name} as well. "
                f"Each basis owes exactly its own sentence — {owed} — and the "
                "other would be prose about a measurement this voice did not make"
            )
    if max_chars_basis not in MAX_CHARS_BASES:
        raise CardError(
            f"voice {manifest.id!r}'s per-arm caps carry no basis and the "
            "packaged schema cannot say. Pass --max-chars-basis "
            f"{'|'.join(sorted(MAX_CHARS_BASES))}: thirdreich shipped a "
            "`higgs_max_chars_mlx: 900` that no sweep ever produced, and a file "
            "that cannot say so ships it as a measured fact"
        )

    lines = [
        f"# {manifest.display} — exported from this build's "
        f"{manifest.path.name} by `crucible voices export`.",
        "#",
        "# The MACHINE facts that file carried are not here: they belong to the",
        "# box that serves the voice, in its config.toml `[tts.<engine>]` table",
        "# (PHASE21 section 2.3). The repo and revision are not here either —",
        "# this file IS the revision, and the local side pins it.",
        "",
        f"schema = {REPO_SCHEMA}",
        "",
        "[voice]",
        f"display         = {_toml_string(manifest.display)}",
        f"kind            = {_toml_string(manifest.kind)}",
        f"narrator_engine = {_toml_string(manifest.narrator_engine)}",
        f"language        = {_toml_string(manifest.language)}",
        f"sample_rate     = {manifest.sample_rate}",
    ]
    if has_band or packs:
        lines += ["", "[voice.pace]"]
        if has_band:
            lines.append(f"basis              = {_toml_string(pace_basis)}")
            lines.append(
                "pace_chars_per_sec = "
                + _toml_number(pace.pace_chars_per_sec)
            )
            lines.append(
                "max_chars_per_sec  = " + _toml_number(pace.max_chars_per_sec)
            )
            lines.append(
                "min_chars_per_sec  = " + _toml_number(pace.min_chars_per_sec)
            )
            if pace_basis == "measured":
                lines.append(
                    f"measured_from      = {_toml_string(measured_from)}"
                )
            else:
                lines.append(
                    f"inherited_from     = {_toml_string(inherited_from)}"
                )
        for key in ("target_chars", "safe_min_chars", "safe_max_chars"):
            value = getattr(pace, key)
            if value is not None:
                lines.append(f"{key:<18} = {value}")
    elif uncertified:
        lines += [
            "",
            "# NO [voice.pace]: nothing was measured on these weights, and this",
            "# file says so by omitting the table rather than by copying a",
            "# predecessor's numbers (PHASE18 section 4.1).",
        ]

    dropped: list[str] = []
    if manifest.serving is not None:
        dropped.append(
            f"[voice.serving] max_num_seqs = {manifest.serving.max_num_seqs} "
            "-> config.toml [tts."
            f"{manifest.narrator_engine}] max_num_seqs (+ its note)"
        )
    for arm in sorted(manifest.backends):
        spec = manifest.backends[arm]
        lines += ["", f"[voice.arms.{arm}]"]
        lines.append(f"max_chars       = {spec.max_chars}")
        lines.append(f"max_chars_basis = {_toml_string(max_chars_basis)}")
        sampling = ", ".join(
            f"{key} = {_toml_number(value)}" for key, value in spec.sampling.items()
        )
        lines.append(f"sampling        = {{ {sampling} }}")
        if spec.sampling_reason is not None:
            lines.append(
                f"sampling_reason = {_toml_string(spec.sampling_reason)}"
            )
        if isinstance(spec.clips, str):
            lines.append(f"clips           = {_toml_string(spec.clips)}")
        elif spec.clips:
            for clip in spec.clips:
                lines += [
                    "",
                    f"[[voice.arms.{arm}.clips]]",
                    f"file       = {_toml_string(clip.file)}",
                    f"transcript = {_toml_string(clip.transcript)}",
                    "seconds    = " + _toml_number(clip.seconds),
                ]
        dropped.append(
            f"[voice.backends.{arm}] memory_bytes_estimate = "
            f"{spec.memory_bytes_estimate}, estimate_basis = "
            f"{spec.estimate_basis!r} -> config.toml [tts."
            f"{manifest.narrator_engine}]"
        )
        if spec.hf_repo is not None:
            dropped.append(
                f"[voice.backends.{arm}] {spec.hf_repo}@{spec.revision} -> "
                f"pins.toml [{manifest.id}]"
            )
        if spec.path is not None:
            dropped.append(
                f"[voice.backends.{arm}] path = {spec.path} (identity "
                f"{spec.identity!r}) -> this is a PHASE18 local voice and has no "
                "repo to carry a manifest; it stays a `PUT /v1/voices/{id}` "
                "override"
            )

    # THE EXPORTED PACE IS READ BACK BY THE LOADER'S OWN CHECKER before this
    # function returns anything. There is one thing `voices/<id>.toml` can state
    # that this file has no field for — `edges = "percentile"`, which is how a
    # band read off a distribution says its lopsidedness is real — because
    # `Pace` deliberately does not carry it (it never reaches the wire). So such
    # a voice would export to a file the loader refuses, and the person
    # converting it has to hear that HERE rather than at the first pull.
    from .voices import _check_pace

    if has_band or packs:
        table = {
            key: value
            for key, value in (
                ("pace_chars_per_sec", pace.pace_chars_per_sec),
                ("max_chars_per_sec", pace.max_chars_per_sec),
                ("min_chars_per_sec", pace.min_chars_per_sec),
                ("target_chars", pace.target_chars),
                ("safe_min_chars", pace.safe_min_chars),
                ("safe_max_chars", pace.safe_max_chars),
            )
            if value is not None
        }
        try:
            _check_pace(f"{manifest.id} [voice.pace]", table)
        except VoiceError as exc:
            raise CardError(
                f"the file this would write is one the loader refuses: {exc}. If "
                "that band's edges really came off a distribution, add "
                'edges = "percentile" to the exported [voice.pace] by hand — '
                "`Pace` does not carry that statement, so this command cannot "
                "carry it across"
            ) from exc

    for take in manifest.takes:
        lines += ["", "[[voice.takes]]"]
        for key, value in sorted(take.overrides.items()):
            lines.append(f"{key} = {_toml_number(value)}")
        if take.reason is not None:
            lines.append(f"reason = {_toml_string(take.reason)}")
    return "\n".join(lines) + "\n", dropped


__all__ = [
    "CardError",
    "FM_OWNED",
    "REPO_MANIFEST_NAME",
    "RETIRED_FM_KEYS",
    "export_manifest",
    "read_frontmatter",
    "render_card",
    "render_frontmatter",
    "render_limits",
]
