from __future__ import annotations

import re

from .voicerepo import REPO_MANIFEST_NAME, REPO_SCHEMA, RepoManifest
from .voices import (
    MAX_CHARS_BASES,
    PACE_BASES,
    Pace,
    Take,
    VoiceBackendSpec,
    VoiceError,
    VoiceManifest,
    check_pace,
)

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

_CAP_KEYS: dict[str, str] = {
    "higgs_max_chars_served": "cuda-linux",
    "higgs_max_chars_mlx": "mlx-darwin",
}

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
_LIMITS_SECTION = re.compile(r"## (?:Measured|Recorded) limits.*?(?=\n## |\Z)", re.S)


class CardError(VoiceError):
    ...


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
        if block is None or "max_chars" not in block:
            continue
        owned.append(f"{key}: {block['max_chars']}")
        owned.append(f"{key}_basis: {repo.max_chars_basis[arm]}")
    return "\n".join(keep + owned)


def render_limits(repo: RepoManifest) -> str:
    lines = [
        "## Measured limits (from this repo's crucible-voice.toml)",
        "",
        *_pace_limit_lines(repo),
        *_cap_limit_lines(repo),
    ]
    lines.append(
        "- **Sampling:** "
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
    return "\n".join(lines) + "\n"


def _pace_limit_lines(repo: RepoManifest) -> list[str]:
    pace = repo.pace or {}
    lines: list[str] = []
    rate = pace.get("pace_chars_per_sec")
    if rate is None:
        lines.append(
            "- **Pace:** not measured on these weights. This voice states no "
            "pace band, so a client packs to the per-chunk cap below if this "
            "repo states one, and narrator guards against its engine's own "
            "default band rather than against a number measured here."
        )
    else:
        lines.append(
            f"- **Pace:** {rate} characters of text per second of audio "
            f"({repo.pace_basis}), with the band running "
            f"{pace['min_chars_per_sec']} to {pace['max_chars_per_sec']}."
        )
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
    return lines


def _cap_limit_lines(repo: RepoManifest) -> list[str]:
    lines: list[str] = []
    for arm in sorted(repo.arms):
        block = repo.arms[arm]
        if "max_chars" not in block:
            lines.append(
                f"- **Per-chunk cap, {arm}:** not measured on these weights. "
                "No sweep has run here, so this voice states no cap and nothing "
                "downstream substitutes one — Crucible's render door does not "
                "refuse a chunk by length."
            )
            continue
        lines.append(
            f"- **Per-chunk cap, {arm}:** {block['max_chars']} characters "
            f"({repo.max_chars_basis[arm]})."
        )
    return lines


def render_card(repo: RepoManifest, existing: str) -> tuple[str, bool]:
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


def _toml_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _toml_string(value: str) -> str:
    if "\n" in value or len(value) > 90:
        escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'"""{escaped}"""'
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def export_manifest(
    manifest: VoiceManifest,
    *,
    pace_basis: str | None,
    measured_from: str | None,
    inherited_from: str | None,
    max_chars_basis: str | None,
    uncertified: bool,
) -> tuple[str, list[str]]:
    pace = manifest.pace
    has_band = pace.pace_chars_per_sec is not None
    packs = pace.target_chars is not None or pace.safe_min_chars is not None
    if not has_band and not uncertified:
        raise CardError(
            f"voice {manifest.id!r} states no pace band, so the file this would "
            "write is an UNCERTIFIED voice (docs/internals/voices.md, \"The voice schema\") — a real state, "
            "and one worth stating on purpose. Pass --uncertified to say so"
        )
    if has_band or packs:
        _check_pace_basis(manifest.id, pace_basis, measured_from, inherited_from)
    _check_cap_basis(manifest, max_chars_basis)
    if has_band or packs:
        _check_exported_pace(manifest)
    lines = _header_lines(manifest)
    if has_band or packs:
        lines += _exported_pace_lines(pace, pace_basis, measured_from, inherited_from)
    elif uncertified:
        lines += _UNCERTIFIED_LINES
    for arm in sorted(manifest.backends):
        lines += _arm_lines(arm, manifest.backends[arm], max_chars_basis)
    lines += _take_lines(manifest.takes)
    return "\n".join(lines) + "\n", _dropped_machine_rows(manifest)


_UNCERTIFIED_LINES = [
    "",
    "# NO [voice.pace]: nothing was measured on these weights, and this",
    "# file says so by omitting the table rather than by copying a",
    "# predecessor's numbers.",
]


def _check_pace_basis(
    voice_id: str,
    pace_basis: str | None,
    measured_from: str | None,
    inherited_from: str | None,
) -> None:
    if pace_basis not in PACE_BASES:
        raise CardError(
            f"voice {voice_id!r} has a pace band and the packaged schema "
            "cannot say how it was got. Pass --pace-basis "
            f"{'|'.join(sorted(PACE_BASES))}: an inherited pace is "
            "indistinguishable from a measured one at the point of use, "
            "which is exactly how deathstalker's 16.64 survived onto weights "
            "that measured 15.91"
        )
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


def _check_cap_basis(manifest: VoiceManifest, max_chars_basis: str | None) -> None:
    capped = any(spec.max_chars is not None for spec in manifest.backends.values())
    if capped and max_chars_basis not in MAX_CHARS_BASES:
        raise CardError(
            f"voice {manifest.id!r}'s per-arm caps carry no basis and the "
            "packaged schema cannot say. Pass --max-chars-basis "
            f"{'|'.join(sorted(MAX_CHARS_BASES))}: thirdreich shipped a "
            "`higgs_max_chars_mlx: 900` that no sweep ever produced, and a file "
            "that cannot say so ships it as a measured fact"
        )
    if not capped and max_chars_basis is not None:
        raise CardError(
            f"voice {manifest.id!r} states no per-arm cap on any arm, and "
            f"--max-chars-basis {max_chars_basis} describes how a cap was got. "
            "There is no cap; drop the flag"
        )


def _check_exported_pace(manifest: VoiceManifest) -> None:
    table = {
        key: value for key, value in manifest.pace.to_dict().items() if value is not None
    }
    try:
        check_pace(f"{manifest.id} [voice.pace]", table)
    except VoiceError as exc:
        raise CardError(
            f"the file this would write is one the loader refuses: {exc}. If "
            "that band's edges really came off a distribution, add "
            'edges = "percentile" to the exported [voice.pace] by hand — '
            "`Pace` does not carry that statement, so this command cannot "
            "carry it across"
        ) from exc


def _header_lines(manifest: VoiceManifest) -> list[str]:
    return [
        f"# {manifest.display} — exported from this build's "
        f"{manifest.path.name} by `crucible voices export`.",
        "#",
        "# The MACHINE facts that file carried are not here: they belong to the",
        "# box that serves the voice, in its config.toml `[tts.<engine>]` table",
        "# (docs/internals/voices.md, \"The repo manifest\"). The repo and revision are not here either —",
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


def _exported_pace_lines(
    pace: Pace,
    pace_basis: str | None,
    measured_from: str | None,
    inherited_from: str | None,
) -> list[str]:
    lines = ["", "[voice.pace]"]
    if pace.pace_chars_per_sec is not None:
        lines.append(f"basis              = {_toml_string(pace_basis)}")
        lines.append("pace_chars_per_sec = " + _toml_number(pace.pace_chars_per_sec))
        lines.append("max_chars_per_sec  = " + _toml_number(pace.max_chars_per_sec))
        lines.append("min_chars_per_sec  = " + _toml_number(pace.min_chars_per_sec))
        if pace_basis == "measured":
            lines.append(f"measured_from      = {_toml_string(measured_from)}")
        else:
            lines.append(f"inherited_from     = {_toml_string(inherited_from)}")
    for key in ("target_chars", "safe_min_chars", "safe_max_chars"):
        value = getattr(pace, key)
        if value is not None:
            lines.append(f"{key:<18} = {value}")
    return lines


def _arm_lines(
    arm: str, spec: VoiceBackendSpec, max_chars_basis: str | None
) -> list[str]:
    lines = ["", f"[voice.arms.{arm}]"]
    if spec.max_chars is not None:
        lines.append(f"max_chars       = {spec.max_chars}")
        lines.append(f"max_chars_basis = {_toml_string(max_chars_basis)}")
    sampling = ", ".join(
        f"{key} = {_toml_number(value)}" for key, value in spec.sampling.items()
    )
    lines.append(f"sampling        = {{ {sampling} }}")
    if spec.sampling_reason is not None:
        lines.append(f"sampling_reason = {_toml_string(spec.sampling_reason)}")
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
    return lines


def _take_lines(takes: tuple[Take, ...]) -> list[str]:
    lines: list[str] = []
    for take in takes:
        lines += ["", "[[voice.takes]]"]
        for key, value in sorted(take.overrides.items()):
            lines.append(f"{key} = {_toml_number(value)}")
        if take.reason is not None:
            lines.append(f"reason = {_toml_string(take.reason)}")
    return lines


def _dropped_machine_rows(manifest: VoiceManifest) -> list[str]:
    engine = manifest.narrator_engine
    dropped: list[str] = []
    if manifest.serving is not None:
        dropped.append(
            f"[voice.serving] max_num_seqs = {manifest.serving.max_num_seqs} "
            f"-> config.toml [tts.{engine}] max_num_seqs (+ its note)"
        )
        for key in ("mem_fraction", "context_length"):
            value = getattr(manifest.serving, key)
            if value is not None:
                dropped.append(
                    f"[voice.serving] {key} = {value} -> config.toml "
                    f"[tts.{engine}] {key} (+ its note)"
                )
    for arm in sorted(manifest.backends):
        dropped += _dropped_arm_rows(manifest, arm)
    return dropped


def _dropped_arm_rows(manifest: VoiceManifest, arm: str) -> list[str]:
    spec = manifest.backends[arm]
    dropped = [
        f"[voice.backends.{arm}] memory_bytes_estimate = "
        f"{spec.memory_bytes_estimate}, estimate_basis = "
        f"{spec.estimate_basis!r} -> config.toml [tts."
        f"{manifest.narrator_engine}]"
    ]
    if spec.hf_repo is not None:
        dropped.append(
            f"[voice.backends.{arm}] {spec.hf_repo}@{spec.revision} -> "
            f"pins.toml [{manifest.id}]"
        )
    if spec.path is not None:
        dropped.append(
            f"[voice.backends.{arm}] path = {spec.path} (identity "
            f"{spec.identity!r}) -> this is a local (path + identity) voice and has no "
            "repo to carry a manifest; it stays a `PUT /v1/voices/{id}` "
            "override"
        )
    return dropped


__all__ = [
    "CardError",
    "FM_OWNED",
    "REPO_MANIFEST_NAME",
    "export_manifest",
    "read_frontmatter",
    "render_card",
    "render_frontmatter",
    "render_limits",
]
