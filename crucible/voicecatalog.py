from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from . import voicerefs, voicerepo
from .config import crucible_home
from .narratorengines import NARRATOR_ENGINE_SAMPLING
from .voices import (
    MANIFEST_OVERRIDE,
    MANIFEST_REPO,
    PINS_FILE,
    VoiceError,
    VoiceManifest,
    engine_voices_path,
    home_voices_dir,
    parse_engine_base,
    parse_voice,
    voice_dirs,
    voices_dir,
    voices_dir_is_overridden,
)


def load_all_voices(directory: Path | None = None) -> dict[str, VoiceManifest]:
    if directory is not None:
        return resolve_weights_of(dict(sorted(voices_in(directory).items())))
    voices: dict[str, VoiceManifest] = {}
    voices.update(voicerepo.pinned_voices())
    if not voices_dir_is_overridden():
        voices.update(engine_voices())
    for root in voice_dirs():
        voices.update(voices_in(root))
    return resolve_weights_of({vid: voices[vid] for vid in sorted(voices)})


def load_voice(voice_id: str, directory: Path | None = None) -> VoiceManifest:
    if directory is not None:
        found = load_voice_file(directory / f"{voice_id}.toml", voice_id)
        if found.weights_of is None:
            return found
        return load_all_voices(directory)[voice_id]
    served = load_all_voices()
    found = served.get(voice_id)
    if found is None:
        unserved = unserved_pins().get(voice_id)
        if unserved is not None:
            raise VoiceError(unserved[1])
        where = ", ".join(str(r) for r in voice_dirs()) or str(home_voices_dir())
        raise VoiceError(
            f"no manifest for voice {voice_id!r} in {where}; this host serves "
            f"{sorted(served)}"
        )
    return found


def unserved_pins() -> dict[str, tuple[str, str]]:
    _, refused = voicerepo.load_pinned()
    served = load_all_voices()
    return {
        voice_id: (pin.revision, why)
        for voice_id, (pin, why) in refused.items()
        if voice_id not in served
    }


def voice_aliases_of(base: VoiceManifest) -> tuple[VoiceManifest, ...]:
    if base.weights_of is not None:
        return ()
    return tuple(v for v in load_all_voices().values() if v.weights_of == base.id)


def declared_voice_ids() -> set[str]:
    return {*voicerepo.packaged_pins(), *engine_voices(), *voices_in(voices_dir())}


def declared_voice_backends(voice_id: str, home: Path | None = None) -> list[str]:
    shipped = {**engine_voices(), **voices_in(voices_dir())}
    if voice_id in shipped:
        return sorted(shipped[voice_id].backends)
    pin = voicerepo.packaged_pins().get(voice_id)
    if pin is None:
        return []
    text, where = voicerepo.fetch_repo_manifest(
        crucible_home() if home is None else home, pin
    )
    return sorted(voicerepo.parse_repo_manifest(text, where).arms)


def engine_voices() -> dict[str, VoiceManifest]:
    found: dict[str, VoiceManifest] = {}
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        path = engine_voices_path(engine)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VoiceError(f"could not read {path}: {exc}") from exc
        found.update(parse_engine_base(text, path, engine))
    return found


def voices_in(root: Path) -> dict[str, VoiceManifest]:
    voices: dict[str, VoiceManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        if path.name == PINS_FILE:
            continue
        voices[path.stem] = replace(
            load_voice_file(path, path.stem), manifest_source=MANIFEST_OVERRIDE
        )
    return voices


def load_voice_file(path: Path, voice_id: str) -> VoiceManifest:
    if not path.is_file():
        known = (
            sorted(p.stem for p in path.parent.glob("*.toml"))
            if path.parent.is_dir()
            else []
        )
        raise VoiceError(
            f"no manifest for voice {voice_id!r} at {path}; that directory holds "
            f"{known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoiceError(f"could not read {path}: {exc}") from exc
    return parse_voice(text, path, voice_id)


def resolve_weights_of(voices: dict[str, VoiceManifest]) -> dict[str, VoiceManifest]:
    for voice in voices.values():
        if voice.weights_of is not None:
            _check_shared_base(voice, voices.get(voice.weights_of))
    resolved = {
        voice_id: (
            voice
            if voice.weights_of is None
            else replace(voice, weights_base=voices[voice.weights_of])
        )
        for voice_id, voice in voices.items()
    }
    aliases: dict[str, list[VoiceManifest]] = {}
    for voice in resolved.values():
        if voice.weights_of is not None:
            aliases.setdefault(voice.weights_of, []).append(voice)
    for base_id, sharing in aliases.items():
        resolved[base_id] = replace(resolved[base_id], weights_aliases=tuple(sharing))
    return resolved


def _check_shared_base(voice: VoiceManifest, base: VoiceManifest | None) -> None:
    where = voice.path.name
    if base is None:
        raise VoiceError(
            f"{where}: weights_of_unknown: voice {voice.id!r} shares "
            f"{voice.weights_of!r}, which this host does not serve"
        )
    if base.weights_of is not None:
        raise VoiceError(
            f"{where}: weights_of_chain: {base.id!r} itself shares {base.weights_of!r}"
        )
    for kind, spec in sorted(voice.backends.items()):
        base_spec = base.backends.get(kind)
        if base_spec is None:
            raise VoiceError(
                f"{where}: weights_of_backend_missing: {base.id!r} has no {kind} block"
            )
        if (spec.hf_repo, spec.revision) != (base_spec.hf_repo, base_spec.revision):
            raise VoiceError(
                f"{where}: weights_of_pin_mismatch on {kind}: "
                f"{spec.hf_repo}@{spec.revision} here, "
                f"{base_spec.hf_repo}@{base_spec.revision} in {base.id!r}"
            )


def following_pin(voice_id: str) -> voicerepo.Pin | None:
    pin = voicerepo.load_pins().get(voice_id)
    return pin if pin is not None and pin.follows_ref else None


def served_revision_of(voice: VoiceManifest) -> str | None:
    revisions = {spec.revision for spec in voice.backends.values()}
    return next(iter(revisions)) if len(revisions) == 1 else None


def refresh_ref(home: Path, voice_id: str) -> voicerefs.RefCheck | None:
    pin = following_pin(voice_id)
    if pin is None:
        return None
    assert pin.ref is not None
    return voicerefs.check(home, pin.hf_repo, pin.ref)


def refresh_unresolved(home: Path, voice_id: str) -> voicerefs.RefCheck | None:
    pin = following_pin(voice_id)
    if pin is None:
        return None
    assert pin.ref is not None
    if voicerefs.served_revision(home, pin.id, pin.hf_repo, pin.ref) is not None:
        return None
    return voicerefs.check(home, pin.hf_repo, pin.ref)


def seed_unresolved(home: Path) -> list[voicerefs.RefCheck]:
    return [
        found
        for found in (refresh_unresolved(home, voice_id) for voice_id in sorted(voicerepo.load_pins()))
        if found is not None
    ]


def _follows(voice: VoiceManifest) -> voicerepo.Pin | None:
    if voice.manifest_source != MANIFEST_REPO:
        return None
    return following_pin(voice.id)


def moves_to(home: Path, voice: VoiceManifest) -> str | None:
    pin = _follows(voice)
    if pin is None:
        return None
    assert pin.ref is not None
    latest = voicerefs.cached(home, pin.hf_repo, pin.ref)
    if latest is None or latest.revision is None:
        return None
    return None if latest.revision == served_revision_of(voice) else latest.revision


def pull_target(home: Path, voice: VoiceManifest) -> VoiceManifest:
    revision = moves_to(home, voice)
    if revision is None:
        return voice
    pin = _follows(voice)
    assert pin is not None
    return voicerepo.voice_for_pin(pin.at(revision))


def ref_state(home: Path, voice: VoiceManifest) -> dict[str, Any]:
    pin = _follows(voice)
    if pin is None:
        return {}
    return _ref_fields(home, pin, served_revision_of(voice))


def unserved_ref_state(home: Path, voice_id: str) -> dict[str, Any]:
    pin = following_pin(voice_id)
    return {} if pin is None else _ref_fields(home, pin, None)


def _ref_fields(home: Path, pin: voicerepo.Pin, current: str | None) -> dict[str, Any]:
    assert pin.ref is not None
    latest = voicerefs.cached(home, pin.hf_repo, pin.ref)
    latest_revision = None if latest is None else latest.revision
    return {
        "ref": pin.ref,
        "latest_revision": latest_revision,
        "update_available": latest_revision is not None and latest_revision != current,
        "update_checked_at": None if latest is None else latest.checked_at,
        "update_error": None if latest is None else latest.error,
    }


def check_updates(home: Path) -> list[dict[str, Any]]:
    following = {
        voice_id: pin
        for voice_id, pin in sorted(voicerepo.load_pins().items())
        if pin.follows_ref
    }
    for pin in following.values():
        assert pin.ref is not None
        voicerefs.check(home, pin.hf_repo, pin.ref)
    served = load_all_voices()
    rows: list[dict[str, Any]] = []
    for voice_id, pin in following.items():
        voice = served.get(voice_id)
        current = None if voice is None else served_revision_of(voice)
        rows.append(
            {
                "id": voice_id,
                "hf_repo": pin.hf_repo,
                "revision": current,
                **_ref_fields(home, pin, current),
            }
        )
    return rows


__all__ = [
    "check_updates",
    "declared_voice_backends",
    "declared_voice_ids",
    "engine_voices",
    "following_pin",
    "load_all_voices",
    "load_voice",
    "load_voice_file",
    "moves_to",
    "pull_target",
    "ref_state",
    "refresh_ref",
    "refresh_unresolved",
    "resolve_weights_of",
    "seed_unresolved",
    "served_revision_of",
    "unserved_pins",
    "unserved_ref_state",
    "voice_aliases_of",
    "voices_in",
]
