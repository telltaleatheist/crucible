"""Which model serves a verb's request: the sizing ruling's precedence
(docs/VERB-SIZING.md section 1a.3), applied at the door of a verb that names no engine.

First that applies: an exact `model` in the request, then a `max_params_b` ceiling in the
request, then the user's per-verb choice in Settings, then Crucible's own pick. The last
two are what this server registered (its capability record's row, which holds the
Settings choice), so the door reads the row. A ceiling caps that pick and never raises it
(Owen, 2026-10-11): the server's own pick is kept when it is at or below the ceiling, and
only a pick above it is replaced by the biggest that fits at or below the ceiling, decided
again against this card with the goal lowered to it. A model and a ceiling in one request
are refused: they are two answers to the one question. A named model the estimate says
does not fit is refused with the verb's own pick named (only a model the USER configured
in Settings is tried past the estimate, section 1a.4). A model in an optional package this
server has not installed is refused by name, whichever way it was reached.

Every refusal here is the caller's or the operator's to repair, so each one is by name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .capabilityclasses import BY_NAME, CLASSES, CapabilityClass, Goal
from .cardfacts import card_for
from .classnames import CHAT_VERBS, DEFAULT_CHAT_VERB
from .errors import ApiError
from .fit import Candidate
from .manifests import ManifestError, UnknownForm, load_manifest
from .memorybudget import available_bytes, gib_text
from .packages import package_not_installed
from .verdict import decide_capabilities

MODEL = "model"
CEILING = "max_params_b"
REGISTERED = "registered"


@dataclass(frozen=True)
class VerbModel:
    """The model a request is served by, and which rule chose it."""

    model: str
    form: str | None
    chosen_by: str
    """`model` (the request named it), `max_params_b` (the request's ceiling replaced a
    registered pick above it) or `registered` (this server's record: the Settings choice
    or Crucible's pick, with or without a ceiling it was already within)."""


SIZED_VERBS: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.goal is not None and entry.candidates is not None
)
"""The verbs a request may address by name instead of a model, and cap with
`max_params_b`: every class with a size goal (docs/VERB-SIZING.md rule 2)."""


VERB_KEY = "verb"


def take_verb(body: dict[str, Any]) -> str | None:
    """A raw body's `verb`, taken off it (it is Crucible's, never forwarded); None when it
    is absent or null."""
    verb = body.pop(VERB_KEY, None)
    if verb is None:
        return None
    if not isinstance(verb, str) or verb == "":
        raise ApiError(
            400,
            "invalid_request",
            f'"verb" names the verb whose model serves this request, a non-empty string; '
            f"got {verb!r}. Leave it out for {DEFAULT_CHAT_VERB!r}",
            {"verb": verb},
        )
    return verb


def take_ceiling(body: dict[str, Any]) -> float | None:
    """A raw body's `max_params_b`, taken off it; None when it is absent or null."""
    ceiling = body.pop(CEILING, None)
    if ceiling is None:
        return None
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or not ceiling > 0:
        raise ApiError(
            400,
            "invalid_request",
            f'"max_params_b" is a ceiling in billions of parameters, a number above 0; '
            f"got {ceiling!r}",
            {"max_params_b": ceiling},
        )
    return float(ceiling)


def refuse_unsized_verb(verb: str, where: str) -> None:
    """A verb a request names to be served by must be one of SIZED_VERBS."""
    if verb in SIZED_VERBS:
        return
    if verb in BY_NAME:
        raise ApiError(
            400,
            "verb_not_sized",
            f"{where} names {verb!r}, which has no size goal, so there is nothing for a "
            f"verb or a ceiling to choose among; the verbs that have one are "
            f"{list(SIZED_VERBS)}. Name a `model` instead",
            {"verb": verb, "verbs": list(SIZED_VERBS)},
        )
    raise ApiError(
        400,
        "unknown_verb",
        f"{where} names {verb!r}, which is not a verb this server knows; the verbs a "
        f"request may name are {list(SIZED_VERBS)}",
        {"verb": verb, "verbs": list(SIZED_VERBS)},
    )


def refuse_unchattable_verb(verb: str) -> None:
    """A chat that names a verb names one whose work is a chat (classnames.CHAT_VERBS)."""
    if verb not in BY_NAME:
        raise ApiError(
            400,
            "unknown_verb",
            f"this chat's `verb` is {verb!r}, which is not a verb this server knows; the "
            f"verbs a chat may name are {list(CHAT_VERBS)}",
            {"verb": verb, "verbs": list(CHAT_VERBS)},
        )
    if verb in CHAT_VERBS:
        return
    raise ApiError(
        400,
        "verb_not_for_chat",
        f"this chat's `verb` is {verb!r}, which is not served through the chat door; the "
        f"verbs a chat may name are {list(CHAT_VERBS)}",
        {"verb": verb, "verbs": list(CHAT_VERBS)},
    )


def _candidate_ids(entry: CapabilityClass, backend_kind: str) -> list[str]:
    assert entry.candidates is not None
    return [candidate.id for candidate in entry.candidates(backend_kind)]


def _refuse_foreign(entry: CapabilityClass, model: str, backend_kind: str) -> None:
    served = _candidate_ids(entry, backend_kind)
    if model in served:
        return
    raise ApiError(
        400,
        "model_not_for_verb",
        f"{model!r} does not serve {entry.name} on this {backend_kind} server; the models "
        f"that do are {served}",
        {"model": model, "verb": entry.name, "models": served},
    )


def _refuse_uninstalled_package(entry: CapabilityClass, config: Any, model: str) -> None:
    try:
        package = load_manifest(model).package
    except ManifestError:
        return
    if package is not None and package not in config.packages:
        raise package_not_installed(model, package, f"{entry.plainly} with it")


def _budget(config: Any, backend: Any) -> int:
    return available_bytes(backend.gpu.vram_bytes, config.desktop_allowance_bytes)


def _refuse_too_big(
    entry: CapabilityClass, config: Any, backend: Any, model: str, form: str | None
) -> None:
    """A named model whose estimate does not fit this card is refused naming the verb's
    own pick, unless it is the model the user chose for the verb in Settings."""
    if config.local_model(entry.name) == model:
        return
    manifest = load_manifest(model)
    try:
        candidate = Candidate.of(manifest, backend.kind, form)
    except UnknownForm as exc:
        raise ApiError(400, exc.code, str(exc), exc.details()) from None
    budget = _budget(config, backend)
    if candidate.holds(entry.work, budget):
        return
    row = None if config.capability is None else config.capability.row(entry.name)
    pick = row.selected if row is not None and row.enabled else None
    need = candidate.need_bytes(entry.work)
    raise ApiError(
        409,
        "model_does_not_fit",
        f"{model!r} needs {gib_text(need)} for {entry.name} and this server's card gives "
        f"a model {gib_text(budget)}, so it is not tried for a request; "
        + (
            f"{entry.name} here is served by {pick!r} (send no model, or that one)"
            if pick
            else f"nothing is registered for {entry.name} here"
        )
        + ". A model a person chooses for the verb in Settings is tried past the estimate",
        {"model": model, "need_bytes": need, "available_bytes": budget, "registered": pick},
    )


def _ceiling_pick(
    entry: CapabilityClass, config: Any, backend: Any, ceiling: float
) -> str:
    """The verb's automatic pick on this card with its goal lowered to the request's
    ceiling (rule 8): the biggest that fits at or below it."""
    assert entry.goal is not None and entry.candidates is not None
    sizes = sorted({c.params_b for c in entry.candidates(backend.kind) if c.params_b is not None})
    if not sizes or sizes[0] > ceiling:
        raise ApiError(
            409,
            "nothing_fits_ceiling",
            f"nothing serves {entry.name} at or below {ceiling:g}B on this server: the "
            + (
                f"smallest model that does is {sizes[0]:g}B"
                if sizes
                else f"build ships no model for it on {backend.kind}"
            ),
            {"verb": entry.name, "max_params_b": ceiling},
        )
    lowered = replace(
        entry,
        goal=Goal(
            params_b=min(ceiling, entry.goal.params_b),
            source=f"the request's max_params_b, {ceiling:g}",
        ),
    )
    decision = decide_capabilities(
        lowered,
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=backend.gpu.vendor,
        chosen=None,
        audio_low_vram=config.audio_low_vram,
        card=card_for(config.home, backend.gpu),
        packages=config.packages,
    )
    if not decision.enabled:
        raise ApiError(
            409,
            "nothing_fits_ceiling",
            f"nothing serves {entry.name} at or below {ceiling:g}B on this server: "
            f"{decision.reason}",
            {"verb": entry.name, "max_params_b": ceiling},
        )
    return decision.selected


def _row(entry: CapabilityClass, config: Any) -> Any:
    """The verb's row in this server's capability record; refused by name when there is
    none (no record, or one written before the class existed)."""
    record = config.capability
    row = None if record is None else record.row(entry.name)
    if row is None:
        raise ApiError(
            503,
            "capability_undecided",
            f"this request names no model, and this server has registered nothing for "
            f"{entry.name}: its capability record "
            + ("does not exist" if record is None else f"predates the {entry.name} class")
            + ". Run `crucible capability --write`, or name a `model`",
            {"capability": entry.name},
        )
    return row


def _registered(entry: CapabilityClass, config: Any) -> str:
    record = config.capability
    row = None if record is None else record.row(entry.name)
    if entry.package is not None and entry.package not in config.packages and (
        row is None or not row.enabled
    ):
        raise ApiError(
            409,
            "package_not_installed",
            f"this server cannot {entry.plainly}: its models are the optional "
            f"{entry.package} package, which it has not installed. `crucible install "
            f"{entry.package}` on the server pulls them and turns it on"
            + (
                "; meanwhile a request may name a decide model (`model`) to rerank with"
                if entry.name == "rerank"
                else ""
            ),
            {"verb": entry.name, "package": entry.package},
        )
    row = _row(entry, config)
    if not row.enabled:
        raise ApiError(
            409,
            "capability_disabled",
            f"this request names no model, and this server cannot {entry.plainly}: "
            f"{row.reason}",
            {"capability": entry.name, "shortfall_bytes": row.shortfall_bytes},
        )
    return row.selected


def _params_b(entry: CapabilityClass, model: str) -> float:
    """How many billions of parameters a registered model has, from its manifest."""
    try:
        return load_manifest(model).params_b
    except ManifestError as exc:
        raise ApiError(
            503,
            "capability_stale",
            f"this server registered {model!r} for {entry.name}, and this build has no "
            f"manifest for it ({exc}). Run `crucible capability --write` to decide the card "
            "again",
            {"capability": entry.name, "model": model},
        ) from None


def _within_ceiling(entry: CapabilityClass, config: Any, ceiling: float) -> str | None:
    """This server's own pick for the verb (the Settings choice, else Crucible's) when it
    is at or below the request's ceiling, which caps the pick and never raises it (Owen,
    2026-10-11). None when it is above the ceiling, or when the server's row for the verb
    is off (nothing of its own serves it, so the ceiling's pick is all there is)."""
    row = _row(entry, config)
    if not row.enabled:
        return None
    if _params_b(entry, row.selected) <= ceiling:
        return row.selected
    return None


def registered(verb: str, config: Any) -> str:
    """The model this server registered for `verb` (its capability record's row: the
    Settings choice, else Crucible's own pick), for a request that names none; refused by
    name when nothing is registered. The door a job type with no ceiling reads (the
    `image` verb, crucible/jobs/__init__.py resolve_model)."""
    return _registered(BY_NAME[verb], config)


def resolve(
    verb: str,
    config: Any,
    backend: Any,
    *,
    model: str | None,
    form: str | None,
    max_params_b: float | None,
    resident: str | None,
) -> VerbModel:
    """The model this request is served by, every refusal made before anything waits.
    `resident` is the model on the card now: one that is loaded is held, whatever its
    estimate says. `verb` is one of SIZED_VERBS (refuse_unsized_verb, at the door)."""
    entry = BY_NAME[verb]
    if model is not None and max_params_b is not None:
        raise ApiError(
            400,
            "model_with_ceiling",
            f"this request names a model ({model!r}) and a ceiling (max_params_b "
            f"{max_params_b:g}); they are two ways to choose the model, so send one: the "
            "model to be served by exactly it, or the ceiling to be served by this "
            "server's pick for the verb, capped at that size",
            {"model": model, "max_params_b": max_params_b},
        )
    if model is not None:
        _refuse_foreign(entry, model, backend.kind)
        _refuse_uninstalled_package(entry, config, model)
        if model != resident:
            _refuse_too_big(entry, config, backend, model, form)
        return VerbModel(model=model, form=form, chosen_by=MODEL)
    if form is not None:
        raise ApiError(
            400,
            "form_without_model",
            "`form` names a form of a named `model`; send the model with it, or neither "
            "to be served by what this server registered",
            {"form": form},
        )
    if max_params_b is not None:
        kept = _within_ceiling(entry, config, max_params_b)
        if kept is not None:
            _refuse_uninstalled_package(entry, config, kept)
            return VerbModel(model=kept, form=None, chosen_by=REGISTERED)
        picked = _ceiling_pick(entry, config, backend, max_params_b)
        _refuse_uninstalled_package(entry, config, picked)
        return VerbModel(model=picked, form=None, chosen_by=CEILING)
    picked = _registered(entry, config)
    _refuse_uninstalled_package(entry, config, picked)
    return VerbModel(model=picked, form=None, chosen_by=REGISTERED)


__all__ = [
    "CEILING",
    "MODEL",
    "REGISTERED",
    "SIZED_VERBS",
    "VerbModel",
    "VERB_KEY",
    "refuse_unchattable_verb",
    "refuse_unsized_verb",
    "registered",
    "resolve",
    "take_ceiling",
    "take_verb",
]
