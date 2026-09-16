"""The settings document, and the one door that changes it.

PHASE15-HOST.md sections 3.1 and 3.2. Owen, 2026-09-14: *"Bookforge and foundry
setup/settings should be able to configure crucible settings. If the user enters
an anthropic api key, it should pass through to crucible … the user shouldn't
have to interact with crucible almost at all but should have access to it if
they want to."*

WHAT THIS MODULE IS FOR
-----------------------
Settings live in the engine and nowhere else. An app is a WINDOW onto them, not
a copy: BookForge's setup page writes the key straight here and reads capability
back, and the app's own settings file never holds one. Two apps and the operator
page can all edit the same server because there is one store — this file, the
config on disk, and nothing between them.

THREE PROPERTIES THE DOOR HAS, AND WHY EACH ONE IS LOAD-BEARING
---------------------------------------------------------------
**A refusal applies NOTHING.** The whole patch is resolved into a candidate
`(routes, upstreams, allowance)` in memory, validated as a whole, and only then
written. A door that applied the upstreams and then refused the routes would
leave a key on a server whose operator believes the request failed — and would
make "configure the upstream AND set the route in one PUT" (which section 5.2
tells the apps to do) a half-transaction.

**A key is write-only.** It goes into the config at 0600 and comes back as
`key_hint`, its last four characters. It is in no response, no log line and no
activity record; `tests/test_settings_api.py` greps for the whole string in all
four.

**A route this server cannot serve is never stored.** `route_upstream_unconfigured`
at the door means `GET /v1/capability` can say `enabled: true` for a routed
class without re-checking, and means the chat door's `upstream_unconfigured`
can only ever be reached by a client naming an upstream model the operator never
routed to — a different mistake, correctly named.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

from . import capability as capability_classes
from . import upstreams as upstream_module
from .config import (
    CapabilityRecord,
    Config,
    LocalModelRecord,
    RouteRecord,
    load_config,
    write_config,
    _advertised,
)
from .errors import ApiError, ConfigError
from .jobs.base import utcnow
from .upstreams import UPSTREAM_NAMES, UpstreamRecord

#: How many settings writes `/v1/activity` remembers. In memory, and a restart
#: forgets — the same rule a task's record follows (PHASE13 3.3: *"a task is not
#: a record anybody keeps"*). This is a bench display of "who changed what just
#: now", not an audit log, and calling it one would promise durability nothing
#: here provides.
HISTORY_LIMIT = 20

#: The top-level keys a patch may carry. Anything else is `invalid_request`
#: naming it, rather than ignored: a caller that sent `desktopAllowanceBytes`
#: believes it changed something.
PATCH_KEYS: frozenset[str] = frozenset(
    {
        "routes",
        "upstreams",
        "local_models",
        "desktop_allowance_bytes",
        "tailscale_advertise",
    }
)


class History:
    """The last few settings writes, for `/v1/activity`. **Never a key.**

    What is recorded is the FIELD PATH that changed, the act the client named
    and the client's own agent string — enough for a person watching two apps
    edit one server to see which of them did it, and nothing that could carry a
    secret. A route's value is recorded because a model id is not one; an
    upstream's value never is, which is why the entry says `set` or `removed`
    rather than what it was set to.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def record(
        self, *, act: str | None, client: str | None, changed: list[str]
    ) -> None:
        row = {
            "at": utcnow(),
            # Null means the client did not say, exactly as it does on a chat
            # row and a job's `client`. Never a guess.
            "act": act,
            "client": client,
            "changed": changed,
        }
        with self._lock:
            self._rows.append(row)
            del self._rows[:-HISTORY_LIMIT]

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._rows))


def local_selection(config: Config, name: str) -> str | None:
    """The local model this class would run on, or None.

    None says one of two things and the document cannot tell them apart:
    nothing fits, or nothing has decided yet (`GET /v1/capability` answers
    `503 capability_undecided`). PHASE15-HOST.md section 3.1 records that
    deliberately — this document has to be readable BEFORE anything probes the
    card, because pasting a key is the first thing an app does, and a window
    that needs the two apart reads capability, which says which by name.
    """
    record = config.capability
    if record is None:
        return None
    row = record.row(name)
    if row is None or row.selected == "":
        return None
    return row.selected


def _choices(
    config: Config, installed: Mapping[str, bool]
) -> dict[str, list[dict[str, Any]]]:
    """What an app may choose from, per class, best-first.

    Computed rather than recorded. `fits` is arithmetic against THIS card and
    `installed` is a fact about THIS disk, and a stored answer to either would
    be wrong the first time the config moved or a download finished.

    `{}` when nothing has decided this host's capability yet, which is the
    true answer then: without a probe there is no backend to list candidates
    for and no budget to measure them against. An app that needs to know WHY
    it is empty reads `/v1/capability`, which says `capability_undecided` by
    name — the same division `local_selection` above explains for its None.
    """
    record = config.capability
    if record is None:
        return {}
    budget = capability_classes.available_bytes(
        record.total_bytes, config.desktop_allowance_bytes
    )
    found: dict[str, list[dict[str, Any]]] = {}
    for name in capability_classes.SELECTABLE_CLASSES:
        entry = capability_classes.BY_NAME[name]
        assert entry.candidates is not None  # SELECTABLE_CLASSES is this
        rows: list[dict[str, Any]] = []
        for candidate in entry.candidates(record.backend_kind):
            if candidate.id not in installed:
                # The catalog and the class table read the SAME manifests, so
                # this cannot happen without one of them being wrong. Saying so
                # beats drawing a chooser with a silent `installed: false` on a
                # model that is sitting on the disk.
                raise ApiError(
                    500,
                    "catalog_incomplete",
                    f"the catalog has no row for {candidate.id!r}, which "
                    f"{name} offers as a candidate; these two read the same "
                    "manifests and must agree",
                    {"capability": name, "model": candidate.id},
                )
            rows.append(
                {
                    "id": candidate.id,
                    "memory_bytes_estimate": candidate.memory_bytes_estimate,
                    "fits": candidate.memory_bytes_estimate <= budget,
                    "installed": installed[candidate.id],
                }
            )
        found[name] = rows
    return found


def document(config: Config, *, installed: Mapping[str, bool]) -> dict[str, Any]:
    """`GET /v1/settings`, and the body every `PUT` answers with.

    The PUT returns this AFTER the write, so a window never has to guess what
    took: it draws the document it was handed and holds no state of its own
    (section 3.7).
    """
    routes: dict[str, Any] = {}
    for name in capability_classes.ROUTABLE_CLASSES:
        model = config.route_model(name)
        if model is None:
            routes[name] = {"route": "local", "model": local_selection(config, name)}
        else:
            routes[name] = {"route": "upstream", "model": model}
    upstreams: dict[str, Any] = {}
    for name in UPSTREAM_NAMES:
        record = config.upstream(name)
        upstreams[name] = (
            upstream_module.blank(name)
            if record is None
            else upstream_module.settings_entry(record)
        )
    return {
        # EVERY selectable class, with null where nobody has chosen. Listing
        # only the chosen ones would make "this class takes the automatic
        # decision" and "this build does not know this class" the same reading.
        "local_models": {
            name: config.local_model(name)
            for name in capability_classes.SELECTABLE_CLASSES
        },
        "local_model_choices": _choices(config, installed),
        "routes": routes,
        "upstreams": upstreams,
        "desktop_allowance_bytes": config.desktop_allowance_bytes,
        "backend_kind": config.backend_kind,
        "tailscale_advertise": list(config.tailscale_advertise),
    }


class Resolved:
    """A validated patch: what the config would become, and what changed.

    Not a dataclass because `changed` is built while resolving and reads better
    appended to than assembled at the end — and nothing outside this module
    constructs one.
    """

    def __init__(self, config: Config) -> None:
        self.upstreams: dict[str, UpstreamRecord] = {
            entry.name: entry for entry in config.upstreams
        }
        self.routes: dict[str, str] = {
            entry.capability: entry.model for entry in config.routes
        }
        self.local_models: dict[str, str] = {
            entry.capability: entry.model for entry in config.local_models
        }
        self.desktop_allowance_bytes = config.desktop_allowance_bytes
        self.tailscale_advertise = config.tailscale_advertise
        self.removed: set[str] = set()
        self.changed: list[str] = []
        self.touched_routes = False

    def as_records(
        self,
    ) -> tuple[
        tuple[RouteRecord, ...],
        tuple[UpstreamRecord, ...],
        tuple[LocalModelRecord, ...],
    ]:
        """The three tables in `config.toml`'s order: class order, then name order.

        A stable order and not insertion order, so a config rewritten twice with
        the same content is byte-identical and a diff of the file says what
        actually changed.
        """
        routes = tuple(
            RouteRecord(capability=name, model=self.routes[name])
            for name in capability_classes.ROUTABLE_CLASSES
            if name in self.routes
        )
        upstreams = tuple(
            self.upstreams[name] for name in UPSTREAM_NAMES if name in self.upstreams
        )
        local_models = tuple(
            LocalModelRecord(capability=name, model=self.local_models[name])
            for name in capability_classes.SELECTABLE_CLASSES
            if name in self.local_models
        )
        return routes, upstreams, local_models


#: The dotted path of the document itself, for a refusal about the WHOLE body.
#: Every refusal this door makes carries `details.field` (PHASE15-HOST.md, and
#: Foundry is built against it), and a body that is not an object has no field
#: inside it to blame — so the root gets a name rather than the key being
#: absent on one refusal out of nine.
ROOT_FIELD = "body"


def _require_object(patch: Any, field: str) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be an object, got {type(patch).__name__}",
            {"field": field},
        )
    return patch


def resolve(config: Config, patch: Any) -> Resolved:
    """A whole patch, applied in memory and validated. Raises rather than half-does.

    The ORDER inside one request is the contract's (section 3.2): upstreams are
    applied, then routes, then the whole is validated. That order is what makes
    *"configure the upstream AND set the route in one PUT"* work, which is
    exactly what an app's AI step does when somebody pastes a key — and it is
    what makes `upstream_in_use` avoidable in the same breath, by re-routing a
    class in the request that removes the key it named.
    """
    body = _require_object(patch, ROOT_FIELD)
    unknown = sorted(set(body) - PATCH_KEYS)
    if unknown:
        raise ApiError(
            400,
            "invalid_request",
            f"unknown settings field(s) {unknown}; this door takes "
            f"{sorted(PATCH_KEYS)}",
            # `field` names the FIRST offender and `unknown` lists them all: a
            # window highlights one control and a log wants the set.
            {"field": unknown[0], "unknown": unknown},
        )
    resolved = Resolved(config)

    if "upstreams" in body:
        table = _require_object(body["upstreams"], "upstreams")
        for name in sorted(table):
            field = f"upstreams.{name}"
            upstream_module.require_name(name, field)
            value = table[name]
            if value is None:
                # Removal. Whether it is ALLOWED is decided below, against the
                # final routes — because the same request may be re-routing the
                # classes that named it, and refusing here would force two
                # round trips for one intention.
                if name in resolved.upstreams:
                    del resolved.upstreams[name]
                    resolved.removed.add(name)
                    resolved.changed.append(f"{field} removed")
                continue
            resolved.upstreams[name] = upstream_module.record_from_patch(
                name, value, field
            )
            resolved.changed.append(f"{field} set")

    if "routes" in body:
        table = _require_object(body["routes"], "routes")
        resolved.touched_routes = True
        for name in sorted(table):
            field = f"routes.{name}"
            if name not in capability_classes.ROUTABLE_CLASSES:
                raise ApiError(
                    400,
                    "route_not_routable",
                    f"{name!r} cannot run anywhere but this server's card. The "
                    f"classes a route may name are "
                    f"{list(capability_classes.ROUTABLE_CLASSES)}; every other "
                    "class is local and there is no upstream that does its kind "
                    "of work",
                    {
                        "field": field,
                        "capability": name,
                        "routable": list(capability_classes.ROUTABLE_CLASSES),
                    },
                )
            value = table[name]
            if not isinstance(value, str) or value == "":
                raise ApiError(
                    400,
                    "invalid_request",
                    f"{field} must be \"local\" or an upstream model id, got "
                    f"{type(value).__name__}",
                    {"field": field},
                )
            if value == "local":
                if name in resolved.routes:
                    del resolved.routes[name]
                    resolved.changed.append(f"{field} = local")
                continue
            upstream_name, _, rest = value.partition("/")
            if upstream_name not in UPSTREAM_NAMES or rest == "":
                raise ApiError(
                    400,
                    "route_bad_model",
                    f"{value!r} is not an upstream model id. A route's value is "
                    f"`<upstream>/<model>` with the upstream one of "
                    f"{list(UPSTREAM_NAMES)}, or the word \"local\"",
                    {
                        "field": field,
                        "model": value,
                        "known": list(UPSTREAM_NAMES),
                    },
                )
            resolved.routes[name] = value
            resolved.changed.append(f"{field} = {value}")

    if "desktop_allowance_bytes" in body:
        value = body["desktop_allowance_bytes"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ApiError(
                400,
                "invalid_request",
                "desktop_allowance_bytes must be a non-negative integer number "
                f"of bytes, got {value!r}",
                {"field": "desktop_allowance_bytes"},
            )
        if value != resolved.desktop_allowance_bytes:
            resolved.desktop_allowance_bytes = value
            resolved.changed.append(f"desktop_allowance_bytes = {value}")

    if "local_models" in body:
        # AFTER the allowance, deliberately. Whether a chosen model fits is
        # arithmetic against a budget THIS SAME PATCH may be changing, and
        # checking the choice first would measure it against a reserve that is
        # about to be gone — the same ordering rule the upstreams/routes pair
        # follows at the top of this function.
        table = _require_object(body["local_models"], "local_models")
        record = config.capability
        for name in sorted(table):
            field = f"local_models.{name}"
            if name not in capability_classes.SELECTABLE_CLASSES:
                raise ApiError(
                    400,
                    "local_model_not_selectable",
                    f"{name!r} has no local models to choose between. The classes "
                    f"that do are "
                    f"{list(capability_classes.SELECTABLE_CLASSES)}",
                    {
                        "field": field,
                        "capability": name,
                        "selectable": list(capability_classes.SELECTABLE_CLASSES),
                    },
                )
            value = table[name]
            if value is None:
                # NULL RESTORES THE AUTOMATIC DECISION. It is not "no model" —
                # that is not a thing an app can ask for — it is the removal of
                # a preference, after which `decide()` walks best-first again.
                if resolved.local_models.pop(name, None) is not None:
                    resolved.changed.append(f"local_models.{name} = automatic")
                continue
            if not isinstance(value, str) or value == "":
                raise ApiError(
                    400,
                    "invalid_request",
                    f"{field} must be a model id, or null for automatic, got "
                    f"{type(value).__name__}",
                    {"field": field},
                )
            if record is None:
                # Nothing has probed this card, so there is no backend to look
                # the id up on and no budget to measure it against. Saying so
                # is the honest answer; accepting the choice unchecked would
                # store a preference this server may never be able to keep.
                raise ApiError(
                    503,
                    "capability_undecided",
                    "This server has not decided its capability yet, so a local "
                    "model cannot be chosen on it. Run `crucible capability "
                    "--write` on the host first",
                    {"field": field, "capability": name},
                )
            entry = capability_classes.BY_NAME[name]
            assert entry.candidates is not None  # SELECTABLE_CLASSES is this
            offered = entry.candidates(record.backend_kind)
            picked = next((c for c in offered if c.id == value), None)
            if picked is None:
                raise ApiError(
                    400,
                    "local_model_unknown",
                    f"{value!r} is not among the {len(offered)} {entry.noun} "
                    f"this build ships for {name} on {record.backend_kind}",
                    {
                        "field": field,
                        "capability": name,
                        "model": value,
                        "choices": [c.id for c in offered],
                    },
                )
            budget = capability_classes.available_bytes(
                record.total_bytes, resolved.desktop_allowance_bytes
            )
            if picked.memory_bytes_estimate > budget:
                # REFUSED WITH THE ARITHMETIC, and nothing is applied. The
                # alternative — storing it and disabling the class — would let
                # an app believe it had configured something that this machine
                # can never run, and the refusal is the only place the numbers
                # can be put in front of whoever chose.
                shortfall = picked.memory_bytes_estimate - budget
                raise ApiError(
                    409,
                    "local_model_does_not_fit",
                    f"{value} needs "
                    f"{picked.memory_bytes_estimate / 2**30:.1f} GiB and there "
                    f"is {budget / 2**30:.1f} GiB available "
                    f"({record.total_bytes / 2**30:.1f} GiB less a "
                    f"{resolved.desktop_allowance_bytes / 2**30:.1f} GiB desktop "
                    f"allowance) — short by {shortfall / 2**30:.1f} GiB",
                    {
                        "field": field,
                        "capability": name,
                        "model": value,
                        "memory_bytes_estimate": picked.memory_bytes_estimate,
                        "available_bytes": budget,
                        "shortfall_bytes": shortfall,
                    },
                )
            # NOT INSTALLED IS NOT A REFUSAL (Owen, 2026-09-16). A choice is a
            # statement of what this app wants to run; preparation is what
            # fetches the weights, and INTENT.md gives the app the choice and
            # Crucible the downloading. Refusing here would force an app to
            # install a model before it was allowed to say it wanted it.
            if resolved.local_models.get(name) != value:
                resolved.local_models[name] = value
                resolved.changed.append(f"local_models.{name} = {value}")

    if "tailscale_advertise" in patch:
        try:
            resolved.tailscale_advertise = _advertised({"server": {"advertise": patch["tailscale_advertise"]}})
        except ConfigError as exc:
            raise ApiError(400, "invalid_request", str(exc), {"field": "tailscale_advertise"}) from exc
        if resolved.tailscale_advertise != config.tailscale_advertise:
            resolved.changed.append("tailscale_advertise")
    _validate(resolved)
    return resolved


def _validate(resolved: Resolved) -> None:
    """The whole, after both halves are applied. The last chance to refuse.

    Two refusals live here and not in the loops above, because both are
    statements about the FINAL document rather than about one field: a route
    may name an upstream the same request configures, and an upstream may be
    removed by a request that also re-routes away from it.
    """
    for name in capability_classes.ROUTABLE_CLASSES:
        model = resolved.routes.get(name)
        if model is None:
            continue
        upstream_name = model.partition("/")[0]
        if upstream_name in resolved.upstreams:
            continue
        if upstream_name in resolved.removed:
            using = sorted(
                other
                for other, value in resolved.routes.items()
                if value.partition("/")[0] == upstream_name
            )
            raise ApiError(
                409,
                "upstream_in_use",
                f"{upstream_name} cannot be removed while "
                f"{', '.join(using)} "
                + ("is" if len(using) == 1 else "are")
                + " routed to it. Re-route "
                + ("it" if len(using) == 1 else "them")
                + " first — in this same request if you like, the routes are "
                "applied after the upstreams",
                {
                    "field": f"upstreams.{upstream_name}",
                    "upstream": upstream_name,
                    # `classes`, the contract's spelling (c5482ff): the
                    # capability CLASSES that name this upstream, which is what
                    # a window lists beside "re-route these first".
                    "classes": using,
                },
            )
        raise ApiError(
            409,
            "route_upstream_unconfigured",
            f"{name} cannot be routed to {model!r}: [upstreams."
            f"{upstream_name}] is not configured on this server. Configure it "
            "in this same request (upstreams are applied before routes) or "
            "before. This server never stores a route it cannot serve",
            {
                "field": f"routes.{name}",
                "capability": name,
                "upstream": upstream_name,
                "model": model,
            },
        )


def recomputed_capability(
    config: Config, resolved: Resolved, *, gpu_vendor: str
) -> CapabilityRecord | None:
    """`[capability]`, decided again with the new routes and allowance applied.

    PHASE15-HOST.md section 2: *"Capability is RECOMPUTED in-process on every
    settings write that touches a route, and re-written to the config, because
    the route is part of the capability answer."* The allowance is in here for
    the same reason — it is the other input `decide()` reads, and a settings
    write that changed it and left the record alone would leave the rows saying
    what the card could hold under the old reserve.

    **The card's own numbers come from the RECORD, not from a fresh probe.**
    `backend_kind` and `total_bytes` are what the decision was made on
    (`crucible/config.py`'s `CapabilityRecord` docstring says why they are
    kept), and a settings write is not the door that re-measures a card —
    `crucible capability --write` is, and it needs the host to do it.

    None when nothing has decided anything here yet, and that stays None: an
    empty record would read as "the card was probed and nothing fit", which is
    a different and false statement about the host.
    """
    record = config.capability
    if record is None:
        return None
    decisions = capability_classes.decide_all(
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        # A LIVE host fact, like the detection that wrote the record in the
        # first place. It is not in `[capability]` because it is not part of
        # the decision's inputs — the pool SIZE is, and that is recorded; the
        # vendor only decides what the pool is CALLED and whether the row
        # carries the cpu-build sentence. `cmd_serve` already refuses to start
        # when the detected backend and the recorded one disagree, so the two
        # cannot drift apart under a running server.
        gpu_vendor=gpu_vendor,
        # The selections as this patch leaves them, not as the config had
        # them: a write that changes a choice must be decided on the NEW one,
        # or the record would describe the model the app just replaced.
        chosen=resolved.local_models,
    )
    return capability_classes.record(
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        decisions=decisions,
        routes=dict(resolved.routes),
    )


def apply(config: Config, resolved: Resolved, *, gpu_vendor: str) -> None:
    """Write the file and adopt it into the Config this process holds.

    **In that order, and into the SAME object.** Every route, the residency,
    the store and every job-type plugin close over this one `Config` (see
    `Config.adopt`), so the chat door forwards to the key that was pasted a
    millisecond ago without a restart and without anybody holding a second
    document. Writing and not adopting would make `GET /v1/settings` answer
    from disk and the chat door answer from memory — one fact, two owners, the
    whole of ARCHITECTURE.md section 1.

    The file is rewritten WHOLE (`write_config`), which is the same mechanism
    `crucible install` uses and for the reason its docstring gives: an in-place
    TOML edit is one more thing that can lose a token.
    """
    routes, upstreams, local_models = resolved.as_records()
    write_config(
        config.home,
        name=config.name,
        host=config.host,
        port=config.port,
        token=config.token,
        backend_kind=config.backend_kind,
        enable_echo=config.enable_echo,
        enable_llm=config.enable_llm,
        enable_asr=config.enable_asr,
        enable_tts=config.enable_tts,
        enable_align=config.enable_align,
        enable_rvc=config.enable_rvc,
        enable_denoise=config.enable_denoise,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        capability=recomputed_capability(config, resolved, gpu_vendor=gpu_vendor),
        routes=routes,
        upstreams=upstreams,
        local_models=local_models,
        advertise=config.advertise,
        tailscale_advertise=resolved.tailscale_advertise,
    )
    config.adopt(load_config(config.home))


__all__ = [
    "HISTORY_LIMIT",
    "History",
    "PATCH_KEYS",
    "Resolved",
    "apply",
    "document",
    "local_selection",
    "recomputed_capability",
    "resolve",
]
