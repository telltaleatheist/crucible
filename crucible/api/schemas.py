from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..tasks import TASK_TYPES


class ArtifactRef(BaseModel):
    """A previous job's artifact on THIS server, taken as an input.

    Owen, 2026-09-25: a render's 2,510 chunk FLACs were downloaded, then read
    back off a share and uploaded again (2.5 minutes, 1.25 GB) to the server
    that made them, for the align. A reference takes them where they already
    are. Usually of a HELD job (`POST /v1/jobs/{id}/hold`); an unheld one works
    while its directory still exists.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str


class JobInput(BaseModel):
    """One named input: an uploaded blob, bytes inline in the request, or an
    artifact of a previous job on this server."""

    model_config = ConfigDict(extra="forbid")

    blob_id: str | None = None
    inline_base64: str | None = None
    artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "JobInput":
        given = [name for name, value in
                 (("blob_id", self.blob_id), ("inline_base64", self.inline_base64),
                  ("artifact", self.artifact))
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                "each input needs exactly one of blob_id, inline_base64 or "
                f"artifact, got {given if given else 'neither'}"
            )
        return self


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, JobInput] = Field(default_factory=dict)
    #: THE CLIENT'S OWN NAME FOR THIS WORK. Echoed back on the job record,
    #: never read by this server, never parsed.
    #:
    #: It is for the restart. An `interrupted` job has to be matched to whatever
    #: the client was doing when its own process went away too, and a job id it
    #: may have lost alongside everything else is a poor key for that.
    #: BookForge puts its queue step id here.
    #:
    #: Bounded like a client name and for the same reason (`_CLIENT_NAME`): it
    #: is printed into logs and benches, so a control character in it is the
    #: caller choosing what somebody's terminal does.
    client_ref: str | None = Field(
        default=None, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$"
    )
    #: HELD FROM BIRTH (2026-09-25). A render's client downloads each chunk as
    #: it lands, so by `done` every artifact has been fetched and a hold taken
    #: afterwards races the fetch-reap. `true` holds the job before it runs,
    #: exactly as `POST /v1/jobs/{id}/hold` would, so there is no window.
    hold: bool = False


class StreamOpen(BaseModel):
    """`POST /v1/tts/stream` — PHASE3-TTS.md section 7.

    Nothing has a default, for the render door's reason: a session opened in the
    wrong language, or on a voice the client did not choose, is a silent
    substitution and a whole afternoon of listening in the wrong accent.
    """

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)


class TaskCreate(BaseModel):
    """`POST /v1/tasks` — one operator operation. PHASE13-OPERATOR.md 3.3.

    One model for three request shapes rather than three routes, because there
    is one lane and one refusal (`task_busy`) governing all of them, and a
    client that had to pick a path before it could be told "busy" would have to
    know which of three doors to retry.

    The validator is `StreamOp`'s in spirit: the `type` word decides which
    fields are required and which are REFUSED. A `narrator_engine` sent with a
    `pull`, or an `id` sent with an `install`, is a client that has confused two
    requests, and accepting it silently would run the wrong one.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    # pull
    kind: str | None = None
    id: str | None = None
    # install
    job_type: str | None = None
    narrator_engine: str | None = None
    # module
    module: dict[str, Any] | None = None
    # engine (PHASE15-HOST.md 4.7)
    target: str | None = None

    #: Which fields each type owns. The validator reads this rather than three
    #: hand-written branches, so a fourth task type is one row.
    FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "pull": ("kind", "id"),
        "install": ("job_type", "narrator_engine"),
        "module": ("module",),
        "engine": ("target",),
        # PHASE17-ORCHESTRATOR.md 4.2: NO fields, and that is the whole
        # request. There is exactly one engine on a machine and the
        # orchestrator knows which — a `target` here would be a client
        # naming a thing it cannot see.
        "engine-restart": (),
    }
    #: ...and which of those may not be omitted. `narrator_engine` is absent
    #: here because whether it is required depends on the job type, which is
    #: `crucible/tasks.py`'s question and not this schema's.
    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "pull": ("kind", "id"),
        "install": ("job_type",),
        "module": ("module",),
        "engine": ("target",),
        "engine-restart": (),
    }

    @model_validator(mode="after")
    def the_type_carries_what_it_needs(self) -> "TaskCreate":
        if self.type not in TASK_TYPES:
            raise ValueError(
                f"type must be one of {list(TASK_TYPES)}, got {self.type!r}"
            )
        mine = self.FIELDS[self.type]
        for name in self.REQUIRED[self.type]:
            if getattr(self, name) is None:
                raise ValueError(f"a {self.type} task needs {name!r}")
        theirs = [
            name
            for group in self.FIELDS.values()
            for name in group
            if name not in mine and getattr(self, name) is not None
        ]
        if theirs:
            raise ValueError(
                f"a {self.type} task takes {list(mine)}; it was also sent "
                f"{sorted(theirs)}, which belong to another task type"
            )
        return self

    def request(self) -> dict[str, Any]:
        """The body as the task echoes it: this type's fields and no others."""
        return {
            "type": self.type,
            **{name: getattr(self, name) for name in self.FIELDS[self.type]},
        }


class LeaseOpen(BaseModel):
    """`POST /v1/models/{id}/lease` — a client saying it intends a run.

    Both fields are required and neither has a default, for the streaming door's
    reason. A default `act` would put a name nobody chose on a bench, which is
    the thing `X-Crucible-Act` is refused for; a default `ttl_seconds` would be
    this server picking how long somebody else's run is, which is the one number
    only the client knows.

    **There is no `kind`.** The id in the path is the resident thing's, of
    whatever kind, and the card holds one thing — so the server reads the kind
    off `Residency.resident` and a client has nothing to disambiguate. A `kind`
    on the body would be a second owner of `resident.kind`, able to disagree with
    it (R1), and would let a client be refused for spelling a fact it was never
    asked to know.
    """

    model_config = ConfigDict(extra="forbid")

    act: str = Field(min_length=1)
    ttl_seconds: int


class StreamOp(BaseModel):
    """`POST /v1/tts/stream/{id}` — one op.

        {"op": "say",    "id": "r12", "text": "...", "take": 0}
        {"op": "cancel", "id": "r12"}
        {"op": "cancel_all"}
        {"op": "close"}

    `take` is **required** on `say` and has no default here. The SDK's
    `say(id, text, take?)` defaults it to 0 in the caller's own code, which is a
    client choosing; a default on the wire would be the server choosing, and now
    that the five fine-tunes declare a second rung that would be a render at a
    take nobody asked for. A take past the end of the voice's ladder is a SEED
    LANE at the voice's own sampling (2026-09-19) and is still never clamped —
    take 4 is never take 2's numbers under take 4's name.
    """

    model_config = ConfigDict(extra="forbid")

    op: str
    id: str | None = None
    text: str | None = None
    take: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def the_op_carries_what_it_needs(self) -> "StreamOp":
        allowed = ("say", "cancel", "cancel_all", "close")
        if self.op not in allowed:
            raise ValueError(f"op must be one of {list(allowed)}, got {self.op!r}")
        if self.op == "say":
            if not (self.id or "").strip():
                raise ValueError("say needs an id; it is how every frame names its row")
            if not (self.text or "").strip():
                # narrator answers an empty generate with a whole-request error,
                # which would take the rest of the batch with it. Refused here.
                raise ValueError(
                    "say needs text that is not blank; narrator refuses an empty "
                    "generate with a whole-request error, which would end the batch"
                )
            if self.take is None:
                raise ValueError("say needs a take; there is no default on the wire")
        elif self.op == "cancel":
            if not (self.id or "").strip():
                raise ValueError("cancel needs the id of the row to cancel")
        else:
            if self.id is not None or self.text is not None or self.take is not None:
                raise ValueError(f"{self.op} takes no id, text or take")
        return self
