from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..jobs.line import DEFAULT_MAX_WAIT_S, MAX_MAX_WAIT_S, MIN_MAX_WAIT_S
from ..queuerequest import QueueRequest
from ..queuesessions import DEFAULT_IDLE_S, MAX_IDLE_S, MIN_IDLE_S, STREAM_IDLE_S
from ..tasks import TASK_TYPES


class ArtifactRef(BaseModel):
    """An artifact of a previous job on this server, taken as an input."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str


class JobInput(BaseModel):
    """One named input: an uploaded blob, inline base64 bytes, or a previous job's
    artifact.
    """

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
    client_ref: str | None = Field(
        default=None,
        max_length=200,
        pattern=r"^[^\x00-\x1f\x7f]+$",
        description="The client's own name for this work, echoed on the job record "
        "and never read by the server.",
    )
    hold: bool = Field(
        default=False,
        description="Hold the job from creation, as `POST /v1/jobs/{id}/hold` would, "
        "so its artifacts outlive being fetched.",
    )
    queue: QueueRequest | None = Field(
        default=None,
        description="Opt in to the server's queue: while the lane is busy the job "
        "waits (status `queued`) instead of being refused `409 server_busy`. "
        "Without it, a busy server refuses as it always has.",
    )


class StreamOpen(BaseModel):
    """`POST /v1/tts/stream`: the voice and language of a streaming session; neither has
    a default.
    """

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)
    idle_s: int = Field(
        default=STREAM_IDLE_S,
        ge=MIN_IDLE_S,
        le=MAX_IDLE_S,
        description="When the client holds no queue session, the stream opens one for "
        "itself with this idle_s: no row being said, no op and no touch for this long "
        "closes the session and the stream with it. Ignored inside the client's own "
        "session.",
    )
    queue: QueueRequest | Literal[False] = Field(
        default_factory=QueueRequest,
        description="How long the stream's queue session may wait in the line to open "
        "(`max_wait_s`). `false`: refuse (`session_open`, `server_busy`) rather than "
        "wait when the server is not free now.",
    )


class TaskCreate(BaseModel):
    """`POST /v1/tasks`: one operator task. `type` decides which fields are required and
    which are refused.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    kind: str | None = None
    id: str | None = None
    job_type: str | None = None
    narrator_engine: str | None = None
    module: dict[str, Any] | None = None
    target: str | None = None

    FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "pull": ("kind", "id"),
        "install": ("job_type", "narrator_engine"),
        "module": ("module",),
        "engine": ("target",),
        "engine-restart": (),
    }
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
        return {
            "type": self.type,
            **{name: getattr(self, name) for name in self.FIELDS[self.type]},
        }


class SessionOpen(BaseModel):
    """`POST /v1/queue/sessions`: ask for the server for a run of requests. `act` has no
    default."""

    model_config = ConfigDict(extra="forbid")

    act: str = Field(
        min_length=1,
        description="The capability class the run is for, as the act header names it.",
    )
    model: str | None = Field(
        default=None,
        description="A model to have resident when the session opens; it is loaded for "
        "the session (a load-model job attributed to it) when it is not.",
    )
    idle_s: int = Field(
        default=DEFAULT_IDLE_S,
        ge=MIN_IDLE_S,
        le=MAX_IDLE_S,
        description="Close the session after this long with nothing in flight, no item "
        "and no touch. A running job or an answer in flight always counts as activity.",
    )
    max_wait_s: int = Field(
        default=DEFAULT_MAX_WAIT_S,
        ge=MIN_MAX_WAIT_S,
        le=MAX_MAX_WAIT_S,
        description="How long it may wait in the line to open before it is removed "
        "`expired`.",
    )


class StreamOp(BaseModel):
    """`POST /v1/tts/stream/{id}`: one op. `say` needs `id`, `text` and `take` (no
    default); `cancel` needs `id`; `cancel_all` and `close` take nothing.
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
