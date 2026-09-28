from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Number = Union[int, float]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Ping(_Open):
    """`GET /v1/ping`: enough for a client to tell a Crucible from anything else."""

    crucible: Literal[True]
    name: str
    api_version: int
    pairing_version: int


class ErrorBody(_Open):
    """What every refusal carries: branch on `code`, show a person `message`."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorEnvelope(_Open):
    """Every error answer: one `error` object."""

    error: ErrorBody


class JobBusyDetails(_Open):
    """`server_busy` from the job door: the job that holds the lane."""

    door: Literal["job"]
    holder: str | None
    job_id: str
    type: str
    model: str | None
    status: str
    since: str
    progress: Number
    message: str | None


class CardHeldDetails(_Open):
    """`server_busy` from the operator door: what holds the card, in the server's words."""

    door: Literal["operator"]
    fact: str
    who: str


ServerBusyDetails = Annotated[
    Union[JobBusyDetails, CardHeldDetails], Field(discriminator="door")
]


class ServerBusyBody(_Open):
    """A `409 server_busy` refusal; `details.door` says which door refused."""

    code: Literal["server_busy"]
    message: str
    details: ServerBusyDetails


class ServerBusy(_Open):
    """The `409 server_busy` envelope."""

    error: ServerBusyBody


class JobFailure(_Open):
    """Why a job failed."""

    code: str
    message: str


class JobStatus(_Open):
    """`GET /v1/jobs/{id}`. A job type adds its own `done_extra` keys beside these."""

    job_id: str
    type: str
    model: str | None
    status: str
    progress: Number
    position: int | None
    error: JobFailure | None
    artifacts: list[str]
    created: str
    started: str | None
    finished: str | None
    client_ref: str | None
    interrupted_at: str | None
    held_by: str | None
    held_since: str | None
    chunks_done: list[int]
    chunks_total: int | None
    chunk_at: str | None
    resume_id: str | None
    resumed: bool
    lease_id: str | None = None
    sampling: dict[str, Any] | None = None


class VoiceInfo(_Open):
    """One row of `GET /v1/voices`: a voice and where it stands on this server."""

    id: str
    display: str
    kind: str | None
    language: str | None
    narrator_engine: str | None
    backend_supported: bool
    installed: bool
    resident: bool
    orphan: bool | None
    loadable: bool
    reason: str | None
    revision: str | None
    fingerprint: str | None
    source: str | None
    identity_basis: str | None
    memory_bytes_estimate: Number | None
    estimate_basis: str | None
    serving: dict[str, Any] | None
    max_chars: int | None
    max_chars_basis: str | None
    pace_basis: str | None
    inherited_from: str | None
    manifest: str | None
    sample_rate: int | None
    takes: int
    needs_reference: bool
    pace: dict[str, Any] | None


class ActivityServer(_Open):
    """Who answered `GET /v1/activity`."""

    name: str
    version: str
    api_version: int
    backend: str
    uptime_s: Number


class ActivityHeld(_Open):
    """What keeps the resident subject on the card."""

    fact: str
    who: str
    details: dict[str, Any]


class ActivityResident(_Open):
    """What is on the card."""

    kind: str
    id: str
    since: str
    memory_bytes_estimate: Number | None
    engine_exit_code: int | None
    reference: Any = None
    held_by: ActivityHeld | None
    unclaimed_since: str | None


class ActivityLease(_Open):
    """The open lease."""

    lease_id: str
    kind: str
    client: str | None
    act: str
    since: str
    expires_at: str


class ActivityChatRow(_Open):
    """One chat completion in flight."""

    id: int
    act: str | None
    model: str
    client: str | None
    since: str


class ActivityChat(_Open):
    """Chat completions in flight and the engine's admission limit."""

    in_flight: int
    max_in_flight: int | None
    max_in_flight_basis: str | None
    rows: list[ActivityChatRow]


class ActivitySlot(_Open):
    """The one accelerated lane."""

    busy: int
    of: int
    queue_depth: int
    accepts_work: bool


class ActivitySlots(_Open):
    """Every lane this server admits work through."""

    accelerated: ActivitySlot


class ActivityJob(_Open):
    """A running or queued job, as `GET /v1/activity` shows it."""

    job_id: str
    type: str
    model: str | None
    status: str
    position: int | None
    progress: Number
    message: str | None
    created: str
    started: str | None
    client: str | None


class ActivitySettings(_Open):
    """Recent writes through `PUT /v1/settings`."""

    writes: list[dict[str, Any]]


class ActivityCatalog(_Open):
    """Recent removals through `DELETE /v1/catalog/{kind}/{id}`."""

    removals: list[dict[str, Any]]


class Activity(_Open):
    """`GET /v1/activity`: what this server is doing, in one read."""

    server: ActivityServer
    resident: ActivityResident | None
    stopping: dict[str, Any] | None
    warming: str | None
    claim: dict[str, Any] | None
    streaming: dict[str, Any] | None
    chat: ActivityChat
    settings: ActivitySettings
    catalog: ActivityCatalog
    lease: ActivityLease | None
    slots: ActivitySlots
    running: list[ActivityJob]
    queued: list[ActivityJob]
    accelerator: dict[str, Any] | None = None


class TerminalStates(_Open):
    """The states after which a job or a task never changes again."""

    jobs: list[str]
    tasks: list[str]


class VoiceSourceLabel(_Open):
    """How to name a voice row's `manifest` source to a person, and the tone to show it in."""

    label: str
    tone: Literal["ok", "warn", "floor"]


class ServiceCommand(_Open):
    """A command, typed on the server itself, that runs it as a machine service."""

    command: str
    does: str


class Info(_Open):
    """`GET /v1/info`: who this server is, what it runs on and what it serves."""

    server: dict[str, Any]
    role: str
    managed_by: dict[str, str] | None
    host: dict[str, Any]
    job_types: list[str]
    capabilities: list[dict[str, Any]]
    pages_engine: dict[str, Any]
    terminal_states: TerminalStates
    voice_sources: dict[str, VoiceSourceLabel]
    service_commands: list[ServiceCommand]


NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorEnvelope, "description": "Refused by name: nothing has that id."},
}

BUSY_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {"model": ServerBusy, "description": "The lane or the card is held."},
}
