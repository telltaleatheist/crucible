from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .jobs.line import DEFAULT_MAX_WAIT_S, MAX_MAX_WAIT_S, MIN_MAX_WAIT_S


class QueueRequest(BaseModel):
    """Wait in the server's queue instead of being refused `409 server_busy`."""

    model_config = ConfigDict(extra="forbid")

    max_wait_s: int = Field(
        default=DEFAULT_MAX_WAIT_S,
        ge=MIN_MAX_WAIT_S,
        le=MAX_MAX_WAIT_S,
        description="How long the job may wait for the lane before it is removed "
        "`expired`.",
    )
