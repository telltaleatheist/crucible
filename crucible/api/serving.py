"""The uvicorn server Crucible runs under, which tells the event stream when it is asked
to stop.

uvicorn waits for every open response to finish before it runs the app's shutdown, and
an SSE stream never finishes by itself: an app holding GET /v1/events open would hold
the server's stop open with it until someone pressed Ctrl+C twice. So the moment uvicorn
is asked to exit — a signal, the controller pipe closing, a test setting `should_exit` —
the hub says `server.stopping` and ends every stream, and the shutdown goes on.
"""

from __future__ import annotations

from typing import Any

import uvicorn

from .. import KEEP_ALIVE_SECONDS
from ..events import EventHub


class Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, events: EventHub) -> None:
        self._events = events
        self._exit_asked = False
        super().__init__(config)

    @property  # type: ignore[override]
    def should_exit(self) -> bool:
        return self._exit_asked

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._exit_asked = value
        if value:
            self._events.stop("the server was asked to stop")


def server_for(app: Any, *, host: str, port: int, log_level: str) -> Server:
    """The one door every Crucible server is started through, for `app` (one create_app
    built): it stops the event stream first, and keeps connections alive for
    KEEP_ALIVE_SECONDS rather than uvicorn's 5 s default."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_keep_alive=KEEP_ALIVE_SECONDS,
    )
    return Server(config, app.state.events)
