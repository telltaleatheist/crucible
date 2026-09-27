from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

from .. import KEEP_ALIVE_SECONDS


def run_owned_server(app: Any, *, host: str, port: int, log_level: str) -> None:
    import uvicorn

    descriptor = sys.stdin.fileno()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            timeout_keep_alive=KEEP_ALIVE_SECONDS,
        )
    )
    finished = threading.Event()

    def watch_controller() -> None:
        try:
            os.read(descriptor, 1)
        except OSError:
            pass
        while not server.started:
            if finished.wait(0.01):
                return
        logging.getLogger("uvicorn.error").info("Controller channel closed; stopping owned engine")
        server.should_exit = True

    threading.Thread(target=watch_controller, name="crucible-controller-pipe", daemon=True).start()
    try:
        server.run()
    finally:
        finished.set()
