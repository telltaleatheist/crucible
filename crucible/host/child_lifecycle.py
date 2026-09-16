"""Private controller-to-engine lifetime channel, inherited only by owned children.

The controller owns the stdin writer. Closing it requests graceful ASGI shutdown;
controller death closes it too. No console signals or public shutdown route are
needed, and the application's lifespan closes resident model workers before exit.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any


def run_owned_server(app: Any, *, host: str, port: int, log_level: str) -> None:
    import uvicorn

    descriptor = sys.stdin.fileno()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=log_level))
    finished = threading.Event()

    def watch_controller() -> None:
        try:
            # Any byte or EOF means stop. os.read avoids a buffered stdin lock
            # held by a daemon thread during Python interpreter shutdown.
            os.read(descriptor, 1)
        except OSError:
            pass  # A lost inherited pipe is also a lost controller.
        # Uvicorn's early should_exit return skips lifespan shutdown when set
        # before startup completes. Preserve a pending stop until it is ready.
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
