"""A stand-in for mlx-vlm's own CLI, faithful to the parts Crucible drives.

`tests/fake_engine.py` serves an OpenAI surface in a THREAD, which is enough
for the proxy. This is the other half of the idea and it is a real PROCESS,
importable as `python -m mlx_vlm server`, because that spelling is precisely
what `crucible/engines/mlx_vlm.py` builds and a double that could not be
reached that way would prove nothing about the command.

It reproduces the four properties of the real server that the engine class
depends on, each measured on the Mac Studio on 2026-09-14 and recorded in that
module's docstring:

1. `--model <dir>` is preloaded BEFORE the socket accepts, so a 200 from
   `/v1/models` means the weights are in memory — which is why `MlxVlmEngine`
   needs no `confirm()` override and mlx-lm does. The sleep below stands in for
   the load and happens before `bind`, not after.
2. `/v1/models` reports the model directory **verbatim as given**, not
   resolved. That is the difference from mlx-lm, and it is the one thing
   `engines.engine_model_name()` has to get right.
3. It refuses to be a text-only server: `--host` and `--port` are required and
   an unknown subcommand is an error, exactly as `mlx_vlm/__main__.py` raises.
4. SIGTERM is honoured with no special handling, so `stop()` never escalates.

    CRUCIBLE_FAKE_MLX_VLM_LOAD_S   seconds to spend "loading" before the socket
                                   is opened. Default 0.0. A test about
                                   readiness sets it, and the point of doing it
                                   here rather than after `bind` is that the
                                   real server's lifespan runs there too.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    model_path: str = "unset"

    def log_message(self, *_args: object) -> None:  # noqa: D102 - quiet
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path not in ("/v1/models", "/models"):
            self.send_error(404)
            return
        body = json.dumps(
            {
                "object": "list",
                # The real route lists every mlx-looking repo in the
                # HuggingFace cache as well; the loaded path is appended to
                # them. Both are here so a double cannot pass by being the only
                # entry — `announced_ready()` asks whether the served name is
                # IN the list, not whether it is the list.
                "data": [
                    {"id": "mlx-community/some-other-model", "object": "model"},
                    {"id": self.model_path, "object": "model"},
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "server":
        raise ValueError("CLI requires a subcommand in {'server'}")
    arguments = sys.argv[2:]
    values = {}
    index = 0
    while index < len(arguments):
        name = arguments[index]
        if not name.startswith("--"):
            raise ValueError(f"unexpected argument {name!r}")
        values[name] = arguments[index + 1]
        index += 2
    for required in ("--model", "--host", "--port"):
        if required not in values:
            raise ValueError(f"{required} is required")

    # The load, before the socket. See point 1 in the module docstring.
    delay = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_LOAD_S")
    if delay:
        time.sleep(float(delay))

    # VERBATIM, not resolved. See point 2.
    _Handler.model_path = values["--model"]
    server = ThreadingHTTPServer((values["--host"], int(values["--port"])), _Handler)
    print(f"fake mlx_vlm serving {values['--model']}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
