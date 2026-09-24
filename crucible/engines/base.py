"""The `Engine` interface and the managed-subprocess machinery behind it.

PHASE2-LLM.md section 3: `start(model_dir, served_name, port, args)`, `ready()`
polling the engine's own `/v1/models`, `stop()` by SIGTERM with a wait and
**never SIGKILL** (a killed CUDA process wedges WSL until Windows reboots), and
`base_url`. The engine binds 127.0.0.1 on a free port; only Crucible talks to it.
Its stdout and stderr go to `~/.crucible/logs/engine-<id>.log`, **appended**:
runs ACCUMULATE in that file and the `=== crucible <id> engine, <date>`
header delimits them. See `start()` for why truncating was a defect.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .. import procgroup
from ..errors import CrucibleError
from ..logtail import tail_of_last_run

#: How long `stop()` waits for SIGTERM to be honoured before it gives up and says
#: so. It never escalates to SIGKILL.
STOP_TIMEOUT_SECONDS = 180.0
#: How often `ready()` asks, and how much of the log a failure reports.
READY_POLL_SECONDS = 2.0
LOG_TAIL_LINES = 40


class EngineError(CrucibleError):
    """An engine would not start, would not answer, or would not stop."""


@runtime_checkable
class Engine(Protocol):
    """What the load-model job needs from vLLM, mlx-lm, or a test double."""

    name: str

    def start(
        self,
        model_dir: Path,
        served_name: str,
        port: int,
        args: list[str],
    ) -> None:
        """Spawn the server. Returns as soon as the process exists, not when ready."""

    def ready(
        self,
        timeout: float,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Block until the engine says it is up, or raise EngineError.

        HOW it says so is the engine's: vLLM and mlx-lm answer `/v1/models`,
        narrator prints a `ready` line on stdout.
        """

    def stop(self) -> None:
        """SIGTERM and wait. Never SIGKILL."""

    @property
    def base_url(self) -> str:
        """`http://127.0.0.1:<port>` — the root the OpenAI routes hang off."""

    @property
    def pids(self) -> frozenset[int]:
        """Every pid this engine owns, for the accelerator guard."""


def find_free_port() -> int:
    """A port nothing is listening on right now.

    Bind-and-release: the engine claims it milliseconds later. Nothing else on
    this host is handing out ports at that rate, and a collision surfaces as the
    engine failing to bind, which `ready()` reports by name rather than hiding.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def logs_dir(home: Path) -> Path:
    directory = home / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class SubprocessEngine:
    """Common body for every engine: spawn, poll, SIGTERM.

    Subclasses supply `name` and `command()`. The process is started in its own
    session so that SIGTERM reaches the workers an engine forks (vLLM forks one
    per GPU) and not just the parent.
    """

    name = "subprocess"

    #: HOW MANY CHAT COMPLETIONS THIS ENGINE REALLY RUNS AT ONCE, or None when
    #: nobody has measured it on this engine.
    #:
    #: `crucible/inflight.py` says a chat record gates nothing, and for a
    #: BATCHING engine that is still exactly right: two cleanup passes on one
    #: resident vLLM genuinely overlap and finish sooner than they would in
    #: sequence. But a SERIAL engine does not overlap them, it queues them — and
    #: an unbounded queue behind a socket the client is holding open is how one
    #: request starves past a deadline while the server reports itself healthy.
    #: Foundry's clean pass hit that on 2026-09-20 against mlx-lm on the Mac: 12
    #: in flight, one starved past the client's 300 s deadline, the pass dead at
    #: block 352 of 940.
    #:
    #: So this is the engine's own truth and nothing more. It is NOT a policy
    #: number and NOT a tuning knob: an engine that has not been read or measured
    #: states None and is not bounded at all, which is what every engine but
    #: mlx-lm does today. A guessed number here would cap work that was never
    #: shown to need capping — the same defect as a `[voice.serving]` field with
    #: no `_note`, which `crucible/voices.py` refuses outright.
    chat_concurrency: int | None = None

    #: Where `chat_concurrency` came from, in a sentence a reader of
    #: `/v1/activity` can check. Required with a number and refused without one
    #: by `engines.chat_admission()`, for `_check_serving_extra`'s reason: a
    #: concurrency with no provenance is a number somebody typed.
    chat_concurrency_basis: str | None = None

    #: CAN THIS ENGINE SERVE A DECISION (`POST /v1/decide`, PHASE22-DECIDE.md)?
    #:
    #: A decision is read off the engine's own `/v1/chat/completions` as
    #: `choices[0].logprobs.content[0].top_logprobs`, so the question is whether
    #: that route returns top logprobs at all, and how many. It is a fact READ
    #: FROM THE ENGINE'S SOURCE at the version Crucible pins, never assumed, so
    #: it is False until somebody has read it — and the door then refuses
    #: `503 decide_not_served` with `decide_basis` as the reason, rather than
    #: sending a request whose reply would have to be guessed at.
    decide_logprobs: bool = False

    #: The most top logprobs one reply carries, or None when the engine has no
    #: small cap (llama-server's `n_probs` is bounded only by the vocabulary).
    #: The reader asks for the labels plus a margin and never more than this; a
    #: question with more options than this is refused before it is sent.
    max_logprobs: int | None = None

    #: Where `decide_logprobs` and `max_logprobs` came from, or why the engine
    #: cannot serve a decision. Required either way (`engines.decide_reading`):
    #: a refusal with no reason is as useless as a number with no provenance.
    decide_basis: str | None = None

    def __init__(self, python: Path, log_path: Path) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._port: int | None = None
        self._served_name: str | None = None

    # ------------------------------------------------------------- subclass

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        """Extra environment for the engine process. Merged over os.environ."""
        return {}

    def announced_ready(self) -> str | None:
        """Has this engine said it is up? The message if so, None if not yet.

        **How an engine announces itself is the subclass's business.** vLLM and
        mlx-lm both answer their own OpenAI `/v1/models`, which is what this
        default does and what PHASE2-LLM.md section 3 specifies. narrator does
        not: its wire is newline-delimited JSON over stdin and stdout and it
        prints a `ready{device,backend}` line (PHASE3-TTS.md section 4), so
        `crucible/engines/narrator.py` overrides this rather than standing up a
        fake HTTP server to fit a probe that assumed one.

        Raising from here is how an engine reports that it is up and serving the
        WRONG thing, which is not a "not yet" and must not be polled through.
        """
        served = self._probe_models(f"{self.base_url}/v1/models")
        if served is None:
            return None
        if self._served_name is not None and self._served_name not in served:
            raise EngineError(
                f"{self.name} is serving {served}, not "
                f"{self._served_name!r}; Crucible will not proxy a model it "
                "did not ask for"
            )
        return f"{self.name} is serving {self._served_name!r}"

    def readiness_description(self) -> str:
        """What `ready()` says the engine failed to do, in a timeout message.

        Reads as "<name> did not <this> within 900s", so it is a verb phrase and
        it names whatever `announced_ready()` was actually watching.
        """
        return f"answer {self.base_url}/v1/models"

    def confirm(
        self, deadline: float, on_progress: Callable[[str], None] | None
    ) -> None:
        """Prove readiness beyond `/v1/models`, if this engine needs it.

        vLLM does not: it binds its OpenAI routes only once the engine core is
        up, so a 200 from `/v1/models` means the weights are on the card.
        """
        return None

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        """How this engine's three standard streams are wired, as Popen kwargs.

        **The second seam PHASE3-TTS.md section 4 said `start()` would need.**
        Every engine before narrator talks HTTP, so its pipes carry nothing but
        diagnostics: stdin is `DEVNULL` and both output streams go to
        `~/.crucible/logs/engine-<id>.log`, which is the default below and is
        byte for byte what vLLM and mlx-lm had. narrator's wire **is** those two
        pipes — newline-delimited JSON in and out — so `crucible/engines/
        narrator.py` overrides this to keep stdin and stdout as pipes and send
        only stderr to the log.

        It returns keyword arguments rather than three streams because the text
        and buffering modes belong to the same decision: a pipe Crucible writes
        JSON lines to is a text-mode, line-buffered pipe, and a log file taking
        an engine's raw output is not.
        """
        return {
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }

    def attach(self, process: subprocess.Popen[Any]) -> None:
        """Called once the process exists, before `start()` returns.

        Where an engine whose wire is its own pipes starts reading them. The
        HTTP engines have nothing to do here and do not override it.
        """
        return None

    def detach(self) -> None:
        """Called once the process is gone, at the end of `stop()`.

        The counterpart to `attach()`: where a reader thread is joined and pipes
        are closed. Runs whether the engine stopped cleanly or not, so the
        engine does not leave a thread holding a dead pipe.
        """
        return None

    # ------------------------------------------------------------- lifecycle

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise EngineError(f"{self.name} has not been started, so it has no url")
        return f"http://127.0.0.1:{self._port}"

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def pids(self) -> frozenset[int]:
        if self._process is None or self._process.poll() is not None:
            return frozenset()
        return frozenset({self._process.pid})

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        if self._process is not None:
            raise EngineError(
                f"{self.name} is already running as pid {self._process.pid}"
            )
        if not self._python.is_file():
            raise EngineError(
                f"no interpreter at {self._python}; the llm env is not installed"
            )
        if not model_dir.is_dir():
            raise EngineError(f"no model directory at {model_dir}")

        command = self.command(model_dir, served_name, port, args)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # APPEND, NOT TRUNCATE (2026-09-20). This was `open("wb")`, so every
        # engine start rewrote the file from the header down — and the one thing
        # an operator does when an engine hangs is reload the voice, which is a
        # start, which destroyed the log of the hang being investigated. Found
        # by a BookForge session hunting what the model server did during a
        # 300 s clean-text timeout and finding a kilobyte written two minutes
        # earlier. A log that a diagnosis erases is worse than no log, because
        # it still looks like evidence.
        #
        # UNBOUNDED rather than a `.1` rotation, deliberately: a rotation keeps
        # exactly one previous run, so the SECOND reload of a hang — which is
        # the normal way one is investigated — destroys it again. That is the
        # same defect with one more step in front of it. Growth is measured,
        # not assumed: the busiest engine log on the Mac after a week of renders
        # is 168 KB (`engine-mistborn.log`), and `serve.log` — which systemd and
        # launchd have always appended to across every restart — is 8.7 MB.
        # `log_tail()` reads from the END, so it stays cheap however long this
        # gets.
        existed = self._log_path.is_file() and self._log_path.stat().st_size > 0
        self._log_handle = self._log_path.open("ab")
        header = (
            # A blank line before every run but the first, so the delimiter is
            # visible to a person scrolling and not only to a parser.
            ("\n" if existed else "")
            + f"=== crucible {self.name} engine, {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=== {' '.join(command)}\n"
        ).encode("utf-8")
        self._log_handle.write(header)
        self._log_handle.flush()

        environment = dict(os.environ)
        environment.update(self.environment())
        try:
            # Its own process group on every platform (`crucible/procgroup.py`):
            # `start_new_session` on POSIX, `CREATE_NEW_PROCESS_GROUP` on win32,
            # where `start_new_session` is silently ignored.
            group = procgroup.own_group()
        except procgroup.ProcessGroupError as exc:
            self._close_log()
            raise EngineError(str(exc)) from exc
        try:
            self._process = subprocess.Popen(
                command,
                env=environment,
                **group,
                **self.stdio(self._log_handle),
            )
        except OSError as exc:
            self._close_log()
            raise EngineError(f"could not spawn {command[0]}: {exc}") from exc
        self._port = port
        self._served_name = served_name
        try:
            self.attach(self._process)
        except Exception as exc:
            # A process exists and nothing is reading it. Tidy it up rather than
            # leaving an engine running that Crucible has no channel to, and
            # report the attach failure, which is the one that explains this.
            try:
                self.stop()
            except EngineError:
                pass
            raise EngineError(
                f"{self.name} started as pid {self._process.pid if self._process else '?'} "
                f"but Crucible could not attach to it: {exc}"
            ) from exc

    def ready(
        self, timeout: float, on_progress: Callable[[str], None] | None = None
    ) -> None:
        if self._process is None or self._port is None:
            raise EngineError(f"{self.name} has not been started")
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            code = self._process.poll()
            if code is not None:
                raise EngineError(
                    f"{self.name} exited {code} before it was ready. Last "
                    f"{LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail()
                )
            announcement = self.announced_ready()
            if announcement is not None:
                if on_progress is not None:
                    on_progress(announcement)
                # Announcing does not always mean the weights are in memory
                # (mlx-lm's list route is served by a thread that does not wait
                # for the load). An engine that needs more proof than that says
                # so here.
                self.confirm(deadline, on_progress)
                return
            if time.monotonic() >= deadline:
                raise EngineError(
                    f"{self.name} did not {self.readiness_description()} within "
                    f"{timeout:.0f}s. Last "
                    f"{LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail()
                )
            attempt += 1
            if on_progress is not None:
                on_progress(self.warming_message(attempt, deadline))
            time.sleep(READY_POLL_SECONDS)

    def warming_message(self, attempt: int, deadline: float) -> str:
        remaining = max(0.0, deadline - time.monotonic())
        last = self.log_tail(1).strip()
        base = (
            f"{self.name} loading; {attempt * READY_POLL_SECONDS:.0f}s elapsed, "
            f"{remaining:.0f}s before give-up"
        )
        return f"{base} — {last}" if last else base

    def _probe_models(self, url: str) -> list[str] | None:
        """The served model ids, or None while the engine is not answering yet."""
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        return [entry.get("id") for entry in data if isinstance(entry, dict)]

    def stop(self) -> None:
        """Ask the engine's process group to stop, then wait.

        POSIX: SIGTERM and never SIGKILL (a killed CUDA process in WSL2 wedges
        the distro). win32: CTRL_BREAK_EVENT, then the tree is terminated if it
        does not go — that reason does not exist for a native Windows process.
        Both halves live in `crucible/procgroup.py`; until 2026-09-23 this
        called `os.killpg` directly, which raised `AttributeError` on win32 and
        left every engine it was asked to stop running.
        """
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                delivered = procgroup.ask_to_stop(process)
                if not delivered and procgroup.platform_kind() == procgroup.WIN32:
                    # No console to route the break through: the polite door
                    # does not exist, so waiting on it would only be a delay.
                    procgroup.terminate_tree(process, self.name)
            except procgroup.ProcessGroupError as exc:
                raise EngineError(
                    f"could not stop {self.name} (pid {process.pid}): {exc}"
                ) from exc
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                if procgroup.platform_kind() == procgroup.WIN32:
                    try:
                        procgroup.terminate_tree(process, self.name)
                    except procgroup.ProcessGroupError as exc:
                        self.detach()
                        self._close_log()
                        raise EngineError(str(exc)) from exc
                else:
                    self.detach()
                    self._close_log()
                    raise EngineError(
                        f"{self.name} (pid {process.pid}) did not exit within "
                        f"{STOP_TIMEOUT_SECONDS:.0f}s of SIGTERM. Crucible does not "
                        "SIGKILL a process holding CUDA — that wedges WSL2 until "
                        f"Windows reboots. Kill it by hand if you must: {self._log_path}"
                    ) from None
        self.detach()
        self._close_log()
        self._process = None
        self._port = None
        self._served_name = None

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        """The last `lines` lines OF THIS RUN, read from the end of the log.

        Not simply the last lines of the file: runs accumulate there now (see
        `start()`), and `logtail.tail_of_last_run` stops at the run header so
        that an earlier run's output is never reported as this one's. That is
        not cosmetic — `LlamaServerEngine._fatal_in_log()` REFUSES a start on a
        fatal line it finds here, and a dead run's "out of memory" would
        otherwise refuse every start after it.
        """
        return tail_of_last_run(self._log_path, lines)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
