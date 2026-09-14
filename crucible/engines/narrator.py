"""narrator — the managed subprocess that is to `tts` what vLLM is to `llm`.

PHASE3-TTS.md section 4. Crucible does not reimplement Higgs's frame budget,
its guard or its codec arithmetic; it runs the code that already has them.
`python/narrator` in the BookForge repo is an installable
package whose `serve` entry point loads a voice once and answers sentence
requests over stdin and stdout, and this file is the client for that wire.

Three things make it unlike every engine before it, and each one is a seam
somewhere else in this package rather than a special case here:

- **Readiness is a line, not a route.** narrator prints
  `{"type": "ready", "device", "backend"}` on stdout when the process is up.
  `SubprocessEngine.announced_ready()` is the seam; the default `/v1/models`
  poll is untouched and neither HTTP engine overrides it.
- **The wire is the pipes.** `SubprocessEngine.stdio()` is the seam: narrator
  keeps stdin and stdout as text-mode pipes and sends only stderr to
  `~/.crucible/logs/engine-<voice>.log`. Everything the HTTP engines do is
  unchanged.
- **Nothing else can reach it.** vLLM binds a loopback port and the proxy talks
  to it; narrator binds nothing, so this object *is* the channel. Hence
  `base_url` refuses rather than returning a port nothing is listening on.

The correlation problem, and why this file does not solve it
------------------------------------------------------------
`generate_batch` retires rows **out of order** — a short row finishes while a
long one is still generating, and `tests/fake_narrator.py` retires in reverse on
purpose so that a consumer relying on arrival order fails in the suite rather
than on a book. The row's identity is the `i` narrator echoes back, and it is the
*caller's* number: the render door sends its chunk indices as `i` and reads them
straight off each `batch_item`.

So `converse()` yields every line as it lands and **reorders nothing**. Buffering
into caller order here would defeat the two things the shape exists for: the
render door writes each FLAC as its row retires, overlapped with the next row's
generation, and the streaming door (section 7) needs `batch_chunk` lines out of
the same iterator *while* a row is still generating. One reader, one iterator,
and the consumer keys on `i`.

What the reader thread does and does not swallow
------------------------------------------------
fd 1 is the wire and nothing else — the same rule, learned from the same
incident, as `crucible/workers.py`: narrator's aligner once had a library log to
stdout on a 401-chunk book and corrupt the result stream. A line on stdout that
is not a JSON object carrying a `type` is therefore a **refusal naming the
line**, never something skipped. narrator's own diagnostics go to stderr and
into the log, which is where a reader should look for them.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..errors import JobCancelled
from .base import LOG_TAIL_LINES, EngineError, SubprocessEngine

if TYPE_CHECKING:  # `crucible.narratorvoices` imports this package; no cycle at runtime
    from ..narratorvoices import VoicesDocument

#: How narrator is started. It takes no configuration on the command line — its
#: whole interface is the protocol on stdin and stdout plus the environment — so
#: the argv is this and nothing else. Which engine it serves is `NARRATOR_ENGINE`
#: below, which is the variable narrator's own `engine_id()` reads.
MODULE = "narrator.serve"

#: The environment variable that decides which of narrator's engines a process
#: serves. narrator refuses an unknown value by name at start-up rather than
#: defaulting, which is why Crucible may set it and then trust the `ready` line.
ENGINE_VARIABLE = "NARRATOR_ENGINE"

#: WHAT A `higgs-v3` WORKER NEEDS BESIDES ITS ENGINE ID, and every one of them
#: is refused BY NAME by narrator rather than defaulted. Crucible's first real
#: `tts` render (2026-09-13) died on the first of them — `narrator (higgs-v3)
#: exited 3` before `ready`, `HIGGS_STACK is not set` — because `environment()`
#: named only NARRATOR_ENGINE and PYTHONUNBUFFERED.
#:
#:   HIGGS_STACK         which serving stack this process renders on.
#:                       `served_common.serving_stack()` raises when it is
#:                       unset, and is called from `HiggsV3Engine
#:                       .detect_backend()` — a CLASSMETHOD `narrator.serve`
#:                       calls before any voice loads, which is why the worker
#:                       dies before `ready` rather than on the first chunk.
#:                       The two stacks place sampling differently and size the
#:                       frame cap against different context windows, so a
#:                       guessed one is a book rendered at sampling nobody
#:                       chose. STATED FROM THE ENV SPEC (`jobenv.tts_env`),
#:                       because the stack is a property of what the recipe
#:                       installed, not of the voice.
#:   HIGGS_ENV           the prefix the SERVER runs out of. narrator's packaged
#:                       `serve_higgs_v3.sh` builds CUDA_HOME, PATH,
#:                       LD_LIBRARY_PATH and `$HIGGS_ENV/bin/vllm-omni` from it
#:                       and refuses (exit 5) when it is unset. For Crucible it
#:                       is the tts env's own venv root — the interpreter's
#:                       `parent.parent`, CHECKED against `pyvenv.cfg` rather
#:                       than assumed, because a path built by walking up from
#:                       a binary is a guess until something confirms it.
#:   HIGGS_MAX_NUM_SEQS  stage 0's admission width AND the width of narrator's
#:                       own batch (`v3_served.serve_concurrency()`, which
#:                       raises by name). STATED FROM THE VOICE MANIFEST's
#:                       `[voice.serving]`.
#:
#: NARRATOR_HIGGS3_SERVE_SCRIPT is deliberately NOT here. narrator ships its own
#: launcher as package data as of BookForge 0eeb0267 and runs it when no
#: override is named; an operator's path into somebody's checkout is exactly
#: what that commit removed the need for.
#:
#: NARRATOR_HIGGS_VOICES (and, on the MLX arm, NARRATOR_HIGGS3_MLX_MODEL) is
#: the fourth thing a `higgs-v3` worker needs, ON BOTH ARMS, and it is not a
#: constant here because it is not a value: it is the PATH of a document
#: Crucible writes at every load, `crucible/narratorvoices.py`, handed to this
#: engine at construction as `voices`. Crucible's first real render found it
#: on the served arm and the keeper found it again on the Mac (2026-09-14):
#: narrator resolves a Higgs v3 voice BY NAME in that document and refuses a
#: `modelDir` on the `load` message by name, so a worker with no document
#: cannot load any voice at all.
STACK_VARIABLE = "HIGGS_STACK"
ENV_PREFIX_VARIABLE = "HIGGS_ENV"
MAX_NUM_SEQS_VARIABLE = "HIGGS_MAX_NUM_SEQS"

#: The narrator engine those three belong to. Written as a constant so the
#: refusals below read as a rule rather than as a special case: they are the
#: shape a narrator engine that reads no `HIGGS_*` variable would arrive into,
#: and since Owen's ruling of 2026-09-14 (`voices.NARRATOR_ENGINE_SAMPLING`)
#: `higgs-v3` is the only engine Crucible names at all.
HIGGS_V3 = "higgs-v3"

#: How long `stop()` gives the `quit` action before falling back on SIGTERM.
#: narrator's teardown releases CUDA from inside the process, and on a loaded
#: SGLang-Omni that takes seconds rather than milliseconds.
QUIT_GRACE_SECONDS = 30.0

#: How often the conversation loop wakes to notice a cancel, a dead process or a
#: missed deadline. Not a timeout on anything: a row that takes ninety seconds is
#: a row that is being generated.
POLL_SECONDS = 0.5

#: How long a `load` may go without narrator saying anything at all. It is a
#: SILENCE timeout and any line resets it, the same discipline `crucible/
#: workers.py` uses: narrator on `cuda-linux` starts SGLang-Omni underneath
#: itself, which is minutes of weight reading before the `loaded` line.
LOAD_SILENCE_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class _Garbled:
    """A line on stdout that is not a message. Carried, not dropped."""

    line: str


class _Ended:
    """narrator's stdout reached end of file."""


_ENDED = _Ended()


class NarratorEngine(SubprocessEngine):
    """`python -m narrator.serve`, and the JSON-lines channel to it.

    One instance per resident voice, built by `build_voice_engine()` from the
    voice manifest's `narrator_engine`. The name carries the engine id, so a log
    line, a timeout and a refusal all say `narrator (higgs-v3)` rather than
    `narrator` — on `cuda-linux` there are two envs and two engines and the
    question a reader has is always which.
    """

    def __init__(
        self,
        narrator_engine: str,
        python: Path,
        log_path: Path,
        *,
        serving_stack: str | None,
        max_num_seqs: int | None,
        voices: VoicesDocument | None,
    ) -> None:
        """`serving_stack` comes from the env spec, `max_num_seqs` from the
        voice manifest, `voices` from `narratorvoices.write_document`, and for
        `higgs-v3` ALL THREE ARE REQUIRED HERE — the first two on the served
        arm, the document on both.

        NONE HAS A DEFAULT, keyword-only and mandatory. `None` is a real
        answer — "narrator starts no server out of this env", "this engine
        reads no document" — and a default would make FORGETTING to pass one
        indistinguishable from saying it, which is precisely how a worker ends
        up spawned without HIGGS_STACK. A caller must state all three;
        `build_voice_engine` is where they come from.

        Refused at CONSTRUCTION and not at spawn, because the alternative is a
        worker that starts, reads 8.5 GB off disk and exits 3 before it says
        `ready` — which is exactly how this was found. A refusal that arrives
        before the process does names the missing thing instead of leaving a
        reader to find `HIGGS_STACK is not set` at the end of an engine log.

        `serving_stack` is None on `mlx-darwin`, where narrator starts no
        server and reads none of these, and for any engine with no row in
        `jobenv.CUDA_LINUX_SERVING_STACK`. `voices` is None for an engine
        outside `narratorvoices.DOCUMENT_READERS` — one whose weights ride the
        `load` message and which reads no `NARRATOR_HIGGS_*` variable.
        """
        super().__init__(python=python, log_path=log_path)
        self._narrator_engine = narrator_engine
        self._serving_stack = serving_stack
        self._max_num_seqs = max_num_seqs
        if narrator_engine == HIGGS_V3:
            if voices is None:
                # THE DOCUMENT IS HOW A HIGGS VOICE IS NAMED, on both arms.
                # Without it there is no load this worker could accept, so the
                # refusal is here and not at the first `load`.
                raise EngineError(
                    f"cannot start {self.name} without a voices document: "
                    "narrator resolves a Higgs v3 voice by name in the "
                    "NARRATOR_HIGGS_VOICES document and refuses a modelDir on the "
                    "load message, on the served arm and the MLX arm alike. "
                    "crucible/narratorvoices.py writes it from the voice "
                    "manifest and the pulled weights at every load"
                )
        elif voices is not None:
            # A DOCUMENT FOR AN ENGINE THAT READS NONE. An engine outside
            # `narratorvoices.DOCUMENT_READERS` takes its weights on the `load`
            # message and never reads the variable; handing it one would leave
            # two statements of where the weights are, one of them read by
            # nothing.
            raise EngineError(
                f"{self.name} was given a voices document ({voices.path}), but "
                f"only {HIGGS_V3!r} resolves a voice by name in one; any other "
                "engine takes its weights on the load message"
            )
        self._voices = voices
        if narrator_engine == HIGGS_V3 and serving_stack is not None:
            # THE SERVED ARM. `serving_stack` set is what "narrator will start a
            # server out of this env" means, and it is the one condition under
            # which all three variables have a reader.
            if max_num_seqs is None:
                raise EngineError(
                    f"cannot start {self.name} on the {serving_stack} stack "
                    f"without {MAX_NUM_SEQS_VARIABLE}: it is stage 0's "
                    "admission width and the width of narrator's own batch, "
                    "and narrator refuses it by name "
                    "(v3_served.serve_concurrency). It comes from the voice "
                    "manifest's [voice.serving].max_num_seqs"
                )
            if max_num_seqs < 1:
                raise EngineError(
                    f"{MAX_NUM_SEQS_VARIABLE}={max_num_seqs} for {self.name} "
                    "must be at least 1"
                )
            # `venv/bin/python` -> `venv`. CHECKED, not assumed: `pyvenv.cfg`
            # is what makes a directory a venv, and handing narrator's launch
            # script a prefix that is not one produces
            # `$HIGGS_ENV/bin/vllm-omni: No such file` at the end of a launch
            # rather than here.
            root = Path(python).resolve().parent.parent
            if not (root / "pyvenv.cfg").is_file():
                raise EngineError(
                    f"cannot start {self.name}: {ENV_PREFIX_VARIABLE} is the "
                    "prefix its server runs out of, and the tts env python "
                    f"{python} does not sit in one — {root / 'pyvenv.cfg'} is "
                    "not there. narrator's launch script builds CUDA_HOME, "
                    "PATH, LD_LIBRARY_PATH and the vllm-omni binary from that "
                    "prefix and refuses when it is unset"
                )
            self._env_prefix: Path | None = root
        elif serving_stack is not None:
            # A STACK ON AN ENGINE THAT HAS NONE. `HIGGS_*` is Higgs v3's
            # vocabulary, and an engine that renders in process reads not one
            # of these names. Silently dropping the value would leave the env
            # recipe and this file disagreeing about what that env starts.
            raise EngineError(
                f"{self.name} was given serving_stack={serving_stack!r}, but "
                f"only {HIGGS_V3!r} starts a server underneath narrator and "
                "reads the HIGGS_* variables. Either the env recipe installed "
                "a stack this engine cannot use, or jobenv.tts_env named one "
                "it should not have"
            )
        else:
            self._env_prefix = None
        #: One writer at a time. narrator holds a lock over its own stdout for
        #: the mirror-image reason (two half-written lines are not two messages);
        #: the render door and a cancel arriving from the queue thread are two
        #: writers, and an interleaved write would be the same corruption in the
        #: other direction.
        self._writer = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any] | _Garbled | _Ended] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._ready_message: dict[str, Any] | None = None

    # ------------------------------------------------------------- the spawn

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"narrator ({self._narrator_engine})"

    @property
    def narrator_engine(self) -> str:
        return self._narrator_engine

    @property
    def base_url(self) -> str:
        """There is none, and saying so is the point.

        Every other engine answers OpenAI routes on a loopback port and the proxy
        forwards to them. narrator answers no HTTP at all, so a `base_url` here
        would be a port nothing is listening on, published on a `/v1/health` row
        and eventually fetched by something. `ResidentVoice` deliberately carries
        no such field either.
        """
        raise EngineError(
            f"{self.name} has no base url: its wire is newline-delimited JSON "
            "over stdin and stdout, not HTTP. Talk to it through this engine "
            "object (PHASE3-TTS.md section 4)"
        )

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        """`<tts env python> -m narrator.serve`, and nothing else.

        `model_dir`, `served_name` and `port` are not on it. The voice travels
        on the `load` message, because narrator is a resident server that
        switches voices without respawning; the weights travel in the
        NARRATOR_HIGGS_VOICES document for `higgs-v3` (see `load`); the port is
        not used at all, and `Residency` finds one anyway for the reason it
        says there.
        """
        return [str(self._python), "-m", MODULE]

    def environment(self) -> dict[str, str]:
        """Everything narrator refuses to start without, and nothing else.

        The three `HIGGS_*` variables are emitted ONLY on the arm that reads
        them — a `higgs-v3` env whose recipe installs a serving stack. On
        `mlx-darwin` narrator builds `HiggsV3MlxEngine` from
        `HiggsV3MlxConfig`, neither of which reads `HIGGS_STACK` or
        `HIGGS_MAX_NUM_SEQS` (the MLX `detect_backend()` returns 'mlx' off an
        import), and there is no launch script for `HIGGS_ENV` to mean anything
        to. Setting them there would be three levers read by nothing.

        A SECOND ENGINE WILL OWE ITS OWN SET HERE, and finding it is that
        engine's first job rather than something guessed in advance: narrator's
        `serve/worker.py` reads each engine's configuration from the
        environment and from the `load` message, and which half Crucible owns
        is the same question `load()` defers on for `caps`.
        """
        environment = {
            ENGINE_VARIABLE: self._narrator_engine,
            # narrator writes its progress to stderr and its protocol to stdout.
            # Without this, a pipe makes CPython block-buffer both, and a `ready`
            # line can sit in a 4 KB buffer for the whole of a load — which reads
            # from here as a readiness timeout on an engine that was up.
            "PYTHONUNBUFFERED": "1",
        }
        if self._env_prefix is not None:
            # `__init__` refuses unless all three are answerable, so this block
            # is all-or-nothing by construction rather than by three checks.
            assert self._serving_stack is not None
            assert self._max_num_seqs is not None
            environment[STACK_VARIABLE] = self._serving_stack
            environment[ENV_PREFIX_VARIABLE] = str(self._env_prefix)
            environment[MAX_NUM_SEQS_VARIABLE] = str(self._max_num_seqs)
        if self._voices is not None:
            # NARRATOR_HIGGS_VOICES on both Higgs arms, plus the MLX arm's base
            # weights when the document has a voice that loads them. The
            # document decides which; see `narratorvoices.VoicesDocument`.
            environment.update(self._voices.environment())
        return environment

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        """stdin and stdout stay pipes; only stderr goes to the log.

        The seam PHASE3-TTS.md section 4 said `start()` would need. UTF-8 is
        stated rather than inherited: the text on these pipes is a book, and the
        locale of whatever shell started the server has no business deciding how
        an em-dash crosses it.
        """
        return {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": log_handle,
            "text": True,
            "encoding": "utf-8",
            "errors": "strict",
            "bufsize": 1,
        }

    # ------------------------------------------------------------ the reader

    def attach(self, process: subprocess.Popen[Any]) -> None:
        self._ready_message = None
        self._reader = threading.Thread(
            target=self._pump,
            args=(process.stdout,),
            name=f"narrator-stdout-{self._narrator_engine}",
            daemon=True,
        )
        self._reader.start()

    def _pump(self, stream: Any) -> None:
        """Every line of narrator's stdout, classified once, on one thread.

        `ready` is taken out here rather than queued, because it is a lifecycle
        announcement and not an answer to anything: `ready()` polls for it while
        `converse()` has not been called yet, and putting it in the same queue
        would make the two compete for it. A SECOND `ready` is a protocol error —
        narrator prints exactly one, and two would mean the process restarted
        underneath a conversation.
        """
        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    self._inbox.put(_Garbled(text))
                    continue
                if not isinstance(message, dict) or not isinstance(
                    message.get("type"), str
                ):
                    self._inbox.put(_Garbled(text))
                    continue
                if message["type"] == "ready" and self._ready_message is None:
                    self._ready_message = message
                    continue
                self._inbox.put(message)
        except (ValueError, OSError):
            # The stream was closed under the reader — `detach()` does exactly
            # that when the engine is stopped. End of file is end of file.
            pass
        finally:
            self._inbox.put(_ENDED)

    def detach(self) -> None:
        """Close the pipes and let the reader go. Order matters here.

        stdin is closed first and unconditionally: Crucible owns the write end
        and nothing else touches it, and closing it is what turns a worker
        blocked on a read it will never satisfy into one that sees EOF.

        **stdout is closed only once the reader has finished**, and that is not
        tidiness. `BufferedReader.close()` takes the object's own lock, which the
        reader thread is holding for as long as it is blocked inside `read()` —
        so closing a pipe another thread is reading deadlocks, and it deadlocks
        exactly in the case this method exists for: a worker that ignored SIGTERM
        and is still alive with its stdout open (measured 2026-09-13, in the test
        that proves `stop()` reports a timeout rather than escalating). When the
        process really is gone the reader is at end of file and joins at once, so
        the ordinary path closes the pipe as it always did; when it is not, the
        fd is left to the daemon thread and the caller has just been told, by
        name, that a process is still running.
        """
        process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        reader = self._reader
        if reader is not None:
            reader.join(timeout=POLL_SECONDS * 4)
        if (
            process is not None
            and process.stdout is not None
            and (reader is None or not reader.is_alive())
        ):
            try:
                process.stdout.close()
            except OSError:
                pass
        self._reader = None
        self._ready_message = None
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break

    # ---------------------------------------------------------- readiness

    def announced_ready(self) -> str | None:
        message = self._ready_message
        if message is None:
            return None
        return (
            f"{self.name} is ready on {message.get('device')} "
            f"(backend {message.get('backend')})"
        )

    def readiness_description(self) -> str:
        return "print a ready line on stdout"

    # ------------------------------------------------------------- the wire

    def send(self, message: dict[str, Any]) -> None:
        """One JSON object, one line, flushed."""
        process = self._process
        if process is None or process.stdin is None:
            raise EngineError(
                f"{self.name} is not running, so there is nothing to send to it"
            )
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._writer:
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} and could not be sent a "
                        f"{message.get('action')!r}. Last {LOG_TAIL_LINES} lines "
                        f"of {self.log_path}:\n" + self.log_tail()
                    ) from None
                raise EngineError(
                    f"could not write to {self.name}: {exc}. Last "
                    f"{LOG_TAIL_LINES} lines of {self.log_path}:\n" + self.log_tail()
                ) from None

    def converse(
        self,
        message: dict[str, Any],
        *,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Send one message and yield narrator's answer, line by line.

        The iterator ends when a message of one of the `terminal` types has been
        yielded. It is a **stream in arrival order and nothing is reordered** —
        see the module docstring on why.

        `cancelled`, when it goes true, sends narrator one `{"action": "cancel"}`
        and then keeps reading. That is narrator's own contract: a cancel aborts
        what is in flight, the rows that will not be rendered come back as
        ordinary per-row failures, and `batch_done` arrives as always. Hanging up
        instead would leave the engine generating into nothing, and killing it
        would take the voice off the card for the next job. `JobCancelled` is
        raised once the terminal message has been seen, so the caller learns the
        run was cancelled rather than that it finished short.
        """
        self.send(message)
        return self._until(terminal, silence_timeout, cancelled)

    def _until(
        self,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None,
    ) -> Iterator[dict[str, Any]]:
        process = self._process
        if process is None:
            raise EngineError(f"{self.name} is not running")
        deadline = time.monotonic() + silence_timeout
        cancel_sent = False
        while True:
            if cancelled is not None and cancelled() and not cancel_sent:
                self.send({"action": "cancel"})
                cancel_sent = True
            try:
                item = self._inbox.get(timeout=POLL_SECONDS)
            except queue.Empty:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} in the middle of a request. "
                        f"Last {LOG_TAIL_LINES} lines of {self.log_path}:\n"
                        + self.log_tail()
                    )
                if time.monotonic() >= deadline:
                    raise EngineError(
                        f"{self.name} said nothing at all for "
                        f"{silence_timeout:.0f}s. Last {LOG_TAIL_LINES} lines of "
                        f"{self.log_path}:\n" + self.log_tail()
                    )
                continue

            # Any line is proof of life, so the silence clock starts again.
            deadline = time.monotonic() + silence_timeout

            if isinstance(item, _Ended):
                raise EngineError(
                    f"{self.name} closed its stdout in the middle of a request "
                    f"(exit {process.poll()}). Last {LOG_TAIL_LINES} lines of "
                    f"{self.log_path}:\n" + self.log_tail()
                )
            if isinstance(item, _Garbled):
                raise EngineError(
                    f"{self.name} wrote a line to stdout that is not a protocol "
                    f"message: {item.line[:300]!r}. stdout carries the wire and "
                    "nothing else; narrator's own diagnostics belong on stderr, "
                    f"which is {self.log_path}"
                )
            if item["type"] == "error":
                # narrator's whole-request refusal: an unknown action, a voice it
                # cannot serve, a load that failed. A per-ROW failure is not this
                # — it is a `batch_item` carrying a `message` — so this ends the
                # conversation rather than being reported alongside the rows.
                raise EngineError(
                    f"{self.name} refused the request: "
                    f"{item.get('message', '(no message)')}"
                )
            yield item
            if item["type"] in terminal:
                if cancel_sent:
                    raise JobCancelled(f"{self.name} was cancelled mid-request")
                return

    # ---------------------------------------------------------------- load

    def load(
        self,
        *,
        voice: str,
        weights_dir: Path,
        warm: bool,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Put a voice on the card and return narrator's own `loaded` line.

        This is what makes `load-voice` mean the weights are resident rather than
        only that a process is up: `ready` says narrator is listening, `loaded`
        says the engine underneath it has a voice in memory.

        **Where the weights ride depends on the engine, and the message says
        only what that engine reads.** `higgs-v3` REFUSES `modelDir` by name on
        both arms
        (`resolve_load_voice`: "the served model is the launch script's
        argument, not a per-load field") and resolves `voice` in the
        NARRATOR_HIGGS_VOICES document this engine was constructed with — so
        the message carries the voice id and `warm`, nothing else, and the
        document's entry is where `weights_dir` already is, as `checkpointDir`.
        `weights_dir` is still taken here so the two can be COMPARED: the
        document and this call are two statements of one fact, and a load that
        checks them agree is the difference between one owner and two.

        **No `caps` are sent, and that is a decision rather than an omission.**
        narrator's caps channel is `register_voice_caps`, whose key vocabulary
        is its older engine's (`temperature`, `topP`, `minP`, `repPenalty`, the
        four `eos*` levers, `maxCharsPerSec`) and which **raises on a key it
        does not know**;
        `higgs_v3_config_from_worker_kwargs` refuses the whole payload by name.
        A Higgs voice's sampling reaches narrator through the DOCUMENT instead
        (`narratorvoices.voice_entry`, key `sampling`), which is the channel
        narrator's `load_voices` reads it from on both arms.
        """
        request: dict[str, Any] = {
            "action": "load",
            "voice": voice,
            # Explicit, though narrator's own default is true: a first load may
            # spend time on discarded warm-up renders, and a load-voice job is
            # an operator's explicit order that would rather pay it here than in
            # the first chunk of a book.
            "warm": warm,
        }
        if self._voices is None:
            request["modelDir"] = str(weights_dir)
        else:
            # Refused HERE, by name, for a voice the document does not carry or
            # a directory it does not agree with — narrator would refuse the
            # first the same way, after the process is up.
            named = self._voices.weights_for(voice)
            if named != weights_dir:
                raise EngineError(
                    f"{self.name} was asked to load {voice!r} from {weights_dir}, "
                    f"but {self._voices.path} names {named} for it. The document "
                    "is what narrator reads; two directories for one voice is a "
                    "load nobody can vouch for"
                )
        loaded: dict[str, Any] | None = None
        for message in self.converse(
            request,
            terminal=frozenset({"loaded"}),
            silence_timeout=LOAD_SILENCE_TIMEOUT_SECONDS,
        ):
            if message["type"] == "loaded":
                loaded = message
            elif on_progress is not None:
                on_progress(f"{self.name}: {message['type']} {message}")
        if loaded is None:  # unreachable: `loaded` is the terminal type
            raise EngineError(f"{self.name} ended its load without a loaded line")
        return loaded

    # ---------------------------------------------------------------- stop

    def stop(self) -> None:
        """`quit` on stdin first, then SIGTERM. Never SIGKILL.

        narrator's own module docstring calls the stdin `quit` action its primary
        teardown: it unwinds the stdin loop from inside the process, runs the
        atexit hooks and releases the GPU. SIGTERM reaches the same place through
        a handler that raises `SystemExit(143)`, and it is the backstop for a
        worker that has stopped reading its stdin. Both are here because the
        first is cleaner and the second always arrives. Neither is SIGKILL —
        force-killing a process stuck in a WSL dxg GPU wait wedges the whole WSL
        VM until Windows reboots, which is why `SubprocessEngine.stop()` reports
        a timeout instead of escalating.
        """
        process = self._process
        if process is not None and process.poll() is None:
            try:
                self.send({"action": "quit"})
            except EngineError:
                # The pipe is gone, so the worker is already on its way out and
                # SIGTERM below is the only thing left to send — which is what
                # would have been sent anyway. Not swallowed silently: whatever
                # killed it wrote to stderr, and that is this engine's log.
                pass
            else:
                try:
                    process.wait(timeout=QUIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        super().stop()
