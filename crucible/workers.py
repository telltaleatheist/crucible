"""Running a job's work in an interpreter that is not this one.

PHASE4-AUDIO.md section 0. `llm` never had this problem — vLLM and mlx-lm are
servers, so Crucible starts one and talks HTTP to it. The phase 4 types are
libraries in envs of their own, so Crucible **runs** them: `<env python>
<crucible/jobs/<type>/worker.py>`, parameters handed over on stdin, newline
-delimited JSON coming back on fd 1, everything else on stderr and into a log.

This module is that plumbing, and only that. It knows nothing about audio,
models or job types; a worker's vocabulary is the job type's business and this
module carries the envelope. `asr` is the first user, `align` and `rvc` are the
next two, and the shape below is chosen so that those are a few dozen lines each.

The three rules this enforces are each a bug that has already happened
--------------------------------------------------------------------
- **fd 1 is results and nothing else.** narrator's aligner learned this on a
  401-chunk book: a library's logger wrote to stdout and corrupted the result
  stream. So a line on fd 1 that is not a JSON object of a kind this module knows
  is a **refusal naming the line**, never something skipped. A worker whose
  library prints is expected to have dup'd fd 1 away before importing it.
- **Results are matched to work by position.** A worker never echoes back an
  index, because an index a worker reports is an index a worker can get wrong.
  The Nth `result` line is the Nth unit of work, and `require_positional_results`
  refuses a run whose count does not match what the caller expected.
- **A worker holding an accelerator is never SIGKILLed.** SIGTERM, wait, and if
  it will not go, say so — killing a process that holds CUDA wedges WSL2 until
  Windows reboots. Same rule, same reason, as `crucible/engines/base.py`.

What `align` and `rvc` will need that `asr` does not
----------------------------------------------------
Two things, and neither of them is a change to the envelope:

- **Many inputs and many outputs.** `asr` takes one file and produces one
  document; `rvc` takes 1,400 sentence FLACs and must produce 1,400 back (a
  missing one is a failed job, not a short answer), and `align` takes one clip
  per chunk. So those types will hand the worker a *list* of input paths in the
  request and have it write each output to a path the server named, reporting
  only the path and the timings on fd 1 — audio does not belong in a JSON line.
  `require_positional_results` is already the check that every unit came back.
- **A model that stays resident across a whole book.** `asr` loads, transcribes
  one file and exits, so one spawn per job is right for it. The Qwen3 aligner is
  resident across hundreds of chunks and one load, and `rvc` recycles its worker
  every 96 files as a *memory* bound rather than a throughput choice. Those need
  a worker that outlives a single job: the same `_Reader` and the same message
  vocabulary, but held open by a residency object, fed a new request on the same
  stdin, and stopped by whatever unloads models. That is an addition to this
  module, not a different one — `run_worker` below is the one-shot case of it,
  and the split to make is `start`/`send`/`stop` with `run_worker` as the three
  in a row.
"""

from __future__ import annotations

import errno
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import CrucibleError, JobCancelled

#: How long `stop()` waits for SIGTERM to be honoured before it gives up and says
#: so. It never escalates to SIGKILL. Same number and same reason as
#: `crucible/engines/base.py`.
STOP_TIMEOUT_SECONDS = 180.0

#: How often the reading loop wakes to notice a cancel or a missed deadline. It
#: is not a timeout on anything: a worker that is working silently for an hour is
#: a worker that is working.
POLL_SECONDS = 0.5

#: How much of the worker's stderr log a failure quotes.
LOG_TAIL_LINES = 40

#: The message kinds a worker may put on fd 1. Anything else is a protocol error
#: rather than something to skip — see the module docstring.
READY = "ready"
PROGRESS = "progress"
RESULT = "result"
FAILED = "failed"
DONE = "done"
MESSAGE_KINDS = frozenset({READY, PROGRESS, RESULT, FAILED, DONE})


class WorkerError(CrucibleError):
    """A worker would not start, broke the protocol, or died. Names which."""


@dataclass(frozen=True)
class WorkerOutcome:
    """What one worker run produced.

    `ready` is the worker's own description of what it is about to do — for `asr`,
    the decoded duration and the number of windows it implies. It arrives before
    any result, so a job type can size its progress against it and refuse early
    if the worker's plan disagrees with the server's.
    """

    ready: dict[str, Any]
    results: tuple[dict[str, Any], ...] = field(default=())


def require_positional_results(
    outcome: WorkerOutcome, expected: int, unit: str
) -> tuple[dict[str, Any], ...]:
    """The results, or a refusal naming both counts.

    The Nth result is the Nth unit of work and there is no id to check that
    against, so the count is the whole check — which is why it is a refusal and
    not a warning.
    """
    if len(outcome.results) != expected:
        raise WorkerError(
            f"the worker returned {len(outcome.results)} {unit} result(s) for "
            f"{expected} {unit}(s); results are matched to work by position, so a "
            "short stream is not a partial answer, it is an answer to a different "
            "question"
        )
    return outcome.results


# --------------------------------------------------------------------- reading


class _Reader:
    """Pumps a worker's fd 1 into a queue on a thread, so the caller can poll.

    Line-by-line `for line in stdout` would block the caller past a cancel and
    past any deadline; a thread plus a queue lets the run loop wake every
    `POLL_SECONDS` and decide whether it still wants to be here.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for line in self._stream:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def lines(self, poll: float) -> Iterator[str | None]:
        """Yields a line, or None once for end-of-stream, or nothing on a timeout."""
        while True:
            try:
                item = self._lines.get(timeout=poll)
            except queue.Empty:
                return
            yield item
            if item is None:
                return


def parse_message(line: str, script: Path) -> dict[str, Any]:
    """One line of fd 1 as a message, or a refusal naming the line.

    Every rejection here quotes the offending line, because the whole class of
    bug this guards against — a library printing to stdout — is only diagnosable
    from the text that appeared.
    """
    text = line.strip()
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkerError(
            f"{script.name} wrote a line to fd 1 that is not JSON ({exc}): {text[:300]!r}. "
            "fd 1 carries results and nothing else; a library that prints must be "
            "imported after the worker has pointed the original fd 1 at stderr"
        ) from None
    if not isinstance(message, dict):
        raise WorkerError(
            f"{script.name} wrote a JSON {type(message).__name__} to fd 1, not an "
            f"object: {text[:300]!r}"
        )
    kind = message.get("type")
    if kind not in MESSAGE_KINDS:
        raise WorkerError(
            f"{script.name} sent a message of type {kind!r}; a worker sends "
            f"{sorted(MESSAGE_KINDS)}"
        )
    return message


# --------------------------------------------------------------------- running


def run_worker(
    *,
    python: Path,
    script: Path,
    request: dict[str, Any],
    log_path: Path,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    environment: dict[str, str] | None = None,
) -> WorkerOutcome:
    """Run one worker to completion and return what it said.

    `ready_silence_timeout` bounds the wait for the `ready` line, and it is a
    **silence** timeout rather than a deadline: any message resets it, so a
    worker that is reporting decode progress for twenty minutes before it can say
    how much work there is never trips it, while a worker that has said nothing
    at all since it started does. After `ready` there is no timeout of any kind,
    because there is no honest one — a worker transcribing an eighteen-hour book
    is quiet for long stretches, and a clock invented here would kill real work.
    What ends a long run early is a cancel, checked every `POLL_SECONDS`.

    Raises `JobCancelled` if `cancelled()` goes true, `WorkerError` for anything
    else. Never returns a partial outcome: a caller that gets a `WorkerOutcome`
    got a worker that said `done`.
    """
    script = Path(script)
    python = Path(python)
    if not python.is_file():
        raise WorkerError(
            f"no interpreter at {python}; this job type's env is not installed"
        )
    if not script.is_file():
        raise WorkerError(f"no worker script at {script}")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    log_handle.write(
        (
            f"=== crucible worker {script.name}, "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=== {python} {script}\n"
        ).encode("utf-8")
    )
    log_handle.flush()

    merged = dict(os.environ)
    if environment is not None:
        merged.update(environment)

    try:
        process = subprocess.Popen(
            [str(python), str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            # Its own session, so a SIGTERM reaches anything the worker forked —
            # ffmpeg, in the `asr` case — and not only the worker itself.
            start_new_session=True,
            env=merged,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        log_handle.close()
        raise WorkerError(f"could not spawn {python} {script}: {exc}") from exc

    try:
        return _converse(
            process=process,
            script=script,
            request=request,
            log_path=log_path,
            ready_silence_timeout=ready_silence_timeout,
            on_ready=on_ready,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    finally:
        log_handle.close()


def _converse(
    *,
    process: subprocess.Popen[str],
    script: Path,
    request: dict[str, Any],
    log_path: Path,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
    cancelled: Callable[[], bool] | None,
) -> WorkerOutcome:
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        # The request is the whole conversation for a one-shot worker, so stdin
        # is closed: a worker blocked on a read it will never satisfy is a hang
        # with no error, and closing turns it into an EOF the worker can act on.
        process.stdin.close()
    except OSError as exc:
        # Much the commonest reason a request cannot be delivered is that the
        # worker is already dead: a broken env dies at import time, before it
        # reads a byte, and the write then fails with EPIPE. Report the death and
        # its log, which is the thing that explains this, rather than the pipe
        # error it caused.
        code = process.poll()
        if code is None:
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
        if code is not None:
            raise WorkerError(
                f"{script.name} exited {code} before it read its request. "
                f"{_log_tail(log_path)}"
            ) from None
        _terminate(process, script, log_path)
        raise WorkerError(
            f"could not send the request to {script.name}: {exc}. "
            f"{_log_tail(log_path)}"
        ) from None

    reader = _Reader(process.stdout)
    ready: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    done = False
    deadline = time.monotonic() + ready_silence_timeout

    while True:
        if cancelled is not None and cancelled():
            _terminate(process, script, log_path)
            raise JobCancelled(f"{script.name} was cancelled")

        ended = False
        for line in reader.lines(POLL_SECONDS):
            if line is None:
                ended = True
                break
            try:
                message = parse_message(line, script)
            except WorkerError:
                # A worker that has broken the protocol is a worker whose later
                # lines cannot be trusted either, and it may be holding the card.
                # Stop it before reporting.
                _terminate(process, script, log_path)
                raise
            kind = message["type"]
            # Any message is proof of life, so the silence clock starts again.
            deadline = time.monotonic() + ready_silence_timeout

            if kind == READY:
                if ready is not None:
                    _terminate(process, script, log_path)
                    raise WorkerError(f"{script.name} sent two ready messages")
                ready = message
                if on_ready is not None:
                    on_ready(message)
                continue

            if kind == PROGRESS:
                if on_progress is not None:
                    on_progress(message)
                continue

            if kind == RESULT:
                if ready is None:
                    _terminate(process, script, log_path)
                    raise WorkerError(
                        f"{script.name} sent a result before it said it was ready; "
                        "the ready message is what tells the server how many "
                        "results to expect"
                    )
                results.append(message)
                continue

            if kind == FAILED:
                _terminate(process, script, log_path)
                raise WorkerError(
                    f"{script.name} failed: {message.get('message', '(no message)')}. "
                    f"{_log_tail(log_path)}"
                )

            # DONE: the worker has said everything it is going to say. Keep
            # draining until end-of-stream so the exit code is the last word.
            if done:
                _terminate(process, script, log_path)
                raise WorkerError(f"{script.name} sent two done messages")
            done = True

        if ended:
            break
        if ready is None and time.monotonic() >= deadline:
            _terminate(process, script, log_path)
            raise WorkerError(
                f"{script.name} said nothing at all for {ready_silence_timeout:.0f}s "
                "and has still not become ready. "
                f"{_log_tail(log_path)}"
            )

    code = process.wait()
    if code != 0:
        raise WorkerError(
            f"{script.name} exited {code}. {_log_tail(log_path)}"
        )
    if ready is None:
        raise WorkerError(
            f"{script.name} exited 0 without ever saying it was ready. "
            f"{_log_tail(log_path)}"
        )
    if not done:
        raise WorkerError(
            f"{script.name} exited 0 without saying done, after "
            f"{len(results)} result(s). An answer that stopped early is not a "
            f"short answer, it is an unfinished one. {_log_tail(log_path)}"
        )
    return WorkerOutcome(ready=ready, results=tuple(results))


def _terminate(process: subprocess.Popen[str], script: Path, log_path: Path) -> None:
    """SIGTERM the worker's process group and wait. Never SIGKILL."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise WorkerError(
                f"could not signal {script.name} (pid {process.pid}): {exc}"
            ) from exc
        return
    try:
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise WorkerError(
            f"{script.name} (pid {process.pid}) did not exit within "
            f"{STOP_TIMEOUT_SECONDS:.0f}s of SIGTERM. Crucible does not SIGKILL a "
            "process that may be holding CUDA — that wedges WSL2 until Windows "
            f"reboots. Kill it by hand if you must; its log is {log_path}"
        ) from None


def _log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """The last lines of the worker's stderr, quoted into the error.

    PHASE4-AUDIO.md section 6: an error body carries what went wrong, not a
    pointer to a file on a machine the client may not be able to read.
    """
    if not Path(log_path).is_file():
        return f"Its log is {log_path} (not written)."
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Its log is {log_path} (unreadable: {exc})."
    tail = "\n".join(text.splitlines()[-lines:])
    return f"Last {lines} lines of {log_path}:\n{tail}"
