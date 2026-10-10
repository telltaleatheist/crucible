from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from . import hosttools, procgroup, workerexit
from .backend import CUDA_LINUX
from .errors import CrucibleError, JobCancelled
from .logtail import led_by_first_error, tail_of_last_run

STOP_TIMEOUT_SECONDS = procgroup.STOP_TIMEOUT_SECONDS

STOP_ON_EOF_SECONDS = 30.0

POLL_SECONDS = 0.5

CANCEL_GRACE_SECONDS = 120.0

LOG_TAIL_LINES = procgroup.LOG_TAIL_LINES

READY = "ready"
PROGRESS = "progress"
RESULT = "result"
FAILED = "failed"
DONE = "done"
MESSAGE_KINDS = frozenset({READY, PROGRESS, RESULT, FAILED, DONE})


class WorkerError(CrucibleError):
    ...


@dataclass(frozen=True)
class WorkerOutcome:
    ready: dict[str, Any]
    results: tuple[dict[str, Any], ...] = field(default=())


def require_positional_results(
    outcome: WorkerOutcome, expected: int, unit: str
) -> tuple[dict[str, Any], ...]:
    if len(outcome.results) != expected:
        raise WorkerError(
            f"the worker returned {len(outcome.results)} {unit} result(s) for "
            f"{expected} {unit}(s); results are matched to work by position, so a "
            "short stream is not a partial answer, it is an answer to a different "
            "question"
        )
    return outcome.results


class _Reader:
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
        deadline = time.monotonic() + poll
        while True:
            try:
                item = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return
            yield item
            if item is None:
                return


def parse_message(line: str, script: Path) -> dict[str, Any]:
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


def _spawn(
    python: Path,
    script: Path,
    log_handle: Any,
    environment: dict[str, str] | None,
) -> subprocess.Popen[str]:
    if not python.is_file():
        raise WorkerError(
            f"no interpreter at {python}; this job type's env is not installed"
        )
    if not script.is_file():
        raise WorkerError(f"no worker script at {script}")

    merged = dict(os.environ)
    if environment is not None:
        merged.update(environment)
    merged["PATH"] = hosttools.worker_path(merged.get("PATH", ""))
    try:
        merged.update(hosttools.compiler_environment(python.parent.parent))
    except hosttools.HostToolError as exc:
        raise WorkerError(str(exc)) from None

    try:
        return subprocess.Popen(
            [str(python), str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            **procgroup.own_group(),
            env=merged,
            text=True,
            bufsize=1,
        )
    except procgroup.ProcessGroupError as exc:
        raise WorkerError(str(exc)) from exc
    except OSError as exc:
        raise WorkerError(f"could not spawn {python} {script}: {exc}") from exc


def _open_log(python: Path, script: Path, log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existed = log_path.is_file() and log_path.stat().st_size > 0
    handle = log_path.open("ab")
    handle.write(
        (
            ("\n" if existed else "")
            + f"=== crucible worker {script.name}, "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=== {python} {script}\n"
        ).encode("utf-8")
    )
    handle.flush()
    return handle


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
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> WorkerOutcome:
    script = Path(script)
    python = Path(python)
    log_path = Path(log_path)
    if not python.is_file():
        raise WorkerError(
            f"no interpreter at {python}; this job type's env is not installed"
        )
    if not script.is_file():
        raise WorkerError(f"no worker script at {script}")

    log_handle = _open_log(python, script, log_path)
    try:
        process = _spawn(python, script, log_handle, environment)
    except WorkerError:
        log_handle.close()
        raise

    conversation = _Conversation(process, script, log_path)
    try:
        return conversation.exchange(
            request,
            ready_silence_timeout=ready_silence_timeout,
            on_ready=on_ready,
            on_progress=on_progress,
            cancelled=cancelled,
            keep_open=False,
            on_result=on_result,
        )
    finally:
        log_handle.close()


class _CancelWatch:
    def __init__(
        self,
        cancelled: Callable[[], bool] | None,
        request: dict[str, Any] | None,
    ) -> None:
        self._cancelled = cancelled
        self._request = request
        self._asked_at: float | None = None

    @property
    def asked(self) -> bool:
        return self._asked_at is not None

    def check(self, conversation: "_Conversation") -> None:
        if self._cancelled is None:
            return
        script = conversation.script
        if self._asked_at is None:
            if not self._cancelled():
                return
            if self._request is None:
                conversation.stop()
                raise JobCancelled(f"{script.name} was cancelled")
            conversation._write(self._request, keep_open=True)
            self._asked_at = time.monotonic()
            return
        if time.monotonic() - self._asked_at > CANCEL_GRACE_SECONDS:
            conversation.stop()
            raise JobCancelled(
                f"{script.name} was asked to stop between steps and had not after "
                f"{CANCEL_GRACE_SECONDS:.0f}s, so it was stopped"
            )


class _Conversation:
    def __init__(
        self, process: subprocess.Popen[str], script: Path, log_path: Path
    ) -> None:
        assert process.stdin is not None and process.stdout is not None
        self.process = process
        self.script = script
        self.log_path = log_path
        self.reader = _Reader(process.stdout)
        # Taken as the worker starts, so its ending can say whether the out-of-memory
        # killer was what ended it (crucible/workerexit.py).
        self.oom_at_start = workerexit.read_oom_count()

    def ending(self, code: int | None) -> workerexit.Ending:
        return workerexit.how_it_ended(code, self.oom_at_start, workerexit.read_oom_count())

    def stop(self) -> None:
        _terminate(self.process, self.script, self.log_path)


    def _write(self, request: dict[str, Any], keep_open: bool) -> None:
        process, script, log_path = self.process, self.script, self.log_path
        assert process.stdin is not None
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            if not keep_open:
                process.stdin.close()
        except OSError as exc:
            code = process.poll()
            if code is None:
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = None
            if code is not None:
                ending = self.ending(code)
                raise WorkerError(
                    f"{script.name} {ending.phrase} before it read its request."
                    f"{ending.sentence()} {_log_tail(log_path)}"
                ) from None
            _terminate(process, script, log_path)
            raise WorkerError(
                f"could not send the request to {script.name}: {exc}. "
                f"{_log_tail(log_path)}"
            ) from None


    def exchange(
        self,
        request: dict[str, Any],
        *,
        ready_silence_timeout: float,
        on_ready: Callable[[dict[str, Any]], None] | None,
        on_progress: Callable[[dict[str, Any]], None] | None,
        cancelled: Callable[[], bool] | None,
        keep_open: bool,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        cancel_request: dict[str, Any] | None = None,
    ) -> WorkerOutcome:
        self._write(request, keep_open)
        watch = _CancelWatch(cancelled, cancel_request)

        process, script, log_path = self.process, self.script, self.log_path
        ready: dict[str, Any] | None = None
        results: list[dict[str, Any]] = []
        done = False
        ended = False
        deadline = time.monotonic() + ready_silence_timeout

        while not (done and keep_open):
            watch.check(self)

            for line in self.reader.lines(POLL_SECONDS):
                if line is None:
                    ended = True
                    break
                try:
                    message = parse_message(line, script)
                except WorkerError:
                    _terminate(process, script, log_path)
                    raise
                kind = message["type"]
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
                            f"{script.name} sent a result before it said it was "
                            "ready; the ready message is what tells the server how "
                            "many results to expect"
                        )
                    results.append(message)
                    if on_result is not None:
                        on_result(message)
                    continue

                if kind == FAILED:
                    _terminate(process, script, log_path)
                    raise WorkerError(
                        f"{script.name} failed: "
                        f"{message.get('message', '(no message)')}. "
                        f"{_log_tail(log_path)}"
                    )

                if done:
                    _terminate(process, script, log_path)
                    raise WorkerError(f"{script.name} sent two done messages")
                done = True

            if ended:
                break
            if ready is None and not watch.asked and time.monotonic() >= deadline:
                _terminate(process, script, log_path)
                raise WorkerError(
                    f"{script.name} said nothing at all for "
                    f"{ready_silence_timeout:.0f}s and has still not become ready. "
                    f"{_log_tail(log_path)}"
                )

        if watch.asked:
            raise JobCancelled(
                f"{script.name} stopped when it was asked to"
                + (f" and then exited {process.wait()}" if ended else "; it keeps running")
            )
        if not keep_open:
            code = process.wait()
            if code != 0:
                ending = self.ending(code)
                raise WorkerError(
                    f"{script.name} {ending.phrase}.{ending.sentence()} {_log_tail(log_path)}"
                )
        elif ended:
            ending = self.ending(process.wait())
            raise WorkerError(
                f"{script.name} {ending.phrase} in the middle of a request, after "
                f"{len(results)} result(s).{ending.sentence()} {_log_tail(log_path)}"
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


class WorkerSession:
    def __init__(
        self,
        *,
        python: Path,
        script: Path,
        log_path: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._python = Path(python)
        self._script = Path(script)
        self._log_path = Path(log_path)
        self._environment = environment
        self._log_handle: Any | None = None
        self._conversation: _Conversation | None = None


    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def pids(self) -> frozenset[int]:
        conversation = self._conversation
        if conversation is None or conversation.process.poll() is not None:
            return frozenset()
        return frozenset({conversation.process.pid})

    @property
    def alive(self) -> bool:
        conversation = self._conversation
        return conversation is not None and conversation.process.poll() is None


    def start(
        self,
        request: dict[str, Any],
        *,
        ready_silence_timeout: float,
        on_ready: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> WorkerOutcome:
        if self._conversation is not None:
            raise WorkerError(
                f"{self._script.name} is already started; a session is one worker, "
                "started once and stopped once"
            )
        if not self._python.is_file():
            raise WorkerError(
                f"no interpreter at {self._python}; this job type's env is not "
                "installed"
            )
        if not self._script.is_file():
            raise WorkerError(f"no worker script at {self._script}")

        self._log_handle = _open_log(self._python, self._script, self._log_path)
        try:
            process = _spawn(
                self._python, self._script, self._log_handle, self._environment
            )
        except WorkerError:
            self._log_handle.close()
            self._log_handle = None
            raise
        self._conversation = _Conversation(process, self._script, self._log_path)
        try:
            return self._exchange(
                request,
                ready_silence_timeout=ready_silence_timeout,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=None,
            )
        except BaseException:
            self._discard()
            raise

    def send(
        self,
        request: dict[str, Any],
        *,
        ready_silence_timeout: float,
        on_ready: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        cancel_request: dict[str, Any] | None = None,
    ) -> WorkerOutcome:
        if self._conversation is None:
            raise WorkerError(
                f"{self._script.name} has not been started; a session takes a "
                "request only after `start`"
            )
        if not self.alive:
            ending = self._conversation.ending(self._conversation.process.poll())
            raise WorkerError(
                f"{self._script.name} is no longer running: it {ending.phrase}."
                f"{ending.sentence()} The resident worker must be reloaded. "
                f"{_log_tail(self._log_path)}"
            )
        try:
            return self._exchange(
                request,
                ready_silence_timeout=ready_silence_timeout,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=cancelled,
                on_result=on_result,
                cancel_request=cancel_request,
            )
        except JobCancelled:
            if not self.alive:
                self._discard()
            raise

    def _exchange(
        self,
        request: dict[str, Any],
        *,
        ready_silence_timeout: float,
        on_ready: Callable[[dict[str, Any]], None] | None,
        on_progress: Callable[[dict[str, Any]], None] | None,
        cancelled: Callable[[], bool] | None,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        cancel_request: dict[str, Any] | None = None,
    ) -> WorkerOutcome:
        assert self._conversation is not None
        return self._conversation.exchange(
            request,
            ready_silence_timeout=ready_silence_timeout,
            on_ready=on_ready,
            on_progress=on_progress,
            cancelled=cancelled,
            keep_open=True,
            on_result=on_result,
            cancel_request=cancel_request,
        )

    def stop(self) -> None:
        conversation = self._conversation
        if conversation is None:
            return
        process = conversation.process
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=STOP_ON_EOF_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        try:
            _terminate(process, self._script, self._log_path)
        finally:
            self._discard()

    def _discard(self) -> None:
        self._conversation = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _terminate(process: subprocess.Popen[str], script: Path, log_path: Path) -> None:
    try:
        procgroup.stop_gracefully(process, script.name, STOP_TIMEOUT_SECONDS, log_path)
    except procgroup.ProcessGroupError as exc:
        raise WorkerError(str(exc)) from exc


def _log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    if not Path(log_path).is_file():
        return f"Its log is {log_path} (not written)."
    tail = tail_of_last_run(Path(log_path), lines)
    if not tail:
        return f"Its log is {log_path} (empty or unreadable)."
    return led_by_first_error(
        Path(log_path), f"Last {lines} lines of the latest run in {log_path}:\n{tail}"
    )


def cuda_library_path(env_dir: Path) -> str | None:
    site = sorted(env_dir.glob("lib/python*/site-packages"))
    if not site:
        return None
    packages = site[0]
    directories: list[str] = []
    nvidia = packages / "nvidia"
    if nvidia.is_dir():
        for child in sorted(nvidia.iterdir()):
            lib = child / "lib"
            if lib.is_dir():
                directories.append(str(lib))
    for bundled in sorted(packages.glob("*.libs")):
        if bundled.is_dir():
            directories.append(str(bundled))
    return os.pathsep.join(directories) if directories else None


TORCH_ALLOC_CONF_VAR = "PYTORCH_CUDA_ALLOC_CONF"
TORCH_ALLOC_CONF = "expandable_segments:True"


def torch_allocator_environment(
    backend_kind: str, inherited: dict[str, str] | None = None
) -> dict[str, str]:
    if backend_kind != CUDA_LINUX:
        return {}
    base = os.environ if inherited is None else inherited
    if TORCH_ALLOC_CONF_VAR in base:
        return {}
    return {TORCH_ALLOC_CONF_VAR: TORCH_ALLOC_CONF}


def torch_memory_cap(backend_kind: str, memory_bytes_estimate: int) -> int | None:
    if backend_kind != CUDA_LINUX:
        return None
    return memory_bytes_estimate


def worker_environment(env_dir: Path, inherited: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if inherited is None else inherited)
    addition = cuda_library_path(env_dir)
    if addition is None:
        return {}
    existing = base.get("LD_LIBRARY_PATH", "")
    return {
        "LD_LIBRARY_PATH": addition + (os.pathsep + existing if existing else "")
    }
