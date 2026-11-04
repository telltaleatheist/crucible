"""Where the ENGINE's log lines go - a decision that belongs to the HOST.

THE BUG THIS EXISTS FOR (found 2026-09-05, driving Higgs v3 through
`python -m narrator.serve` on the Mac):

    RuntimeError: non-JSON on stdout: '[HIGGS-MLX] loading Higgs v3 from ...'

`narrator.serve`'s **stdout IS the JSON-lines protocol**. Any bare `print` from
the engine layer lands between two protocol messages and breaks the client's
parse. The engine has 148 of them (111 under `orpheus/`, 37 under
`higgs/`) - counted by AST in
`tests/test_engine_log_stream.py::LogCallCountTest`, which is where that
number lives so it cannot drift apart across four docstrings again. It was 126
at the migration below, and the count moving when the engine gains a log line is
the test working.

THIS SENTENCE IS ITSELF UNDER TEST since 2026-09-13
(`test_the_prose_agrees_with_the_count`), because it drifted: it said 136/25
while the test said 28 and the truth was 32. Measuring one copy of a fact does
not stop a second copy disagreeing - so both are checked now. Change the numbers
here and in that file in the same commit that adds a log call.

WHY THIS IS NOT "MOVE THEM ALL TO STDERR". One engine serves hosts with
INCOMPATIBLE stdout contracts, and the other host parses the very lines that
break the first one. Measured in `electron/parallel-tts-bridge.ts`, whose
worker `stdout` handler runs FIVE parsers that its `stderr` handler does not:

    MODEL_LOAD_START_RE     /Loading Orpheus model with/  -> "Loading model weights..."
    MODEL_LOAD_DONE_RE      /model loaded!/               -> the load bar completing
    REPAIR_START_RE         /hit the MLX audio-token cap/ -> the re-split ladder bar
    parseMlxHeartbeat()     "[ORPHEUS] MLX batch generating: ..."  -> the batch bar
    parseOrpheusGuardEvent()                              -> the guard-event index

Every one of those strings is printed by `engine/orpheus/`. Moving them to
stderr unconditionally would fix the serve protocol and silently break the
audiobook progress UI - a different bug, in a place nobody would look.

SO THE STREAM IS THE HOST'S TO SET, and the two hosts differ:

    narrator.serve            stdout is JSON. Engine logs -> STDERR (the default).
    narrator.compat.worker    stdout is what parallel-tts-bridge.ts PARSES, so
    (and render.worker)       it calls `set_log_stream(sys.stdout)` and today's
                              behaviour is preserved byte for byte.

**NO STRING CHANGES.** The bridge's regexes match these lines exactly as they
are; this module only decides which file descriptor they are written to.

THE DEFAULT IS STDERR ON PURPOSE. A host that forgets to choose gets the safe
answer - logs on the log stream - rather than corrupting a structured stdout it
never told anyone about. The host that genuinely needs stdout is the one that
has to say so, and `tests/test_engine_log_stream.py` pins both directions.

`flush=True` always: a watchdog reading lines to decide whether a worker is
alive must not be waiting on a block buffer. (The bridge already exports
PYTHONUNBUFFERED, so this is a safety net rather than a change.)

Imports `sys` and nothing else - `narrator.engine` must stay importable with no
torch, no mlx and no transformers (`tests/test_engine_lazy_imports.py`).
"""
import sys

#: The stream engine log lines are written to. None means "stderr, resolved at
#: call time" - deliberately not a captured `sys.stderr` reference, so a host
#: (or a test) that swaps `sys.stderr` is honoured.
_STREAM = None


def set_log_stream(stream) -> None:
    """Point the engine's log lines at `stream`.

    Called ONCE by a host, before any engine is built. `None` restores the
    default (stderr). A host whose stdout carries structured data must NOT call
    this with stdout - see the module docstring for the two that do and why.
    """
    global _STREAM
    if stream is not None and not hasattr(stream, 'write'):
        raise ValueError(
            f'set_log_stream() needs a writable stream or None; got '
            f'{type(stream).__name__}.')
    _STREAM = stream


def log_stream():
    """The stream in effect right now. Resolves the default at CALL time."""
    return sys.stderr if _STREAM is None else _STREAM


def current_log_stream():
    """What `set_log_stream` was last given - `None` meaning "the default".

    Distinct from `log_stream()`, which resolves `None` to stderr. A caller that
    wants to put the setting BACK (a test, a nested scope) needs the setting,
    not the resolution: restoring `log_stream()`'s answer would pin the stream
    to whatever `sys.stderr` happened to be, turning a temporary override into a
    permanent one.
    """
    return _STREAM


def log(*args, **kwargs) -> None:
    """`print`, to the host's chosen stream, flushed.

    Signature-compatible with the builtin on purpose: all 126 call sites were
    rewritten mechanically from `print(...)` to `log(...)` with every string
    untouched, so `sep` / `end` still work.

    NOT AN IDENTICAL DIFF, THOUGH, AND THE FIRST WRITE-UP SAID SO TOO LOOSELY.
    Exactly:

        94  bare `print(...)` in engine/orpheus/  -> call name only
        17  `print(..., file=sys.stderr)` in engine/orpheus/   } the kwarg was
        14  `print(..., file=sys.stderr)` in engine/higgs/     } DELETED
         1  engine/higgs/mlx_backend.py::_log, authored as log()
       ---
       126

    The 31 that named `file=sys.stderr` therefore CHANGED DESTINATION - from
    stderr always, to whatever the host chose (stdout under
    `narrator.compat.worker`). Calling that "nothing but call-name swaps" was
    wrong. It was deliberate - a call that names its own stream cannot be
    routed, which is the whole point of this module - and it was checked before
    it was made, twice: none of the 31 is reachable from `compat.worker`, and
    only one of their strings matches any bridge pattern (one the bridge already
    runs on both streams). See PORT_NOTES 13.5.
    """
    kwargs.setdefault('file', log_stream())
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


__all__ = ['log', 'log_stream', 'set_log_stream']
