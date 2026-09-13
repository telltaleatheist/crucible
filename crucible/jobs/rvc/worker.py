"""The `rvc` worker: ultimate-rvc, run in its own interpreter, in batches.

PHASE4-AUDIO.md sections 0 and 4. This module is **standalone**. It imports the
standard library and nothing else — not even ultimate-rvc, which it *runs* rather
than imports — and nothing from `crucible`, because the env it runs in has no
`crucible` installed and never will.

Why it spawns urvc instead of importing it
------------------------------------------
**Batching is a memory bound, not a throughput choice.** BookForge recycles a
convert process every 96 files, proven necessary on a 64 GB Mac on 2026-07-17: a
ten-minute chunk grew the process by about 1.5 GB and never released it, unified
memory ballooned to ~50 GB, the machine hit swap and RVC slowed about 5x. The env
carries a per-file `torch.mps.empty_cache` patch and it is *not sufficient*. What
is sufficient is the process dying, because then the OS reclaims everything it
leaked regardless of what leaked. So each batch is its own child that exits, and
that is only possible if the engine is a command rather than an import.

This is engine knowledge. The client never sees `batch_size`, never sees a batch
boundary, and gets one output per input either way (`crucible/jobs/rvc/__init__.py`).

**Never `urvc.exe`, and never the `urvc` console script.** pip's console-script
launcher bakes the interpreter path in at install time as a shebang, and an env
that was installed by extracting into a temp directory and moving it into place
then points at a python that no longer exists — which fails with exit 1 and
*zero output*, because the launcher dies before python starts
(`electron/rvc-bridge.ts:53`). `sys.executable -m ultimate_rvc.cli.main` is the
same program reached a way that cannot go stale, and this paragraph is here so
that nobody simplifies it back.

The wire, in full
-----------------
    stdin   one object: {models_dir, model_name, output_dir, input_dir, inputs,
                         index_rate, protect_rate, n_semitones, batch_size,
                         extension, f0_method?, hop_length?}
            `f0_method` and `hop_length` are THE ONLY optional keys in any phase
            4 worker's request, and their absence is meaningful rather than a
            refusal — see `_convert_args`.

    fd 1    {"type": "ready", "files", "batches", "batch_size", "model"}
            {"type": "progress", "stage", "processed", "total", "batch",
             "batches"}
            {"type": "result", "bytes"}              one per input, BY POSITION
            {"type": "result", "error": "..."}       an input with no output
            {"type": "failed", "message"}            the whole run
            {"type": "done"}

fd 1 is results and nothing else
--------------------------------
The first thing this file does is dup fd 1 somewhere safe and point the original
at stderr. Whose lesson it is: narrator's aligner, Owen's 401-chunk witches book,
2026-09-05 — a library logged a warning to stdout between two result lines and
the parent's `json.loads` died with "Extra data". Here the risk is even plainer,
because urvc's own progress goes to *its* stdout and this worker reads it; that
stream is parsed, never forwarded.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

#: urvc's own progress line, and the app's regex verbatim
#: (`electron/rvc-bridge.ts:257`). It counts within the batch, so the worker adds
#: the offset of the batches already done.
PROGRESS_LINE = re.compile(r"^\[RVC\]\s+(\d+)/(\d+)")


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def require(request: dict, key: str, kind):
    """One required key, or a refusal naming it."""
    if key not in request:
        raise KeyError(
            f"the rvc request has no {key!r}; every parameter except 'f0_method' "
            "and 'hop_length' is required, and those two are absent on purpose"
        )
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        raise KeyError(
            f"the rvc request's {key!r} must be "
            f"{'/'.join(k.__name__ for k in kinds)}, got {type(value).__name__}"
        )
    return value


def _convert_args(request: dict, input_dir: str, output_dir: str) -> list[str]:
    """The `python -m ultimate_rvc.cli.main generate convert-dir ...` argv.

    **An absent `f0_method` or `hop_length` means the flag is OMITTED**, so urvc
    keeps its own default. It does not mean a value Crucible chose. This is the
    one place in the whole server where "absent" is a meaningful wire value
    rather than a refusal, and the reason is that urvc's defaults are the tuned
    ones: a server that filled them in would be substituting its guess for the
    engine's measurement and the output would not say which it got.

    `hop_length` is tested with `is not None` and not for truthiness, because its
    range is 1-512 and nothing meaningful in it is falsy — but read the way the
    others are, a future 0 would silently mean "unset" instead of being refused
    by urvc's own CLI, which is the louder failure.
    """
    argv = [
        sys.executable,
        "-m",
        "ultimate_rvc.cli.main",
        "generate",
        "convert-dir",
        input_dir,
        output_dir,
        require(request, "model_name", str),
        "--index-rate",
        str(require(request, "index_rate", (int, float))),
        # protect_rate's scale is INVERTED — see `jobs/rvc/__init__.py`, where it
        # is validated. Nothing is adjusted here; the number goes across as the
        # client sent it.
        "--protect-rate",
        str(require(request, "protect_rate", (int, float))),
        "--input-glob",
        f"*.{require(request, 'extension', str)}",
        "--output-ext",
        require(request, "extension", str),
    ]
    f0_method = request.get("f0_method")
    if f0_method is not None:
        argv += ["--f0-method", str(f0_method)]
    hop_length = request.get("hop_length")
    if hop_length is not None:
        argv += ["--hop-length", str(hop_length)]
    n_semitones = require(request, "n_semitones", int)
    if n_semitones:
        # Zero is urvc's own default and the flag is omitted for it, matching the
        # app (`rvc-bridge.ts`): a pitch shift of zero and no pitch shift are the
        # same thing, and sending `--n-semitones 0` would only add a way for the
        # two to stop being the same thing.
        argv += ["--n-semitones", str(n_semitones)]
    return argv


def _stage(batch: list[str], input_dir: str, staging: str) -> None:
    """Hardlink this batch's files into a fresh directory, copying if we must.

    Hardlinks because a book's sentences are hundreds of megabytes and copying
    them per batch would double the I/O for nothing. A copy is the fallback and
    not a silent one — it happens only when the link fails, which on one
    filesystem means it cannot be done at all.
    """
    for name in batch:
        source = os.path.join(input_dir, name)
        destination = os.path.join(staging, name)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)


def _run_batch(argv: list[str], models_dir: str, on_line) -> None:
    """One urvc process, its stdout read line by line, raising on a bad exit.

    The child inherits this worker's environment, which the SERVER set
    (`URVC_SKIP_INIT`, `HF_HUB_OFFLINE`, `KMP_DUPLICATE_LIB_OK`,
    `OMP_NUM_THREADS`) — see `jobs/rvc/__init__.py` for what each one is for.
    `URVC_MODELS_DIR` is added here because it is the one that depends on this
    job's staged model root.
    """
    environment = dict(os.environ)
    environment["URVC_MODELS_DIR"] = models_dir
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None and process.stderr is not None
    tail: list[str] = []
    for line in process.stdout:
        stripped = line.rstrip("\n")
        tail.append(stripped)
        del tail[:-40]
        # urvc's stdout is READ, never forwarded: fd 1 here is the result stream.
        print(f"[urvc] {stripped}")
        on_line(stripped)
    errors = process.stderr.read()
    code = process.wait()
    if code != 0:
        print(errors)
        detail = (errors.strip() or "\n".join(tail)).strip()[-1500:]
        raise RuntimeError(f"urvc convert-dir exited {code}: {detail}")


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return fail("the rvc worker was given no request on stdin")
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return fail(f"the rvc request is not JSON: {exc}")
    if not isinstance(request, dict):
        return fail(f"the rvc request must be a JSON object, got {type(request).__name__}")

    try:
        models_dir = require(request, "models_dir", str)
        model_name = require(request, "model_name", str)
        input_dir = require(request, "input_dir", str)
        output_dir = require(request, "output_dir", str)
        inputs = require(request, "inputs", list)
        batch_size = require(request, "batch_size", int)
        require(request, "extension", str)
        require(request, "index_rate", (int, float))
        require(request, "protect_rate", (int, float))
        require(request, "n_semitones", int)
    except KeyError as exc:
        return fail(str(exc.args[0]))
    if not inputs:
        return fail("the rvc request lists no inputs")
    if batch_size < 1:
        return fail(f"batch_size must be at least 1, got {batch_size}")

    os.makedirs(output_dir, exist_ok=True)
    batches = [inputs[at : at + batch_size] for at in range(0, len(inputs), batch_size)]
    send(
        "ready",
        files=len(inputs),
        batches=len(batches),
        batch_size=batch_size,
        model=model_name,
    )

    done_before = 0
    for number, batch in enumerate(batches, start=1):
        # A fresh temp directory per batch, and a child that EXITS when it is
        # finished with it. That exit is the memory bound; see the module
        # docstring for what happens without it.
        staging = tempfile.mkdtemp(prefix="crucible-rvc-")
        try:
            _stage(batch, input_dir, staging)

            def on_line(text: str, offset: int = done_before) -> None:
                match = PROGRESS_LINE.match(text.strip())
                if match is None:
                    return
                send(
                    "progress",
                    stage="converting",
                    processed=offset + int(match.group(1)),
                    total=len(inputs),
                    batch=number,
                    batches=len(batches),
                )

            _run_batch(
                _convert_args(request, staging, output_dir), models_dir, on_line
            )
        except Exception as exc:
            # A batch that could not run at all is the whole job's problem: every
            # file in it is missing an output, and the run cannot tell which of
            # them the engine would have managed. Say so once, with the engine's
            # own words, rather than 96 identical per-file errors.
            return fail(
                f"batch {number} of {len(batches)} failed: {type(exc).__name__}: {exc}"
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        done_before += len(batch)

    # One result per input, IN THE ORDER THE INPUTS ARRIVED. A result carries no
    # name and no index, on purpose: the server matches by position and already
    # knows what it sent, and a name a worker echoes back is a name a worker can
    # get wrong. The failure case names the file in its MESSAGE, which is prose
    # for a human and not a field anything matches on.
    for name in inputs:
        produced = os.path.join(output_dir, name)
        if not os.path.isfile(produced):
            send("result", error=f"urvc wrote no output for {name}")
            continue
        send("result", bytes=os.path.getsize(produced))

    send("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
