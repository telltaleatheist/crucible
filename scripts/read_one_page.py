#!/usr/bin/env python3
"""Read ONE page through a Crucible server and say what came back.

PHASE15-HOST.md section 8, T6 and T7. The two stages differ in which server
they point at and nothing else, which is the exit condition being tested:
*"an app cannot tell whether a page was read by vLLM in WSL, llama.cpp on
Windows, or mlx-vlm on a Mac, and must not be able to"*.

So this script builds its request from `crucible/pages.py` — the one owner of
the prompt, the dpi, the pixel budget and the ceiling (3.10 fact 7) — and
writes two files beside its output:

* `answer.json`, the model's raw answer, and
* `shape.json`, the DIALECT: the keys of each block and their categories, in
  reading order, with no text in it.

`shape.json` is what T7 diffs against T6's. Two engines reading one page are
allowed to disagree about a character; they are not allowed to disagree about
the dialect, and comparing the text would make a flaky test out of a real
one.

**`--load` OWNS THE RESIDENCY** (found by the first Windows run, 2026-09-14).
Crucible never loads a model to answer a chat request — it refuses
`model_not_resident` by name, on every backend — so a caller that wants a page
read submits the `load-model` job itself, waits for its `done`, asks, and
submits `unload-model` so the card is released. T6 pointed at the WSL server
without `--load` and got that refusal; the server was right and this script
was wrong. The unload always runs, because a llama-server still holding 6 GB
after a failed read is what breaks the NEXT stage.

**THE READ IS RECORDED BEFORE THE UNLOAD IS ATTEMPTED** (the second T6 run,
2026-09-15). It was not, and the consequence was the whole stage: the page WAS
read, `unload-model` came back `409 engine_in_use` because the server's own
settlement had already begun clearing the card, the `finally` raised, and that
refusal was the only thing the report carried — no `answer.json`, no
seconds/page, no block count, for a page that had been read successfully. The
server's half of that is fixed (`crucible/settle.py`: a clearance of the same
model is the same intent, not a conflict); this script's half is that a
measurement is written and printed the moment it exists, and the tidy-up
afterwards can only ADD a line. The exit code follows the same split: a failed
READ is a failure, and so is an unload that failed for any reason other than
the card already being clear of the model — which is not a failure at all, it
is the thing the unload asked for.

RASTERISING IS THE APP'S WORK (3.10: *"rasterising, parsing, EPUB assembly
stay in the app"*), so a PNG is taken as it is and a PDF needs `pypdfium2` —
which this script does not install and does not fall back from. Without it,
it says so and exits 2, and the caller reports the stage SKIPPED by name.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crucible import pages  # noqa: E402

#: `python -m pytest` exits 2 for "could not collect"; this uses the same
#: number for "could not even try", so a caller can tell a SKIP from a FAIL
#: without parsing a sentence.
EXIT_CANNOT_TRY = 2

#: The unload refusal that is NOT a failure. `unload-model` for a model that is
#: not resident means the card is already clear of it, which is the whole of
#: what the unload was asking for — so it is reported and the run carries on.
#: Every other code is a real failure of the tidy-up.
ALREADY_CLEAR = "model_not_resident"


def error_code(detail: str) -> str | None:
    """The `error.code` in a Crucible refusal or a failed job, if it has one.

    Read as a FIELD and never matched as a substring of a message: the caller
    branches on this, and a branch taken on the server's prose is a branch that
    breaks the next time somebody improves a sentence.
    """
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


class Refused(Exception):
    """A Crucible door or job said no, carrying the code it said it by."""

    def __init__(self, what: str, detail: str, *, status: int | None = None) -> None:
        self.detail = detail
        self.status = status
        self.code = error_code(detail)
        where = "" if status is None else f"HTTP {status} "
        super().__init__(f"{what}: {where}{detail[:600]}")


def rasterise(path: Path) -> bytes:
    """The page as a PNG at `pages.DPI`, under `pages.MAX_PIXELS`."""
    if path.suffix.lower() == ".png":
        return path.read_bytes()
    try:
        import pypdfium2
    except ImportError:
        print(
            f"{path} is a PDF and this interpreter has no pypdfium2. Crucible "
            "ships no rasteriser — that is the app's work (3.10) — so either "
            "`pip install pypdfium2` here, or drop a PNG at "
            f"{path.with_suffix('.png')} instead.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_CANNOT_TRY)
    document = pypdfium2.PdfDocument(str(path))
    page = document[0]
    # `scale` is in points-per-pixel terms: 72 pt to the inch, so DPI/72.
    bitmap = page.render(scale=pages.DPI / 72)
    image = bitmap.to_pil()
    if image.width * image.height > pages.MAX_PIXELS:
        # The processor would scale it anyway; doing it here means the bboxes
        # the parser scales back are against a frame this script knows.
        ratio = (pages.MAX_PIXELS / (image.width * image.height)) ** 0.5
        image = image.resize((int(image.width * ratio), int(image.height * ratio)))
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def post(url: str, token: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Crucible-Api": "1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise Refused(
            f"{url} refused",
            exc.read().decode("utf-8", "replace"),
            status=exc.code,
        ) from None
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"{url} did not answer: {type(exc).__name__}: {exc}")


def run_job(base: str, token: str, request: dict, timeout: float) -> dict:
    """Submit one job and poll it to a terminal state. Never returns a failure.

    Polled rather than streamed: this is a script, and a load that takes
    minutes is a load whose only interesting fact is when it stopped.
    """
    job = post(f"{base}/v1/jobs", token, request, timeout)
    job_id = job.get("job_id") or job.get("id")
    if job_id is None:
        raise SystemExit(f"{request['type']} was accepted without an id: {job}")
    while True:
        poll = urllib.request.Request(
            f"{base}/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}", "X-Crucible-Api": "1"},
        )
        with urllib.request.urlopen(poll, timeout=60) as response:
            status = json.loads(response.read().decode("utf-8"))
        if status["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(2.0)
    if status["status"] != "done":
        raise Refused(
            f"{request['type']} {request.get('model', '')} {status['status']}",
            json.dumps(status),
        )
    return status


def parse_blocks(text: str) -> list[dict]:
    """The dots dialect: a JSON array, sometimes fenced."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"the answer is not the {pages.DIALECT} dialect — it does not parse "
            f"as JSON: {exc}\nFirst 400 characters:\n{body[:400]}"
        )
    if not isinstance(parsed, list):
        raise SystemExit(
            f"the answer parses but is a {type(parsed).__name__}, and the "
            f"{pages.DIALECT} dialect is an array of blocks"
        )
    return parsed


def shape_of(blocks: list[dict]) -> list[dict]:
    """The DIALECT, with no text in it. What T7 diffs against T6's."""
    shape = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise SystemExit(f"block {index} is a {type(block).__name__}, not an object")
        bbox = block.get("bbox")
        shape.append(
            {
                "keys": sorted(block),
                "category": block.get("category"),
                "bbox_length": len(bbox) if isinstance(bbox, list) else None,
                "has_text": "text" in block,
            }
        )
    return shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True, help="a .png, or a .pdf with pypdfium2")
    parser.add_argument("--out", required=True, help="directory for the artifacts")
    parser.add_argument("--server", required=True, help="e.g. http://127.0.0.1:7100")
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--load",
        action="store_true",
        help="OWN THE RESIDENCY for this run: submit a load-model job first, "
        "and an unload-model job at the end so the card is released whatever "
        "happened in between. Crucible NEVER loads a model to answer a chat "
        "request (it refuses `model_not_resident` by name), so every server "
        "this script reads a page from needs this — the WSL one as much as "
        "the staged Windows one.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.server.rstrip("/")

    png = rasterise(Path(args.page))
    (out / "page.png").write_bytes(png)

    if args.load:
        started = time.monotonic()
        print(f"load-model {pages.MODEL_ID}")
        try:
            run_job(
                base,
                args.token,
                {"type": "load-model", "model": pages.MODEL_ID},
                args.timeout,
            )
        except Refused as exc:
            # Nothing has been measured and nothing is on the card: this is the
            # ordinary "could not even start" exit, not a result to record.
            raise SystemExit(str(exc)) from None
        print(f"  loaded in {time.monotonic() - started:.1f}s")

    # THE READ, AND EVERYTHING IT PRODUCES, BEFORE THE CARD IS TIDIED UP. Not a
    # `finally` around the read: a `finally` runs before the block's own value
    # is used, so the unload got to speak first and its refusal replaced a page
    # that had been read. The measurement is written and printed HERE; the
    # unload below can only add a line to it.
    reading: SystemExit | Refused | None = None
    try:
        read_the_page(base, args.token, png, out, args.timeout)
    except (SystemExit, Refused) as exc:
        reading = exc
        print(f"THE READ FAILED: {exc}", file=sys.stderr)

    tidying: Refused | None = None
    if args.load:
        # THE CARD IS RELEASED WHATEVER HAPPENED. A page that came back
        # truncated, or in the wrong dialect, is a result to record; a
        # llama-server still holding 6 GB afterwards is a test run that broke
        # the next stage.
        print(f"unload-model {pages.MODEL_ID}")
        try:
            run_job(
                base,
                args.token,
                {"type": "unload-model", "model": pages.MODEL_ID},
                args.timeout,
            )
            print("  unloaded")
        except Refused as exc:
            if exc.code == ALREADY_CLEAR:
                # NOT A FAILURE. The card is clear of the model, which is what
                # this asked for — the server's settlement having got there
                # first is the server doing its job.
                print(f"  the card was already clear of {pages.MODEL_ID}")
            else:
                tidying = exc
                print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)
        except SystemExit as exc:
            # The server stopped answering. A real failure of the tidy-up, and
            # still not a reason to lose the page that was already recorded.
            tidying = Refused("unload-model", str(exc))
            print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)

    if reading is not None:
        return 1
    return 1 if tidying is not None else 0


def read_the_page(
    base: str, token: str, png: bytes, out: Path, timeout: float
) -> None:
    """Read the page and RECORD it — the artifacts and the figures, in one go.

    Every measurement this stage exists for is written to disk and printed from
    inside here, so that by the time it returns there is nothing left for a
    later failure to lose.
    """
    body = pages.request_body(pages.data_uri(png))
    started = time.monotonic()
    answer = post(f"{base}/v1/openai/chat/completions", token, body, timeout)
    seconds = time.monotonic() - started
    (out / "answer.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")

    choice = answer["choices"][0]
    if pages.was_truncated(choice):
        raise SystemExit(
            f"the page came back TRUNCATED ({pages.TRUNCATED_FINISH_REASON}) at "
            f"the {pages.MAX_TOKENS}-token ceiling. That is a real answer about "
            "this page and not a defect in the run — record it."
        )
    blocks = parse_blocks(choice["message"]["content"])
    shape = shape_of(blocks)
    (out / "shape.json").write_text(json.dumps(shape, indent=2), encoding="utf-8")

    categories = sorted({block.get("category") for block in blocks})
    print(f"server:        {base}")
    print(f"seconds/page:  {seconds:.1f}")
    print(f"blocks:        {len(blocks)}")
    print(f"categories:    {categories}")
    print(f"dialect:       {pages.DIALECT} — parsed")
    print(f"artifacts:     {out}")


if __name__ == "__main__":
    raise SystemExit(main())
