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
was wrong. The unload runs in a `finally`, because a llama-server still
holding 6 GB after a failed read is what breaks the NEXT stage.

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
        raise SystemExit(
            f"{url} refused: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:600]}"
        )
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
        raise SystemExit(
            f"{request['type']} {request.get('model', '')} "
            f"{status['status']}: {json.dumps(status)[:600]}"
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
        run_job(
            base,
            args.token,
            {"type": "load-model", "model": pages.MODEL_ID},
            args.timeout,
        )
        print(f"  loaded in {time.monotonic() - started:.1f}s")

    reading: BaseException | None = None
    try:
        body = pages.request_body(pages.data_uri(png))
        started = time.monotonic()
        answer = post(
            f"{base}/v1/openai/chat/completions", args.token, body, args.timeout
        )
        seconds = time.monotonic() - started
    except BaseException as exc:
        reading = exc
        raise
    finally:
        if args.load:
            # THE CARD IS RELEASED WHATEVER HAPPENED. A page that came back
            # truncated, or in the wrong dialect, is a result to record; a
            # llama-server still holding 6 GB afterwards is a test run that
            # broke the next stage.
            print(f"unload-model {pages.MODEL_ID}")
            try:
                run_job(
                    base,
                    args.token,
                    {"type": "unload-model", "model": pages.MODEL_ID},
                    args.timeout,
                )
            except SystemExit as exc:
                # A failed unload must not OVERWRITE the failure that is
                # already on its way out — the reason the page could not be
                # read is the one a person needs. Both are said; the first one
                # is the one that exits.
                if reading is None:
                    raise
                print(f"AND THE UNLOAD FAILED TOO: {exc}", file=sys.stderr)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
