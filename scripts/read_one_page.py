#!/usr/bin/env python3

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

from crucible import pages

EXIT_CANNOT_TRY = 2

ALREADY_CLEAR = "model_not_resident"


def error_code(detail: str) -> str | None:
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

    def __init__(self, what: str, detail: str, *, status: int | None = None) -> None:
        self.detail = detail
        self.status = status
        self.code = error_code(detail)
        where = "" if status is None else f"HTTP {status} "
        super().__init__(f"{what}: {where}{detail[:600]}")


def rasterise(path: Path) -> bytes:
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
    bitmap = page.render(scale=pages.DPI / 72)
    image = bitmap.to_pil()
    if image.width * image.height > pages.MAX_PIXELS:
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
    parser = argparse.ArgumentParser(
        description="Read one page through a Crucible server and record the answer and its shape.",
    )
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
            raise SystemExit(str(exc)) from None
        print(f"  loaded in {time.monotonic() - started:.1f}s")

    reading: SystemExit | Refused | None = None
    try:
        read_the_page(base, args.token, png, out, args.timeout)
    except (SystemExit, Refused) as exc:
        reading = exc
        print(f"THE READ FAILED: {exc}", file=sys.stderr)

    tidying: Refused | None = None
    if args.load:
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
                print(f"  the card was already clear of {pages.MODEL_ID}")
            else:
                tidying = exc
                print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)
        except SystemExit as exc:
            tidying = Refused("unload-model", str(exc))
            print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)

    if reading is not None:
        return 1
    return 1 if tidying is not None else 0


def read_the_page(
    base: str, token: str, png: bytes, out: Path, timeout: float
) -> None:
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
