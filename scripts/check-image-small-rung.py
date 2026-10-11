#!/usr/bin/env python3
"""Album covers from the image verb's pick, through a RUNNING server's job API on its card:
which model the server registered, what each picture took in time and memory, and whether
the memory stayed inside the model's estimate.

  scripts/check-image-small-rung.py --url http://127.0.0.1:7100 --token <token> --out covers/
  scripts/check-image-small-rung.py --url ... --token ... --model dreamshaper-xl-lightning --count 12

GPU use: it loads the image model and makes --count pictures on the server's card. Run it
only with Owen's go for the card it uses (Victoria's 8 GiB laptop is the one it is for).

Why (2026-10-10): dreamshaper-xl-lightning is the small rung of the `image` verb for 8 GiB
CUDA cards. Its recipe was measured beside Crucible (E:\\claude-files\\scratch\\dreamshaper\\
covers.py on owens-pc, the 3090 Ti held to a laptop card's room): 4.14 GiB resident, 6.15
GiB peak reserved at 1024x1024, 3.2 s a picture warm. This is the same check through the
worker, on the card the rung is for (docs/internals/image.md "The small rung").

What it does:

  1. GET /v1/playground: prints each image page's standing, `registered` and `facts`, and
     which model a request that names none gets (the registered page; --model overrides).
  2. Opens a queue session (act "image"), so the model loads once and stays for the batch.
  3. Sends --count `image` jobs one after another with album-cover prompts (six moods,
     seeds --seed-base, +1, ...), no size, steps or guidance (the model's own defaults),
     unless --width/--height are given. While each runs it samples GET /v1/accelerator every
     --sample seconds for the card's used bytes and the bytes Crucible's own processes hold.
  4. Saves each image.png to --out and prints one line per picture: wall seconds, the
     worker's seconds and stage seconds, the worker's peak (torch reserved) against the
     estimate, the card's peak use and Crucible's peak, the prompt cache, and the effective
     steps, guidance, size and negative prompt.
  5. Closes the session (the model comes off the card) and prints a summary: the first
     picture's wall time (which includes the load) against the warm median, the largest
     worker and card peaks, and every miss.

It passes (exit 0) when every picture is `done`, every worker peak is at or under the
model's `memory_bytes_estimate`, and with --expect-model, the registered model is that one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPTS = [
    "album cover art, melancholy indie, rain on a window at dusk, muted blues and greys, soft painterly brushwork, lonely, quiet",
    "album cover art, 80s synthwave, neon magenta and cyan sunset grid, chrome palm trees, retro-futuristic, glowing haze",
    "album cover art, watercolour folk landscape, rolling green hills, small farmhouse, warm morning light, loose watercolour washes on textured paper",
    "album cover art, abstract colour field painting, large soft-edged blocks of deep orange and indigo, rothko style, contemplative",
    "album cover art, grainy 1970s film photograph, empty desert highway, faded warm colours, light leaks, nostalgic",
    "album cover art, surreal paper collage, a whale floating above a city at night, vintage cutouts, dreamlike, textured paper",
]
TERMINAL = {"done", "failed", "cancelled", "interrupted"}
GIB = 2**30


def gib(value: Any) -> str:
    return "-" if value is None else f"{value / GIB:.2f}"


class Server:
    def __init__(self, args: argparse.Namespace) -> None:
        self.url = args.url.rstrip("/")
        self.token = args.token
        self.session: str | None = None

    def call(self, method: str, path: str, body: dict[str, Any] | None = None, raw: bool = False) -> tuple[int, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-image-small-rung",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.session is not None:
            headers["X-Crucible-Session"] = self.session
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                return response.status, data if raw else json.loads(data.decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


class CardSampler:
    """GET /v1/accelerator on a pace while a picture runs: the card's peak use, and the
    peak of what Crucible's own processes hold."""

    def __init__(self, server: Server, every: float) -> None:
        self._server = server
        self._every = every
        self._stop = threading.Event()
        self.used_peak: int | None = None
        self.owned_peak: int | None = None
        self.errors = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "CardSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            status, state = self._server.call("GET", "/v1/accelerator")
            if status == 200:
                used = state.get("used_bytes")
                owned = sum(h["bytes"] for h in state.get("holders", []) if h.get("owned_by_crucible"))
                if used is not None:
                    self.used_peak = max(used, self.used_peak or 0)
                self.owned_peak = max(owned, self.owned_peak or 0)
            else:
                self.errors += 1
            self._stop.wait(self._every)


def pick(server: Server, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    status, body = server.call("GET", "/v1/playground")
    if status != 200:
        raise SystemExit(f"GET /v1/playground answered {status}: {body}")
    pages = [page for page in body["pages"] if page["job_type"] == "image"]
    print("image pages (in the server's order):")
    for page in pages:
        facts = page.get("facts") or {}
        print(f"  {page['id']:<26} {page['standing']:<11} registered={page.get('registered')} "
              f"native {facts.get('native_width')}x{facts.get('native_height')} "
              f"min side {facts.get('quality_min_side')} steps {facts.get('default_steps')} "
              f"guidance {facts.get('default_guidance')} negative {facts.get('negative_prompt')}"
              + ("" if page["standing"] == "ready" else f"  ({page.get('reason')})"))
    registered = next((page for page in pages if page.get("registered")), None)
    model = args.model or (registered or {}).get("id")
    if model is None:
        raise SystemExit("the server registered no image model (run `crucible capability --write` on it), and no --model was given")
    page = next((page for page in pages if page["id"] == model), None)
    if page is None:
        raise SystemExit(f"{model!r} is not an image page on this server")
    print(f"\nmaking covers with {model}" + (" (named)" if args.model else " (the server's pick)"))
    return model, page


def cover(server: Server, args: argparse.Namespace, model: str | None, index: int) -> dict[str, Any]:
    params: dict[str, Any] = {"prompt": PROMPTS[index % len(PROMPTS)], "seed": args.seed_base + index}
    if args.width is not None:
        params.update(width=args.width, height=args.height)
    body: dict[str, Any] = {"type": "image", "params": params}
    if model is not None:
        body["model"] = model
    started = time.monotonic()
    with CardSampler(server, args.sample) as card:
        status, submitted = server.call("POST", "/v1/jobs", body)
        if status != 202:
            return {"index": index, "status": f"refused {status}", "error": submitted.get("error")}
        job_id = submitted["job_id"]
        while True:
            status, job = server.call("GET", f"/v1/jobs/{job_id}")
            if status == 200 and job["status"] in TERMINAL:
                break
            if time.monotonic() - started > args.timeout:
                return {"index": index, "job_id": job_id, "status": "timed out"}
            time.sleep(args.poll)
    row = {
        "index": index,
        "job_id": job_id,
        "status": job["status"],
        "wall_seconds": round(time.monotonic() - started, 2),
        "card_used_peak": card.used_peak,
        "card_owned_peak": card.owned_peak,
        "image": job.get("image") or {},
        "error": job.get("error"),
    }
    if job["status"] == "done":
        status, png = server.call("GET", f"/v1/jobs/{job_id}/artifacts/image.png", raw=True)
        if status == 200:
            name = f"{index:02d}-{model or 'pick'}-{args.seed_base + index}.png"
            (Path(args.out) / name).write_bytes(png)
            row["file"] = name
    return row


def misses(row: dict[str, Any]) -> list[str]:
    if row["status"] != "done":
        return [f"ended {row['status']}: {row.get('error')}"]
    image = row["image"]
    peak, estimate = image.get("peak_bytes"), image.get("memory_bytes_estimate")
    if peak is not None and estimate is not None and peak > estimate:
        return [f"worker peak {gib(peak)} GiB over the estimate {gib(estimate)} GiB"]
    return []


def line(row: dict[str, Any]) -> str:
    if row["status"] != "done":
        return f"FAIL #{row['index']}: {row['status']} {row.get('error')}"
    image = row["image"]
    stages = " ".join(f"{k} {v}" for k, v in (image.get("stage_seconds") or {}).items())
    missed = misses(row)
    return (
        f"{'ok  ' if not missed else 'MISS'} #{row['index']:<2} {image.get('model')} "
        f"{image.get('width')}x{image.get('height')} steps {image.get('steps')} guidance {image.get('guidance')} "
        f"wall {row['wall_seconds']} s worker {image.get('seconds')} s ({stages}) "
        f"peak {gib(image.get('peak_bytes'))} / est {gib(image.get('memory_bytes_estimate'))} GiB "
        f"card {gib(row['card_used_peak'])} GiB (crucible {gib(row['card_owned_peak'])}) "
        f"cache {image.get('prompt_cache')} neg-default {image.get('negative_prompt_is_default')}"
        + (f"  <- {'; '.join(missed)}" if missed else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default=None, help="name a model; default: send none (the server's pick)")
    parser.add_argument("--expect-model", default=None, help="fail unless the pick is this model")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=11)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out", default="covers")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for one picture")
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--sample", type=float, default=0.5, help="seconds between card samples")
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height go together")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    server = Server(args)
    model, _ = pick(server, args)
    if args.expect_model is not None and model != args.expect_model:
        print(f"MISS the pick is {model}, expected {args.expect_model}")
        return 1
    status, opened = server.call("POST", "/v1/queue/sessions", {"act": "image"})
    if status not in (200, 201, 202):
        raise SystemExit(f"could not open a queue session: {status} {opened}")
    server.session = opened["session_id"]
    rows: list[dict[str, Any]] = []
    try:
        for index in range(args.count):
            row = cover(server, args, args.model, index)
            rows.append(row)
            print(line(row), flush=True)
    finally:
        session, server.session = server.session, None
        server.call("DELETE", f"/v1/queue/sessions/{session}")
    done = [row for row in rows if row["status"] == "done"]
    if done:
        warm = [row["image"]["seconds"] for row in done[1:] if row["image"].get("seconds") is not None]
        print(f"\nfirst picture (with the load): wall {done[0]['wall_seconds']} s; "
              + (f"warm worker median {statistics.median(warm):g} s over {len(warm)}" if warm else "no warm pictures"))
        print(f"largest worker peak {gib(max(r['image'].get('peak_bytes') or 0 for r in done))} GiB, "
              f"card peak {gib(max(r['card_used_peak'] or 0 for r in done))} GiB, "
              f"Crucible's processes {gib(max(r['card_owned_peak'] or 0 for r in done))} GiB; pictures in {args.out}")
    failed = [row for row in rows if misses(row)]
    print(f"{len(rows)} pictures: {len(done)} done, {len(failed)} missed")
    return 1 if failed or len(done) != args.count else 0


if __name__ == "__main__":
    sys.exit(main())
