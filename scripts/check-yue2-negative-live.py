#!/usr/bin/env python3
"""One YuE2 song rendered twice at cfg 2.0 against a RUNNING server on its card: once with
YuE2's stock CFG negative, once with a negative branch the client wrote (`negative_tags`,
`negative_lyrics`; docs/AUDIO.md "The negative branch"). The first run of it through Crucible.

  scripts/check-yue2-negative-live.py --url http://127.0.0.1:7100 --token <token> --out <dir>
  scripts/check-yue2-negative-live.py --url ... --token ... --out ... --seed 3306096576 --cfg 2.0

GPU use: two songs on yue2-3b (1.5 to 6 minutes each on the 3090 Ti, more under
[audio] low_vram). Run it only with Owen's go for the card it uses.

What it does:
1. Reads yue2-3b's playground page and refuses to go on unless it lists the `negative_tags`
   and `negative_lyrics` fields (a server without the knob).
2. Renders an instrumental (--tags, --seed, --cfg) with the stock negative and saves
   `stock.<format>` and `stock.abc` in --out.
3. Writes the negative as the 2026-10-12 experiment did (E:\\claude-files\\scratch\\voice-hum\\
   negative\\run_negative.py): vocal tags in front of the song's own tags with its
   "instrumental"/"no vocals" words taken out, and the first four sung lines of the planning
   lyrics the stock render was planned from. The words are this script's, as they will be a
   client's (B-Sides'); Crucible writes none.
4. Renders the same params and seed with that negative and saves `negative.<format>` and
   `negative.abc`.

It checks the records, not the sound: the stock render's `audio.negative` is null and its
`negative_prefix_tokens` a count; the written one's `audio.negative` is exactly what was
sent and its prefix longer than stock's; both composed with 2 CFG branches; and it says
whether the two scores are identical (same seed and params plan the same score on one card;
if they differ the pair is not a clean comparison and it says so). Listening is Owen's: it
prints the two files to compare. Exit 1 when a song did not finish or a record is wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "yue2-3b"
TAGS = "Instrumental, slow somber piano and strings, no vocals, no singing, no choir, 66 BPM"
TERMINAL = {"done", "failed", "cancelled", "interrupted"}
# The experiment's negative, as a client writes it.
STRIP = re.compile(r"\s*,?\s*\b(instrumental|no vocals|no singing|no choir|no spoken words)\b\.?", re.I)
NEG_FRONT = "lead vocals, female vocals, male vocals, singing, choir, spoken word"


def negative_tags(tags: str) -> str:
    rest = STRIP.sub("", tags).strip(" ,.")
    return NEG_FRONT + (", " + rest if rest else "") + "."


def negative_lyrics(planning: str) -> str:
    lines = [line for line in planning.splitlines() if line.strip() and not line.strip().startswith("[")]
    return "[Verse]\n" + "\n".join(lines[:4]) + "\n"


def call(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None,
         raw: bool = False) -> tuple[int, Any]:
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-yue2-negative-live",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
            return response.status, data if raw else json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def require_the_knob(args: argparse.Namespace) -> None:
    status, answer = call(args, "GET", "/v1/playground")
    if status != 200:
        raise SystemExit(f"GET /v1/playground answered {status}: {answer}")
    for page in answer["pages"]:
        if page.get("id") == MODEL:
            names = {field["name"] for field in page["fields"]}
            missing = {"negative_tags", "negative_lyrics"} - names
            if missing:
                raise SystemExit(f"{MODEL}'s playground page has no {sorted(missing)}; this server "
                                 "does not take a written negative branch")
            return
    raise SystemExit(f"this server's playground has no {MODEL} page")


def render(args: argparse.Namespace, name: str, extra: dict[str, Any]) -> dict[str, Any]:
    params = {"tags": args.tags, "instrumental": True, "seed": args.seed, "cfg": args.cfg,
              "format": args.format, **extra}
    status, submitted = call(args, "POST", "/v1/jobs", {"type": "audio", "model": MODEL, "params": params})
    if status != 202:
        raise SystemExit(f"{name}: refused {status}: {submitted.get('error')}")
    job_id = submitted["job_id"]
    print(f"{name}: job {job_id} submitted", flush=True)
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            break
        if time.monotonic() - started > args.timeout:
            raise SystemExit(f"{name}: job {job_id} did not finish in {args.timeout:g} s")
        time.sleep(args.poll)
    if job["status"] != "done":
        raise SystemExit(f"{name}: job {job_id} ended {job['status']}: {job.get('error')}")
    audio = job["audio"]
    for artifact, target in ((audio["artifact"], f"{name}.{args.format}"), (audio["score"], f"{name}.abc")):
        status, data = call(args, "GET", f"/v1/jobs/{job_id}/artifacts/{artifact}", raw=True)
        if status != 200:
            raise SystemExit(f"{name}: artifact {artifact} answered {status}")
        (args.out / target).write_bytes(data)
    composing = (audio.get("decode_stages") or {}).get("composing") or {}
    print(f"{name}: {audio['audio_seconds']} s of audio in {round(time.monotonic() - started)} s; "
          f"negative {json.dumps(audio.get('negative'))}, negative prefix "
          f"{audio.get('negative_prefix_tokens')} tokens, positive prefix "
          f"{composing.get('prefix_tokens')}, cfg branches {composing.get('cfg_branches')}, "
          f"composing ended {composing.get('ended')}", flush=True)
    return {"job_id": job_id, "audio": audio}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", required=True, type=Path, help="a directory for the two renders")
    parser.add_argument("--tags", default=TAGS)
    parser.add_argument("--seed", type=int, default=3306096576)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--format", default="flac", choices=["flac", "wav", "mp3"])
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds to wait for one song")
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()
    if not args.cfg > 1:
        raise SystemExit("a written negative runs only with cfg above 1")
    args.out.mkdir(parents=True, exist_ok=True)
    require_the_knob(args)

    stock = render(args, "stock", {})
    planning = stock["audio"].get("planning_lyrics") or {}
    if not planning.get("lyrics"):
        raise SystemExit("the stock render names no planning lyrics to write the negative from")
    written = {"tags": negative_tags(args.tags), "lyrics": negative_lyrics(planning["lyrics"])}
    print(f"negative tags: {written['tags']}\nnegative lyrics:\n{written['lyrics']}", flush=True)
    negative = render(args, "negative", {"negative_tags": written["tags"],
                                         "negative_lyrics": written["lyrics"]})

    wrong = []
    a, b = stock["audio"], negative["audio"]
    if a.get("negative") is not None:
        wrong.append(f"the stock render's audio.negative is {a.get('negative')!r}, not null")
    if b.get("negative") != written:
        wrong.append(f"the written render's audio.negative is {b.get('negative')!r}, not what was sent")
    if not isinstance(a.get("negative_prefix_tokens"), int):
        wrong.append("the stock render has no negative_prefix_tokens count")
    elif not isinstance(b.get("negative_prefix_tokens"), int) or not b["negative_prefix_tokens"] > a["negative_prefix_tokens"]:
        wrong.append(f"the written negative prefix ({b.get('negative_prefix_tokens')}) is not longer "
                     f"than the stock one ({a['negative_prefix_tokens']})")
    for name, audio in (("stock", a), ("negative", b)):
        branches = ((audio.get("decode_stages") or {}).get("composing") or {}).get("cfg_branches")
        if branches != 2:
            wrong.append(f"the {name} render composed with {branches} CFG branches, not 2")
    same = (args.out / "stock.abc").read_bytes() == (args.out / "negative.abc").read_bytes()
    print(f"\nscores {'identical' if same else 'DIFFER: not a clean comparison of the negative alone'}")
    print(f"listen: {args.out / f'stock.{args.format}'}  vs  {args.out / f'negative.{args.format}'}")
    for line in wrong:
        print(f"WRONG: {line}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
