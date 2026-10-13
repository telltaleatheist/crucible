#!/usr/bin/env python3
"""One YuE2 song rendered with its score header fixed - key Em, meter 3/4, 76 BPM by default -
against a RUNNING server on its card, and the score's own K, M and Q reported
(`key`, `meter`, `bpm`; docs/AUDIO.md "Key, meter and tempo"). The first run of it through
Crucible.

  scripts/check-yue2-header-live.py --url http://127.0.0.1:7100 --token <token> --out <dir>
  scripts/check-yue2-header-live.py --url ... --token ... --out ... --key F#m --meter 6/8 --bpm 84
  scripts/check-yue2-header-live.py --url ... --token ... --out ... --instrumental

GPU use: one song on yue2-3b (1.5 to 6 minutes on the 3090 Ti, more under [audio]
low_vram). Run it only with Owen's go for the card it uses.

What it does:
1. Reads yue2-3b's playground page and refuses to go on unless it lists the `key`, `meter`
   and `bpm` fields (a server without the knob).
2. Renders one song - sung from --lyrics, or with --instrumental planned from the pool -
   with the tags of the case Victoria measured (a minor-key 3/4 song at 76 BPM) and the
   three fixed, and saves `song.<format>` and `song.abc` in --out.
3. Reads the score's header lines itself, from the artifact, and prints them beside
   `audio.header`.

It checks the records and the score, not the sound: `audio.header` holds exactly the
values sent, `audio.header.score` agrees with the score's own K:, M: and Q:1/4= lines, and
those are the values sent; the score's L: and V: lines are printed as the model wrote them.
Listening is Owen's (and Victoria's): it prints the file. Exit 1 when the song did not finish
or a record or the score is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "yue2-3b"
TAGS = "English, slow waltz, melancholic, warm piano, strings, expressive female voice, E minor, 3/4, 76 BPM"
LYRICS = (
    "[Verse]\nThe kettle cools beside the door\nThe lamp is low, the hour is late\n"
    "I count the boards across the floor\nAnd listen for the garden gate\n\n"
    "[Chorus]\nTurn, turn, the night is turning\nSlow, slow, the candle burning\n"
)
TERMINAL = {"done", "failed", "cancelled", "interrupted"}


def call(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None,
         raw: bool = False) -> tuple[int, Any]:
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-yue2-header-live",
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
            missing = {"key", "meter", "bpm"} - names
            if missing:
                raise SystemExit(f"{MODEL}'s playground page has no {sorted(missing)}; this server "
                                 "does not fix a score's header")
            return
    raise SystemExit(f"this server's playground has no {MODEL} page")


def header_lines(score: str) -> dict[str, str]:
    """The score's header as written: each field letter's first value, up to the body."""
    found: dict[str, str] = {}
    for line in score.splitlines():
        if len(line) < 2 or line[1] != ":" or not line[0].isalpha():
            break
        found.setdefault(line[0], line[2:].strip())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", required=True, type=Path, help="a directory for the render")
    parser.add_argument("--key", default="Em")
    parser.add_argument("--meter", default="3/4")
    parser.add_argument("--bpm", type=int, default=76)
    parser.add_argument("--tags", default=TAGS)
    parser.add_argument("--lyrics", default=LYRICS)
    parser.add_argument("--instrumental", action="store_true",
                        help="an instrumental planned from the server's pool instead of --lyrics")
    parser.add_argument("--seed", type=int, default=3306096576)
    parser.add_argument("--format", default="flac", choices=["flac", "wav", "mp3"])
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds to wait for the song")
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    require_the_knob(args)

    sent = {"key": args.key, "meter": args.meter, "bpm": args.bpm}
    params: dict[str, Any] = {"tags": args.tags, "seed": args.seed, "format": args.format, **sent}
    if args.instrumental:
        params["instrumental"] = True
    else:
        params["lyrics"] = args.lyrics
    status, submitted = call(args, "POST", "/v1/jobs", {"type": "audio", "model": MODEL, "params": params})
    if status != 202:
        raise SystemExit(f"refused {status}: {submitted.get('error')}")
    job_id = submitted["job_id"]
    print(f"job {job_id} submitted: {json.dumps(sent)}", flush=True)
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            break
        if time.monotonic() - started > args.timeout:
            raise SystemExit(f"job {job_id} did not finish in {args.timeout:g} s")
        time.sleep(args.poll)
    if job["status"] != "done":
        raise SystemExit(f"job {job_id} ended {job['status']}: {job.get('error')}")
    audio = job["audio"]
    for artifact, target in ((audio["artifact"], f"song.{args.format}"), (audio["score"], "song.abc")):
        status, data = call(args, "GET", f"/v1/jobs/{job_id}/artifacts/{artifact}", raw=True)
        if status != 200:
            raise SystemExit(f"artifact {artifact} answered {status}")
        (args.out / target).write_bytes(data)

    score = (args.out / "song.abc").read_text(encoding="utf-8")
    written = header_lines(score)
    tempo = written.get("Q", "")
    bpm = int(tempo.split("=", 1)[1]) if tempo.startswith("1/4=") and tempo.split("=", 1)[1].isdigit() else None
    read = {"key": written.get("K"), "meter": written.get("M"), "bpm": bpm}
    scoring = (audio.get("decode_stages") or {}).get("scoring") or {}
    print(f"{audio['audio_seconds']} s of audio in {round(time.monotonic() - started)} s; scoring "
          f"{scoring.get('tokens')} tokens, ended {scoring.get('ended')}")
    print(f"score header: K:{read['key']}  M:{read['meter']}  Q:{tempo or None}  "
          f"(L:{written.get('L')}, the model's)")
    print(f"audio.header: {json.dumps(audio.get('header'))}")
    if audio.get("length"):
        print(f"score length: {audio['length'].get('score_seconds')} s")

    wrong = []
    header = audio.get("header") or {}
    if {name: header.get(name) for name in sent} != sent:
        wrong.append(f"audio.header holds {header!r}, not the values sent {sent!r}")
    if header.get("score") != read:
        wrong.append(f"audio.header.score {header.get('score')!r} disagrees with the score's own lines {read!r}")
    if read != sent:
        wrong.append(f"the score states {read!r}, not the fixed {sent!r}")
    print(f"listen: {args.out / f'song.{args.format}'}")
    for line in wrong:
        print(f"WRONG: {line}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
