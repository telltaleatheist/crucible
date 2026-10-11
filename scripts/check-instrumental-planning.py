#!/usr/bin/env python3
"""YuE2 instrumentals planned from the pool's planning lyrics, against a RUNNING server on its
card: do their scores land in the sung range, with no score at its cap?

  scripts/check-instrumental-planning.py --url http://127.0.0.1:7100 --token <token>
  scripts/check-instrumental-planning.py --url ... --token ... --count 20 --seed-base 5000

GPU use: every song loads and runs yue2-3b (about 4 to 6 minutes a song on an 8 GiB laptop
under [audio] low_vram). Run it only with Owen's go for the card it uses.

Why (Victoria's 8 GiB laptop, 2026-10-10): planned from empty sections, 4 of 13
instrumentals ran the score to its 4096-token cap (75 to 92 s at ~54 tok/s) and the rest
scored 1023 to 3911 tokens (137 to 360 s of audio); sung songs score 1800 to 2600. Each
instrumental now plans its score from a pool set picked by its seed (set seed mod 10,
crucible/audio/planning/yue2.toml), so consecutive seeds walk the pool in file order.

It renders --count instrumentals one after another (seeds --seed-base, +1, ...; the style
tags cycle through --tags), waits for each, and prints one line per song:

  seed, the pool set the server picked, how the job ended, score tokens and how the score
  ended (eos or cap), song tokens and how they ended, seconds of audio, wall seconds

then a summary per set and overall. It passes (exit 0) when every song finished, no score
or song ran to its cap, and every score is within --band (default 1800-2600 tokens, the
sung range); each line that misses says which. A failed song's error is printed as the
server said it (an empty or truncated score keeps its plan in the job's failed-plan/).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

MODEL = "yue2-3b"
STYLES = [
    "Instrumental, slow somber piano and strings, no vocals, no singing, no choir, 66 BPM",
    "Instrumental, warm acoustic guitar folk, light drums, no vocals, no singing, 96 BPM",
    "Instrumental, synthwave, analog synth lead, driving bass, no vocals, no singing, 112 BPM",
    "Instrumental, lo-fi jazz, electric piano, brushed drums, no vocals, no singing, 78 BPM",
]
TERMINAL = {"done", "failed", "cancelled", "interrupted"}


def call(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-instrumental-planning",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def render(args: argparse.Namespace, seed: int, tags: str) -> dict[str, Any]:
    status, submitted = call(args, "POST", "/v1/jobs", {
        "type": "audio",
        "model": MODEL,
        "params": {"tags": tags, "instrumental": True, "seed": seed},
    })
    if status != 202:
        return {"seed": seed, "status": f"refused {status}", "error": submitted.get("error")}
    job_id = submitted["job_id"]
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            break
        if time.monotonic() - started > args.timeout:
            return {"seed": seed, "job_id": job_id, "status": "timed out", "error": None}
        time.sleep(args.poll)
    audio = (job.get("done_extra") or {}).get("audio") or {}
    kept = job.get("request") or {}
    planning = audio.get("planning_lyrics") or (kept.get("settled") or {}).get("planning_lyrics") or {}
    stages = audio.get("decode_stages") or {}
    score, song = stages.get("scoring") or {}, stages.get("composing") or {}
    return {
        "seed": seed,
        "job_id": job_id,
        "status": job["status"],
        "set": planning.get("id"),
        "source": planning.get("source"),
        "score_tokens": score.get("tokens"),
        "score_ended": score.get("ended"),
        "song_tokens": song.get("tokens"),
        "song_ended": song.get("ended"),
        "audio_seconds": audio.get("audio_seconds"),
        "wall_seconds": round(time.monotonic() - started, 1),
        "error": job.get("error"),
    }


def misses(row: dict[str, Any], band: tuple[int, int]) -> list[str]:
    found = []
    if row["status"] != "done":
        found.append(f"ended {row['status']}")
        return found
    if row["score_ended"] == "cap":
        found.append("score at its cap")
    if row["song_ended"] == "cap":
        found.append("song at its cap")
    tokens = row["score_tokens"]
    if tokens is None or not band[0] <= tokens <= band[1]:
        found.append(f"score {tokens} outside {band[0]}-{band[1]}")
    if row["source"] != "pool":
        found.append(f"planned from {row['source']!r}, not the pool")
    return found


def line(row: dict[str, Any], missed: list[str]) -> str:
    if row["status"] != "done" or row.get("score_tokens") is None:
        error = row.get("error") or {}
        return (f"FAIL seed {row['seed']} set {row.get('set')}: {row['status']} "
                f"score {row.get('score_tokens')} {row.get('score_ended')} "
                f"{error.get('code', '')} {error.get('message', '')}".rstrip())
    return (
        f"{'ok  ' if not missed else 'MISS'} seed {row['seed']:>10} set {str(row['set']):<10} "
        f"score {row['score_tokens']:>5} {row['score_ended']:<3}  song {row['song_tokens']:>5} "
        f"{row['song_ended']:<3}  audio {row['audio_seconds']:>6} s  wall {row['wall_seconds']:>6} s"
        + (f"  <- {'; '.join(missed)}" if missed else "")
    )


def summary(rows: list[dict[str, Any]], band: tuple[int, int]) -> None:
    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(str(row.get("set")), []).append(row)
    print("\nper set: songs, score tokens (min / median / max), audio seconds (min / max), misses")
    for name, group in sorted(by_set.items()):
        tokens = [r["score_tokens"] for r in group if r.get("score_tokens") is not None]
        seconds = [r["audio_seconds"] for r in group if r.get("audio_seconds") is not None]
        missed = sum(1 for r in group if misses(r, band))
        print(f"  {name:<10} {len(group):>3}  "
              + (f"{min(tokens)} / {statistics.median(tokens):g} / {max(tokens)}" if tokens else "-")
              + "  " + (f"{min(seconds)} / {max(seconds)}" if seconds else "-")
              + f"  {missed}")
    tokens = [r["score_tokens"] for r in rows if r.get("score_tokens") is not None]
    capped = sum(1 for r in rows if r.get("score_ended") == "cap")
    failed = sum(1 for r in rows if r["status"] != "done")
    inside = sum(1 for t in tokens if band[0] <= t <= band[1])
    print(f"\n{len(rows)} songs: {failed} not done, {capped} scores at the cap, "
          f"{inside} of {len(tokens)} scores within {band[0]}-{band[1]} tokens"
          + (f" (median {statistics.median(tokens):g})" if tokens else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--count", type=int, default=10, help="songs to render (10 covers the pool once)")
    parser.add_argument("--seed-base", type=int, default=1000, help="the first seed; each song adds 1")
    parser.add_argument("--tags", action="append", default=None,
                        help="a style to cycle through (repeatable); default four instrumental styles")
    parser.add_argument("--band", default="1800-2600", help="score tokens a sung song lands in")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for one song")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--jsonl", default=None, help="also write every row to this file")
    args = parser.parse_args()
    low, high = (int(part) for part in args.band.split("-"))
    styles = args.tags or STYLES
    rows = []
    for index in range(args.count):
        seed = args.seed_base + index
        row = render(args, seed, styles[index % len(styles)])
        rows.append(row)
        print(line(row, misses(row, (low, high))), flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    summary(rows, (low, high))
    return 1 if any(misses(row, (low, high)) for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
