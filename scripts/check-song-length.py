#!/usr/bin/env python3
"""YuE2 instrumentals asked for a length range, against a RUNNING server on its card: how close
does one pass land?

  scripts/check-song-length.py --url http://127.0.0.1:7100 --token <token>
  scripts/check-song-length.py --url ... --token ... --range 120-180 --range 180-270 --per-range 3
  scripts/check-song-length.py --url ... --token ... --named-sets   # planning_set per track

GPU use: every song loads and runs yue2-3b (1.5 to 6 minutes a song on the 3090 Ti, more
under [audio] low_vram). Run it only with Owen's go for the card it uses.

What it measures (docs/AUDIO.md "Song length"): every song is one score and one render. For
an instrumental planned from the server's pool, the set is sized toward [min_duration_s,
max_duration_s] before that one score; the done record's `audio.length` says the structure
it was sized to, the score's nominal seconds (its bars at its tempo) and whether it landed.

It renders --per-range instrumentals for each --range (default 3 in 120-180 s and 3 in
180-270 s; seeds --seed-base, +1, ...; styles cycle through --tags; with --named-sets each
track names a pool set in turn, as B-Sides does for an album) one after another, waits for
each, and prints one row per song: range, seed, set, structure (lines and sections), score
seconds, audio seconds, audio / score, and whether the score and the audio landed in range.

The summary gives the in-range rate (score and audio), the mean distance from the range of
the songs that missed, and audio / score (min / median / max). A miss is a measurement, not
a failure: it exits non-zero only when a song did not finish (failed, refused, timed out).
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
            "User-Agent": "check-song-length",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def pool_ids(args: argparse.Namespace) -> list[str]:
    """The pool's set ids, from the model's playground page (`planning_set`'s options)."""
    status, answer = call(args, "GET", "/v1/playground")
    if status != 200:
        raise SystemExit(f"GET /v1/playground answered {status}: {answer}")
    for page in answer["pages"]:
        if page.get("id") == MODEL:
            for field in page["fields"]:
                if field["name"] == "planning_set":
                    return [option for option in field["options"] if option]
    raise SystemExit(f"{MODEL}'s playground page has no planning_set field; is this server current?")


def distance(seconds: float | None, low: float, high: float) -> float | None:
    """How far `seconds` is outside [low, high]; 0 inside, None when unknown."""
    if seconds is None:
        return None
    return max(low - seconds, seconds - high, 0.0)


def render(args: argparse.Namespace, seed: int, tags: str, low: float, high: float,
           planning_set: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"tags": tags, "instrumental": True, "seed": seed,
                              "min_duration_s": low, "max_duration_s": high}
    if planning_set is not None:
        params["planning_set"] = planning_set
    row: dict[str, Any] = {"range": [low, high], "seed": seed, "named": planning_set}
    status, submitted = call(args, "POST", "/v1/jobs", {"type": "audio", "model": MODEL, "params": params})
    if status != 202:
        return {**row, "status": f"refused {status}", "error": submitted.get("error")}
    job_id = submitted["job_id"]
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            break
        if time.monotonic() - started > args.timeout:
            return {**row, "job_id": job_id, "status": "timed out", "error": None}
        time.sleep(args.poll)
    audio = job.get("audio") or {}
    kept = job.get("request") or {}
    planning = audio.get("planning_lyrics") or (kept.get("settled") or {}).get("planning_lyrics") or {}
    length = audio.get("length") or {}
    return {
        **row,
        "job_id": job_id,
        "status": job["status"],
        "set": planning.get("id"),
        "resized": planning.get("resized"),
        "lines": length.get("lines"),
        "structure": length.get("structure"),
        "score_seconds": length.get("score_seconds"),
        "in_range": length.get("in_range"),
        "audio_seconds": audio.get("audio_seconds"),
        "wall_seconds": round(time.monotonic() - started, 1),
        "error": job.get("error"),
    }


def ratio(row: dict[str, Any]) -> float | None:
    if row.get("audio_seconds") is None or not row.get("score_seconds"):
        return None
    return row["audio_seconds"] / row["score_seconds"]


def line(row: dict[str, Any]) -> str:
    low, high = row["range"]
    head = f"{low:g}-{high:g} s  seed {row['seed']:>10}  set {str(row.get('set')):<10}"
    if row["status"] != "done":
        error = row.get("error") or {}
        return f"ERROR {head}  {row['status']} {error.get('code', '')}: {error.get('message', '')}"
    structure = ", ".join(row.get("structure") or []) or "as written"
    audio_in = distance(row.get("audio_seconds"), low, high) == 0.0
    r = ratio(row)
    return (f"{'in  ' if row.get('in_range') else 'MISS'}  {head}  {row.get('lines')} lines "
            f"({structure})  score {row.get('score_seconds')} s  audio {row.get('audio_seconds')} s"
            f"  audio/score {'-' if r is None else f'{r:.3f}'}  audio {'in' if audio_in else 'out'}"
            f"  wall {row['wall_seconds']} s")


def summary(rows: list[dict[str, Any]]) -> None:
    done = [r for r in rows if r["status"] == "done"]
    scored = [r for r in done if r.get("score_seconds") is not None]
    landed = [r for r in scored if r.get("in_range")]
    audio_in = [r for r in done if distance(r.get("audio_seconds"), *r["range"]) == 0.0]
    missed = [distance(r["score_seconds"], *r["range"]) for r in scored if not r.get("in_range")]
    audio_missed = [d for d in (distance(r.get("audio_seconds"), *r["range"]) for r in done) if d]
    ratios = [x for x in (ratio(r) for r in done) if x is not None]
    print(f"\n{len(rows)} songs, {len(done)} done, {len(rows) - len(done)} errors")
    if scored:
        print(f"score in range: {len(landed)} of {len(scored)} ({100 * len(landed) / len(scored):.0f}%)"
              + (f"; the misses were {statistics.mean(missed):.1f} s outside on average"
                 f" (max {max(missed):.1f})" if missed else ""))
    if done:
        print(f"audio in range: {len(audio_in)} of {len(done)}"
              + (f"; the misses were {statistics.mean(audio_missed):.1f} s outside on average"
                 if audio_missed else ""))
    if ratios:
        print(f"audio / score: min {min(ratios):.3f}  median {statistics.median(ratios):.3f}  "
              f"max {max(ratios):.3f}  (measured 2026-10-10: 0.943 / 0.987 / 1.069)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--range", action="append", default=None,
                        help="a range as MIN-MAX seconds (repeatable); default 120-180 and 180-270")
    parser.add_argument("--per-range", type=int, default=3, help="songs to render in each range")
    parser.add_argument("--seed-base", type=int, default=3000, help="the first seed; each song adds 1")
    parser.add_argument("--named-sets", action="store_true",
                        help="name a pool set per track (planning_set), cycling the pool")
    parser.add_argument("--tags", action="append", default=None,
                        help="a style to cycle through (repeatable); default four instrumental styles")
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds to wait for one song")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--jsonl", default=None, help="also write every row to this file")
    args = parser.parse_args()
    ranges = [tuple(float(part) for part in text.split("-")) for text in (args.range or ["120-180", "180-270"])]
    styles = args.tags or STYLES
    sets = pool_ids(args) if args.named_sets else None
    rows = []
    index = 0
    for low, high in ranges:
        for _ in range(args.per_range):
            named = None if sets is None else sets[index % len(sets)]
            row = render(args, args.seed_base + index, styles[index % len(styles)], low, high, named)
            rows.append(row)
            print(line(row), flush=True)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            index += 1
    summary(rows)
    return 1 if any(row["status"] != "done" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
