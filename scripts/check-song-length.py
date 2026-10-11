#!/usr/bin/env python3
"""YuE2 instrumentals asked for a length range, against a RUNNING server on its card: does each
score land in its range, in how many attempts, and how close is the audio to the score?

  scripts/check-song-length.py --url http://127.0.0.1:7100 --token <token>
  scripts/check-song-length.py --url ... --token ... --range 120-180 --range 180-270 --per-range 3
  scripts/check-song-length.py --url ... --token ... --named-sets   # planning_set per track

GPU use: every song loads and runs yue2-3b (1.5 to 6 minutes a song on the 3090 Ti, more
under [audio] low_vram, plus up to two extra scores of 10 to 30 s each). Run it only with
Owen's go for the card it uses.

What it checks (docs/AUDIO.md "Song length"): after the score and before composing, the
server reads the score's nominal length (its bars at its tempo) and, for an instrumental
planned from its pool, grows or cuts the pool set by whole sections and re-plans the score
until it lands in [min_duration_s, max_duration_s], at most 3 scores. On the PC's first nine
pool instrumentals (2026-10-10) the audio ran 0.943 to 1.069 of the score's nominal length.

It renders --per-range instrumentals for each --range (default 3 in 120-180 s and 3 in
180-270 s; seeds --seed-base, +1, ...; styles cycle through --tags; with --named-sets each
track names a pool set in turn, as B-Sides does for an album) one after another, waits for
each, and prints one line per song:

  the range, seed, pool set (and whether it was named, and whether it was resized), every
  attempt as `lines -> nominal s`, the final score's nominal seconds, the audio's seconds,
  audio / nominal, and wall seconds

then a summary: songs done, songs whose score landed in range, attempts per song, songs
refused `instrumental_length_not_reached` (with their attempts), audio inside the range,
and the audio / nominal ratios (min / median / max). It passes (exit 0) when every song is
done with its score in range and no score or song ran to its cap.
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
    error = job.get("error") or {}
    planning = audio.get("planning_lyrics") or (kept.get("settled") or {}).get("planning_lyrics") or {}
    length = audio.get("length") or {}
    attempts = length.get("attempts") or (error.get("details") or {}).get("attempts") or []
    stages = audio.get("decode_stages") or {}
    return {
        **row,
        "job_id": job_id,
        "status": job["status"],
        "set": planning.get("id"),
        "requested": planning.get("requested"),
        "resized": planning.get("resized"),
        "attempts": [{"lines": a.get("lines"), "score_seconds": a.get("score_seconds"),
                      "structure": a.get("structure"), "in_range": a.get("in_range"),
                      "score_ended": a.get("score_ended")} for a in attempts],
        "score_seconds": length.get("score_seconds"),
        "score_in_range": length.get("in_range"),
        "song_ended": (stages.get("composing") or {}).get("ended"),
        "audio_seconds": audio.get("audio_seconds"),
        "wall_seconds": round(time.monotonic() - started, 1),
        "error": job.get("error"),
    }


def ratio(row: dict[str, Any]) -> float | None:
    if row.get("audio_seconds") is None or not row.get("score_seconds"):
        return None
    return row["audio_seconds"] / row["score_seconds"]


def misses(row: dict[str, Any]) -> list[str]:
    if row["status"] != "done":
        code = (row.get("error") or {}).get("code", "")
        return [f"ended {row['status']} {code}".rstrip()]
    found = []
    if not row.get("score_in_range"):
        found.append(f"score {row.get('score_seconds')} s outside {row['range'][0]:g}-{row['range'][1]:g}")
    if any(a.get("score_ended") == "cap" for a in row.get("attempts", [])):
        found.append("a score at its cap")
    if row.get("song_ended") == "cap":
        found.append("song at its cap")
    return found


def line(row: dict[str, Any], missed: list[str]) -> str:
    low, high = row["range"]
    tried = ", ".join(f"{a['lines']} lines -> {a['score_seconds']} s" for a in row.get("attempts", []))
    head = (f"{'ok  ' if not missed else 'MISS'} {low:g}-{high:g} s seed {row['seed']:>10} "
            f"set {str(row.get('set')):<10}{' (named)' if row.get('requested') else ''}"
            f"{' resized' if row.get('resized') else ''}")
    if row["status"] != "done":
        error = row.get("error") or {}
        return f"{head}  attempts [{tried}]  {error.get('code', '')}: {error.get('message', '')}"
    r = ratio(row)
    return (f"{head}  attempts [{tried}]  score {row['score_seconds']} s  audio {row['audio_seconds']} s"
            f"  audio/score {r:.3f}  wall {row['wall_seconds']} s"
            + (f"  <- {'; '.join(missed)}" if missed else ""))


def summary(rows: list[dict[str, Any]]) -> None:
    done = [r for r in rows if r["status"] == "done"]
    landed = [r for r in done if r.get("score_in_range")]
    refused = [r for r in rows if (r.get("error") or {}).get("code") == "instrumental_length_not_reached"]
    audio_inside = [r for r in done if r.get("audio_seconds") is not None
                    and r["range"][0] <= r["audio_seconds"] <= r["range"][1]]
    attempts = [len(r.get("attempts", [])) for r in rows if r.get("attempts")]
    ratios = [x for x in (ratio(r) for r in done) if x is not None]
    print(f"\n{len(rows)} songs: {len(done)} done, {len(landed)} with the score in range, "
          f"{len(audio_inside)} with the audio in range, {len(refused)} refused "
          "instrumental_length_not_reached")
    if attempts:
        print(f"attempts per song: {', '.join(str(a) for a in attempts)} "
              f"(mean {statistics.mean(attempts):.2f})")
    if ratios:
        print(f"audio / score: min {min(ratios):.3f}  median {statistics.median(ratios):.3f}  "
              f"max {max(ratios):.3f}  (measured 2026-10-10: 0.943 / 0.987 / 1.069)")
    for row in refused:
        tried = ", ".join(f"{a['lines']} lines -> {a['score_seconds']} s" for a in row.get("attempts", []))
        print(f"  refused seed {row['seed']} set {row.get('set')} {row['range']}: {tried}")


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
            print(line(row, misses(row)), flush=True)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            index += 1
    summary(rows)
    return 1 if any(misses(row) for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
