#!/usr/bin/env python3
"""yue2-3b-instrumental against yue2-3b on a RUNNING server's card: the same instrumental
request and seed on both, so the only difference is the adapter folded in while composing.

  scripts/check-yue2-instrumental-live.py --url http://127.0.0.1:7100 --token <token> --out ./ab
  scripts/check-yue2-instrumental-live.py --url ... --token ... --out ./ab \
      --seed 1297391292 --tags "Instrumental, ..." --planning-set harbor --no-vocals

GPU use: every seed renders two songs (yue2-3b, then yue2-3b-instrumental: each switch is a
full YuE2 load, docs/AUDIO.md "Switching between the two") and, unless --no-vocals, separates
each with vocals-roformer. Run it only with Owen's go for the card it uses.

Why (2026-10-12): a yue2-3b instrumental invents a singer while composing, even from a score
with no vocal note and section-tag lyrics. B-Sides measured the adapter on their own fold:
Iron Lanterns (seed 1297391292) sang 80 s on the base and 7.5 s with it; a soul/Hammond piece
13.5 s and 0 s, three of three. This is the first render through Crucible's own fold.

For each seed it renders both models with "hold": true, saves each song and its score.abc
under --out, and checks what can be checked without ears:

  - both songs finished;
  - yue2-3b-instrumental's done record names the pinned adapter (sha256) with
    `applied_to: ["compose"]`, `linears: 196` and `weights_of: "yue2-3b"`; yue2-3b's names none;
  - the two scores are the same text: the score is planned on the base weights, so the same
    seed and request must plan the same score on both (the fold reaching the planning stage
    would change it; if they differ, render yue2-3b twice with the seed to rule out a card
    whose kernels are not deterministic);
  - unless --no-vocals: each song's vocals stem (vocals-roformer, by reference to the held
    song) has its voiced seconds counted (50 ms windows above --voiced-dbfs), and the
    instrumental's must not exceed the base's.

Exit 0 when every check holds for every seed; each line that misses says which. Listening is
still the verdict: the files are in --out.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

BASE = "yue2-3b"
INSTRUMENTAL = "yue2-3b-instrumental"
ADAPTER_SHA256 = "e408fd3148b75b1165f7ddbf63db575d83bb6402a0b5f876fcb767dbcb2c5414"
SEPARATOR = "vocals-roformer"
DEFAULT_TAGS = "Instrumental, soul, Hammond organ, warm bass, brushed drums, no vocals, 92 BPM"
TERMINAL = {"done", "failed", "cancelled", "interrupted"}
WINDOW_S = 0.05


def call(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None,
         raw: bool = False) -> tuple[int, Any]:
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-yue2-instrumental-live",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            if raw:
                return response.status, payload
            return response.status, json.loads(payload.decode("utf-8")) if payload else None
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def wait(args: argparse.Namespace, job_id: str) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            return job
        if time.monotonic() - started > args.timeout:
            raise SystemExit(f"job {job_id} did not finish in {args.timeout:g} s")
        time.sleep(args.poll)


def submit(args: argparse.Namespace, body: dict[str, Any]) -> dict[str, Any]:
    status, answer = call(args, "POST", "/v1/jobs", body)
    if status != 202:
        raise SystemExit(f"{body['model']} was refused {status}: {answer}")
    return wait(args, answer["job_id"])


def fetch(args: argparse.Namespace, job_id: str, name: str) -> bytes:
    status, payload = call(args, "GET", f"/v1/jobs/{job_id}/artifacts/{name}", raw=True)
    if status != 200:
        raise SystemExit(f"job {job_id} has no artifact {name!r} ({status}: {payload})")
    return payload


def params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    found: dict[str, Any] = {"tags": args.tags, "instrumental": True, "seed": seed, "format": "flac"}
    if args.planning_set:
        found["planning_set"] = args.planning_set
    return found


def voiced_seconds(wav: bytes, dbfs: float) -> float:
    """Seconds of the stem whose 50 ms windows are louder than `dbfs` (RMS, all channels)."""
    import numpy

    with wave.open(io.BytesIO(wav)) as handle:
        width, channels, rate = handle.getsampwidth(), handle.getnchannels(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width == 3:
        raw = numpy.frombuffer(frames, dtype=numpy.uint8).reshape(-1, 3)
        ints = (raw[:, 0].astype(numpy.int32) | (raw[:, 1].astype(numpy.int32) << 8)
                | (raw[:, 2].astype(numpy.int32) << 16))
        samples = numpy.where(ints >= 1 << 23, ints - (1 << 24), ints) / float(1 << 23)
    elif width in (2, 4):
        kind = numpy.int16 if width == 2 else numpy.int32
        samples = numpy.frombuffer(frames, dtype=kind) / float(numpy.iinfo(kind).max)
    else:
        raise SystemExit(f"the vocals stem is {width * 8}-bit PCM, which this script does not read")
    samples = samples.reshape(-1, channels)
    window = int(rate * WINDOW_S)
    count = len(samples) // window
    blocks = samples[: count * window].reshape(count, window * channels)
    rms = numpy.sqrt(numpy.mean(blocks ** 2, axis=1))
    loud = rms > 10 ** (dbfs / 20)
    return round(float(loud.sum()) * WINDOW_S, 2)


def vocals(args: argparse.Namespace, song: dict[str, Any]) -> float:
    job = submit(args, {
        "type": "denoise", "model": SEPARATOR, "params": {"stems": "all", "format": "wav"},
        "inputs": {"song.flac": {"artifact": {"job_id": song["job_id"], "name": "audio.flac"}}},
    })
    if job["status"] != "done":
        raise SystemExit(f"separating {song['job_id']} ended {job['status']}: {job.get('error')}")
    stem = next(name for name in job["artifacts"] if "(vocals)" in name)
    return voiced_seconds(fetch(args, job["job_id"], stem), args.voiced_dbfs)


def render(args: argparse.Namespace, model: str, seed: int) -> dict[str, Any]:
    job = submit(args, {"type": "audio", "model": model, "hold": True, "params": params(args, seed)})
    row: dict[str, Any] = {"model": model, "seed": seed, "job_id": job["job_id"],
                           "status": job["status"], "error": job.get("error")}
    if job["status"] != "done":
        return row
    audio = job["audio"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{seed}-{model}.flac").write_bytes(fetch(args, job["job_id"], "audio.flac"))
    score = fetch(args, job["job_id"], "score.abc").decode("utf-8")
    (out / f"{seed}-{model}.abc").write_text(score, encoding="utf-8")
    row.update(score=score, adapter=audio.get("adapter"), weights_of=audio.get("weights_of"),
               audio_seconds=audio.get("audio_seconds"), planning=(audio.get("planning_lyrics") or {}).get("id"))
    if not args.no_vocals:
        row["voiced_s"] = vocals(args, row)
    call(args, "DELETE", f"/v1/jobs/{job['job_id']}/hold")
    return row


def misses(base: dict[str, Any], mine: dict[str, Any]) -> list[str]:
    found = [f"{row['model']} ended {row['status']}: {row['error']}"
             for row in (base, mine) if row["status"] != "done"]
    if found:
        return found
    adapter = mine["adapter"] or {}
    if (adapter.get("sha256"), adapter.get("applied_to"), adapter.get("linears")) != (
            ADAPTER_SHA256, ["compose"], 196) or mine["weights_of"] != BASE:
        found.append(f"{INSTRUMENTAL}'s record is {adapter} / weights_of {mine['weights_of']!r}")
    if base["adapter"] is not None:
        found.append(f"{BASE} names an adapter: {base['adapter']}")
    if base["score"] != mine["score"]:
        found.append("the scores differ: planning did not run on the base weights alone")
    if "voiced_s" in mine and mine["voiced_s"] > base["voiced_s"]:
        found.append(f"more voice with the adapter ({mine['voiced_s']} s) than without ({base['voiced_s']} s)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", required=True, help="where each song and score is saved")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="a seed to render on both models (repeatable); default 1297391292")
    parser.add_argument("--tags", default=DEFAULT_TAGS)
    parser.add_argument("--planning-set", default=None, help="a pool set id; default the seed's")
    parser.add_argument("--no-vocals", action="store_true", help="skip separating and counting voice")
    parser.add_argument("--voiced-dbfs", type=float, default=-40.0,
                        help="a 50 ms window of the vocals stem louder than this is voice")
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds to wait for one job")
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()
    failed = False
    for seed in args.seed or [1297391292]:
        base, mine = render(args, BASE, seed), render(args, INSTRUMENTAL, seed)
        missed = misses(base, mine)
        failed = failed or bool(missed)
        print(f"{'ok  ' if not missed else 'MISS'} seed {seed} set {mine.get('planning')}: "
              f"voice {base.get('voiced_s', '-')} s -> {mine.get('voiced_s', '-')} s, "
              f"audio {base.get('audio_seconds')} s / {mine.get('audio_seconds')} s"
              + (f"  <- {'; '.join(missed)}" if missed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
