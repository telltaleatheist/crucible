#!/usr/bin/env python3
"""One cleanup chunk through a Crucible server, timed. PHASE15 section 8, T7.

What T7 wants from the text half is one number — seconds per chunk on the
`llama-windows` engine — and one fact: that a GGUF served by `llama-server`
answers the same door, with the same `model` id, as the safetensors served by
vLLM. So this loads the model, sends ONE chunk of the kind BookForge's
cleanup pass sends, and prints the figure.

**IT OWNS THE RESIDENCY.** Crucible never loads a model to answer a chat
request, so this loads one by name; and it unloads it at the end, in a
`finally`, so the card is free for the stage after this one. Found by the
first Windows run, 2026-09-14: a script that loads and never unloads leaves a
`llama-server` holding the card for as long as the server lives.

`thinking: false` travels in `chat_template_kwargs`, which is what BookForge's
crucible provider sends on every cleanup request and what
`models/qwen3.5-9b.toml`'s `[defaults]` states — Qwen3.5 otherwise spends a
bounded budget entirely on reasoning and returns a message with no content at
all. Sent here for the same reason: a measurement of a reasoning trace is not
a measurement of a cleanup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: A chunk the shape of the ones a book is cleaned in: prose with the two
#: defects the pass exists for, an em-dash split across a line break and a
#: hyphenated word broken over one. Invented here rather than taken from a
#: book, because a test run must not carry somebody's copyrighted page.
CHUNK = """Chapter Four

The morning came in grey and the harbour was still. Nobody had
told them what to expect, and so they waited — as people do — for
someone with more authority to arrive and explain the situa-
tion. By noon the tide had turned twice and the explanation had
not come. "We could go in ourselves," said the youngest of them,
and the others looked at the water and said nothing at all."""

PROMPT = (
    "Clean this passage for text-to-speech narration. Join words broken "
    "across line breaks, join lines that are one paragraph, and change "
    "nothing else. Output only the cleaned text."
)


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
            f"{url} refused: HTTP {exc.code} "
            f"{exc.read().decode('utf-8', 'replace')[:600]}"
        )
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"{url} did not answer: {type(exc).__name__}: {exc}")


def run_job(base: str, token: str, request: dict, timeout: float) -> dict:
    """Submit one job and poll it to a terminal state. Never returns a failure."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.server.rstrip("/")

    started = time.monotonic()
    run_job(base, args.token, {"type": "load-model", "model": args.model}, args.timeout)
    load_seconds = time.monotonic() - started

    cleaning: BaseException | None = None
    try:
        started = time.monotonic()
        answer = post(
            f"{base}/v1/openai/chat/completions",
            args.token,
            {
                "model": args.model,
                "temperature": 0,
                "max_tokens": 512,
                # See the module docstring: without this the budget goes
                # entirely on reasoning and the message comes back with no
                # content.
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": CHUNK},
                ],
            },
            args.timeout,
        )
        seconds = time.monotonic() - started
    except BaseException as exc:
        cleaning = exc
        raise
    finally:
        # THE CARD IS RELEASED WHATEVER HAPPENED, and a failed unload never
        # overwrites the failure already on its way out.
        try:
            run_job(
                base,
                args.token,
                {"type": "unload-model", "model": args.model},
                args.timeout,
            )
        except SystemExit as exc:
            if cleaning is None:
                raise
            print(f"AND THE UNLOAD FAILED TOO: {exc}", file=sys.stderr)
    (out / "cleanup.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")

    content = answer["choices"][0]["message"].get("content") or ""
    if content.strip() == "":
        raise SystemExit(
            "the model answered with no content at all. That is the "
            "reasoning-budget failure `[defaults] thinking = false` exists "
            f"for; the raw answer is in {out / 'cleanup.json'}"
        )
    joined = "situation" in content and "situa-" not in content
    print(f"server:          {base}")
    print(f"model:           {args.model}")
    print(f"seconds to load: {load_seconds:.1f}")
    print(f"seconds/chunk:   {seconds:.1f}")
    print(f"joined the broken word: {joined}")
    print(f"artifacts:       {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
