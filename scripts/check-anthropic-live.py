#!/usr/bin/env python3
"""The anthropic upstream against a RUNNING server and Anthropic itself: what only a live
call can prove. The server must already hold an Anthropic key (`PUT /v1/settings`
`upstreams.anthropic.key`); this script never takes, reads or prints one. It talks to the
server only.

  scripts/check-anthropic-live.py --url http://127.0.0.1:7100 --model claude-opus-5-5
  scripts/check-anthropic-live.py --url http://host:7100 --token <crucible token> \
      --model claude-sonnet-5-5 --max-tokens 256 --effort low

Billing: three Messages API calls, each capped at --max-tokens (default 256). Thinking
counts toward it, so a model that thinks on a trivial prompt can end `length` with
little or no text; the check says so rather than failing it.

What it checks, each a numbered check that passes or fails by itself:
 1. a plain chat comes back 200 in OpenAI's shape: its `model` is the one asked for,
    `content` is text, `finish_reason` is said, and `usage` counts prompt and completion
    tokens.
 2. a streamed chat with `stream_options.include_usage` sends content chunks, a chunk
    carrying `finish_reason`, then exactly one usage chunk (empty `choices`), then
    `[DONE]`.
 3. a `json_schema` chat (sent as the Messages API's `output_config.format`) answers
    JSON that parses and has the schema's one required field.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

SCHEMA = {
    "type": "object",
    "properties": {"capital": {"type": "string"}},
    "required": ["capital"],
    "additionalProperties": False,
}

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)


def _request(args: argparse.Namespace, body: dict[str, Any]) -> urllib.request.Request:
    headers = {
        "Content-Type": "application/json",
        "X-Crucible-Api": "1",
        "User-Agent": "check-anthropic-live",
    }
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    return urllib.request.Request(
        args.url.rstrip("/") + "/v1/openai/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _body(args: argparse.Namespace, prompt: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": f"anthropic/{args.model}",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        **extra,
    }
    if args.effort:
        body["reasoning_effort"] = args.effort
    return body


def post(args: argparse.Namespace, body: dict[str, Any]) -> tuple[int, Any, dict[str, str]]:
    try:
        with urllib.request.urlopen(_request(args, body), timeout=args.timeout) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw), dict(error.headers)
        except ValueError:
            return error.code, raw.decode("utf-8", "replace"), dict(error.headers)


def stream(args: argparse.Namespace, body: dict[str, Any]) -> tuple[int, list[str]]:
    try:
        with urllib.request.urlopen(_request(args, body), timeout=args.timeout) as response:
            lines = [line.decode("utf-8").rstrip("\n") for line in response]
            return response.status, [line for line in lines if line.startswith("data: ")]
    except urllib.error.HTTPError as error:
        return error.code, [error.read().decode("utf-8", "replace")]


def _length_note(finish: Any) -> str:
    if finish == "length":
        return (" (finish_reason length: --max-tokens ran out, thinking counts toward it;"
                " raise it or pass --effort low)")
    return ""


def check_plain(args: argparse.Namespace) -> None:
    status, document, headers = post(
        args, _body(args, "Reply with the single word: forge.")
    )
    if status != 200:
        check("1 plain chat", False, f"HTTP {status}: {json.dumps(document)[:600]}")
        return
    choice = document["choices"][0]
    content = choice["message"]["content"]
    usage = document.get("usage") or {}
    print(f"      content={content!r} finish_reason={choice['finish_reason']!r} "
          f"usage={usage} X-Crucible-Sampling={headers.get('X-Crucible-Sampling')}")
    check(
        "1 plain chat",
        document["model"] == f"anthropic/{args.model}"
        and isinstance(content, str)
        and choice["finish_reason"] is not None
        and isinstance(usage.get("prompt_tokens"), int)
        and isinstance(usage.get("completion_tokens"), int),
        _length_note(choice["finish_reason"]),
    )


def check_stream(args: argparse.Namespace) -> None:
    status, frames = stream(
        args,
        _body(args, "Count from one to five in words.", stream=True,
              stream_options={"include_usage": True}),
    )
    if status != 200:
        check("2 streamed chat with usage", False, f"HTTP {status}: {frames[:1]}")
        return
    if not frames or frames[-1] != "data: [DONE]":
        check("2 streamed chat with usage", False,
              f"no [DONE] last; the last frame is {frames[-1:]!r}")
        return
    payloads = [json.loads(frame[len("data: "):]) for frame in frames[:-1]]
    errors = [p["error"] for p in payloads if "error" in p]
    if errors:
        check("2 streamed chat with usage", False, f"error frame: {errors[0]}")
        return
    with_choices = [p for p in payloads if p["choices"]]
    usage = [p for p in payloads if not p["choices"]]
    text = "".join(p["choices"][0]["delta"].get("content") or "" for p in with_choices)
    finishes = [p["choices"][0]["finish_reason"] for p in with_choices
                if p["choices"][0]["finish_reason"] is not None]
    print(f"      text={text!r} finish_reasons={finishes} usage={[u['usage'] for u in usage]}")
    check(
        "2 streamed chat with usage",
        len(finishes) == 1
        and len(usage) == 1
        and payloads[-1] is usage[0]
        and isinstance(usage[0]["usage"].get("completion_tokens"), int),
        _length_note(finishes[0] if finishes else None),
    )


def check_schema(args: argparse.Namespace) -> None:
    status, document, _ = post(
        args,
        _body(args, "What is the capital of France?",
              response_format={"type": "json_schema",
                               "json_schema": {"name": "capital", "schema": SCHEMA}}),
    )
    if status != 200:
        check("3 json_schema chat", False, f"HTTP {status}: {json.dumps(document)[:600]}")
        return
    choice = document["choices"][0]
    content = choice["message"]["content"]
    print(f"      content={content!r} finish_reason={choice['finish_reason']!r}")
    try:
        parsed = json.loads(content)
    except ValueError as exc:
        check("3 json_schema chat", False,
              f"content is not JSON ({exc}){_length_note(choice['finish_reason'])}")
        return
    check("3 json_schema chat",
          isinstance(parsed, dict) and isinstance(parsed.get("capital"), str),
          f"parsed={parsed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="the Crucible server, e.g. http://127.0.0.1:7100")
    parser.add_argument("--token", default=None,
                        help="the Crucible token, only for a server with auth on (never the Anthropic key)")
    parser.add_argument("--model", required=True, help="the Anthropic model id, without anthropic/")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--effort", default=None,
                        help="sent as reasoning_effort (low, medium, high, xhigh, max)")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    check_plain(args)
    check_stream(args)
    check_schema(args)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
