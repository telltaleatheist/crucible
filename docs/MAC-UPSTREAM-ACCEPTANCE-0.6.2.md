# Published 0.6.2: Mac Ollama upstream acceptance

On 2026-09-16, the installed published Crucible 0.6.2 runtime successfully forwarded a known-answer completion to an existing local Ollama model. This checks a local Ollama upstream, not cloud providers or every model.

## Isolation and setup

- Mac: Apple M1 Ultra; installed Crucible 0.6.2 service remained running on port 7100 throughout.
- Both Crucible activity and Ollama `/api/ps` were idle before each request.
- A separate foreground Crucible process used the installed packed Python, a fresh private `CRUCIBLE_HOME`, a random loopback port, and cooperative stdin shutdown. No launchd registrations or installed configuration were changed.
- Authenticated `PUT /v1/settings` configured `upstreams.ollama.url = http://127.0.0.1:11434` and `routes.translate = ollama/qwen3.5:0.8b`. Readback confirmed the upstream route.
- Ollama 0.32.14 already held `qwen3.5:0.8b` (1,036,046,583 bytes, digest `f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`). No pulls were made.

## Requests and results

Both requests used the configured route model in authenticated `POST /v1/openai/chat/completions` and asked: “What is 2+2? Answer with only the digit.”

1. With default reasoning and a 256-token budget, forwarding succeeded, but Qwen spent all 256 tokens reasoning and returned empty content with `finish_reason: length` after 14.108 seconds. This proved transport only, not the known-answer assertion. The model and temporary server were then stopped.
2. One authorized repeat added `reasoning_effort: none`, a field documented in [Ollama's OpenAI compatibility API](https://docs.ollama.com/api/openai-compatibility). It returned content `4`, `finish_reason: stop`, and two completion tokens in 1.978 seconds. The response retained `model: ollama/qwen3.5:0.8b` and reported `system_fingerprint: fp_ollama`.

A preliminary helper run stopped before either request because its assertion looked for the server name at the wrong JSON level. Correcting the helper to inspect `info.server.name` required no product changes and loaded no model.

## Storage and cleanup proof

The successful temporary home contained only `config.toml` (468 bytes) and `pairing` (103 bytes): no model weights or inference environments. Ollama model names and digests matched their pre-test inventory. Crucible activity recorded the settings write and showed no resident model, lease, running job, queued job, or chat in flight after completion.

The temporary service exited normally with code 0 after stdin closure. Only the model loaded by this test was released through Ollama's `keep_alive: 0`; `/api/ps` was empty afterward. SHA-256 hashes of the installed Crucible configuration and pairing file were unchanged. The original service remained authenticated, healthy, and idle. The Mac GPU was released after the test.

Private evidence on the Mac:

- Successful response and checks: `~/.crucible-release-validation/0.6.2-upstream-_ti74fi1/result.json`.
- Initial reasoning-budget result: `~/.crucible-release-validation/0.6.2-upstream-ehfi35n_/result.json`.
- Each directory retains isolated service logs and configuration. Pairing tokens are not included in this report.

The test required no runtime, release, version, or application changes. It does not change the release's prerelease status or replace pending Windows acceptance.

Separately, the full [CI run for source commit ce2684e](https://github.com/telltaleatheist/crucible/actions/runs/35064195077) passed Linux and macOS Python 3.11/3.12 server jobs, SDK checks, and wheel/sdist verification. That source commit adds the installer readiness gate; it is newer than the immutable 0.6.2 release assets tested here and does not retroactively change them.
