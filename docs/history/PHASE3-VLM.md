# Phase 3c: page reading

Contract for `vlm-pages`. Extends PHASE2-LLM.md; where the two disagree this file wins for
image input. Written 2026-09-13 from `docs/CLIENT-SURFACES.md` sections 7 and 10 (tier 2).

## 1. The finding that shrinks this phase

DESIGN.md section 3 describes `vlm-pages` as "PDF or page images in, per-page structured
markup out. Send the PDF; the server rasterises." That is a real job type and it is not the
one either app needs.

What Foundry actually does (CLIENT-SURFACES.md section 7.1) is send **an ordinary chat
completion** whose first content part is a data-URI PNG:

```
POST <endpoint>/chat/completions
{ "model": "<model>", "temperature": 0, "max_tokens": <adaptive, <= 8192>,
  "messages": [{ "role": "user", "content": [
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,<page>" } },
    { "type": "text", "text": "<the layout-all prompt>" } ] }] }
```

One page per request, twelve in flight, nothing retried. Rasterisation is local on **every**
route — PyMuPDF at a pinned `VLM_DPI = 200`, `--python <interpreter>` mandatory — so DPI and
page range are not wire parameters and never reach a server. `resolveVlmRoute` already picks
a typed URL over every other route.

So **Crucible receives pictures, never PDFs**, and the whole of phase 3c is: serve dots.ocr
through the `llm` proxy that already exists, and prove an image part survives it.

**There is no `vlm-pages` job type.** Inventing one would mean a second residency, a second
proxy and a second set of refusals for a request that is a chat completion in every respect
but its content parts. The capability a client needs is not "a different door" but "which of
your models takes pictures" — so that is what gets added. DESIGN.md's send-the-PDF job stays
future work, and this file is not it.

## 2. `modalities` on the model row

`models/<id>.toml` gains `[model] modalities`, a list, required in every manifest:

```toml
[model]
id = "dots-ocr"
modalities = ["text", "image"]
```

It appears on `GET /v1/models` rows and on `/v1/info`'s `llm` capability rows, which are the
same rows (PHASE2-LLM.md section 5). A client picks an image-capable model from that list
rather than knowing one by name.

It is required rather than defaulted to `["text"]`, for the reason every other required key
in a manifest is required: a manifest that forgets it must not quietly load as text-only and
have a page reader refused at request time with an error about content parts.

## 3. `--skip-mm-profiling` is a text-model flag and must not appear here

`qwen3.5-9b`'s manifest carries `--skip-mm-profiling`, worth a measured 1.90 GiB, and its
comment already says the condition: *"It does not make the engine text-only, it only stops
it RESERVING for an image. Nothing in Crucible sends one today. If the `llm` proxy is ever
given image input, this line must come out and the utilisation be measured again."*

This is that day, for **image-capable models only**. The rule, enforced by the manifest
loader rather than by a reviewer's memory: a manifest whose `modalities` contains `image`
and whose `engine_args` contains `--skip-mm-profiling` is **refused at load**, naming both.
A text-only model keeps the flag and keeps the 1.90 GiB.

## 4. The dots.ocr manifest

```toml
[model]
id = "dots-ocr"
family = "dots"
params_b = 3
context_default = 32768
modalities = ["text", "image"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "dots-studio/dots.ocr"
revision = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"
memory_bytes_estimate = 12_878_610_432   # DECLARED, not measured — see below
engine_args = [
  "--trust-remote-code",
  "--max-model-len", "32768",
  "--gpu-memory-utilization", "0.5",     # DECLARED, not measured — see below
  "--max-num-seqs", "16",
]
```

**The repo has been renamed, and this file had the old name.** As of 2026-09-13
`GET /api/models/rednote-hilab/dots.ocr` answers **307 to
`/api/models/dots-studio/dots.ocr`**, and `?author=rednote-hilab` lists nothing at all —
the org was renamed with its weights and history intact. Both names report `main` at
`c0111ce6bc07803dbc267932ffef0ae3a51dc951`, which is the pin. `hf_repo` names the repo as
it is named now: pinning the redirect would work today and would be a fallback, a name that
resolves only because somebody else's server is still forwarding it. What both apps *send*
is a separate question and unaffected — section 5 already rules that the client sets
`--vlm-endpoint-model dots-ocr`.

**The two numbers that could not be measured are labelled DECLARED in the manifest itself,
at length.** The 3090 Ti was on an overnight fine-tune and the Mac on an audio job, so the
choice was between a number with its derivation shown and no manifest at all. The
derivation, in brief:

- `0.5` is not invented. Both apps that serve this model on this card converge on it for
  the same stated, incident-backed reason: BookForge's `RESERVE_CAP_MB = 12_288`
  (`electron/vlm-page-server.ts`; set after a ~20 GiB reservation held the machine at 93%
  commit on 2026-08-11 and OOM-killed bun, ffmpeg and python, because under WSL's dxg layer
  every reserved GiB is also committed host RAM), and Foundry's flat `GPU_UTIL = 0.5`
  (`app/electron/vllm-server.ts`). 12_288 MiB of a 24_564 MiB card *is* 0.5.
- `memory_bytes_estimate` is that budget, because `--gpu-memory-utilization` is a budget
  and vLLM spends whatever the weights, the activation peak and the graphs leave of it on
  KV. Against the two blocks where both figures exist, the budget over-states the measured
  peak by 0.6 and 0.5 GiB — the safe direction for a guard. The floor it must clear is
  COMPUTED from `config.json` at the pinned sha: 6_078_431_736 B of bf16 safetensors plus
  28 layers x 2 KV heads x 128 head_dim x 2 x 2 B x 32768 tokens = 7_017_955_832 B.

Both are replaced by a measured peak the first time a card is free; the manifest says so in
the comment that carries them.

Two smaller findings while writing it. Neither app passes `--max-num-seqs` at all today, so
twelve in flight is served by vLLM's default rather than by anything anyone chose; 16 is set
here for the 9B's measured CUDA-graph reason. And no `--dtype` is passed: the checkpoint's
config already says `bfloat16` and neither working launch line overrides it.

Four requirements, all exact, all from CLIENT-SURFACES.md section 10 row 9:

- **Twelve concurrent pages minimum.** That is Foundry's ungated default and BookForge takes
  it by passing `concurrency: 0`. `--max-num-seqs` must be at least 12; 16 is the number the
  other manifests use and the number BookForge's own text server runs on this card.
- **32k context**, because the prompt plus an 11.3 MP page's image tokens plus an 8192-token
  answer do not fit less.
- **`--trust-remote-code`**, because dots.ocr ships its modeling class in its repo.
- **No `--skip-mm-profiling`** (section 3), and the utilisation measured *after* it comes
  out, because the profiled activation peak with an image in it is a different number from
  the one the text models measured.

`mlx-darwin` gets no block until `mlx-community/dots.ocr-4bit` is measured on the Mac.
Foundry's `mlx-local` route stays exactly where it is meanwhile — it is the Mac's only route
today and nothing in this phase touches it.

## 5. The served id is a client setting, not a server special case

Foundry sends whatever `--vlm-endpoint-model` says, defaulting to the registry's
`rednote-hilab/dots.ocr`. Crucible's ids are its own (`^[a-z0-9][a-z0-9._-]*$` — no slashes,
deliberately, because an id with a slash in it is a path waiting to be traversed), and the
proxy refuses a chat whose `model` is not the resident id, by name, with no rewriting.

So the client sets `--vlm-endpoint-model dots-ocr`. That is one settings value and no code,
and it is the right way round: a client that points at a Crucible names Crucible's model,
the same as every other client of every other model. The alternative — teaching the server
an alias table so a request for `rednote-hilab/dots.ocr` quietly reaches `dots-ocr` — is a
fallback, and a fallback in the one field that says what produced a book's markup is exactly
the wrong place for one.

## 6. What has to be proved

Against a fake engine, in pytest:

- A chat body whose `content` is a **list of parts** (`image_url` with a `data:` URI, plus
  `text`) reaches the engine byte-identical except for `model`. The proxy is meant to be
  verbatim; a data URI is the largest thing it will ever carry and the one most likely to
  meet a size limit that nobody wrote down.
- A ~1300x2112 PNG as a data URI — about 11 MB of base64 — goes through without truncation
  and without buffering the whole body twice.
- Twelve concurrent chat requests against one resident model all complete, and the proxy
  adds no serialisation of its own. (The engine batches; Crucible must not stand in front of
  that with a lock it did not need. `load-model` runs on the exclusive lane; chat does not,
  and this is the test that says so.)
- A manifest with `image` in `modalities` and `--skip-mm-profiling` in `engine_args` is
  refused, naming both.

All four are proved, in `tests/test_vlm_pages.py`, and the suite went from 155 to 176. Three
things the list above did not know:

**"Verbatim" is not "byte for byte", and now it is written down.** The proxy decodes the
request (`_chat_body`) and httpx re-encodes it, so the body on the second hop is the same
JSON with compact separators. On Foundry's exact body that is MEASURED at **18 bytes** out
of 10_986_363 — one per `", "` and one per `": "`, every separator in it and nothing else.
The test asserts against the compact encoding rather than a tolerance, so it is a statement
about whitespace and not about roughly the right size. Every value, the whole data URI
included, is identical; the base64 still decodes to the same PNG byte for byte.

**Yes, the request body is buffered more than once — four times over.** MEASURED on the
11.0 MB body by counting full-size materialisations inside the server: `Request.body()`
(bytes, and Starlette caches it on the request for the rest of the handler), `json.loads`
(a str), then httpx's `json_dumps` (a str) and its `.encode()` (bytes). All four are live
at the moment of the second hop, so one page in flight peaks at roughly **4x its own
size** — and the ceiling is twelve of them at once. `tracemalloc` around one request read a
peak of 76.98 MB, 7.0x the body, though that figure includes the in-process fake engine
reading and parsing it again, which a real out-of-process engine would not charge to
Crucible's heap. **Reported, not fixed**: another builder is in `api.py` tonight, and a
streaming pass-through of the body is a change to the proxy's shape rather than a patch.

**A required key is a claim on every manifest, including ones not written yet.** `modalities`
is now required in `[model]`, so any manifest a later phase adds to `models/` must declare
it or the loader refuses the build's whole model list by name.

Live, on the card, and **owed**: pull dots.ocr, load it, read one real page through the
proxy, compare the markup to what the `dots` env produces today for the same page, and write
the measured utilisation and the measured peak into the manifest in place of the two
DECLARED numbers.

## 7. What this deletes, once it is live

`electron/vlm-page-server.ts` (479 lines of WSL spawn, VRAM arithmetic and guest `pkill`),
`wslVlmRefusal`, `useWsl2ForVlm`, `wslVlmCondaEnv`, `wslVlmModel`, the `wsl-server` arm of
`resolveVlmRoute`, and `foundry-app/electron/vllm-server.ts` — **including the one path in
the app that reserves half the card with no arbitration at all** (`GPU_UTIL = 0.5`, no
`gpu-arbiter` import, and `foundry-job.ts` brackets only the *text* server, so a `read` gets
no lease). `stopVlmPageServer` has no caller today, which is its own small argument for
deleting the file rather than fixing it.

`mlx-local` stays. It is the Mac's only route until `mlx-darwin` has a measured block.

One thing to design around, from the audit: a queue row in a project's `project.json`
**freezes its `model` / `server` / `concurrency` at enqueue time**. Pointing a machine at
Crucible does not repoint rows already queued, and the migration has to say so out loud
rather than discovering it on a resumed job.
