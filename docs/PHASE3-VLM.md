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
hf_repo = "rednote-hilab/dots.ocr"
revision = "<40 hex, pinned by reading the repo>"
engine_args = [
  "--trust-remote-code",
  "--max-model-len", "32768",
  "--gpu-memory-utilization", "<measured>",
  "--max-num-seqs", "16",
]
```

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

Live, on the card, and **owed**: pull dots.ocr, load it, read one real page through the
proxy, compare the markup to what the `dots` env produces today for the same page, and write
the measured utilisation into the manifest.

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
