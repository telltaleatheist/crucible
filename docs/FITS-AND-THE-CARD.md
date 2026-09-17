# What fits on this card — the design

Owen, 2026-09-16: *"this is one of those things that crucible is supposed to
adjudicate on its own. to decide how much it can fit… we need to come up with a
creative way of measuring the user's card and determining what would fit, and if
kv cache+overhead+model weights will fit on their card."*

**STATUS 2026-09-16: steps 1, 2, 3 and the published ceiling are BUILT.** What
is left is step 4 (two-point calibration, which wants the card), step 5
(`context_exceeded` refused before the engine) and step 6 (the loaded context as
a settings knob). The sections below are the design as written; where the build
went further than the design said, the section says so.

What landed:

* `[backends.<kind>.memory]` in the manifests — `weights_bytes`,
  `overhead_bytes`, `kv_bytes_per_token`, `basis`, `measured_at_context` — on
  seven of eleven blocks, every number quoted from prose already in the file.
  The parser refuses terms that do not add back up to `memory_bytes_estimate`
  within 5%, and refuses terms taken at a context the block does not serve.
* `[model] trained_context` — what the WEIGHTS support, read from each pinned
  checkpoint's config.json. Required, and it bounds both the model's own
  `context_default` and any backend override.
* `CapabilityClass.work` — five classes now state the context and concurrency
  they actually use, each with a `source`.
* `decide()` asks `candidate.need_bytes(entry.work)` instead of reading a stored
  estimate, and every refusal names all four terms.
* `max_context` on the `llm` rows: `{tokens, card_affords, weights_allow,
  limited_by, concurrency, basis}`.

The four blocks WITHOUT terms are not an oversight and each says why in place:
`dots-ocr`'s cuda-linux estimate is a budget rather than a sum, its
llama-windows block has no cited KV rate, and the 27B-4bit on MLX has a residual
nothing measured says is flat or rising. A candidate with no terms answers every
question with its collapsed estimate, exactly as before.

## 0. The correction this starts from

`memory_bytes_estimate` does **not** omit the KV cache. `models/qwen3.5-9b.toml`
line 3 is the contract: *"what the whole engine holds when the model is resident
and a request has filled `context_default` tokens of KV: weights + KV + the
engine's own allocator overhead."* I said otherwise twice before reading it, and
the claim reached a wire type's doc comment and two other sessions. It is wrong.

Two real defects live where that wrong claim was pointing.

**a. The budget is measured against the wrong denominator.** vLLM's
`--gpu-memory-utilization 0.86` is a fraction of the card's TOTAL, not of what is
free, and WSL shares the card with the Windows desktop. Measured: the same argv
gave `Available KV cache memory: -0.08 GiB` and `engine_failed` on one run and
`2.44 GiB` and a clean load minutes later. A ~2.5 GiB desktop swing flips it, and
the failure lands after weights and CUDA-graph capture, so it reads as a late
crash rather than a sizing refusal.

**b. An estimate is only true at the context it was measured at.** A backend
block that omits `context_default` inherits the MODEL's. `llama-windows`
inherited 98304 — which `qwen3.8-27b-4bit.toml` itself calls "a Mac fact" — where
KV is 6.44 GB against a declared 1.5 GB allowance. Caught and pinned to 16384;
nothing structurally stops the next one.

## 1. The reframe: the free variable is the WORK, not the model

    engine_total = weights + overhead + kv_bytes_per_token x context x concurrency

The first three belong to the model and the backend. The last two belong to the
work. The card gives the budget. Today all five are collapsed into ONE number
keyed to ONE context, and that number is then used for every capability class.

So a 27B is refused on a 24 GB card for a 98304-token working context that
nothing in translate ever asks for. **Owen's fact makes this concrete:**
translate and simplify send roughly a paragraph at a time, batched, and each
block is independent of the one before it — so they need thousands of tokens of
KV, not a hundred thousand.

Stop asking "does this model fit". Ask "what can this card afford, for this
class" — and refuse only when the answer is less than the class needs.

## 2. The terms are already known; they are just not fields

Every term is already written down in `models/qwen3.8-27b-4bit.toml`, in prose:

    weights on the card          17.68 GiB
    non-KV demand                19.12 GiB     => overhead = 1.44 GiB
    KV really costs              86_251 B/token   MEASURED
    the arithmetic said          65_536 B/token   24% light

Promoting them to fields costs no new measurement:

    [backends.cuda-linux.memory]
    weights_bytes       = 18_568_108_256
    overhead_bytes      =  1_546_188_226
    kv_bytes_per_token  =         86_251
    basis               = "measured"
    measured_at_context = 16384

**Why the computed number was 24% light is the argument for section 4.** vLLM
pads the attention page up to the linear layers' recurrent state. No amount of
reading `config.json` finds that.

## 3. A class declares its working context; a model does not

    translate, simplify   ~4k tokens, batched          a paragraph, independent
    clean                 ~8k tokens, 2 in flight      a longer run of text
    pages                 32k tokens, 1 in flight      a whole OCR page

Owen's ruling above is the source for the first row. The rest are placeholders
until measured.

`fits` then stops being a stored boolean and becomes arithmetic that can STATE
ITSELF, in this repo's refusal voice:

> `qwen3.8-27b-4bit` for translate needs 19.4 GiB — 17.3 weights + 1.4 overhead
> + 0.7 KV for 4096 tokens x 4 in flight — and this card has 21.0 GiB.

## 4. The creative half: two-point calibration on the actual card

Do not compute the slope from the architecture. MEASURE it, once per model per
host, by loading at two small contexts (2k and 8k) and reading the engine's share
from the accelerator probe at each:

    slope     = (share_8k - share_2k) / (8192 - 2048)     bytes per token of KV
    intercept =  share_2k - slope x 2048                  weights + overhead

Two points give both numbers with no knowledge of layer counts, head dims,
attention intervals or quantization. It self-corrects for the engine version, the
driver, the GPU generation and the page padding that made the hand arithmetic
24% wrong. The result goes into the capability record for THIS host, which turns
`estimate_basis` from `declared` into `measured` for real rather than by
declaration.

Cost: two loads, a couple of minutes, once.

## 5. Decide against what is FREE, not against a guess about the desktop

`GET /v1/accelerator` already reports `free_bytes` and `unattributed_bytes` —
VRAM held by something that is not us. `CapabilityRecord` stores only
`total_bytes` and a 3 GiB `desktop_allowance_bytes` guess, which is precisely why
a desktop swing flips a load. Decide against measured-free, keep the allowance as
a FLOOR, and record both numbers so a stale decision is visible.

## 6. The ceiling, and how Ollama gets it wrong

Owen, 2026-09-16: *"there are some situations in which foundry and bookforge
could conceivably change the chunk size for things like translate. if thats the
case, crucible should have the upper limit for each crucible server. if bookforge
tries to send an entire book through on one translate call, that would work on
the mac but not on the PC. maybe kv cache should be a configurable number with a
maximum set. how does ollama handle this?"*

### 6.1 What Ollama actually does — MEASURED, Owen's PC, 2026-09-16

Not "it OOMs". Ollama never errors on context at all. It answers 200 and lies.

**Ask for a context bigger than the weights were trained for.** `num_ctx:
1_000_000` to `qwen3.5:9b-q8_0`:

    HTTP 200 in 49.4 s, response "ok"
    /api/ps: context_length=262144, size 14.71 GB, size_vram 14.71 GB (all GPU)

It clamped a million to 262144 — the checkpoint's `max_position_embeddings` —
and said nothing. The client asked for one thing, got another, and has no field
on the response that tells it so.

**Send a prompt longer than the context.** ~5,600 tokens of text whose FIRST line
is `REMEMBER THIS WORD: pomegranate.` and whose last line asks for that word
back, at `num_ctx: 512`:

    HTTP 200 in 8.6 s
    prompt_eval_count: 1026        (of ~5628 sent)
    response: "RE"

It threw away the front of the prompt — the instruction included — evaluated
about a fifth of what was sent, and answered anyway. No error, no warning, no
flag. A wrong answer that looks exactly like a right one.

There is a third silent shape not measured here: a `num_ctx` that fits the
trained maximum but not the card makes Ollama offload layers to system RAM and
run an order of magnitude slower, again with a 200 and no field saying so.

So Ollama's answer to "let the user set it to anything" is: accept anything,
clamp or truncate whatever does not fit, and never tell the caller. **This is the
model to not copy.** An OOM would be more honest than what it actually does.

### 6.2 The number is already on the wire; two things behind it are not

`/v1/info`'s `llm` rows already carry the ceiling per server per model, and they
already carry it twice for two different reasons (`crucible/jobs/llm/__init__.py`):

* `context_default` — the context THIS host intends to serve for this model,
  from the backend block or the model.
* `max_model_len` — what the resident engine is serving RIGHT NOW. A client
  sizes a request against this one; Foundry's `capFor` is
  `max_model_len − (⌈chars/2.5⌉ + 256)` with **no clamp** when the field is
  absent, which is how an unclamped request becomes a 400.

Measured on the PC today: `qwen3.5-9b` 16384, `qwen3.8-27b-4bit` 16384,
`qwen3.8-27b` 12288, `dots-ocr` 32768. So an app never has to guess a chunk
size — it reads the ceiling off the server it is about to talk to. Owen's
Mac-vs-PC example is already visible in that field.

Two things behind it are missing.

**(a) The Mac's number is a claim, not a limit.** `models/qwen3.5-9b.toml`'s
`mlx-darwin` block says so in its own comment: mlx-lm has no `--max-model-len`
and `Residency._engine_args` sends that flag to vLLM only, so the Mac takes its
context from the checkpoint's `max_position_embeddings` (262144) and the reported
number enables nothing — it only stops the server under-reporting. The
consequence is that the two backends do not merely have different ceilings, they
have different KINDS of ceiling: on the PC vLLM rejects an over-long prompt with
a 400; on the Mac nothing rejects anything and the machine runs until it cannot.
One field, two meanings. That is the asymmetry, and it is worse than Owen's
version of it.

**(b) Crucible does not adjudicate; it forwards the engine's verdict.** Today an
over-long request reaches vLLM and comes back wearing vLLM's sentence — and on
mlx-darwin there is no sentence at all. Crucible should count the prompt and
refuse by name before the engine sees it, the same refusal on every backend:

    context_exceeded: this request is 41,208 tokens and this server serves
    qwen3.5-9b at 16,384. Send it in blocks, or load the model with a taller
    context — this card affords up to 49,152 at concurrency 16.

Both numbers in the sentence, and the third one — what the card *could* afford —
is the fits arithmetic talking.

### 6.3 The ceiling should be derived, not typed

Every `context_default` in the tree today is a hand-written number with a comment
arguing for it. `qwen3.5-9b`'s own comment is forty lines of archaeology about a
12288 that turned out to be BookForge's 32B tier reaching a 9B. That is the cost
of a typed ceiling: it has no source, so it takes a day to disprove.

The same terms that answer *does it fit* answer *how tall can it be*. Section 1's
identity, solved for context instead of for total:

    max_context = (available − weights − overhead) / (kv_bytes_per_token × concurrency)

Every term on the right already exists — stated as prose in the 27B manifest
(weights 17.68 GiB, non-KV demand 19.12 GiB, KV 86,251 B/token MEASURED), and
obtainable on any card by section 4's two-point calibration without knowing a
single architectural fact. So the server can say *"on this card, at this
concurrency, this model tops out at N"* rather than reciting a table, and the
Mac/PC difference stops being two opinions and becomes two cards.

### 6.4 Yes to a configurable KV with a maximum — and where the knob lives

Owen's instinct is right, with one correction about what the knob IS.

On vLLM the context is **fixed when the engine starts**, not negotiated per
request: `--max-model-len` sizes the KV pool at load. So "BookForge changes the
chunk size for translate" can only ever move WITHIN the started ceiling; going
above it is a reload, not a bigger request. That is good news rather than a
limitation — there is exactly one number per resident engine, the app reads it
and chunks to it, and no per-request negotiation is needed.

So the configurable number is the context the model is LOADED with, it belongs in
settings (which the apps own, per the Ollama standard — Crucible is set and
forget), and Crucible computes the maximum from 6.3 and **refuses a value above
it by name, before the engine starts.** That refusal already exists in a worse
form: the 9B block records that `--gpu-memory-utilization 0.79` on the 3090 Ti
produces a −0.07 GiB pool and vLLM dies with `No available memory for the cache
blocks`. Same fact, discovered by the engine after a minute of loading instead of
stated by Crucible in a millisecond.

And per section 3 this knob barely matters for the work Owen actually runs.
Translate and simplify send ~4k batched blocks that never approach any of these
ceilings; raising the context buys DEPTH — more requests in flight against a
bigger pool — not reach. It is `pages` at 32k that has to fit, and `pages`
already declares it.

## 7. Order of work

1. Split the manifest number into its terms. Pure promotion of existing prose;
   no GPU.
2. Per-class working context, from Owen's translate ruling. This alone stops the
   wrong refusals.
3. `fits` as arithmetic that names its terms.
4. Calibration (section 4). Wants the card.
5. `context_exceeded` as Crucible's own refusal, counted before the engine sees
   the request, identical on every backend (section 6.2b). This is what closes
   the mlx-darwin hole, where today nothing refuses at all.
6. The loaded context as a settings knob with a DERIVED maximum (section 6.4),
   refused above the ceiling before the engine starts rather than after.

1–3 are a day and need no GPU. 4 is what makes Crucible adjudicate rather than
recite a table. 5 needs no GPU and is worth doing early — it is the difference
between Crucible's verdict and vLLM's. 6 waits on 4, because its maximum is 4's
output.
