# What fits on this card — the design

Owen, 2026-09-16: *"this is one of those things that crucible is supposed to
adjudicate on its own. to decide how much it can fit… we need to come up with a
creative way of measuring the user's card and determining what would fit, and if
kv cache+overhead+model weights will fit on their card."*

NOT BUILT. This is the design, written while it was sharp.

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

## 6. Order of work

1. Split the manifest number into its terms. Pure promotion of existing prose;
   no GPU.
2. Per-class working context, from Owen's translate ruling. This alone stops the
   wrong refusals.
3. `fits` as arithmetic that names its terms.
4. Calibration (section 4). Wants the card.

1–3 are a day and need no GPU. 4 is what makes Crucible adjudicate rather than
recite a table.
