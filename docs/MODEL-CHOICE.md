# Who picks the model — the ruling

Owen, 2026-09-16. This supersedes the binary translate ruling of 2026-09-13 and
is the contract the settings doors in BookForge and Foundry are built against.

NOT BUILT except where a section says otherwise. Written the day it was given.

## 0. The ruling, in his words

> *"the user can pick a model to use for translate/simplify/etc, but they cant
> pick smaller than 9b. this is a reversal from what it was previously - before,
> it was never smaller than 27b. i think 9b could do an ok job at translation.
> bookforge/foundry should give the user the option of using 3.5:9b or 3.8:27b IF
> their system can manage it, and it should give them quantizing options IF their
> system can handle it. so for some things, id want to use a 9b quantized to 16
> bit instead of a 27b 4 bit because its faster. that should be a choice the user
> can make in setup/settings when picking which models to use and which to
> download for foundry or bookforge. if host foundry chooses 9b 4 bit, and
> vendored foundry in bookforge picks 27b, both models should exist in the system
> and the user should be faced with a choice on trnaslation/simplify jobs (or in
> settings instead of giving them the choice every single time). they should be
> able to download both if they want, and pick which to use in settings as a
> permanent setting that can be changed in the settings page of foundry and/or
> bookforge. it can show them other models available, but theyll be grayed out
> but clickable if they cant run it. there can be an override that allows them to
> run something that crucible is telling them they cant run beceause sometimes i
> can see a case where maybe it could. actually, we should go with what ollama
> does here. also, if nothing fits their card, it should give them the option of
> using api keys for claude or openai. and regarding ollama - if its possible to
> use the ollama copies that already exist on disk then we should do that. i dont
> want to have 16 copies of giant models sitting around"*

and

> *"they sohuld have a way to delete models from crucible, too. probably through
> bookforge/foundry settings"*

## 1. What is reversed

`crucible/capability.py` says, in `translate`'s `binary_note`:

> *"Translation is binary per server: it needs a 27B and the smallest this build
> ships is already 4-bit, so this host cannot translate."*

That was Owen's ruling of 2026-09-13 — *"translation is binary per server as
well. it should use a 27b to translate. if 27b doesnt fit on the card then it
cant translate."* It is now withdrawn. **The floor for translate, simplify and
analysis is the 9B**, not the 27B. `clean` is unchanged: it was always 9B-class.

The three classes stay three classes. That part of the 2026-09-13 ruling stands
for its own reason, which was never about size: *"they can't lie to the user and
say a translate job is running when it's actually a simplify job."*

## 2. Quality is the USER'S axis now, not the card's

The old shape had Crucible pick: candidates ordered by declared size, best-first,
take the first that fits. That is still the DEFAULT and is still what an app gets
by choosing nothing. What changes is that the order is no longer the only
argument, because size is not the only axis a person cares about:

> *"for some things, id want to use a 9b quantized to 16 bit instead of a 27b 4
> bit because its faster"*

A 9B at bf16 and a 27B at 4-bit are roughly the same bytes and are NOT the same
trade. One is faster and less capable; the other is slower and knows more. No
arithmetic in this repo can rank them, because the ranking depends on what the
person is doing. So the catalog offers both, Crucible says which of them this
card can hold, and the choice is the user's.

**Crucible's job is unchanged and is the whole of its job: say what fits, and
say it with numbers.** Picking is the app's, and behind the app it is Owen's.

## 3. Two apps, two choices, one machine

> *"if host foundry chooses 9b 4 bit, and vendored foundry in bookforge picks
> 27b, both models should exist in the system"*

So a model choice is per APP, not per machine, and the weights store holds
whatever any app chose. This is already the shape `/v1/settings`'s
`local_models` has — a chosen model id per capability — and the settings document
is per client. What it needs is that **choosing does not evict**: today's
"a subject is never stored twice on one machine" rule (PHASE15-HOST.md 3.5) is
about not duplicating ONE subject, and it does not say a machine holds one model
per class. Nothing has to change for two models to coexist; what has to change is
that nothing may delete one because another was chosen.

The choice is a SETTING, not a prompt:

> *"or in settings instead of giving them the choice every single time"*

Settings. A per-job picker is a per-job decision, and a person translating a book
makes it four hundred times.

## 4. Greyed but clickable, and the override

> *"it can show them other models available, but theyll be grayed out but
> clickable if they cant run it. there can be an override… actually, we should go
> with what ollama does here."*

Care is needed here, because "what Ollama does" was MEASURED two hours before
this ruling and section 6.1 of `FITS-AND-THE-CARD.md` is a list of the ways it
lies. The two must not be confused, and they are easy to separate:

* **What Ollama does that is right, and what Owen means:** it does not refuse.
  There is no allow-list of models your card is big enough for. You ask for a
  27B on a 12 GB card and it tries — offloading layers to system RAM and running
  slowly rather than telling you no. Sometimes that is exactly what a person
  wants, and Owen is right that the server cannot know when.
* **What Ollama does that is wrong, and must not be copied:** it does all of that
  SILENTLY. `num_ctx: 1_000_000` comes back 200, clamped to 262144, with no field
  saying so. A prompt past the context is truncated from the front and answered
  anyway. The user is not told they are in the degraded case.

So the override is: **Crucible's refusal becomes a warning the user can accept,
and accepting it is recorded.** The numbers stay on the screen, the estimate
stays in the settings document, the job's provenance says the guard was
overridden and by how much. A run that was taken past the guard must never be
indistinguishable afterwards from one that fit.

Mechanically this is a field on the settings document beside the chosen model —
an override is per (client, capability, model), not a global "stop checking" —
and `decide()`'s refusal already carries `shortfall_bytes`, which is the number
the warning shows.

## 5. When nothing fits

> *"if nothing fits their card, it should give them the option of using api keys
> for claude or openai"*

This exists. `CapabilityClass.routable` is true for exactly the five chat-shaped
`llm` classes — `clean`, `translate`, `simplify`, `analysis` and (since
2026-09-23) `generate` — and a routed class
keeps its local answer whole underneath (`LOCAL_ANSWER_PREFIX`: *"the local
answer would be: …"*), so routing back loses nothing. Keys live IN the engine and
apps write them through `/v1/settings` (PHASE15-HOST.md; the apps store no key).

What is missing is not the machinery, it is the OFFER: a disabled class today
says "the smallest of 2 qwen3.8 variants is … and there is only … — short by …"
and stops. It should end by naming the door that is open. `pages` is deliberately
NOT routable and stays that way: sending page images to Anthropic is a different
feature with a different body that nobody has asked for.

## 6. Reusing Ollama's copies — MEASURED, and the answer is "partly"

> *"i dont want to have 16 copies of giant models sitting around"*

Measured on Owen's PC, 2026-09-16: `~/.ollama/models` is **64 GB**, and the
largest blob begins with the four bytes `47 47 55 46` — `GGUF`. Ollama stores
GGUF and nothing else.

That decides it per backend, and the answer is not the same on all three:

| backend | engine | wants | can it read Ollama's store? |
| --- | --- | --- | --- |
| `llama-windows` | llama.cpp | GGUF | **yes** — it is the same format, and `-m` takes any path |
| `cuda-linux` | vLLM | safetensors | no |
| `mlx-darwin` | mlx-lm | mlx safetensors | no |

So on a Windows box the whole 64 GB is usable as it stands, and the manifests
already name the exact tags to look for — every `[local]` table has `kind =
"ollama"` and the published tag. The join is: read
`~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>`, take the
layer whose mediaType is the model, and point `-m` at
`~/.ollama/models/blobs/sha256-<digest>`.

**On the machine Owen actually runs, it does not help.** His server is
`cuda-linux` inside WSL, which needs safetensors; those 64 GB cannot feed it at
any price short of re-quantizing. That is worth saying plainly rather than
promising a saving that will not arrive: the duplication he is seeing on the PC
is Ollama's copy and Crucible's copy of two DIFFERENT formats of the same
weights, and no amount of pathing makes one into the other.

Two honest things can still be done about the size on that machine:

1. `llama-windows` — the Windows-native backend, used when there is no WSL —
   should resolve weights through Ollama's store when a `[local] kind = "ollama"`
   table names a tag that is present. That is a real saving for every user who
   has Ollama and no WSL, which is most of them.
2. The `DELETE` door in section 7 is how the PC's duplication actually gets
   fixed: pick one, delete the other.

## 7. Deleting — ALREADY BUILT on the server

`DELETE /v1/catalog/{kind}/{subject_id}` exists, is authenticated, and refuses by
name in the job door's order (what is wrong with the REQUEST first, then what is
wrong with this server's STATE, so a misspelled id is not answered with "it is in
use"). It was built for PHASE15-HOST.md 3.5a, so that the host never reaches into
`crucible/weights.py`'s layout from outside.

So Owen's *"they should have a way to delete models from crucible, too. probably
through bookforge/foundry settings"* is an APP-SIDE job only. Nothing is owed in
Crucible: the settings pages need a row per installed subject with its size and a
delete button, wired to that route.

## 8. What the catalog has to grow

The choice Owen describes cannot be offered out of today's four manifests,
because the variants he names do not all exist:

| he said | exists today |
| --- | --- |
| 3.5:9b | `qwen3.5-9b` (bf16) — yes |
| 3.8:27b | the bf16 is GONE (2026-09-17) — it fit nothing either of us owns, so `qwen3.8-27b-8bit` replaced it: FP8 on cuda-linux, MLX 8-bit on the Mac, where it fits |
| 9b quantized to 16 bit | that IS bf16 — `qwen3.5-9b` |
| a 9B 4-bit | **no** — `capability.py` currently says out loud that this build ships none |
| 27b 4-bit | `qwen3.8-27b-4bit` — yes |

So the gap is a 4-bit 9B, and it is the variant that makes translate possible on
a small card at all. Adding it is a manifest with a pinned repo, a revision, and
`memory_bytes_estimate` — and, since 2026-09-16, a `[backends.<kind>.memory]`
table and a `trained_context`.

`clean`'s `binary_note` also says this build ships no 4-bit 9B; that sentence
comes out with the same change.

## 9. Order of work

1. **Drop the translate/simplify/analysis floor to the 9B.** A candidates change
   and three `binary_note` rewrites. No new weights, and it is what makes every
   later step mean anything.
2. **End a disabled routable class's refusal with the upstream offer** (section
   5). Prose in one place; no new machinery.
3. **The override** (section 4): a recorded, per-(client, capability) acceptance
   of a named shortfall, carried into job provenance.
4. **A 4-bit 9B manifest** (section 8).
5. **`llama-windows` resolves weights through Ollama's store** (section 6.1).
6. **App settings pages**: pick per capability, download either, delete either,
   show what does not fit greyed with its numbers.

1 and 2 are small and unblock the apps. 3 is the one with a contract to get right
— an override that is not recorded is worse than no override, because the run it
produces cannot be told apart afterwards from one that fit.

## Addendum, 2026-09-23: the floor is a number on the class

Section 1's floor — *"they cant pick smaller than 9b"* — was, until today, a side effect:
`clean` read the `qwen3.5` family and `translate`/`simplify`/`analysis` read `qwen3.8` and
`qwen3.5`, and the 9B was the floor only because it was the smallest model either family
shipped. PHASE22 added `qwen3.5-4b` and `qwen3.5-0.8b` for the decision door, and a family
filter alone would have put both under all four classes without anybody deciding it.

So the floor is now explicit: `CapabilityClass.min_params_b`, applied by the candidate
source (`CatalogCandidates.min_params_b`) against each manifest's `[model] params_b`, and set
to `capability.NINE_B_FLOOR` (9) on `clean`, `translate`, `simplify` and `analysis`. Their
candidate lists did not move; `tests/test_decide_lineup.py` asserts them exactly, per
backend. The new `decide` class has **no floor** — a decision is the one text act a 0.8B
does well enough to offer (PHASE22-DECIDE.md sections 8a and 2.9) — so it lists every tier.
