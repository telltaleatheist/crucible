# Every verb, sized to the card — the plan

Owen, 2026-10-09. Phase 1 (§3 item 1) BUILT and merged (§7). Decide's image form, the
9B marker's removal and the API-key recommendation BUILT on `feat/decide-vision-and-api-hint`,
not merged (§8). The rest NOT BUILT. This extends `MODEL-CHOICE.md` and `FITS-AND-THE-CARD.md`;
where it disagrees with them, it says so (§6).

## 0. The ruling, in his words

> *"i think the goal behind crucible is to resize/pick the models the system is capable of
> running. … we'd have verbs, and crucible would pick whichever model it was capable of using
> by default. … OR the app can programmatically call specific models if they want. even
> models that are too big for the system … nothing should actually stop them … the decide
> verb pretty much universally calls the 9b … crucible should reconfigure its verbs and models
> to fit on the card its running. … b sides on her laptop still calls lyrics, as does the mac
> and the pc. they all use the same verb. its crucible who decides what fits on the card. …
> each job should have a goal - 9b 16 bit for decide, for example - but it uses whatever
> crucible has registered for that verb. and it registers when we download/set up the
> package, and can be configured directly in config files"*

> *"pretty much no matter what, we should always have access to every verb. even if that
> means we're running decide on an 0.8b 4 bit. crucible should decide on the biggest model
> size that works, prioritizing high parameters and low quant, and adding quant when there's
> more room. up to a limit. chat should shoot for 27b. decide shoots for 9b. lyrics is a set
> size already. we should have a measurement tool that determines the largest size that fits
> on that system. OR, the user can configure it manually by picking which models apply to
> which verbs. and that ui settings too, not just in a config file. 27b will never fit in an
> 8 gb card. thats obvious. have crucible use common sense about what will fit but dont
> undershoot it"*

> *"crucible should never assign a 27b to an 8 gb card. but if the user configures it
> directly, yes. try it. if it fails, give them the error why"*

## 1. The rules

1. **Every verb is always available.** A verb's lineup runs down to a model that fits any
   supported card (the 0.8B at 4-bit for the text verbs). Below its goal a verb is *smaller*,
   never *off*. (Only a card that cannot hold even the smallest is refused, by name.)
2. **Each verb has a goal**, the size it is meant to run at, and never auto-picks above it:
   | verb | goal | notes |
   |---|---|---|
   | `generate` (chat) | 27B | also `translate`, `simplify`, `analysis` |
   | `decide` | 9B | today it auto-picks the 27B on the PC and the Mac; that ends |
   | `clean` | 9B | |
   | `lyrics` (new) | qwen3.5-4b-bside | fixed size; only its precision steps down |
   | the media verbs | as today | `pages`, `asr`, `song`, `music`, ... keep their lineups |
3. **The automatic pick:** among the lineup's variants at or below the goal that FIT, take the
   most parameters, then the highest precision. So a 9B at 4-bit beats a 4B at 16-bit, and the
   27B at 8-bit beats the 27B at 4-bit when there is room. Precision floor: 4 bits.
4. **Fit is common sense and is measured, not guessed high.**
   - A size the card plainly cannot hold (a 27B on 8 GiB) is never auto-picked.
   - Declared estimates run high (YuE2 declared 16 GB, measured 8.7; Stable Audio declared
     8 GB, measured 6.2), and over-estimating is undershooting the card.
   - So `crucible ladder` grows from ASR to every verb: on the card itself it measures the
     largest variant of each lineup that actually loads and works. The pick reads measured
     figures where it has them, estimates where it doesn't.
5. **The registration happens at setup** (install, the card's first measure, `capability
   --write`), is recorded, and is re-made when the card or the catalog changes.
6. **The user can set it per verb**, in `config.toml` (`[routes]`, grown to cover every verb)
   and in the app's settings page: every variant is listed, with the ones that don't fit greyed
   out but selectable (`MODEL-CHOICE.md` §4).
7. **Configured by the user, Crucible tries.** A model the user set for a verb in Crucible's
   settings is loaded even when the estimate says it won't fit (narrowed by §1a.4: an app's
   named model is not, and the try has guards). If the load fails, the error says why, with the numbers (engine out of
   memory, the bytes it wanted, what the card had).
8. **Apps call verbs, and may narrow them per request.** Owen: *"any app can programmatically
   tell crucible which model OR which model maximum to use with a verb. so if 27b is more than
   necessary for a chat job, the app can programmatically request the 9b. or they can set the
   maximum model size programmatically, so if they request the 9b instead of the 27b but
   theyre on an 8 gb card, it wont automatically try to use the 9b, itll use the biggest
   available up to 9b."* A request may carry:
   - a **verb** alone, served by what this server registered for it;
   - a **model**: exactly that, when it fits; one the estimate says won't fit is refused
     with the verb's ceiling named (only a user's configured model is tried past it, §1a.4);
   - a **ceiling** (a size such as 9B): the automatic pick (rule 3) runs with the goal lowered
     to the ceiling, so on an 8 GiB card a 9B ceiling gets the biggest variant that fits up
     to 9B, never a 9B that doesn't fit. A ceiling above the verb's goal changes nothing.

   A chat request may name a verb instead of a model, and is served by whatever this server
   registered for it. B-Side calls `lyrics` and gets the bf16 model on the
   PC and the Mac and the 4-bit one on an 8 GiB laptop. Its own chooser (b-side 798c6bf) comes
   out.

## 1a. The holes, and Owen's answers (2026-10-09)

Five holes raised in review; Owen's answers, verbatim, then what they mean.

1. **One card, many verbs (swap churn).** *"i figured this would be on the calling app to use
   crucible efficiently, and on crucible to allow sessions to use the same model back-to-back,
   which it already does."* The pick does not trade quality for fewer swaps. A queue session
   keeps its model on the card between its items; an app that wants no churn batches its work
   by verb, or asks for one model across verbs (rule 8).
2. **Fit includes working room, not just weights.** *"i agree on the fix. maybe crucible
   dynamically decides which is the biggest model thatll fit based on the context being
   delivered. also, thats the purpose of the "maximum" setting, where a user can pick the max
   model parameters, for speed purposes or context purposes."* A model fits a verb only with
   room for the verb's working context and concurrency (FITS-AND-THE-CARD.md). Where a request
   states its context (`?context_tokens=`, or the prompt's own length), the pick is made
   against that, so a long request may get a smaller model than a short one. The per-verb and
   per-request maximum (rule 8) is how a person or an app chooses speed or context over size.
3. **Who wins when settings disagree.** *"i agree with your fix."* Precedence, first that
   applies: an exact model in the request, then a ceiling in the request, then the user's
   per-verb setting, then Crucible's own pick.
4. **"Try it" hurting what's running.** *"only try this if the user directly configures a model
   in crucible settings. this is a kind of "dont stop the user from hitting themselves in the
   face" type of thing. but yes, we can put sensible guards/protections on it."* This narrows
   rule 7:
   - Only a model the USER configured in Crucible's settings is tried past the estimate. An app
     naming a model that the estimate says won't fit gets a refusal that names the verb's
     ceiling instead.
   - Guards on the try: only when the card is idle; never evicting a job, session or stream in
     progress; a load that fails stops cleanly, leaves the card as it was, and says why with
     the numbers.
5. **Measuring every size of every verb.** *"we dont need to measure every size of every verb.
   we just need to measure the highest model weights/quant it can handle, and we can assign
   every verb within that range. might only take one or two measurements total. and that
   measurement might just be "how much memory does this card have?" and downloading the right
   model for that, and seeing how much headroom is there when it runs. then setting the final
   maximum model size based on that single measurement"*
   - The ladder becomes ONE measurement per card: from the card's memory, pick the largest text
     variant the estimate says fits; download it; load it under its verb's working context;
     read the real headroom.
   - That headroom sets the card's CEILING in bytes, recorded with the card's facts.
   - Every verb is then assigned within it from the catalog's figures, with no further
     measurement. A second measurement runs only if the first variant didn't load, or left
     enough headroom for the next size up.

## 1b. How close to the wall (Owen, 2026-10-09)

> *"in most cases the os will start pushing things into swap if it needs more room for a
> model. we shouldnt plan to hit the 8 gb wall, but we can get pretty close in most cases"*

- Fit plans close to the card's real limit with a modest margin, never a large blanket
  reserve. A flat 3 GiB desktop allowance on an unmeasured 8 GiB card throws away most of the
  headroom (the low_vram agent's open question, 2026-10-09). Where the desktop's share hasn't
  been measured, the default allowance scales with the card instead of being a flat 3 GiB, and
  the measurement (§1a.5) replaces it as soon as it runs.
- Spilling into system memory is a safety net, not a plan. It's slower, and whether CUDA
  inside WSL2 spills at all (rather than failing) is unmeasured; measure it before relying on
  it. The pick never chooses a model that only fits by spilling.

## 2. What exists and what doesn't

Already there:
- Verbs: capability classes (`crucible/capabilityclasses.py`).
- Lineups: candidates per class.
- The recorded pick: the capability record, plus `[routes]` for the routable classes and a
  chosen model for the selectable ones.
- Variants as separate manifests: `qwen3.8-27b-4bit` / `-8bit`. (`qwen3.5-4b-bside-4bit` is
  gone: on cuda-linux `qwen3.5-4b-bside` is a Q8_0 GGUF on llama-server, about 5.4 GB at 16k,
  which an 8 GiB card holds.)
- The ladder, for ASR widths.
- Settings choices and a greyed-but-clickable list (`settings.local_model_choices`).

Missing:
1. A per-verb **goal** and a fit-gated pick against it (today the pick is the biggest that fits).
   **Built in phase 1 (§7).**
2. **Small variants**, so every verb reaches every card:
   - cuda-linux (vLLM): W4A16 and W8A16 checkpoints of the 9B, the 4B and the 0.8B, built the
     way `qwen3.5-4b-bside-w4a16` was (compressed-tensors, round-to-nearest; Marlin on Ampere).
   - mlx-darwin: mlx-community's 4- and 8-bit conversions.
3. The **`lyrics`** verb.
4. **Verb-addressed chat**: a request naming a verb, resolved to the registered model, optionally narrowed by a model or a size ceiling (rule 8).
5. The **ladder for every verb**: measure the largest fitting variant per lineup, record it.
6. **`[routes]` for every verb**, and a Settings page in the app.
7. **Named loads that try** instead of being refused upfront, with an error that says why when
   they fail.

## 3. Order of work

1. Goals + the pick + always-available lineups, on the variants that exist today. Then decide
   moves to the 9B on the PC and the Mac. **BUILT 2026-10-09 on `feat/verb-goals` (§7); not
   yet measured on a card.**
2. The `lyrics` verb and verb-addressed chat; B-Side calls `lyrics`, its chooser comes out.
3. Small variants: build, upload (private HF, like the bside 4-bit), measure, ship.
4. The ladder for every verb, so picks come from measurement.
5. `[routes]` for every verb and the Settings page.
6. Named loads that try, and their errors.

Each phase ships on its own and is measured on the PC (24 GiB), the Mac (unified memory) and
an 8 GiB card (Victoria's laptop, when it's awake) before the next.

## 4. Measured so far (2026-10-08 / 09)

- yue2-3b: 8.73 GiB whole, 6.4–6.6 GiB with `[audio] low_vram`.
- stable-audio-3-medium: 6.2 GB at 380 s (declared 8).
- qwen3.5-4b-bside-4bit: 3.11 GiB of weights, about 5.3 GB serving one request at 8k context.

## 5. Settled

- **The 9B floor (MODEL-CHOICE.md §0, 2026-09-16) is gone.** Owen, 2026-10-09: *"regarding
  translate and simplify - i think we should make them available. it will be low quality, but
  let it work anyway. rely on the user to know that some models are more powerful than others
  and theyll get bad results if they try weaker models"*. translate, simplify and analysis run
  down the same lineup as every verb; below the goal they are smaller, not refused. Crucible
  states which model served the request, as it does for every verb, and makes no
  quality judgement of its own.

## 6. What this reverses

- `decide` stops auto-picking the biggest model; its goal is 9B.
- A model the estimate says won't fit is no longer refused when named; it's tried.
- B-Side's own tag-model chooser (b-side 798c6bf) is replaced by the `lyrics` verb.
- The 9B floor for translate/simplify (MODEL-CHOICE.md §0) and the class floors
  (MODEL-CHOICE.md addendum 2026-09-23) no longer refuse a smaller model.

## 7. Phase 1, as built (2026-10-09, `feat/verb-goals`)

Built on the variants that exist today. Nothing was run on a card: every figure below is the
catalog's estimate against each card's budget, computed by the tests, not measured.

**Where each fact lives (one owner each).**

- **The goal** is the class's: `CapabilityClass.goal`, a `Goal(params_b, source)` in
  `crucible/capabilityclasses.py`. `CHAT_GOAL` (27) on `generate`, `translate`, `simplify`,
  `analysis`; `DECIDE_GOAL` (9) on `decide`; `CLEAN_GOAL` (9) on `clean`. The media classes
  have none and pick as before (the first that fits by declared size).
- **The pick order** is the class's too: `CapabilityClass.pick_order`. At or below the goal;
  most `[model] params_b`; then most bits; then a model's own form before its `weights_of`
  alias (the same weights with a vision tower to hold); the catalog's order (largest need
  first) settles the rest. `verdict._best_fit` takes the first of these that fits.
- **Bits are a stated manifest fact**: `bits` on each `[backends.<kind>]` block, stated on every
  qwen manifest. `precision.weight_bits` reads it first; the parse refuses a value under 4,
  over 32, or one that disagrees with what the block's GGUF file, repo name or `--dtype` implies
  (`precision.implied_bits`, the reading that was the only source before). `dots-ocr` states none
  (its class has no goal; its cuda-linux block's precision is not stated anywhere today).
- **The floors are gone**: `min_params_b`, `NINE_B_FLOOR`, `NINE_B_TEXT_MODELS` and every
  text class's `binary_note`. `clean` reads the qwen3.5 family down to the 0.8B; the other
  four chat-shaped classes and `decide` read qwen3.8 and qwen3.5 down to the 0.8B.
- **Fit** is unchanged (`Candidate.holds`, with the class's working context). It never counts
  system memory, so nothing is picked that fits only by spilling (§1b).
- **The unmeasured card reserve scales with the card** (§1b): `config.default_desktop_allowance_bytes`
  gives a card an eighth of its VRAM, between 1 GiB and 3 GiB (`CARD_DESKTOP_ALLOWANCE_FRACTION`).
  An eighth matches both cards measured: the 24 GiB PC at 3 GiB, Victoria's 8 GiB laptop at
  1 GiB. A measured or stated reserve is used as it is; a CPU build keeps the flat 3 GiB (its
  pool is system memory, shared with the OS); the Mac keeps 25%. It applies when a config is
  written (`crucible init`): an existing `declared` 3 GiB stays until
  `crucible capability --measure-desktop` or a fresh init.

**What the surfaces say.** The record's `reason` and `summary`, so `crucible capability`,
`/v1/capability`, doctor and the install plan all say it: *"can decide, using qwen3.5-9b (goal 9B;
bf16 fits with 2.2 GiB to spare)"*, and below the goal *"(goal 27B; the largest that fits this
card)"*. A fitting model above the goal is named in the reason (*"qwen3.8-27b-4bit also fits, and
is above the 9B goal, which the automatic pick never exceeds; Settings can still choose it."*).
`/v1/capability` rows carry `goal` (`params_b`, `source`); install-plan rows carry `goal`, and
their `best` is the best within the goal. `Decision.chosen` says a Settings choice decided it.
Doctor reports a record whose model differs from what this build decides (`capability_stale`),
since a deploy does not re-record by itself: **the PC and the Mac keep deciding on the 27B
until `crucible capability --write` runs on each.**

**Precedence (§1a.3).** Unchanged where it exists: a model chosen in settings (`local_models`)
wins over the automatic pick, above or below the goal; `[routes]` sends the class upstream. The
request-level model and ceiling are phase 2.

**What each card picks** (estimates; PC = 24 GiB cuda-linux with its stated 3 GiB reserve, Mac =
64 GiB unified with the 25% reserve, 8 GiB = cuda-linux):

| verb | PC before | PC now | Mac before | Mac now | 8 GiB before (3 GiB reserve) | 8 GiB now (1 GiB reserve) |
|---|---|---|---|---|---|---|
| decide | qwen3.8-27b-4bit | **qwen3.5-9b** | qwen3.8-27b-8bit | **qwen3.5-9b** | qwen3.5-0.8b | qwen3.5-0.8b |
| clean | qwen3.5-9b | qwen3.5-9b | qwen3.5-9b | qwen3.5-9b | off | **qwen3.5-0.8b** |
| translate, simplify, analysis | qwen3.8-27b-4bit | same | qwen3.8-27b-8bit | same | off | **qwen3.5-0.8b** |
| generate | qwen3.8-27b-4bit | same | qwen3.8-27b-8bit | same | off | **qwen3.5-0.8b** |
| sfx | small-sfx | same | small-sfx | same | small-sfx | small-sfx |
| music | stable-audio-3-medium | same | same | same | off (short 1.3 GiB) | **stable-audio-3-medium** |
| song | yue2-3b | same | same | same | off (short 1.8 GiB with low_vram) | **yue2-3b** with `[audio] low_vram` |

On the 8 GiB card the 2B and 4B do not fit even at 7 GiB: their cuda-linux estimates carry the
3.5 GB image reserve of a block that serves images (the 2B needs 7.8 GiB at decide's context).
So every text verb lands on the 0.8B there until the 4- and 8-bit variants of phase 3 exist.

**decide on the small models.** The small tiers were added for decide (history/PHASE22-DECIDE.md
2.9) and the door's reading is per engine, not per model: vLLM (`--logprobs-mode raw_logprobs`,
run live on the 0.8B, 8a), mlx-lm (the patched logprobs, a live 2B triage on the Mac, 2.10) and
llama-server. Nothing found blocks a small model from decide. Two things to know: on the Mac the
small tiers serve text only (an image decision needs `qwen3.5-9b-vl`), and on vLLM the prefix
cache's 544-token blocks are the 0.8B's own measurement.

**Left for later phases or for Owen.**
- Measure phase 1 on the three cards (§3's rule), and re-record the PC and the Mac.
- ~~`qwen3.5-9b`'s `[local] minimum_for = ["translate", "simplify"]` still marks it the
  Foundry lineup's floor.~~ Removed (§8b).
- On a tie of size and bits the pick takes a model's own form before its vision alias, so the
  Mac decides on `qwen3.5-9b` (mlx-lm, 16 wide), not `qwen3.5-9b-vl` (mlx-vlm, width 1), though
  both fit. PHASE22's picker note said "the vision form when it fits"; that was the app's rule,
  and this is Crucible's. A decision that carries images now switches by itself (§8a).

## 8. Decide with images, the 9B marker, and an API key below 4B (2026-10-09, `feat/decide-vision-and-api-hint`)

Two rulings of Owen's from 2026-10-09, built together. Nothing was run on a card; the picks
below are the catalog's estimates against each card's budget, computed by the tests.

### 8a. Decide switches to a vision form by itself when a request carries images

> *"we do use vision sometimes for decide, so thatll have to be an option at a bare minimum."*

The vision form is the same model, so no less capable, but it costs throughput: on a Mac
mlx-vlm serves one request at a time against mlx-lm's 16, and on a 24 GB PC card the tower and
image reserve leave about 1,700 tokens of KV. So decide stays on the text form, and:

- **The pick.** `CapabilityClass.takes_images` (true on `decide` only) makes the verdict
  record a second model per granted decision, `Decision.with_images` (`verdict._image_pick`):
  1. the vision form of the text pick, when this host has one and it fits: a candidate that
     serves `image` here and is the pick itself or a `weights_of` alias of it
     (`Candidate.form_of`; `Candidate` now carries `weights_of` and `serves_images`);
  2. else the first of the class's pick order (at or below the 9B goal, most parameters,
     then most bits) that serves `image` and fits;
  3. else nothing (`""`), and the reason names what was passed over and what the smallest
     image model would need, and says to send without images or to a bigger server.

  A choice made in Settings is served with images by its own vision form the same way.
- **What each card gets** (estimates):

  | card | decide | with images | why |
  |---|---|---|---|
  | PC, 24 GiB cuda-linux, 3 GiB reserve | qwen3.5-9b | **qwen3.5-4b** | qwen3.5-9b-vl needs 21.5 GiB at decide's working context; there are 21.0 |
  | Mac Studio, 64 GiB | qwen3.5-9b | qwen3.5-9b-vl | the vision form of the same weights fits |
  | 32 GiB cuda-linux | qwen3.5-9b | qwen3.5-9b-vl | fits |
  | 16 GiB llama-windows | qwen3.5-9b | qwen3.5-9b-vl | fits (Q8_0 with its projector) |
  | 8 GiB cuda-linux | qwen3.5-0.8b | qwen3.5-0.8b | its own cuda-linux block reads images |
  | 16 GiB Mac | qwen3.5-4b | **nothing** | the small tiers serve text only on mlx; the 9B vision form is the smallest that reads images there |

- **The record shows both.** A capability row carries `with_images` and
  `with_images_reason` (written together, only on decide's granted row; absent from records
  written before this). `/v1/capability` answers both on every row (null where a class takes no
  images); decide's summary reads *"can decide, using qwen3.5-9b (goal 9B; …); with images,
  qwen3.5-4b"*, its reason (and so the doctor line) adds *"With images: …"*, and the install
  plan's row and line name it. Doctor reports a record whose image model differs from this
  build's, or that has none, as `capability_stale`.
- **The door.** `POST /v1/decide`'s `model` is optional now (`crucible api decide --model`
  and the TS SDK's `decide`/`decideItems` follow). No `model`: the decide row's `selected`,
  or with images its `with_images`. A named `model` keeps today's rules (`400
  model_text_only` when it serves no images). Refusals by name: `503 capability_undecided`
  (no record, or one from before the image pick: run `crucible capability --write`), `409
  capability_disabled`, `409 no_image_model_fits` (with the record's reason). The answer's
  `model` already names which weights served it.
- **Residency.** The door already handled a model that is not resident: the decision waits
  in the server's line and the model is loaded for it when its turn comes (`"queue": false`
  refuses `409 model_not_resident` instead). One model is resident at a time, so switching
  between the text form and the vision form is a reload; `qwen3.5-9b-vl` is `weights_of =
  "qwen3.5-9b"`, so it is served from the 9B's one download (no second pull). The tests load
  the 9B, decide with images (the 4B is loaded), decide without (the 9B again), and on a
  32 GiB card serve `qwen3.5-9b-vl` from the 9B's stamped weights alone.

### 8b. The 9B floor marker is gone

> *"we can remove it, yes."*

`[local] minimum_for` is no longer a manifest key (a manifest that still carries it is
refused as an unknown key), and everything that read it went with it:
`foundry-lineup.json` drops `floors` and each row's `minimum` and `minimumFor` (schema 3);
`modules.resolve_class` no longer resolves a class to a floor, so a module that needs a class
with several candidates names its model, as `clean` already did; the operator page's
"minimum for" chip is gone. `GET /v1/catalog` still sends `floors`, always `[]`, because the
SDKs released before this demand the key (as `license` is always null).

### 8c. An API key recommended below 4B

> *"but it can recommend using an api key if the models are under a certain size."*

- When a routable text verb's automatic pick (`clean`, `translate`, `simplify`, `analysis`,
  `generate`) has fewer than 4B parameters, its reason, summary, doctor line and install-plan
  line add one sentence: *"Models under 4B give weaker results, so for better ones add an API
  key for Anthropic or OpenAI in Settings, under "Accounts this engine may spend", and send
  translate to it in the same page (or set [upstreams] and [routes] in config.toml)."*
- The threshold has one owner: `API_KEY_ADVICE_BELOW_PARAMS_B = 4` in
  `crucible/capabilityclasses.py`, read by `CapabilityClass.advises_api_key`. 4B is the
  chosen default: the 0.8B and the 2B are what small cards get today.
- It never refuses and never changes the pick. `decide` is never advised (no upstream returns
  the logprobs it reads), and neither is a model a person chose in Settings.
