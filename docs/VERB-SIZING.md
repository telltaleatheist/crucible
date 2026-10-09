# Every verb, sized to the card — the plan

Owen, 2026-10-09. NOT BUILT. This extends `MODEL-CHOICE.md` and `FITS-AND-THE-CARD.md`;
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
7. **Asked for by name, Crucible tries.** An app naming a model, or a user's per-verb choice,
   is loaded even when the estimate says it won't fit. Only Crucible's own automatic pick
   respects the card. If the load fails, the error says why, with the numbers (engine out of
   memory, the bytes it wanted, what the card had).
8. **Apps call verbs, and may narrow them per request.** Owen: *"any app can programmatically
   tell crucible which model OR which model maximum to use with a verb. so if 27b is more than
   necessary for a chat job, the app can programmatically request the 9b. or they can set the
   maximum model size programmatically, so if they request the 9b instead of the 27b but
   theyre on an 8 gb card, it wont automatically try to use the 9b, itll use the biggest
   available up to 9b."* A request may carry:
   - a **verb** alone, served by what this server registered for it;
   - a **model**: exactly that, tried even when it won't fit (rule 7);
   - a **ceiling** (a size such as 9B): the automatic pick (rule 3) runs with the goal lowered
     to the ceiling, so on an 8 GiB card a 9B ceiling gets the biggest variant that fits up
     to 9B, never a 9B that doesn't fit. A ceiling above the verb's goal changes nothing.

   A chat request may name a verb instead of a model, and is served by whatever this server
   registered for it. B-Side calls `lyrics` and gets the bf16 model on the
   PC and the Mac and the 4-bit one on an 8 GiB laptop. Its own chooser (b-side 798c6bf) comes
   out.

## 2. What exists and what doesn't

Already there:
- Verbs: capability classes (`crucible/capabilityclasses.py`).
- Lineups: candidates per class.
- The recorded pick: the capability record, plus `[routes]` for the routable classes and a
  chosen model for the selectable ones.
- Variants as separate manifests: `qwen3.8-27b-4bit` / `-8bit`, `qwen3.5-4b-bside-4bit`.
- The ladder, for ASR widths.
- Settings choices and a greyed-but-clickable list (`settings.local_model_choices`).

Missing:
1. A per-verb **goal** and a fit-gated pick against it (today the pick is the biggest that fits).
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
   moves to the 9B on the PC and the Mac.
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

## 5. Open

- **The 9B floor (MODEL-CHOICE.md §0, 2026-09-16):** "they cant pick smaller than 9b" for
  translate/simplify. Rule 1 ("always have access to every verb … even … an 0.8b 4 bit") reads
  as superseding it for the *automatic* pick. To confirm with Owen: does a small card run
  translate on a sub-9B model, or show it below its floor?

## 6. What this reverses

- `decide` stops auto-picking the biggest model; its goal is 9B.
- A model the estimate says won't fit is no longer refused when named; it's tried.
- B-Side's own tag-model chooser (b-side 798c6bf) is replaced by the `lyrics` verb.
