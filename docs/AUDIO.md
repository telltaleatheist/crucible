# Audio generation: the `audio` job

Words in, sound out: sound effects, instrumental music, and full songs with sung vocals. One job
type, `audio`, serves four models; which one you name decides what you get. How it runs and
why is [internals/audio.md](internals/audio.md); this page is what a caller sends, what comes
back, and how to write a prompt each model understands.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | longest | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `stable-audio-3-small-sfx` | sound effects | `sfx` | yes | yes | 120 s | Stability AI Community (gated) |
| `stable-audio-3-medium` | instrumental music, stems, also effects | `music` | yes | yes | 380 s (6 min 20 s) | Stability AI Community (gated) |
| `yue2-3b` | songs with vocals, from lyrics and style tags | `song` | yes | **no** (below) | about 6 min, set by the lyrics | CC BY-NC 4.0 + creator addendum |
| `yue2-3b-instrumental` | instrumentals only: yue2-3b plus an adapter that composes without a singer ([below](#yue2-3b-instrumental-instrumentals-that-do-not-invent-a-singer-2026-10-12)) | none (named by the client) | yes | no | as yue2-3b | CC BY-NC 4.0 |

`GET /v1/capability` has one row per class: "can make sound effects, using
stable-audio-3-small-sfx", "can make music, using stable-audio-3-medium", "can make songs with
vocals, using yue2-3b", each with the fit reason, or why not.

**YuE2 runs on the Mac too (2026-10-03), on the official code with a newer torch.** Its
pinned torch (2.10) can silently corrupt the bfloat16 causal attention it runs on Apple's
Metal backend - a query sees up to three future tokens (YuE issue #176, pytorch#195910,
fixed in torch 2.13). The Mac's env therefore pins torch 2.14.0 and installs `yue2-infer`
without its own dependencies (`# crucible: no-deps` in the recipe). Because the bug depends
on the chip and macOS (it showed on an M4 Pro and an M5 Max, not on the M1 Ultra), the
worker re-proves the kernel on every load - YuE2's own `sdpa` with `is_causal` against an
explicit causal mask - and refuses to generate if they disagree; each render's provenance
carries the figures (`versions.mps_causal_check`). The community MLX ports are not used.

## Turning it on

```bash
crucible install audio                # builds one env per audio engine this machine runs,
                                      # places ffmpeg, and turns [jobs] enable_audio on
crucible models pull stable-audio-3-small-sfx
crucible models pull stable-audio-3-medium
crucible models pull yue2-3b
```

None of these is needed by hand: a job for a missing env or model starts the install and
answers `409 installing`, like every other type. `crucible jobs enable audio` and
`crucible jobs disable audio` turn the type on or off without touching anything else in
config.toml (the token stays); `enable` is refused by name when the card cannot hold any
audio model or the envs are not built. On a Windows PC these run in the Linux engine:
`crucible guest jobs enable audio`. The one step Crucible cannot take for you is
accepting a licence (next section).

### A card too small for YuE2 whole: `[audio] low_vram`

YuE2's 7.26 GB backbone is two halves that never run together: the AR half (2.83 GB, plus
the 1.51 GB of embeddings and output layer) writes the score and the song, and the NAR half
(the `nar_*` modules, 2.82 GB) solves the synthesis. On a host whose card cannot hold the
model whole (an 8 GiB laptop), Crucible turns `[audio] low_vram` on by itself, and only the
half a stage uses is on the card; the other waits in host memory. Measured on
the 3090 Ti on 2026-10-08 through the worker, capped as an 8 GiB card: 6.37 to 6.62 GiB of
card over the desktop for songs of 204 to 312 s, against 8.73 GiB holding YuE2 whole; about
the same render time; and the audio within 5.4e-6 (-105 dB) of the whole model's. The load is admitted against the manifest's
`low_vram_memory_bytes_estimate` instead of `memory_bytes_estimate`, and the `done` event's
`audio.low_vram` says which ran.

The capability verdict weighs the same figure: `audiomodels.held_need` is the one rule for
which need a host uses, and the audio job, `crucible capability`, `crucible install audio`,
`crucible doctor`, the Settings model choices and `/v1/info`'s `vram_bytes` all read it. On
an 8 GiB card with a 1 GiB desktop allowance, `song` is granted "with [audio] low_vram"
(6.8 GiB of 14.9 GiB whole); with a person's off it is refused, and the refusal names
`crucible audio low-vram on` (or `auto`) as the fix rather than a 7.9 GiB shortfall alone. A capability record decided
before the setting changed is reported stale by `crucible doctor`.

**Who turns it on.** A friend's 8 GiB laptop (2026-10-08) needed it, and it had to be set by
hand inside a WSL distribution its owner did not know existed. So whenever Crucible decides
the card (`crucible install audio`'s capability step, `crucible capability --write`, the
server deciding a card it has no record for, a new desktop allowance in Settings), it
applies one rule, `crucible/lowvram.py`: a model that declares a low-VRAM figure and does
not fit whole in what the card gives a job, but does fit at that figure, turns it on. It is
written with the capability record, in the same write, and said in one sentence:

```
Crucible turned [audio] low_vram on: yue2-3b needs 14.9 GiB whole and this card gives a job
7.0 GiB, so it now holds only the part each stage uses (6.8 GiB); the audio is the same, and
`crucible audio low-vram off` turns it off
```

On a card that holds the model whole it stays off and config.toml gets no `[audio]` table
(Owen, 2026-10-08: *"this would be a configuration for systems with low ram, not for high
ram systems like this pc"*), and Crucible's own on comes back off if the same config is
decided on a bigger card. The rule weighs what the card gives a job, after the desktop
reserve: an 8 GiB card whose reserve was never measured keeps Crucible's 3 GiB default and
gives 5.0 GiB, which YuE2 does not fit even split, so the setting stays off and `song` says
it is short (`crucible capability --measure-desktop` measures the reserve; Victoria's laptop
measured 1 GiB).

**Who owns it.** The file says, in two keys:

| `[audio]` | means |
|---|---|
| neither key | Crucible decides, and it is off |
| `low_vram = true`, `low_vram_auto = true` | Crucible turned it on for this card, and decides it again with the card |
| `low_vram = true` or `false` alone | a person set it; nothing but a person changes it |

```bash
crucible audio low-vram               # the setting, who set it, and what this card makes of it
crucible audio low-vram on            # yours: on, whatever the card
crucible audio low-vram off           # yours: off, and Crucible will not turn it back on
crucible audio low-vram auto          # hand it back to Crucible
```

(`crucible guest audio low-vram ...` on a Windows PC.) The operator console's Settings panel
has the same switch, through `PUT /v1/settings/audio/low-vram`, which writes through the
same door. Each writes the setting and decides the card's audio verdict again (the
capability record and `[jobs] enable_audio`), so nothing is left stale; after a hand edit
of config.toml, run `crucible capability --write`. `crucible doctor` says which of the
three it is, and reports `capability_stale` when Crucible would change its own setting for
this card and has not written it yet. A running server reads a change on its next request;
a model already resident keeps the way it was loaded until it comes off the card.

Only a model whose manifest declares a low-VRAM figure honours it: today `yue2-3b` on
cuda-linux, and `on` is refused (`low_vram_not_offered`) on a backend where no model does.
`[audio]` is not a table `crucible install` writes, so a reinstall keeps it. Two keys rather
than `low_vram = "auto"`: every reader acts on the one boolean, and an older Crucible reads
`low_vram` and ignores `low_vram_auto`, so rolling back keeps the card working.

### The Stable Audio models are gated

Hugging Face serves `stabilityai/stable-audio-3-small-sfx` and `stabilityai/stable-audio-3-medium`
only to an account that has accepted the Stability AI Community License. Until then a job or a
pull is refused `409 model_gated`, and the refusal says exactly what to do:

1. Signed in to Hugging Face, open https://huggingface.co/stabilityai/stable-audio-3-small-sfx
   and https://huggingface.co/stabilityai/stable-audio-3-medium and accept the licence on each
   (acceptance is immediate).
2. Make a read token at https://huggingface.co/settings/tokens and give it to the server: set
   `HF_TOKEN` in the environment Crucible runs in, or put it under `[hf] token` in the config
   file the refusal names.
3. Run `crucible models pull stable-audio-3-small-sfx` (and `-medium`) again, or resend the job.

YuE2's repos (`m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`) are not gated.

## The request

A sound effect:

```json
{"type": "audio",
 "model": "stable-audio-3-small-sfx",
 "params": {"prompt": "TrackType: SFX. A heavy oak door creaks open slowly in a stone hallway, close mic, dry",
            "duration_s": 4, "seed": 1}}
```

Music:

```json
{"type": "audio",
 "model": "stable-audio-3-medium",
 "params": {"prompt": "TrackType: Music, VocalType: Instrumental. Warm lo-fi hip hop, dusty Rhodes, soft vinyl crackle, laid-back boom bap drums, 84 BPM",
            "duration_s": 120}}
```

A song:

```json
{"type": "audio",
 "model": "yue2-3b",
 "params": {"tags": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, 88 BPM",
            "lyrics": "[Verse]\nThe kettle sings the morning in\nThe window fogs, the day begins\n\n[Chorus]\nStay, stay a while\nThe light is soft, the hour is mild\n",
            "seed": 3}}
```

An instrumental (YuE2 writes the score, its vocal melody moves note for note to the
instrument, and that score is rendered with only section tags - nothing is sung; YuE2's own
yue2-music workflow, vendored in `crucible/jobs/audio/yue2music/`):

```json
{"type": "audio",
 "model": "yue2-3b",
 "params": {"tags": "Instrumental, slow somber piano and strings, no vocals, no singing, no choir, 66 BPM",
            "instrumental": true,
            "seed": 3}}
```

The render's `effective_params.notes` carries YuE2's transfer report (how many notes moved)
and the score it first planned.

### Planning lyrics: what an instrumental is planned from

YuE2 plans an instrumental's score the way it plans a sung song's: from lyrics. It then
moves the vocal melody to the instrument and re-plans from that fixed score with section
tags only, so **the words are never sung** (the yue2-music skill's own design, its
`--planning-lyrics-file`). The words are there to give the melody a sung song's bounded
phrase structure. Planned from empty sections (`[Intro] [Verse] [Chorus] [Outro]` with no
lines), nothing bounded how many bars YuE2 wrote: on Victoria's 8 GiB laptop (2026-10-10)
4 of 13 instrumentals ran the score to its 4096-token cap and failed, and the rest scored
anywhere from 1023 to 3911 tokens (137 to 360 s of audio), where sung songs score 1800 to
2600.

An instrumental is planned from, in order:

1. **`planning_lyrics`**, when the client sends them: its own words, never sung.
2. **`lyrics`** holding only section tags (`[Intro]\n\n[Verse]\n\n[Chorus]\n`), when the
   client wants to set the form itself. Empty sections bound nothing, so this is the old
   behaviour and can run the score to its cap; prefer `planning_lyrics`.
3. Otherwise **a set from the server's pool**: ten original planning songs in
   `crucible/audio/planning/yue2.toml`, the one named by **`planning_set`** when the client
   sends it, else picked **from the seed**: set number `seed mod 10`, in file order. The
   same params and seed plan the same song; a different seed usually plans from a different
   set. Seeds can still collide (two seeds 10 apart pick the same set), so a client making
   an album names the set per track instead (B-Sides: track index to set). The set ids are
   the `planning_set` field's options on the model's page of `GET /v1/playground`, and an
   unknown id is refused `planning_set_unknown` with every id in the message and in
   `details.planning_sets`. `planning_set` goes only with `instrumental: true` and never
   beside `planning_lyrics` or `lyrics` (`audio_param_conflict`).

The ten sets differ in structure, not only in words (verse/chorus, verse/chorus/bridge,
AABA, through-composed, chorus-first; with and without an intro, interlude or outro; 4 to
8 sections; 3 to 6 lines a section; lines of 4-6 or 8-11 syllables), and each is sized
like B-Sides' sung songs (`[Verse] [Chorus] [Verse] [Chorus]`, four lines of 6 to 9
syllables, which score 1800 to 2600 tokens): about 95 to 140 syllables. The words are
deliberately generic, since words colour the melody. Adding or reordering sets changes
which set a seed picks; the record below carries the text, which sent back as
`planning_lyrics` plans the same song whatever the pool says by then.

The `done` event's `audio.planning_lyrics` (and a kept request's `settled.planning_lyrics`)
says what the score was planned from:

```json
"planning_lyrics": {"source": "pool", "id": "harbor", "requested": true, "resized": false,
                    "lyrics": "[Intro]\n\n[Verse]\nBoats along the shore\n…"}
```

`source` is `pool` or `request` (`id` is null for the client's own); `requested` is true when
the client named the set (`planning_set`), false when the seed picked it, null for the
client's own words. In the `done` event, `resized` is true when the set was grown or cut
toward a length range ("Song length" below) and `lyrics` is then the text the score was
actually planned from, so sent back as `planning_lyrics` it plans the same song. The field is
null for a sung song, a sound without a score, and an instrumental shaped by section tags in
`lyrics`.

`planning_lyrics` rules, each refused by name: only with `instrumental: true`
(`audio_param_conflict`; send the words as `lyrics` to sing them), never beside `lyrics`
(`audio_param_conflict`: plan from one or the other), and refused by a model that makes no
instrumentals (`audio_param_unsupported`). The text itself (`invalid_params`, saying why):
starts with a section tag; tags only from YuE2's own vocabulary (`[Intro] [Verse]
[Pre-Chorus] [Chorus] [Bridge] [Interlude] [Outro] [Instrumental]`, any case), each on a line
of its own and never the same tag twice in a row (the skill refuses a score whose section
labels repeat back to back); at least one line of words (tags alone go in `lyrics`); at most
36 lines of words and 2000 characters. Sung `lyrics` have no limit of Crucible's own (YuE2's
caps bound them); planning lyrics exist to keep the score inside its 4096 tokens, and a
16-line sung song already uses 1800 to 2600 of them, so 36 lines is past what any score holds.

`scripts/check-instrumental-planning.py` renders N instrumentals from the pool on a running
server and reports each one's set, score tokens, how the score and the song ended and its
seconds of audio, passing when every score is in the sung range with no cap hit.

### Song length: `min_duration_s`, `max_duration_s`

YuE2 has no length input: a song lasts what its score says. It writes the score (ABC) first,
then composes and synthesizes the song from it. A client may send a **length to aim at** with
`min_duration_s` and `max_duration_s` (either alone; each 30 to 360 s, the model's longest;
the minimum below the maximum). It is a nudge and a record, never a guarantee:

- **Every song is made in one pass**: one score, one render. A range never re-plans a
  score, refuses a song or fails it (Owen, 2026-10-10: "strictly one pass"; "if its close,
  thats good enough").
- **An instrumental planned from the server's pool** (no `planning_lyrics`, no `lyrics`; the
  seed's set or the one `planning_set` names) has its set **sized toward the range** before
  its one score. Planning lyrics are never sung, so their size is a free lever here. The set
  comes in sizes counted in **body sections** - everything between a leading `[Intro]` and a
  trailing `[Outro]`: a smaller size cuts sections from the body's end, a larger one carries
  the body on from its start (never the same section twice in a row), so a longer song
  repeats the set's own verses and choruses in its own order, and every size passes the
  planning-lyrics rules (at most 36 lines). The size chosen is the one whose estimate - its
  lines at **8.6 s of score a planning line** - lands at least 10% inside each end of the
  range (the middle of a range too narrow for that), the fewest sections from the set as
  written first, so a set that already fits is planned from verbatim; out of reach, the
  nearest size. The same params and seed size the same way, so a kept request reproduces the
  song. A `note` event says the size ("the planning set is sized toward the 120-180 s asked:
  12 lines (verse, chorus, verse)").
- **Every other song** - sung `lyrics`, an instrumental's own `planning_lyrics`, section-tag
  `lyrics` - is made from its words exactly as sent. The range has no effect on making it; it
  is only recorded against what the score measured, and a `note` event says so.

**How the length is read.** After the score, Crucible reads its nominal length - its bars at
its tempo - with `crucible/jobs/audio/scorelength.py`, which reads the score as written: each
bar lasts what its notes and rests add up to at the unit length (`L:`) and tempo (`Q:`) in
force there, a whole-bar rest (`Z`) lasts a bar of the meter (`M:`), and the score lasts as
long as its longest voice. It does not use the yue2-music skill's strict parser, which refuses
scores YuE2 really writes (a 4-quarter bar in 3/4); on a score the strict parser accepts the
two give exactly the same seconds. What it cannot time (repeats, tuplets, no tempo; none is in
YuE2's dialect) leaves `score_seconds` null with the reason in `unread`, and the song is made
all the same.

**How close it is, measured.**

- *Score to audio:* the audio ran **0.943 to 1.069** of the score's nominal length, median
  **0.987**, on the PC's first nine pool instrumentals (2026-10-10, kept as
  `tests/fixtures/yue2-scores/`), and **0.959 to 0.982** on the four songs the 1.0.137 card
  check made. The score is a sound measure of the song.
- *Planning lines to score:* this is the loose part. 21 scores so far ran **4.7 to 14.4 s a
  planning line** (median 8.9; the prior is 8.6), because the tempo the model picks and the
  bars it gives a line move a score far more than the line count does: one set made 158.7,
  158.7 and 165.3 s from 24, 31 and 35 lines. Aimed with the prior alone, 2 of the card
  check's 6 first scores landed in their range (120-180 s and 180-270 s); the others missed
  by 4.6, 7.6, 19.2 and 21.3 s. Expect "close", not "inside".

**The record.** Every YuE2 song's `done` event carries `audio.length`:

```json
"length": {"min_duration_s": 120, "max_duration_s": 180, "sized": true,
           "body_sections": 3, "structure": ["verse", "chorus", "verse"], "lines": 12,
           "score_seconds": 144.0,
           "score": {"seconds": 144.0, "bars": 48, "quarters": 192.0, "bpm": 80.0, "meter": "4/4"},
           "unread": null, "in_range": true}
```

`min_duration_s` and `max_duration_s` are the range asked (null ends when not sent); `sized`
says whether the range shaped the plan (true only for a pool-planned instrumental with a
range); `body_sections`, `structure` and `lines` are what the set was sized to (null when not
sized); `score_seconds` is the score's nominal length (null, with `unread`, only for a score
that could not be timed); `in_range` whether it landed in the range (null when none was
asked). A kept request carries the range as `settled.length_range`. Stable Audio's `length` is
null, and it refuses both params (`audio_param_unsupported`): it makes exactly `duration_s`
seconds, so send that.

`scripts/check-song-length.py` renders instrumentals across ranges on a running server and
reports each song's structure, score seconds, audio seconds and whether it landed, with the
in-range rate and the mean distance from the range of the ones that missed.

| param | who takes it | default | rule |
| --- | --- | --- | --- |
| `prompt` | sfx, music | required | not blank; a song model refuses it by name (send `tags`) |
| `tags` | song | required | the style: comma-separated genre, instruments, voice, language, tempo. The playground shows it as chips (type a phrase and a comma, or click a suggestion from `crucible/audio/tags/song.toml`) |
| `instrumental` | song | false | true renders the planned melody on an instrument instead of a voice (above) |
| `lyrics` | song | required (optional when `instrumental`) | sections tagged `[Intro] [Verse] [Pre-Chorus] [Chorus] [Interlude] [Bridge] [Outro]`, separated by blank lines; English or Chinese. With `instrumental`, section tags only. Refused by name on sfx and music |
| `planning_lyrics` | song, with `instrumental` | a pool set picked by the seed | words the instrumental's score is planned from and never sung ("Planning lyrics" above); at most 36 lines and 2000 characters |
| `planning_set` | song, with `instrumental` | picked by the seed | a pool set's id (`GET /v1/playground` lists them); never beside `planning_lyrics` or `lyrics`; unknown: `planning_set_unknown` |
| `min_duration_s`, `max_duration_s` | song | none | a length to aim at ("Song length" above): sizes a pool-planned instrumental's set, recorded for every song, never refuses one; each 30 to 360, the minimum below the maximum. Stable Audio refuses them: send `duration_s` |
| `score` | song | none | the name of the job's one input that carries a score to cover ("Covers" below); never beside `planning_lyrics`, `planning_set` or a length range |
| `chords` | song, with `score` | `keep` for an instrumental cover, `strip` for a sung one | `keep` or `strip` the score's chord symbols: cot `full` (harmony fixed) or `melody` (accompaniment free) |
| `duration_s` | sfx, music | sfx 10, music 60 | above 0, at most 120 (sfx) or 380 (music): `audio_too_long`. A song refuses it: its length follows its lyrics |
| `steps` | sfx, music | 8 | 1 to 50. Stability: 8 is what the post-trained models were made for, and more does not necessarily sound better |
| `cfg` | song | 1.0 | 0 to 20; above 1 guides harder towards the tags and lyrics and runs the model twice per token (YuE2 suggests trying 1.2). Stable Audio refuses it: its post-trained checkpoints ignore guidance |
| `negative_prompt` | nobody yet | | refused by name: the post-trained Stable Audio checkpoints ignore it (only Stability's `-base` checkpoints read it) and YuE2 has none |
| `seed` | all | chosen and reported | 0 to 4294967295; the same seed and params on the same model and machine give the same sound |
| `format` | all | `flac` | `flac` (24-bit) or `wav` (24-bit PCM) |

Unknown params are refused, never ignored. A param the named model does not take is refused
`audio_param_unsupported` with the reason and the list of what it does take; a missing one is
`audio_param_missing`; a value past the model's ceiling is `audio_param_out_of_range`. An audio
job takes no input files (`invalid_inputs`), except a cover's one score.

### Covers: a song rendered from a given score (`score`)

YuE2 writes a score and then makes the song from it, and it will take the score instead of
writing one: given an ABC, it decodes no score and composes the song from that one. Two
covers come from it (Owen, 2026-10-11), both made in one pass like every song - nothing is
re-planned and nothing is retried:

- **An instrumental cover** (`instrumental: true`): the score's Vocal melody moves note for
  note to the instrument - the same vendored skill the planned instrumental uses
  (`instrumentalize.convert_score`, checked by `instrumental.validate_score`) - and the result
  is rendered with lyrics that are only the score's own section tags (`% verse` becomes
  `[Verse]`), so nothing is sung.
- **A sung cover** (no `instrumental`): the score's Vocal melody is sung with new `lyrics`.
  This is upstream's own recipe (YuE repo at the pinned commit, `skills/yue2-music`: "Cover
  from a score ... use a validated melody-only ABC and the desired lyrics", rendered with
  `cot="melody"`).

The score is YuE2's own two-voice ABC dialect, as YuE2 writes it into `score.abc` and as the
`score` job writes from a recording: `X:1`, a blank `T:`, `M:`, an `L:1/32`-style unit, a
`Q:1/4=` tempo, the `Vocal` and `Ins` voices, `% section` comments. **No `w:` lines**: the
dialect carries no words, and a score with any is refused; a sung cover's words go in
`lyrics`.

It arrives as **the job's one input**, and `score` names it - uploaded:

```json
{"type": "audio",
 "model": "yue2-3b",
 "params": {"tags": "Instrumental, warm piano and strings, no vocals, no singing, 72 BPM",
            "instrumental": true, "score": "score.abc", "seed": 3},
 "inputs": {"score.abc": {"blob_id": "<from POST /v1/uploads>"}}}
```

or **by reference** to another job's artifact, so a score the `score` job made (or a song's
`score.abc`) is covered without the client handling it; the source job is sent with
`"hold": true` (or held with `POST /v1/jobs/{id}/hold`) so its artifact outlives being
fetched:

```json
{"type": "audio",
 "model": "yue2-3b",
 "params": {"tags": "English, intimate acoustic folk, warm male voice, fingerpicked guitar, 72 BPM",
            "lyrics": "[Verse]\nThe lantern by the harbour wall\nStill burns when no one comes\n\n[Chorus]\n…",
            "score": "melody.abc"},
 "inputs": {"melody.abc": {"artifact": {"job_id": "<the score job>", "name": "score.abc"}}}}
```

`inline_base64` works too. The job carries exactly that one input; any other is
`invalid_inputs`.

**`chords`** keeps or strips the score's chord symbols (vendored `abc_tools.strip_chords`).
YuE2 renders with cot `full` when the Vocal voice still carries chords (the harmony is fixed)
and `melody` when it does not (the accompaniment is free). Left out, an instrumental cover
keeps them (the skill's conversion keeps the harmony) and a sung cover strips them (the
skill's melody-only cover).

**What the app must provide.** For a sung cover, `lyrics` written to the score's phrases:
sections tagged as the score's `% section` comments are (`[Verse]`, `[Chorus]`, ...), a line
for each phrase, syllables fitted to its notes. Crucible does not write or fit them - it
sends the words to YuE2 as given - and a sung cover without words to sing (no `lyrics`, or
section tags alone) is refused `audio_param_missing`. An instrumental cover takes no
`lyrics` (`audio_param_conflict`): its section tags come from the score.

**What a score turns off.** The score is the plan, so nothing that plans or sizes one is
taken beside it, each refused `audio_param_conflict` naming the param: `planning_lyrics`,
`planning_set` (no pool set is picked either) and `min_duration_s`/`max_duration_s` (the
score's bars at its tempo are the song's length).

**Refused by name, before anything renders.** Checked on the server before YuE2 is loaded,
failing the job:

- `score_invalid`: not UTF-8, carries `w:` lines, or the skill's parser refuses it - its own
  reason is in the message (e.g. `group 9, Vocal, bar 33: note/rest exceeds meter
  duration`); for an instrumental cover, the transfer or `validate_score` refusing the
  result; for a sung cover, a Vocal voice with no notes to sing.
- `score_too_long`: the score's nominal length (its bars at its tempo, `scorelength.read`)
  is past the 360 s YuE2 composes.

Checked by the worker, which has YuE2's tokenizer, before it plans anything:

- `score_too_many_tokens`: the score is more than 4096 tokens, the budget YuE2 plans a score
  within. yue2-infer does not hold a given score to it; the skill's own `generate()`
  refuses one past it, by the same count (`len(pipe.tokenizer.encode(abc))`). The server
  has no tokenizer (it would need tiktoken and the model's `qwen.tiktoken`), so this one
  needs the worker loaded; the worker answers the refusal and stays loaded.

**The result.** As for any song: `audio.<format>` and `score.abc` - **what was rendered**
(after the transfer or the chord removal) - plus `given-score.abc`, the score as it was
sent. The `done` event's `audio.cover` says how:

```json
"cover": {"kind": "instrumental", "source": "reference", "input": "melody.abc",
          "reference": {"job_id": "4b60493b…", "name": "score.abc"},
          "given": "given-score.abc", "cot": "full", "chords": "kept",
          "chords_chosen_by": "default", "score_tokens": 1180, "score_seconds": 95.0,
          "score": {"seconds": 95.0, "bars": 26, "quarters": 104.0, "bpm": 66, "meter": "4/4"}}
```

`source` is `upload` (a blob or inline bytes; `reference` null) or `reference`.
`score_seconds` is the nominal length the server read; `audio.length.score_seconds` is the
same reading of the rendered score. An instrumental cover's `audio.notes` carries the
transfer report (`instrumental_transfer`) and `score_before_transfer` (the given score), as a
planned instrumental's does. `audio.planning_lyrics` is null, and `audio.decode_stages` has
no `scoring` stage: nothing decoded a score. A kept request (`request`, "A sound that did not
finish") carries the same `cover` under `settled`, with `score_tokens` null.

### `yue2-3b-instrumental`: instrumentals that do not invent a singer (2026-10-12)

**Why.** A `yue2-3b` instrumental can still sing. Its score has no vocal note left (the
melody moved to the instrument) and its lyrics are only section tags, yet the composing pass
invents a singer anyway: the AR model that writes the song's tokens learned that songs have
voices. B-Sides measured an adapter that fixes it, folded into the composing pass only, with
the stock synthesis and VAE, the same score and the same seed: *Iron Lanterns* (seed
1297391292) sang for 80 s on the base and 7.5 s with the adapter; a soul/Hammond piece for
13.5 s and then 0 s, three runs of three. Owen listened and approved it (2026-10-12).

**What it is.** `yue2-3b-instrumental` is yue2-3b's weights plus that adapter:
[Mothersuperior/YuE2-instrumental-cot-full-loras](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras)
at commit `947f2f4b`, the bfloat16 file `ar_lora_inst_v3abc.bf16.safetensors` (139,502,088
bytes, sha256 `e408fd31…5414`, pinned from the Hugging Face API). It is rank 64 on
`self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` of all 28 layers: 392 tensors, 196
linears, each changed by `W += B @ A` at scale 1.0. Nothing else changes - not the
embeddings, `lm_head`, `llm2vae`/`vae2llm`, the `nar_*` half or the VAE. The adapter's
README also offers a NAR adapter and a "realaudio" tokenizer; this model uses neither.

**The rule is the model's, in its manifest** (`crucible/audio/yue2-3b-instrumental.toml`,
`[backends.cuda-linux.adapter]`, `applies_to = ["compose"]`), not the request's:

1. **Plan on the base.** The score is planned with yue2-3b's own weights (left to write its
   own score, the adapter loops one riff), then the instrumental melody move
   (`yue2_worker._instrumental_plan`) runs as it always has. A cover's given score is
   likewise planned on the base.
2. **Compose with the delta.** The adapter is folded in for the AR pass that writes the
   song's tokens *and* for synthesis, whose prefill runs the same backbone; both ran with the
   delta in the approved renders.
3. **Then the base again, exactly.** When synthesis ends - or anything in it raises - every
   folded weight is the base again.

**How it is folded** (`crucible/jobs/audio/lorafold.py`, `yue2_worker.HostHomes.folded`).
Never on disk: `weights_manifest.json`'s hashes stay yue2-3b's. The CUDA worker already keeps
one host copy of every weight for the life of the model (`HostHomes`), and every copy on the
card is made from it. While the adapter is folded, each target linear's copy on the card gets
`addmm_(B, A)` in place - one bfloat16 GEMM that accumulates in float32 and rounds once -
and every copy that *arrives* on the card during composing gets it as it arrives. That second
part is what keeps `[audio] low_vram` right: there the AR half comes and goes between stages,
and synthesis streams it one layer at a time, so a delta applied once to "the weights on the
card" would be lost on the next arrival. Taking it out is pointing each weight back at its
host copy, which nothing ever folded into: exact (no subtraction, so no bfloat16 round trip),
no allocation, and nothing that can fail halfway; the next stage copies the base to the card
as it already does every stage.

**What it costs.** No second copy of any weight anywhere. On the card: one linear's A and B
(at most 0.8 MB each) for the length of its fold, and no float32 temporary. In host memory:
the adapter itself, 0.14 GB, for the life of the worker (the manifest's host figure is
yue2-3b's 10.5 GB plus that, declared). The card figures are yue2-3b's, low_vram's 7.3 GB
included: the fold adds nothing that grows. None of this has run on a card yet.

**What it takes.** Exactly what an instrumental on yue2-3b takes - `tags`, planning lyrics, a
`planning_set`, section-tag `lyrics`, a length range, `cfg`, an instrumental cover - and
`instrumental: true` on every request. Anything else is refused before anything loads:

```json
{"error": {"code": "instrumental_only",
  "message": "yue2-3b-instrumental makes instrumentals only: its adapter composes without a singer. Send `instrumental: true`, or send a sung song to yue2-3b",
  "details": {"model": "yue2-3b-instrumental", "kind": "song", "backend": "cuda-linux",
              "param": "instrumental", "sung_model": "yue2-3b"}}}
```

A sung cover is refused the same way: the adapter is trained to compose without a voice, so
it is not quietly made instrumental. It is never the server's pick for the `song` verb
(a `weights_of` model is not a verb candidate); a client names it.

**What it records.** `audio.model` is `yue2-3b-instrumental`, `audio.weights_of` is
`yue2-3b`, `audio.hf_repo`/`revision` are the base's, and `audio.adapter` names the adapter
and where it ran:

```json
"adapter": {"name": "instrumental-lora",
            "hf_repo": "Mothersuperior/YuE2-instrumental-cot-full-loras",
            "revision": "947f2f4b28978b2b6c3e316e6a87925c76bf3c4b",
            "file": "ar_lora_inst_v3abc.bf16.safetensors",
            "sha256": "e408fd3148b75b1165f7ddbf63db575d83bb6402a0b5f876fcb767dbcb2c5414",
            "bytes": 139502088, "rank": 64, "scale": 1.0,
            "applied_to": ["compose"], "linears": 196}
```

The worker re-hashes the file at load and refuses it if it is not those bytes; `applied_to`
is the worker's account of what it did, and the server fails the song `worker_failed` if it
differs from the manifest in any field. On `yue2-3b`, `adapter` and `weights_of` are null and
nothing about it changed: a yue2-3b record still reproduces, and B-Sides can still A/B.

**Pulling it.** The manifest says `weights_of = "yue2-3b"` (Ollama's FROM) and carries the
adapter as a companion (its ADAPTER): the same two mechanisms Crucible already uses for an
LLM or ASR model that shares another's download and for YuE2's VAE. So
`crucible models pull yue2-3b-instrumental` with yue2-3b present downloads only the adapter,
into yue2-3b's folder (`models/yue2-3b/cuda-linux/instrumental-lora/`); without it, yue2-3b
is pulled first, once, into its own folder. Its catalogue row has `shares_weights_of:
"yue2-3b"`, `installed` only with the base, its VAE and the adapter all present,
`installed_bytes` the adapter's alone, and `expected_bytes` 139,502,088 when the base is
present (unknown otherwise: a main repo's size is not pinned). Removing it takes only the
adapter; removing yue2-3b is refused `weights_shared` while it is pulled.

**Switching between the two.** The card holds one audio model id at a time, so going from
yue2-3b to yue2-3b-instrumental (or back) unloads one worker and loads the other: the same
base weights are read again, a full YuE2 load. A worker that held the base once and folded
the adapter in only for the songs that ask would avoid it, but it would make one resident
answer to two ids, which residency, keep-calls-together and the card events all key on; not
built. Within one model, its calls are kept together as yue2-3b's are.

**Only on the PC.** The fold's exact way back is the CUDA worker's host homes. The Mac's
worker keeps none (the card and host memory are one pool there, so a home would hold the
model twice), so it has no exact way back to the base yet, and the model declares no
`mlx-darwin` arm; the manifest parser refuses an adapter on one.

## The result

One artifact, `audio.flac` (or `audio.wav`); a song adds `score.abc`, the chord-annotated ABC
score YuE2 writes before it composes the audio. The `done` event carries `audio`, every
effective parameter, so a sound can be made again:

```json
{"artifacts": ["audio.flac", "score.abc"],
 "audio": {"model": "yue2-3b", "kind": "song", "hf_repo": "m-a-p/YuE2-3B",
           "revision": "c044757a011169583f363168348ae380946efff8",
           "backend": "cuda-linux", "engine": "yue2", "dtype": "bfloat16",
           "prompt": null, "tags": "English, warm piano pop, …", "lyrics": "[Verse]\n…",
           "duration_s": null, "seed": 3, "steps": null, "cfg": 1.0, "format": "flac",
           "instrumental": false, "planning_lyrics": null,
           "length": {"min_duration_s": null, "max_duration_s": null, "sized": false,
                      "body_sections": null, "structure": null, "lines": null,
                      "score_seconds": 183.0, "score": {"…": "…"}, "unread": null,
                      "in_range": null},
           "artifact": "audio.flac", "score": "score.abc",
           "audio_seconds": 182.4, "sample_rate": 48000, "channels": 2,
           "seconds": 71.2,
           "stage_seconds": {"scoring": 9.1, "composing": 50.2, "synthesizing": 8.4, "decoding": 2.1, "saving": 0.4},
           "peak_bytes": 12000000000,
           "stage_peak_bytes": {"scoring": 9000000000, "composing": 12000000000, "synthesizing": 11000000000, "decoding": 3000000000},
           "memory_bytes_estimate": 16000000000, "memory_basis": "declared",
           "low_vram": false,
           "versions": {"yue2-infer": "0.1.6", "torch": "2.10.0", "transformers": "4.57.6"},
           "notes": null,
           "decode_stages": {
             "scoring": {"tokens": 1180, "cap": 4096, "ended": "eos", "execution": "cuda_graph",
                         "attention": "sdpa", "low_vram": false, "prefix_tokens": 212,
                         "cfg_branches": 1, "seconds": 9.0, "prefill_seconds": 0.1,
                         "tokens_per_second": 131.1},
             "composing": {"tokens": 9000, "cap": 9000, "ended": "cap", "execution": "cuda_graph",
                           "attention": "sdpa", "low_vram": false, "prefix_tokens": 1395,
                           "cfg_branches": 1, "seconds": 50.0, "prefill_seconds": 0.2,
                           "tokens_per_second": 180.0}},
           "stages_at_cap": ["composing"],
           "host_memory": {
             "before": {"rss_bytes": 9760000000, "anon_bytes": 1950000000, "file_bytes": 7720000000},
             "after": {"rss_bytes": 9780000000, "anon_bytes": 1980000000, "file_bytes": 7720000000},
             "peak_rss_bytes": 9780000000,
             "host_homes_bytes": {"model": 7261000000, "vae": 270000000}}},
 "resident": "yue2-3b"}
```

### The worker's host memory: `host_memory`

The worker's own memory in the machine (on a PC, the WSL guest), from `/proc/self/status`:
`before` as the song began, `after` once it was saved and its audio released, and
`peak_rss_bytes` between them. `anon_bytes` is what the kernel's OOM killer weighs;
`file_bytes` is mapped weights the kernel can drop and read again. `host_homes_bytes` is what
the engine keeps in host memory on purpose, by part (YuE2 keeps the host copy each part of the
model was loaded into; null for Stable Audio). A worker that makes song after song should
show the same `after` each time: YuE2 once grew here until the OOM killer took it on track 12
of an album (docs/internals/audio.md "Host memory"). `host_memory` is null on a Mac.

### How each token stage ended: `decode_stages`, `stages_at_cap`

YuE2 writes two token streams, one pass each over one prefix (neither is decoded in
segments): `scoring`, the ABC score, capped at 4096 tokens, and `composing`, the song's
codec tokens, capped at 9000. Normally the model ends each with its end token. A bad seed can
keep it going until the cap - on an RTX 3070 one song came out 6:00 long that way, while the
same seed ended normally on a 3090 Ti (2026-10-09). The finished job says which, from
yue2-infer's own account of each stage, so nobody has to re-run the seed to find out:

| field | meaning |
|---|---|
| `tokens` | every token the model wrote in the stage, its end token included |
| `cap` | the most the stage may write (yue2-infer's own setting, not a Crucible copy of it) |
| `ended` | `eos`: the model ended the stage. `cap`: it ran to `cap` without ending |
| `execution` | `cuda_graph` or `eager` |
| `attention` | the attention kernel the stage ran on |
| `low_vram` | whether `[audio] low_vram` held half of YuE2 on the card for this render |
| `prefix_tokens`, `cfg_branches` | the prompt it decoded after, and 2 when `cfg` is above 1 |
| `seconds`, `prefill_seconds`, `tokens_per_second` | wall time of the stage and its speed |

`stages_at_cap` lists the stages whose `ended` is `cap`, in order: `[]` when the model ended
every stage, null for Stable Audio, which decodes no tokens (`decode_stages` is null too). A
stage at its cap does not fail the job and nothing re-runs it: the audio is real, only longer
than the model meant. The last progress event says it in words too ("…; composing at its
9000-token cap without ending"). An instrumental reports the score YuE2 decoded, not the fixed
score it re-plans from (which decodes nothing).

(The numbers above show the shape; no audio model has been measured through Crucible yet.)
`steps`, `duration_s` and `cfg` are `null` where the model does not take them. Stable Audio
reports stages `encoding`, `denoising` (one progress event per step), `decoding`, `saving`;
YuE2 reports `scoring` and `composing` (every 64 tokens), `synthesizing`, `decoding`, `saving`.
Every progress event carries `fraction`. `DELETE /v1/jobs/{id}` stops the job between two
steps or tokens; the model stays loaded if a queue session holds it.

### A sound that did not finish: `request`

The `done` event's `audio` is the record of a finished sound. A job that fails, is cancelled
or is interrupted by a restart never reaches it, so an audio job keeps its request on disk
from the moment it starts: `request.json` in the job's directory, and `request` on
`GET /v1/jobs/{id}`. It is the params as sent with the seed this run uses written in (the
one the server chose when the client sent none), what the server settled around them, and a
`reproduce` sentence:

```json
{"job_id": "1f3da14c…", "type": "audio", "model": "yue2-3b",
 "params": {"tags": "English, warm piano pop, …", "lyrics": "[Verse]\n…", "seed": 2771032915},
 "seed": 2771032915, "seed_chosen_by": "server",
 "settled": {"duration_s": null, "steps": null, "cfg": 1.0, "instrumental": false, "seed": 2771032915,
             "planning_lyrics": null, "length_range": null},
 "low_vram": true, "revision": "c044757a…", "backend": "cuda-linux",
 "recorded": "2026-10-10T07:12:03+00:00",
 "reproduce": "POST /v1/jobs with this record's `type`, `model` and `params` (the seed is in them) runs this again with seed 2771032915; it ran on revision c044757a… (cuda-linux, [audio] low_vram on)"}
```

Submitting its `type`, `model` and `params` again makes the same sound on the same model and
machine. The request clears once the job finishes successfully: a job that ends `done` drops
it (`request` is null, `audio` says the same and more). One that ends `failed`, `cancelled`
or `interrupted` keeps it until the job itself is reaped ([jobs] retention_days). It is the
audio job's alone: no other job type keeps anything of its request on disk, since a
narration's params are a chapter of somebody's book (docs/internals/jobs-runtime.md
"Durability and restart").

## Many sounds in a row

The model comes off the card when the job that loaded it ends, unless something holds it. For a
batch, open a queue session first (`POST /v1/queue/sessions` with `{"act": "sfx"}`; see
[QUEUE.md](QUEUE.md)), optionally warm up with `load-audio`:

```json
{"type": "load-audio", "model": "stable-audio-3-small-sfx"}
```

then send the batch as ordinary `audio` jobs and `DELETE /v1/queue/sessions/{id}` at the end;
the session keeps the model loaded between them and nothing from another app comes in between.
`unload-audio` takes the model off at once when nothing holds it. The `image` job's page,
[IMAGE.md](IMAGE.md), walks through the same flow in more detail.

## Writing a prompt

### Stable Audio 3 (Stability's own prompting guide, `docs/guides/prompting.md` in their repo)

- **Say what makes the sound, how it is triggered and how long it lasts, and how it was
  recorded.** Source, action, production: "a heavy oak door, pushed open slowly, close mic in a
  stone hallway".
- **Sound effects: start with `TrackType: SFX`** for more semantically sensible effects, and
  ask for a short duration.
- **Music: name the genre, the instruments, the mood and energy, and the tempo in BPM**
  ("124 BPM"). `TrackType: Music, VocalType: Instrumental` gives higher quality, more coherent
  music; tags like `Genre: Funk, Genre: Jazz` and `Instruments: Guitar, Saxophone` help.
- **Stems:** start with `TrackType: Instrument` (add `Format: Duo` for two).
- **Write like the training metadata:** the models learned from Freesound and AudioSparx
  descriptions, so plain descriptive phrases work better than instructions.
- **Set a realistic duration** for what you describe: a door slam is 2 s, not 60.
- **No intelligible vocals.** Stable Audio does not sing words; use `yue2-3b` for songs.

### YuE2 (the YuE2 model card and `protocol.py`)

- **Tags are comma-separated** and cover genre, instruments, vocal character, language and
  tempo, e.g. "English, warm piano pop, expressive female voice, acoustic piano, rounded bass
  and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM".
- **Lyrics are sections**: `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` and so on, each block
  separated by a blank line. The number and length of the sections set the song's length.
- **English and Chinese** are the languages it was trained for.
- **The score comes first.** YuE2 writes `score.abc` (melody and chords) and then composes the
  audio from it; read the score to see what it planned.

## Licences

- **Stable Audio 3 Small SFX and Medium:** Stability AI Community License
  (https://stability.ai/community-license-agreement). Free, commercial use included, for
  individuals and organisations under US$1,000,000 annual revenue, after registering with
  Stability AI; above that an Enterprise licence. The outputs are yours. The bundled T5Gemma
  text encoder is also under the Gemma Terms of Use.
- **YuE2 3B:** weights CC BY-NC 4.0 (non-commercial), with the authors' addendum of
  2026-09-16 letting individual creators and musicians, acting for themselves, publish and
  monetise the songs they make. A company needs a commercial licence from the authors. The
  `yue2-infer` code is Apache-2.0.
- **YuE2 3B Instrumental's adapter** (Mothersuperior/YuE2-instrumental-cot-full-loras): CC
  BY-NC 4.0, derived from YuE2-3B; its README says "non-commercial use only". It carries no
  creator addendum of its own, and its author does not say whether YuE2's addendum reaches
  songs made with it; read it as non-commercial.
