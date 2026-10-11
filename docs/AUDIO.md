# Audio generation: the `audio` job

Words in, sound out: sound effects, instrumental music, and full songs with sung vocals. One job
type, `audio`, serves three models; which one you name decides what you get. How it runs and
why is [internals/audio.md](internals/audio.md); this page is what a caller sends, what comes
back, and how to write a prompt each model understands.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | longest | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `stable-audio-3-small-sfx` | sound effects | `sfx` | yes | yes | 120 s | Stability AI Community (gated) |
| `stable-audio-3-medium` | instrumental music, stems, also effects | `music` | yes | yes | 380 s (6 min 20 s) | Stability AI Community (gated) |
| `yue2-3b` | songs with vocals, from lyrics and style tags | `song` | yes | **no** (below) | about 6 min, set by the lyrics | CC BY-NC 4.0 + creator addendum |

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
client's own words. In the `done` event, `resized` is true when the set was grown or cut to
land in a length range ("Song length" below) and `lyrics` is then the text the score was
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
in 10 to 30 s, then composes and synthesizes the song from it, which is the expensive part.
A client asks for a **range to shoot for** with `min_duration_s` and `max_duration_s` (either
alone; each 30 to 360 s, the model's longest; the minimum below the maximum). With neither,
nothing below happens and the song is made exactly as before.

**The check.** After the score and before anything is composed, Crucible reads the score's
nominal length - its bars at its tempo - and holds it against the range. The reader
(`crucible/jobs/audio/scorelength.py`) reads the score as written: each bar lasts what its
notes and rests add up to at the unit length (`L:`) and tempo (`Q:`) in force there, a
whole-bar rest (`Z`) lasts a bar of the meter (`M:`), and the score lasts as long as its
longest voice. It does not use the yue2-music skill's strict parser, which refuses scores
YuE2 really writes (a 4-quarter bar in 3/4, the first planning-lyrics song on the PC); on a
score the strict parser accepts, the two give exactly the same seconds. What it cannot time it
refuses by name (repeats, tuplets, a score with no tempo); none is in YuE2's dialect.

**How close the nominal length is** (the PC's first nine planning-lyrics instrumentals,
2026-10-10, one per pool set, kept as `tests/fixtures/yue2-scores/`): the audio ran **0.943
to 1.069** of the score's nominal length, **median 0.987** (113.7 s of score made 113.3 s,
213.6 made 212.2, 154.3 made 165.0). So a score inside the range makes a song within about
7% of it; aim the range that much wide of a hard limit.

**An instrumental planned from the server's pool** (no `planning_lyrics`, no `lyrics`) is
sized to land in the range, deterministically from the seed:

1. Its set (the seed's, or `planning_set`'s) comes in sizes counted in **body sections** -
   everything between a leading `[Intro]` and a trailing `[Outro]`. A smaller size cuts
   sections from the body's end; a larger one carries the body on from its start (never the
   same section twice in a row), so a longer song repeats the set's own verses and choruses
   in its own order. Every size passes the planning-lyrics rules (at most 36 lines).
2. The first score is aimed with the pool's measured rate, **8.6 s of score per planning
   line** (the median of the nine above; they ran 5.7 to 12.0, since the tempo the model
   picks and the bars it gives a line both move it). The size chosen is the one whose
   estimate lands at least 10% inside each end of the range (the middle of a range too
   narrow for that), the fewest sections from the set as written first, so a set that
   already fits is planned from verbatim.
3. If the score lands outside, the **score only** is planned again, at a size not yet tried,
   aimed with this request's own measured seconds a line. At most **3 scores** in all (a
   score is 10 to 30 s of the card; composing never repeats).
4. If none lands, the job fails **`instrumental_length_not_reached`**: the message and
   `error.details.attempts` give every score's lines and seconds, and every plan is kept in
   the job's `failed-plan/attempt-1/`, `attempt-2/`... Send it again with another seed or
   `planning_set`, or a wider range.

The same params and seed plan the same attempts, so a kept request reproduces the song. Each
re-plan is a `note` event ("score 1 is 192.0 s, outside the 120-180 s asked; planning it
again with 12 lines (verse, chorus, verse)").

**A song planned from the client's words** - sung `lyrics`, an instrumental's own
`planning_lyrics`, or section-tag `lyrics` - is never altered. If its score lands outside the
range, nothing is composed and the job fails **`song_length_out_of_range`**, with the plan
kept in `failed-plan/`; the worker answers the refusal and lives on (held by a queue session, YuE2 stays on the card), so the client can resize the words and
send it again:

```json
{"code": "song_length_out_of_range",
 "message": "the score YuE2 wrote for these words lasts 212.4 s (its bars at its tempo), outside the 120-180 s asked, so nothing was composed. …",
 "details": {"min_duration_s": 120, "max_duration_s": 180, "score_seconds": 212.4,
             "direction": "shorter", "ratio_needed": 0.847, "ratio_to_middle": 0.706,
             "score": {"seconds": 212.4, "bars": 104, "quarters": 417.7, "bpm": 118.0, "meter": "4/4"},
             "attempts": [{"attempt": 1, "score_seconds": 212.4, "in_range": false, …}]}}
```

`ratio_needed` is what the length must be multiplied by to reach the nearer end of the range;
`ratio_to_middle` (both ends sent) to reach its middle, the safer target. A job refused for
its length also keeps the refusal in its kept request (`request.refused` on
`GET /v1/jobs/{id}`, beside the params and seed). A score whose length
cannot be read at all, with a range asked, is `score_length_unreadable` (never seen yet).

**Every finished song** carries `audio.length` in its `done` event, whether a range was asked
or not, so a client can calibrate how it sizes lyrics:

```json
"length": {"min_duration_s": 120, "max_duration_s": 180, "score_seconds": 144.0, "in_range": true,
           "attempts": [
             {"attempt": 1, "body_sections": 4, "structure": ["verse", "chorus", "verse", "chorus"],
              "lines": 16, "score_seconds": 192.0, "score": {"seconds": 192.0, "bars": 64, "quarters": 256.0, "bpm": 80.0, "meter": "4/4"},
              "unread": null, "score_tokens": 1326, "score_ended": "eos", "in_range": false},
             {"attempt": 2, "body_sections": 3, "structure": ["verse", "chorus", "verse"],
              "lines": 12, "score_seconds": 144.0, "…": "…", "in_range": true}]}
```

`body_sections`, `structure` and `lines` are null for words that were not resized; `in_range`
is null when no range was asked; `score_seconds` is null (with `unread` saying why) only for
a score the reader could not time. Stable Audio's `length` is null, and it refuses both
params (`audio_param_unsupported`): it makes exactly `duration_s` seconds, so send that.

| param | who takes it | default | rule |
| --- | --- | --- | --- |
| `prompt` | sfx, music | required | not blank; a song model refuses it by name (send `tags`) |
| `tags` | song | required | the style: comma-separated genre, instruments, voice, language, tempo. The playground shows it as chips (type a phrase and a comma, or click a suggestion from `crucible/audio/tags/song.toml`) |
| `instrumental` | song | false | true renders the planned melody on an instrument instead of a voice (above) |
| `lyrics` | song | required (optional when `instrumental`) | sections tagged `[Intro] [Verse] [Pre-Chorus] [Chorus] [Interlude] [Bridge] [Outro]`, separated by blank lines; English or Chinese. With `instrumental`, section tags only. Refused by name on sfx and music |
| `planning_lyrics` | song, with `instrumental` | a pool set picked by the seed | words the instrumental's score is planned from and never sung ("Planning lyrics" above); at most 36 lines and 2000 characters |
| `planning_set` | song, with `instrumental` | picked by the seed | a pool set's id (`GET /v1/playground` lists them); never beside `planning_lyrics` or `lyrics`; unknown: `planning_set_unknown` |
| `min_duration_s`, `max_duration_s` | song | none | the range the song's score must land in before it is composed ("Song length" above); each 30 to 360, the minimum below the maximum. Stable Audio refuses them: send `duration_s` |
| `duration_s` | sfx, music | sfx 10, music 60 | above 0, at most 120 (sfx) or 380 (music): `audio_too_long`. A song refuses it: its length follows its lyrics (ask a range instead) |
| `steps` | sfx, music | 8 | 1 to 50. Stability: 8 is what the post-trained models were made for, and more does not necessarily sound better |
| `cfg` | song | 1.0 | 0 to 20; above 1 guides harder towards the tags and lyrics and runs the model twice per token (YuE2 suggests trying 1.2). Stable Audio refuses it: its post-trained checkpoints ignore guidance |
| `negative_prompt` | nobody yet | | refused by name: the post-trained Stable Audio checkpoints ignore it (only Stability's `-base` checkpoints read it) and YuE2 has none |
| `seed` | all | chosen and reported | 0 to 4294967295; the same seed and params on the same model and machine give the same sound |
| `format` | all | `flac` | `flac` (24-bit) or `wav` (24-bit PCM) |

Unknown params are refused, never ignored. A param the named model does not take is refused
`audio_param_unsupported` with the reason and the list of what it does take; a missing one is
`audio_param_missing`; a value past the model's ceiling is `audio_param_out_of_range`. An audio
job takes no input files (`invalid_inputs`).

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
           "length": {"min_duration_s": null, "max_duration_s": null, "score_seconds": 183.0,
                      "in_range": null, "attempts": [{"attempt": 1, "score_seconds": 183.0, "…": "…"}]},
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
