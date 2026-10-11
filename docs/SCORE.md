# Scores from songs: the `score` job

A recorded song in, a score YuE2 can play out: the melodies (and, if asked, the chords) as
YuE2's own two-voice ABC, with YuE2's section labels. It is what a cover needs before YuE2 makes
one. Crucible transcribes the song and makes SheetSage2's score fit YuE2; the cover itself - its
lyrics, its style, the song YuE2 renders from the score - is the app's.

| model | transcribes with | class | PC (cuda-linux) | Mac (mlx-darwin) | licence |
| --- | --- | --- | --- | --- | --- |
| `sheetsage2` | SheetSage2 (m-a-p/SheetSage2, a 229 MB LoRA and BART decoder) on the MERT-v2-FullSong encoder (632 M parameters, 2.53 GB float32) | `score` | yes: float32 weights, bfloat16 autocast | declared, **unverified**: float32 on Metal (MPS) | CC BY-NC 4.0 |

`GET /v1/capability` has the row "can transcribe a song into a score, using sheetsage2", with
the fit reason, or why not.

## Turning it on

```bash
crucible init --enable-score        # or [jobs] enable_score = true
crucible install score              # SheetSage2's own env, and Crucible's ffmpeg
crucible models pull sheetsage2     # SheetSage2 and its encoder
```

None of these is needed by hand: a job for a missing env or model starts the install and
answers `409 installing`, like every other type. Neither repo is gated.

## The request

The song is the job's one input, in any format ffmpeg decodes (WAV, FLAC, MP3, M4A, OGG, a
video's audio track), sent the way every job sends a file: inline, as an uploaded blob, or as
another job's artifact.

```json
{"type": "score",
 "model": "sheetsage2",
 "params": {"keep_harmony": false},
 "inputs": {"song.mp3": {"blob_id": "…"}}}
```

| param | default | what it does |
| --- | --- | --- |
| `keep_harmony` | `false` | `true` keeps the chords: SheetSage2's `full` task, chord symbols in the score's Vocal voice. `false` is its `melody-full` task (`melody_only`): both melodies, no chords, so YuE2 harmonises the cover itself - what YuE's cover skill does unless told `--keep-harmony` |
| `start_s` | `0` | seconds into the recording to start at |
| `max_seconds` | to the end | at most this many seconds from `start_s`; at most 360 |

Unknown params are refused, never ignored (`invalid_params`). SheetSage2's other knobs (its
`paper` preset, the window overlap and lookahead, the tensor exports, rendering) are its
defaults and not on the wire.

### Refused by name, before anything loads

The server decodes the input itself, with its own ffmpeg, into exactly what SheetSage2's own
decoder makes (mono, 24 kHz, float32), and hands the model the samples. So a bad input is
refused before the card is touched - before a resident YuE2 is evicted for it:

| code | when |
| --- | --- |
| `invalid_inputs` | not exactly one input |
| `audio_undecodable` | ffmpeg cannot read it (its own words are in the message) |
| `audio_too_short` | fewer than SheetSage2's 1,025 samples after the crop (`start_s` past the end, say) |
| `audio_too_long` | longer than 360 s after `start_s`, or `max_seconds` above 360 |
| `ffmpeg_missing` | the server has no ffmpeg (`crucible install score` places Crucible's) |

**Why 360 s.** It is the longest song YuE2 composes (yue2-3b's `max_duration_s`: about six
minutes, 9,000 codec tokens at 25 a second): the score is for YuE2, and a longer one is a song
it cannot make whole. It also bounds the work: SheetSage2 reads 300 s windows overlapping by
200 s, so 360 s is at most two windows (0-300 s and the last 300 s), each decoded to at most
5,120 tokens. A longer recording is transcribed in parts with `start_s` and `max_seconds`.

## What comes back

| artifact | what it is |
| --- | --- |
| `score.abc` | the YuE2-native score: SheetSage2's, with its section labels mapped onto YuE2's. The Vocal voice still sings the sung melody - the form a sung cover is written against |
| `instrumental.abc` | the same score with the sung melody moved to the instrument voice (`instrumentalize.convert_score`, chords kept or not as `keep_harmony` says): the form YuE2's instrumental workflow plays a cover from, and the one YuE2's validator passed |
| `sheetsage2.abc` | SheetSage2's score exactly as it came out, for evidence |
| `transcription.mid` | SheetSage2's playback MIDI: the melodies, and the chords when kept |
| `score.json` | the report, below |

The `done` event's `score` is the same report:

- what ran: `model`, `revision`, `encoder` (repo, revision), `backend`, `device`, `dtype`
  (`bf16` is SheetSage2's autocast on CUDA; `fp32` elsewhere), `task`, `keep_harmony`,
  `prompts`, and `recording` (input name, crop, seconds decoded);
- what SheetSage2 said: `warnings` (verbatim), and under `sheetsage2` its `diagnostics`,
  `abc_error`, note counts, `elapsed_seconds` and `peak_gpu_mib` (torch's
  `max_memory_allocated`; 0 off CUDA);
- `windows`: one row per window - its span, what of it was kept, `tokens`, `token_cap`
  (5,120) and `reached_token_limit` (SheetSage2's own test, more tokens than the cap, which is
  also when it warns "Window N reached the token limit"); `windows_at_token_limit` lists them;
- `yue2_native`: `labels` (every label line read: SheetSage2's label, the YuE2 section, why,
  and whether it was kept or dropped and why), `checks` (each of YuE2's checks and whether it
  passed), `instrumental_transfer` (YuE2's own record of the voice transfer), and `error`;
- `lengths` for `sheetsage2`, `score` and `instrumental`: `seconds`, `bars`, `bpm`, `meter`
  (the score's nominal length, read with the same reader the audio job checks a YuE2 song's
  length with, crucible/jobs/audio/scorelength.py), and `chars`. `yue2_tokens` is null: the
  YuE2 tokenizer is not in the score env, and YuE2 tokenizes a score inside its own pipeline,
  so the score's length is given in characters instead. YuE2 writes scores of up to 4,096 of
  its own tokens; whether a transcribed score longer than that plays whole is unmeasured;
- `peak_bytes` (the worker's own reading), `seconds`, `stage_seconds`, the declared
  `memory_bytes_estimate` and `memory_basis`, `verified`, `versions` and `licence`.

What to do with the score - which phrases to write words for, the style, the song - is the
app's. Parsing it is too: `score.abc` is plain ABC in YuE2's dialect.

## Making it YuE2-native

SheetSage2 and YuE2 write the same two-voice ABC dialect, but not the same section labels.
SheetSage2 labels sections from its own 23-label vocabulary, and can open one group of bars with
several labels; YuE2 knows eight (`intro`, `verse`, `pre-chorus`, `chorus`, `bridge`,
`interlude`, `outro`, `instrumental`) and refuses an unknown label and the same label twice in
a row. The job maps them deterministically (crucible/jobs/score/yue2native.py):

| SheetSage2 | YuE2 | why |
| --- | --- | --- |
| the eight YuE2 has | the same | |
| `silence` | `intro` first, `outro` last, `interlude` between | where it stands is what it is |
| `post-chorus` | `chorus` | the hook's tail joins the chorus it follows |
| `fade-out`, `pre-outro` | `outro` | the ending (MIREX reduction for `fade-out`) |
| `loop` | `chorus` | a repeated hook (MIREX) |
| `rap`, `irregular` | `verse` | MIREX |
| `preshot` | `interlude` | a short instrumental pickup (MIREX: inst) |
| `solo` | `instrumental` | |
| `intro and verse`, `pre-chorus and chorus`, `verse and pre-chorus` | the first part | a compound takes the section it begins with |
| `theme`, `variation` | `verse` | the main material |
| `development` | `bridge` | the contrasting middle |

Then: where one group opens with several labels, the last is kept (the section in force through
the group); a label equal to the section already in force is dropped (the stretches are one
section to YuE2). Only label lines change; every bar is SheetSage2's.

The result must then pass YuE2's own tools, vendored unchanged in crucible/jobs/audio/yue2music/:
`abc_tools.parse_abc` (the strict two-voice parser) on `score.abc`, and, after
`instrumentalize.convert_score` moves the sung melody to the instrument voice,
`instrumental.validate_score` on `instrumental.abc`. YuE2's validator is written for the
instrumental score - it refuses a Vocal voice with notes in it - which is why it judges the
instrumental form, and the sung score is held to the parser and the label rule. A score that
fails is not repaired by guessing: the job fails by name, and `sheetsage2.abc`,
`transcription.mid` and `score.json` (with every label decision and the check that refused it)
stay with the failed job.

| code | when |
| --- | --- |
| `score_label_unknown` | SheetSage2 wrote a label outside the table (a vocabulary this build has not seen) |
| `score_not_yue2_native` | the relabelled score fails YuE2's parser, the voice transfer or the validator; `checks` names which and what it said |
| `score_not_transcribed` | SheetSage2 wrote no ABC at all (no decoded key, fewer than two beats, ...); its reason is in the message |

## On the card

`sheetsage2` is a resident like every other model: one resident engine at a time, so loading it
takes whatever is resident off the card first - a YuE2 kept for the cover's song included - and
the job's `warming` events say so. Outside a queue session it comes off when the job ends;
`load-score` in a session keeps it for the next song, and `unload-score` takes it off
(`score_transcriber_not_resident` when it is not the resident).

The worker loads offline from the pulled files (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`):
SheetSage2's own loader checks the encoder's code and weight hashes (the same ones the manifest
pins for the pull), loads both in float32 on the CPU, merges its LoRA adapters there, and only
the merged model goes to the card. On CUDA it computes under a bfloat16 autocast; on the Mac
everything is float32, with `PYTORCH_ENABLE_MPS_FALLBACK=1` so an op Metal lacks runs on the
CPU rather than failing.

Memory is **declared, not measured**: 6 GB on CUDA (also the worker's hard cap on the card) and
8 GB on the Mac; the manifest (crucible/score/sheetsage2.toml) says how each was reached. Every
job reports SheetSage2's own `peak_gpu_mib`; replace the figure with a measured one after the
first runs.

## The env

`crucible install score` builds `envs/score`, a worker env of its own: SheetSage2's
requirements.txt at the pinned revision (torch 2.8.0, torchaudio 2.8.0, transformers 4.45.2,
huggingface-hub 0.36.0, safetensors 0.5.3, numpy 1.24.3, scipy 1.13.1, mir_eval 0.8.2,
pretty_midi 0.2.10, mido 1.3.3, setuptools 78.1.1) - older than every other Crucible env's, and
SheetSage2's remote code is written against them. crucible/envs/score/{cuda-linux,mlx-darwin}.txt
are `uv pip compile` locks for Python 3.11 (`--python-platform x86_64-manylinux_2_28` and
`aarch64-apple-darwin`); torch 2.8.0 from PyPI is the CUDA 12.8 build. Their `archive-bytes`
headers are the wheels of each resolution summed from PyPI on 2026-10-11: 4.00 GB on
cuda-linux, 0.15 GB on mlx-darwin. pretty_midi ships only an sdist (pure Python).

## The weights

| repo | revision | what | how it is verified |
| --- | --- | --- | --- |
| m-a-p/SheetSage2 | 32c7c747 (2026-10-06) | the adapter and decoder (model.safetensors, 228,738,564 bytes, sha256 b235f680…, unchanged since eab522a8, the revision YuE's cover skill pins), its config and its Python | pinned by revision |
| m-a-p/MERT-v2-FullSong | d8ba1c74 (the release SheetSage2's config names) | the encoder (model.safetensors, 2,529,812,848 bytes) and its code | every file by sha256 in the manifest; SheetSage2's loader checks again |

The encoder is a companion: pulled into `<model dir>/mert/` and handed to SheetSage2 as its
`base_model_path`. `crucible models pull sheetsage2` fetches both, only what is missing.

## Licence

Both repos are **CC BY-NC 4.0**: non-commercial use only, of the weights and of what they make.
Neither carries a creator addendum like YuE2's. SheetSage2 loads its own Python
(`trust_remote_code`), and neither repo states a licence for that code apart from the weights'
CC BY-NC 4.0; Crucible pins it by revision and runs it unmodified, and does not vendor or
redistribute it.

## Unverified

Nothing here has run SheetSage2 on a card through Crucible (no card time has been given for
it). Until it has:

- the memory figures (6 GB CUDA, 8 GB Mac), the seconds per window and the load time;
- the offline load from the pulled files (a local `base_model_path` and transformers' remote
  code cache);
- how often a real SheetSage2 score passes YuE2's checks, and whether the label table meets
  labels it does not name;
- whether a transcribed score longer than YuE2's 4,096 score tokens plays whole;
- the Mac: whether every MERT-v2 op has a Metal kernel, and how float32 on MPS compares with
  CUDA's bfloat16 autocast.

`scripts/check-score-live.py` is the first real run: it submits one song to a server and checks
the artifacts and the report.
