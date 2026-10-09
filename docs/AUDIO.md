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
crucible init --enable-audio          # or [jobs] enable_audio = true
crucible install audio                # builds one env per audio engine this machine runs
crucible models pull stable-audio-3-small-sfx
crucible models pull stable-audio-3-medium
crucible models pull yue2-3b
```

None of these is needed by hand: a job for a missing env or model starts the install and
answers `409 installing`, like every other type. The one step Crucible cannot take for you is
accepting a licence (next section).

### A card too small for YuE2 whole: `[audio] low_vram`

YuE2's 7.26 GB backbone is two halves that never run together: the AR half (2.83 GB, plus
the 1.51 GB of embeddings and output layer) writes the score and the song, and the NAR half
(the `nar_*` modules, 2.82 GB) solves the synthesis. A host whose card cannot hold the model
whole (an 8 GiB laptop) sets

```toml
[audio]
low_vram = true
```

and only the half a stage uses is on the card; the other waits in host memory. Measured on
the 3090 Ti on 2026-10-08 through the worker, capped as an 8 GiB card: 6.37 to 6.62 GiB of
card over the desktop for songs of 204 to 312 s, against 8.73 GiB holding YuE2 whole; about
the same render time; and the audio within 5.4e-6 (-105 dB) of the whole model's. The load is admitted against the manifest's
`low_vram_memory_bytes_estimate` instead of `memory_bytes_estimate`, and the `done` event's
`audio.low_vram` says which ran.

The capability verdict weighs the same figure: `audiomodels.held_need` is the one rule for
which need a host uses, and the audio job, `crucible capability`, `crucible install audio`,
`crucible doctor`, the Settings model choices and `/v1/info`'s `vram_bytes` all read it. On
an 8 GiB card with a 1 GiB desktop allowance, `song` is granted "with [audio] low_vram"
(6.8 GiB of 14.9 GiB whole); with the setting off it is refused, and the refusal names the
setting as the fix rather than a 7.9 GiB shortfall alone. A capability record decided
before the setting changed is reported stale by `crucible doctor`.

It is off unless a host's config says so (Owen, 2026-10-08: *"this would be a configuration
for systems with low ram, not for high ram systems like this pc"*), and only a model whose
manifest declares a low-VRAM figure honours it: today `yue2-3b` on cuda-linux. `[audio]` is
not a table `crucible install` writes, so a reinstall keeps it. After changing it, run
`crucible capability --write` and restart the server.

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

`lyrics` is optional for an instrumental: send section tags (`[Intro]

[Verse]

[Chorus]
...`)
to shape its form; without them YuE2 plans `[Intro] [Verse] [Chorus] [Outro]`. The render's
`effective_params.notes` carries YuE2's transfer report (how many notes moved) and the score it
first planned.

| param | who takes it | default | rule |
| --- | --- | --- | --- |
| `prompt` | sfx, music | required | not blank; a song model refuses it by name (send `tags`) |
| `tags` | song | required | the style: comma-separated genre, instruments, voice, language, tempo. The playground shows it as chips (type a phrase and a comma, or click a suggestion from `crucible/audio/tags/song.toml`) |
| `instrumental` | song | false | true renders the planned melody on an instrument instead of a voice (above) |
| `lyrics` | song | required (optional when `instrumental`) | sections tagged `[Intro] [Verse] [Pre-Chorus] [Chorus] [Interlude] [Bridge] [Outro]`, separated by blank lines; English or Chinese. Refused by name on sfx and music |
| `duration_s` | sfx, music | sfx 10, music 60 | above 0, at most 120 (sfx) or 380 (music): `audio_too_long`. A song refuses it: its length follows its lyrics |
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
           "artifact": "audio.flac", "score": "score.abc",
           "audio_seconds": 182.4, "sample_rate": 48000, "channels": 2,
           "seconds": 71.2,
           "stage_seconds": {"scoring": 9.1, "composing": 50.2, "synthesizing": 8.4, "decoding": 2.1, "saving": 0.4},
           "peak_bytes": 12000000000,
           "stage_peak_bytes": {"scoring": 9000000000, "composing": 12000000000, "synthesizing": 11000000000, "decoding": 3000000000},
           "memory_bytes_estimate": 16000000000, "memory_basis": "declared",
           "versions": {"yue2-infer": "0.1.6", "torch": "2.10.0", "transformers": "4.57.6"}},
 "resident": "yue2-3b"}
```

(The numbers above show the shape; no audio model has been measured through Crucible yet.)
`steps`, `duration_s` and `cfg` are `null` where the model does not take them. Stable Audio
reports stages `encoding`, `denoising` (one progress event per step), `decoding`, `saving`;
YuE2 reports `scoring` and `composing` (every 64 tokens), `synthesizing`, `decoding`, `saving`.
Every progress event carries `fraction`. `DELETE /v1/jobs/{id}` stops the job between two
steps or tokens; the model stays loaded if a queue session holds it.

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
