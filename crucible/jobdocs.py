"""What every job type does, what it takes and what it gives back - served by the server.

Owen, 2026-10-04: "every command should have documentation to go along with it, and
crucible should have a list of every verb/command so nobody has to ask you how to do
something, they can just check the documentation on the running server."

`GET /docs` (a page) and `GET /v1/docs` (JSON) read this table; tests/test_api_docs.py
refuses a job type declared in crucible/jobtypes.py without an entry here, so a new
verb cannot ship undocumented. The params table is not written here: it is read off
the job type's own pydantic model, the one its submit is validated with, so it cannot
drift from what the server accepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .jobs.align import AlignParams
from .jobs.alignlongform import AlignLongformParams
from .jobs.asr import AsrParams
from .jobs.audio import LoadAudioParams
from .jobs.audio.params import AudioParams
from .jobs.denoise import DenoiseParams
from .jobs.echo import EchoParams
from .jobs.image import ImageParams, LoadImageParams
from .jobs.llm import LoadParams
from .jobs.rvc import RvcParams
from .jobs.segment import LoadSegmentParams
from .jobs.segment.params import SegmentParams
from .jobs.tts import LoadVoiceParams
from .jobs.tts.render import TtsParams
from .jobs.unload import UnloadParams
from .jobs.video import LoadVideoParams
from .jobs.video.params import VideoParams

# A job type's done event reports `resident` as the queue writes it, after the
# settlement (crucible/settle.py) has cleared a card nothing holds.
RESIDENT_AFTER_A_JOB = (
    "(what is on the card as the job ends: the model while a session, a claim, a "
    "chat or a waiting call holds the card, otherwise null, because the server "
    "took it off first and said so in a `note`)"
)


@dataclass(frozen=True)
class JobDoc:
    """One job type, as a client author needs it."""

    # One sentence: what the job does.
    summary: str
    # What `model` names, or None when the job takes no model.
    model: str | None
    # The pydantic model the submit's `params` is validated with; None when it takes none.
    params: type[BaseModel] | None
    # What `inputs` must hold: how many files, which formats, how they are named.
    inputs: str
    # What comes back: the artifacts (names, formats) and the fields of the `done` event.
    returns: str
    # A whole `POST /v1/jobs` body that works.
    example: dict[str, Any]
    # Anything else a caller trips on (limits, refusals by name, what is never done).
    notes: tuple[str, ...] = field(default=())


BLOB = {"blob_id": "<from POST /v1/uploads>"}

INSTALLS_ON_SUBMIT = (
    "A model or env not installed yet is installed on submit where the server allows "
    "it: the submit answers 409 `installing`; submit again when it is done."
)

UNHELD_AFTER_THE_JOB = (
    "The model comes off the card when the job ends unless something holds it. To run "
    "several jobs without a reload between them, open a queue session first "
    "(POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end."
)

LOAD_STAYS = (
    "A load that ends `done` leaves the subject resident and held by nothing: the next "
    "job to end, a queue session closing or the last chat returning takes it off unless "
    "something holds it. Load inside a queue session to keep it for the session."
)

ENGINE_IN_USE = (
    "Refused 409 `engine_in_use` while something holds the card (a TTS stream session); "
    "the refusal names the holder."
)


def _load(
    job_type: str, params: type[BaseModel], noun: str, model: str, example_model: str
) -> JobDoc:
    return JobDoc(
        summary=f"Puts {_article(noun)} on the card and leaves it there, so the first "
        "job that uses it starts at once. Nothing is generated.",
        model=model,
        params=params,
        inputs="None.",
        returns=f"No artifacts. The `done` event gives `resident`, the {noun} now on "
        "the card.",
        example={"type": job_type, "model": example_model, "params": {}},
        notes=(LOAD_STAYS, INSTALLS_ON_SUBMIT, ENGINE_IN_USE),
    )


def _unload(job_type: str, noun: str, loaded_by: str, example_model: str) -> JobDoc:
    code = f"{noun.replace(' ', '_')}_not_resident"
    return JobDoc(
        summary=f"Takes the resident {noun} off the card now, whichever job put it "
        f"there ({loaded_by}).",
        model=f"The id of the {noun} that is resident (GET /v1/activity's `resident`, "
        "or the resident rows of GET /v1/catalog).",
        params=UnloadParams,
        inputs="None.",
        returns="No artifacts. The `done` event gives `resident`: what is on the card "
        "afterwards, normally null.",
        example={"type": job_type, "model": example_model, "params": {}},
        notes=(
            f"Refused 409 `{code}` when that id is not the resident {noun}; the "
            "refusal says what is resident instead.",
            ENGINE_IN_USE,
            "An unload of the subject the server is already clearing is admitted and "
            "ends `done`.",
        ),
    )


def _article(noun: str) -> str:
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


JOB_DOCS: dict[str, JobDoc] = {
    "echo": JobDoc(
        summary="Copies every input to an artifact of the same name after a delay. "
        "The smoke test: it proves submit, events and artifact download without a GPU.",
        model=None,
        params=EchoParams,
        inputs="Any number of files, any names and formats.",
        returns="One artifact per input, byte for byte the same, under the input's name.",
        example={
            "type": "echo",
            "params": {"delay_ms": 25},
            "inputs": {"hello.txt": {"inline_base64": "aGVsbG8="}},
        },
    ),
    "load-model": JobDoc(
        summary="Starts an LLM engine for a model and leaves it resident, so the chat "
        "door (POST /v1/openai/chat/completions) and POST /v1/decide can use it. "
        "Loading the resident model again at a new `context` is a reload.",
        model="An LLM id (GET /v1/catalog, or GET /v1/models for the ones this server "
        "holds), e.g. `qwen3.5-9b`.",
        params=LoadParams,
        inputs="None.",
        returns="No artifacts. The `done` event gives `resident`, the model now on the "
        "card.",
        example={
            "type": "load-model",
            "model": "qwen3.5-9b",
            "params": {"context": 65536},
        },
        notes=(
            "There is no `llm` job type: chat and decide never load a model "
            "(`model_not_resident`), so load it here first.",
            "`context` (at least 2048) is refused above this host's ceiling "
            "(`context_over_limit`; GET /v1/capability's `generate` row lists it per "
            "model); without it the model's own default is used. A plan that leaves "
            "too little KV cache is refused 409 `insufficient_kv_cache`.",
            "`form` loads one form of a model that comes in more than one (GET "
            "/v1/models, the row's `forms`); without it, the form this card takes. "
            "An unknown name is 400 `unknown_form`; a form this card does not take and "
            "that is not pulled is 409 `form_not_installed`, with its pull command "
            "(docs/FITS-AND-THE-CARD.md section 8).",
            LOAD_STAYS,
            INSTALLS_ON_SUBMIT,
            ENGINE_IN_USE,
        ),
    ),
    "unload-model": _unload("unload-model", "model", "load-model", "qwen3.5-9b"),
    "load-voice": JobDoc(
        summary="Starts narrator with a TTS voice and leaves it resident, for `tts` "
        "jobs and serialized TTS streams. A `zeroshot` voice can only be loaded "
        "this way, because only this job carries its reference clip.",
        model="A voice id (GET /v1/voices; each row gives its `kind` and `takes`), "
        "e.g. `deathstalker`.",
        params=LoadVoiceParams,
        inputs="None. A zeroshot voice's reference clip travels in `params.reference`, "
        "not as an input.",
        returns="No artifacts. The `done` event gives `resident` (the voice id), "
        "`fingerprint`, and `reference` (the clip's name, sha256 and seconds, or null).",
        example={"type": "load-voice", "model": "deathstalker", "params": {}},
        notes=(
            "A `zeroshot` voice needs `reference`: `{\"data\": \"<base64 WAV, no data: "
            "prefix>\", \"transcript\": \"<the exact words spoken in it>\"}`, at most "
            "30 s and 32 MiB (`reference_required`, `reference_malformed`). Any other "
            "kind of voice refuses one (`reference_not_allowed`).",
            LOAD_STAYS,
            INSTALLS_ON_SUBMIT,
            ENGINE_IN_USE,
        ),
    ),
    "unload-voice": _unload("unload-voice", "voice", "load-voice or tts", "deathstalker"),
    "tts": JobDoc(
        summary="Renders a batch of text chunks with a TTS voice, one FLAC per chunk. "
        "It loads the voice itself when it is not resident.",
        model="A voice id (GET /v1/voices), e.g. `deathstalker`. For Higgs a voice is "
        "the merged checkpoint, so the voice id is the model.",
        params=TtsParams,
        inputs="None; the text is in `params.chunks`.",
        returns="`<index>.flac` per chunk that rendered: mono FLAC at the voice's own "
        "rate. A `chunk` event per row gives `seconds`, `chars`, `chars_per_sec`, "
        "`tokens`, `capped`, `take`, `guard` and `pause_cuts`. The `done` event gives "
        "`rendered` (a count), `failed` (a list of `{index, message}`), `take`, "
        "`sample_rate`, `sampling` (the full triple applied), `voice` (`id`, "
        "`identity`, `identity_basis`) and `width` (the width you sent, or null).",
        example={
            "type": "tts",
            "model": "deathstalker",
            "params": {
                "language": "en",
                "take": 0,
                "chunks": [
                    {"index": 41, "text": "He had been walking for some time."},
                    {"index": 42, "text": "The road did not appear to end."},
                ],
            },
        },
        notes=(
            "`take` has no default: it is a rung of the voice's retake ladder "
            "(`takes` on GET /v1/voices). Chunk indexes must be unique and text not "
            "blank.",
            "A chunk that fails is listed in `failed` and does not fail the job; "
            "re-submit only those indexes, at the next take if you like.",
            "`retake: true` needs `band` (`pace_chars_per_sec`, `min_chars_per_sec`, "
            "`max_chars_per_sec`, min < pace < max): `retake_without_band`, "
            "`band_malformed`.",
            "`width` (chunks in flight) above the voice's serving width is refused "
            "`width_over_serving` on cuda-linux; left out, the engine runs at the width "
            "it was started with.",
            "A `zeroshot` voice must be loaded with `load-voice` first "
            "(`voice_kind_unsupported`).",
            "Needs ffmpeg on the server (`ffmpeg_missing`).",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
            ENGINE_IN_USE,
        ),
    ),
    "asr": JobDoc(
        summary="Transcribes one audio file to timed text with Whisper (faster-whisper "
        "on cuda-linux, mlx-whisper on a Mac) or Qwen3-ASR.",
        model="An ASR id (GET /v1/catalog), e.g. `qwen3-asr-1.7b`, "
        "`whisper-large-v3-turbo`, or `qwen3-asr-1.7b-mlx` on a Mac.",
        params=AsrParams,
        inputs="Exactly one audio file in any format ffmpeg decodes (m4b, mp3, FLAC, "
        "WAV, …); the server decodes it.",
        returns="`transcript.json`: the model and revision, the language, "
        "`duration_s`, and `segments` (`start`, `end` in seconds, `text`, and `words` "
        "with their own times when `word_timestamps` is true); with `speech_only`, the "
        "stretches taken out are listed in `removed`. Qwen3-ASR with word timestamps "
        "first publishes `transcript.text.json` (each piece's text before alignment). "
        "A Qwen3-ASR job's `done` event gives `context_echo_pieces` and "
        "`decode_loop_pieces`.",
        example={
            "type": "asr",
            "model": "qwen3-asr-1.7b",
            "params": {"language": "en", "vad_filter": False, "word_timestamps": True},
            "inputs": {"chapter.m4b": BLOB},
        },
        notes=(
            "`language`, `vad_filter` and `word_timestamps` are required. Qwen3-ASR "
            "takes one of en, de, fr, es, it, pt, ru, ja, ko, zh, yue "
            "(`language_unsupported_by_engine`); `\"auto\"` is Whisper's only.",
            "Engine-specific knobs are refused on the other engine by name: "
            "`initial_prompt` is Whisper's, `context`, `piece_s` and `overlap_s` "
            "Qwen3-ASR's; `vad_filter: true` only on faster-whisper; `overlap_s` > 0 "
            "needs `word_timestamps`.",
            "`speech_only` (Crucible's speech detector) defaults to the opposite of "
            "`vad_filter`, so it is ON unless you send `vad_filter: true`; send "
            "`speech_only: false` to transcribe every stretch. Its knobs "
            "(`speech_threshold`, `speech_pad_s`, `speech_min_gap_s`) are refused "
            "without it.",
            "Qwen3-ASR keeps a resume journal: the submit answers `resume_id` beside "
            "`job_id`; after a failure or cancel, submit the same job with "
            "`params.resume` set to it (docs/RESUMABLE-JOBS.md). Whisper refuses "
            "`resume` (`resume_unsupported`).",
            "More than one input is `invalid_inputs`.",
            "Pieces decode 8 at a time on cuda-linux and one at a time on a Mac; each "
            "job loads its own engine. See Throughput.",
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "align": JobDoc(
        summary="Forced alignment: places known text in time inside short audio "
        "windows with Qwen3-ForcedAligner, one window per chunk, all in one job.",
        model="An aligner id (GET /v1/catalog): `qwen3-aligner`.",
        params=AlignParams,
        inputs="One audio file per chunk, named `<index>.<ext>` (e.g. `0.flac`), each "
        "at most 300 s, any format ffmpeg decodes. Every chunk needs a file and every "
        "file a chunk.",
        returns="`alignment.json`: model, revision, language, and `chunks`, each "
        "`{index, items: [{text, start, end}]}` in seconds from that window's start, or "
        "`{index, error}`. Items are the model's own tokens, not your words. A `cue` "
        "event goes out as each chunk lands. The `done` event gives `chunks`, `failed` "
        "(the failed indexes) and `resident` " + RESIDENT_AFTER_A_JOB + ".",
        example={
            "type": "align",
            "model": "qwen3-aligner",
            "params": {
                "language": "en",
                "chunks": [{"index": 0, "text": "He had been walking for some time."}],
            },
            "inputs": {"0.flac": BLOB},
        },
        notes=(
            "`language` is one of en, de, fr, es, it, pt, ru, ja, ko, zh, yue; there "
            "is no fallback to English.",
            "A chunk that fails is reported alone in `failed`; the job still ends "
            "`done`.",
            "Cutting a long recording into windows is the caller's job; for a whole "
            "audiobook use `align-longform`.",
            "Needs ffmpeg on the server (`ffmpeg_missing`).",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
            ENGINE_IN_USE,
        ),
    ),
    "unload-aligner": _unload("unload-aligner", "aligner", "align", "qwen3-aligner"),
    "align-longform": JobDoc(
        summary="Aligns a whole audiobook to its book text: a rough faster-whisper "
        "transcript places each sentence, then Qwen3-ForcedAligner places the words, "
        "and out comes a WebVTT with one cue per sentence.",
        model="The aligner id (GET /v1/catalog): `qwen3-aligner`. The rough pass's "
        "Whisper model is `params.rough_model`.",
        params=AlignLongformParams,
        inputs="Exactly one audio file, the whole audiobook (the m4b as it is). The "
        "book never crosses: its sentences are in `params.sentences`.",
        returns="`alignment.vtt` (a cue per placed sentence; `kind: \"heading\"` cues "
        "carry a `NOTE heading`) and `align-report.json` (`sentences`, `placed`, "
        "`dropped`, `rate_tokens_per_second`, `chunks`, `capped`, `duration_s`, "
        "`rough_model`, `aligner`). Progress events name the stage: `transcribe`, "
        "`coarse-align`, `align`, `write`.",
        example={
            "type": "align-longform",
            "model": "qwen3-aligner",
            "params": {
                "language": "en",
                "rough_model": "whisper-large-v3-turbo",
                "sentences": [
                    {"index": 0, "text": "Chapter One", "kind": "heading"},
                    {"index": 1, "text": "He had been walking for some time."},
                ],
            },
            "inputs": {"book.m4b": BLOB},
        },
        notes=(
            "Send `rough_model`: it names a faster-whisper ASR manifest installed on "
            "this server (e.g. `whisper-large-v3-turbo`, `whisper-tiny`). The default, "
            "`small`, names no manifest this build ships, so leaving it out is refused "
            "`unknown_rough_model`; `rough_model_not_installed` and "
            "`rough_model_not_on_this_backend` are the other refusals.",
            "Sentence indexes must be unique and in reading order, and text not blank.",
            "`chunk_s` (default 240) is at most 300, the aligner's window.",
            "`nothing_narrated` / `no_cues` mean no sentence could be placed: the "
            "wrong book or the wrong language.",
        ),
    ),
    "rvc": JobDoc(
        summary="Voice conversion: re-voices every input through an RVC model, "
        "keeping each input's container, sample format and exact duration.",
        model="An RVC voice id (GET /v1/catalog), e.g. `sigma`, "
        "`deathstalker-rvc-v3`.",
        params=RvcParams,
        inputs="One or more audio files of any length (WAV, FLAC, OGG, MP3 or AIFF, "
        "read from the bytes, so names need no extension). Each is converted on its "
        "own.",
        returns="One artifact per input, under the input's name, in its container and "
        "sample format (a WAV past 4 GiB comes back RF64), at `output_rate` "
        "(`native`: max of the input rate and the model's, 48 kHz for the published "
        "models; `input`: the input's) and `output_channels` (`input` or `mono`). The "
        "`done` event gives `files`, `pieces`, `model_name`, and `outputs`: per input "
        "its `frames`, `sample_rate`, `channels`, `format` and `subtype`.",
        example={
            "type": "rvc",
            "model": "sigma",
            "params": {
                "index_rate": 0.3,
                "protect_rate": 0.1,
                "n_semitones": -2,
                "f0_method": "rmvpe",
            },
            "inputs": {"c000.flac": BLOB},
        },
        notes=(
            "`index_rate`, `protect_rate` and `n_semitones` are required. "
            "`protect_rate` is inverted: lower protects more, 0.5 turns protection "
            "off.",
            "A model with no .index refuses `index_rate` above 0 "
            "(`model_has_no_index`).",
            "Long inputs are cut at quiet points into pieces of `piece_s` (default "
            "60, 10 to 600) with `overlap_s` each side (default 0.5, under half a "
            "piece) and joined with a `crossfade_s` fade (default 0.02, at most twice "
            "the overlap), so a 12-hour master can be sent whole.",
            "A stereo input gets one converted voice in both channels; it is not a "
            "per-channel conversion.",
            "If any input produces no output the job fails `rvc_output_missing`, "
            "naming them.",
            "Needs ffmpeg and ffprobe on the server (`ffmpeg_missing`).",
            "Send many files as one job: up to 96 pieces share one conversion process "
            "and one load of the voice. See Throughput.",
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "denoise": JobDoc(
        summary="Separates audio into stems with a source-separation model: "
        "`vocals-roformer` splits vocals from the instrumental, `denoise-roformer` "
        "splits dry speech from noise.",
        model="A separator id: `vocals-roformer` or `denoise-roformer` "
        "(GET /v1/catalog lists them and whether each is installed).",
        params=DenoiseParams,
        inputs="Exactly one audio file libsndfile reads (WAV, FLAC), at the separator's "
        "own rate, 44100 Hz. Convert a video's audio first, e.g. "
        "`ffmpeg -i in.mp4 -vn -ar 44100 -c:a pcm_s16le in.wav`.",
        returns="WAV stems at 44100 Hz with the input's channel count and exactly its "
        "frame count, so they line up sample for sample. `stems: \"primary\"` (the "
        "default) returns only the primary stem (`…_(Vocals)_….wav` or `…_(Dry)_….wav`); "
        "`stems: \"all\"` returns every stem, the primary first. The `done` event names "
        "`primary_stem`, lists `stems`, and gives `sample_rate`, `frames`, "
        "`separate_seconds` and `load_seconds`.",
        example={
            "type": "denoise",
            "model": "vocals-roformer",
            "params": {"stems": "all"},
            "inputs": {"song.wav": {"blob_id": "<from POST /v1/uploads>"}},
        },
        notes=(
            "Nothing is resampled: another sample rate is refused by name.",
            "A model not installed yet is installed on submit where the server allows "
            "it: the submit answers 409 `installing`; submit again when it is done.",
            "`vocals` means every voice in the track, singing included.",
            "The separator stays loaded between jobs; send a file whole rather than "
            "in chunks. See Throughput.",
        ),
    ),
    "unload-denoiser": _unload(
        "unload-denoiser", "separator", "denoise", "vocals-roformer"
    ),
    "image": JobDoc(
        summary="Makes one picture from a prompt (text-to-image), redraws an input "
        "picture (image-to-image), or regenerates the masked region of one "
        "(inpainting and outpainting).",
        model="An image model id (GET /v1/catalog): `qwen-image-2.1`.",
        params=ImageParams,
        inputs="None for text-to-image. Exactly one PNG, JPEG or WebP with "
        "`image_strength`. With `mask`, exactly two: the image and the mask, `mask` "
        "naming the mask input; the mask is the image's exact size, white (128 and "
        "up) is regenerated.",
        returns="`image.png`, and with a mask also `generated.png` (the model's "
        "picture before the paste-back). The `done` event gives `image`, every "
        "effective parameter (seed included) plus timings and memory, and "
        "`resident` " + RESIDENT_AFTER_A_JOB + ".",
        example={
            "type": "image",
            "model": "qwen-image-2.1",
            "params": {
                "prompt": "An ordinary documentary photograph of a kitchen table with "
                "one red apple on it, natural window light. No text.",
                "width": 1280,
                "height": 720,
                "seed": 1,
                "steps": 40,
            },
        },
        notes=(
            "Width and height are 256 to 2048 and multiples of 16 (32 on cuda-linux, "
            "`image_size_not_supported`), at most 1,048,576 pixels "
            "(`image_too_large`).",
            "`guidance` above 1.0 needs `negative_prompt`, and `negative_prompt` "
            "needs it; `mask_blur` needs `mask`.",
            "Inputs that do not match the mode are `invalid_inputs`; a mask of a "
            "different size is `mask_size_mismatch`, one with no white `mask_empty`.",
            "The seed reproduces a picture on the same backend only.",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "unload-image": _unload(
        "unload-image", "generator", "image or load-image", "qwen-image-2.1"
    ),
    "load-image": _load(
        "load-image",
        LoadImageParams,
        "image model",
        "An image model id (GET /v1/catalog): `qwen-image-2.1`.",
        "qwen-image-2.1",
    ),
    "audio": JobDoc(
        summary="Makes sound from words: sound effects and instrumental music with "
        "Stable Audio 3, songs with sung vocals (or instrumentals) with YuE2.",
        model="An audio model id (GET /v1/catalog): `stable-audio-3-small-sfx` (sound "
        "effects), `stable-audio-3-medium` (music), `yue2-3b` (songs).",
        params=AudioParams,
        inputs="None; an audio job reads no files (`invalid_inputs`).",
        returns="`audio.flac` (24-bit, the default), `audio.wav` or `audio.mp3` (192 "
        "kbps), per `format`; a song adds `score.abc`, the score YuE2 writes first. "
        "The `done` event gives `audio`, every effective parameter (seed included) "
        "plus timings and memory, and `resident` " + RESIDENT_AFTER_A_JOB + ".",
        example={
            "type": "audio",
            "model": "stable-audio-3-small-sfx",
            "params": {
                "prompt": "TrackType: SFX. A heavy oak door creaks open slowly in a "
                "stone hallway, close mic, dry",
                "duration_s": 4,
                "seed": 1,
            },
        },
        notes=(
            "Which params a model takes is its own: Stable Audio reads `prompt` and "
            "takes `duration_s` and `steps`; YuE2 reads `tags` and `lyrics` (sections "
            "like `[Verse]`, optional with `instrumental: true`) and takes `cfg`. "
            "Anything else is refused `audio_param_unsupported` with the list it "
            "does take; a missing one `audio_param_missing`.",
            "Past a model's ceiling: `audio_too_long` (120 s sfx, 380 s music), "
            "`audio_param_out_of_range`. A song's length follows its lyrics.",
            "A host with `[audio] low_vram = true` in its config holds only half of YuE2 "
            "on the card at a time; Crucible turns it on by itself on a card too small "
            "to hold YuE2 whole (an 8 GiB card), and `audio.low_vram` in the `done` "
            "event says which ran (docs/AUDIO.md).",
            "A song's `done` event says how each token stage ended: `audio.decode_stages` "
            "gives `scoring` and `composing` their tokens, `cap`, `ended` (`eos`, or `cap` "
            "when the model never ended the stage), execution path, `low_vram` and "
            "tokens per second, and `audio.stages_at_cap` names the stages that ran to "
            "their cap (`[]` when none; null for Stable Audio). A stage at its cap still "
            "finishes the job and is never re-run (docs/AUDIO.md).",
            "The Stable Audio repos are gated: until the licence is accepted on "
            "Hugging Face and the server has a token, `409 model_gated` says what to "
            "do (docs/AUDIO.md).",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "unload-audio": _unload(
        "unload-audio", "audio generator", "audio or load-audio", "stable-audio-3-medium"
    ),
    "load-audio": _load(
        "load-audio",
        LoadAudioParams,
        "audio model",
        "An audio model id (GET /v1/catalog), e.g. `stable-audio-3-small-sfx`.",
        "stable-audio-3-small-sfx",
    ),
    "segment": JobDoc(
        summary="Makes a mask from a picture: the main subject by itself (`birefnet`, "
        "background removal), or the object under your points or inside your box "
        "(`sam2.1-hiera-large`).",
        model="A segment model id (GET /v1/catalog): `birefnet` or "
        "`sam2.1-hiera-large`.",
        params=SegmentParams,
        inputs="Exactly one PNG, JPEG or WebP of at most 40,000,000 pixels.",
        returns="`mask.png` (8-bit grey, 255 = selected; soft edges from birefnet, "
        "hard from SAM) and `cutout.png` (the input as RGBA with the mask as alpha), "
        "both at the input's size. The `done` event gives `segment` (effective "
        "parameters, `score`, `coverage`, timings, memory) and `resident` "
        + RESIDENT_AFTER_A_JOB + ".",
        example={
            "type": "segment",
            "model": "sam2.1-hiera-large",
            "params": {"points": [{"x": 412, "y": 300, "label": 1}]},
            "inputs": {"photo.jpg": BLOB},
        },
        notes=(
            "`birefnet` takes no params (`segment_param_unsupported`); "
            "`sam2.1-hiera-large` needs `points` (1 to 64, `label` 1 keeps, 0 leaves "
            "out), `box` `[x0, y0, x1, y1]`, or both (`segment_param_missing`).",
            "Coordinates are the input's stored pixels from the top-left; EXIF "
            "orientation is not applied. Outside the picture is "
            "`segment_prompt_outside_picture`.",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "unload-segment": _unload(
        "unload-segment", "segmenter", "segment or load-segment", "birefnet"
    ),
    "load-segment": _load(
        "load-segment",
        LoadSegmentParams,
        "segment model",
        "A segment model id (GET /v1/catalog): `birefnet` or `sam2.1-hiera-large`.",
        "birefnet",
    ),
    "video": JobDoc(
        summary="Makes a video clip with its own sound from a prompt (text-to-video), "
        "or brings a start picture to life (image-to-video), with LTX-2.5.",
        model="A video model id (GET /v1/catalog): `ltx-2.5-distilled`.",
        params=VideoParams,
        inputs="None for text-to-video, or exactly one PNG, JPEG or WebP: the first "
        "frame, cropped to the clip's shape and resized.",
        returns="`video.mp4`: H.264 (yuv420p) with AAC stereo at 48 kHz, faststart. "
        "The `done` event gives `video`, every effective parameter (mode, size, "
        "`num_frames`, `fps`, `duration_s`, seed, …) plus timings and GPU-busy "
        "figures, and `resident` " + RESIDENT_AFTER_A_JOB + ".",
        example={
            "type": "video",
            "model": "ltx-2.5-distilled",
            "params": {
                "prompt": "A red fox trots through fresh snow at dawn, the camera "
                "tracking alongside at knee height; each step crunches.",
                "width": 1280,
                "height": 704,
                "duration_s": 5,
            },
        },
        notes=(
            "Send `width` and `height` together or neither (default 1280x704); "
            "`duration_s` or `num_frames`, not both. `num_frames` is 8k+1 "
            "(`video_frames_not_supported`).",
            "Every limit is refused by name before anything loads: "
            "`video_size_not_supported` (multiples of 32, 64 on a Mac; sides 256 to "
            "1280), `video_too_large`, `video_too_long`, and "
            "`video_param_unsupported` (`negative_prompt`, `steps` other than 8, "
            "`fps` other than 24 or 25). docs/VIDEO.md has the per-backend figures.",
            "The model repo is gated: `409 model_gated` says how to accept the "
            "licence.",
            UNHELD_AFTER_THE_JOB,
            INSTALLS_ON_SUBMIT,
        ),
    ),
    "unload-video": _unload(
        "unload-video", "video generator", "video or load-video", "ltx-2.5-distilled"
    ),
    "load-video": _load(
        "load-video",
        LoadVideoParams,
        "video model",
        "A video model id (GET /v1/catalog): `ltx-2.5-distilled`.",
        "ltx-2.5-distilled",
    ),
}


__all__ = ["JOB_DOCS", "JobDoc"]
