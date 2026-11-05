"""What a Higgs engine is built from, and every number the audition measured.

Sources (all read-only): the campaign
`E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\` - `render_v2.py`
(the measured render path), `smoke_v2.py` (the processor probe),
`v2_pokemon_para_log.json` (the nine-chunk audition), narrator's Higgs notes (the
template, the token ids, the licence) and narrator's Higgs v3 lever notes (the v3 lever
sweep, which is DESIGN input here - v3 is not built).

THE VOICE IS CLIPS, AND CLIPS COME FROM A FILE. Orpheus's voice is a token that
rides in the prompt, so its whole configuration is a string. Higgs's voice is
reference clips WITH BOOK-EXACT TRANSCRIPTS, which is too much to put on a
command line and must not be guessed, so it is a JSON document named by
`NARRATOR_HIGGS_VOICES`:

    {
      "deathstalker": {
        "scene": "Audio is recorded from a quiet room.",
        "clips": [
          {"path": "/home/<user>/xtts_ft/ds_ad4s/wavs/cd_cd_00000008.wav",
           "transcript": "The purpose here is not sectarian, ...",
           "seconds": 14.02}
        ]
      }
    }

A load for a voice the file does not carry FAILS, naming the file and the
voices it does carry. There is no default voice: rendering a book in the
model's own narrator instead of the one asked for is a silent failure.

LICENCE (Boson Higgs Audio 2 Community License, release date 2025-06-20):
usable at BookForge's scale, but it obliges the attribution "Built with Higgs
Materials licensed from Boson AI USA, Inc." plus the Meta Llama 3 attribution,
and ANY FINE-TUNE WE SHIP MUST CARRY "Higgs Audio 2" IN ITS NAME. Recorded here
because this is the file a packager reads.
"""
import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional

from ..protocol import ClipsVoice, DefaultVoice, ReferenceClip, StopPolicy

MODEL_ID = 'bosonai/higgs-audio-v2-generation-3B-base'
TRANSFORMERS_CLASS = 'HiggsAudioV2ForConditionalGeneration'

#: Where the voice document lives. No default path: an absent variable is a
#: refusal, not a search.
VOICES_ENV = 'NARRATOR_HIGGS_VOICES'


class HiggsDefaults:
    """Every Higgs v2 number, with the measurement it came from.

    They are class attributes for the same reason Orpheus's are (EngineDefaults):
    a test reads and overrides them by name, and a post-mortem can print them.
    """

    # Matches its registry id. The suffix is the whole point: nothing should
    # select this engine by accident (see narrator/engine/registry.py).
    ENGINE_ID = 'higgs-v2-scaffold'
    MODEL_ID = MODEL_ID
    SAMPLE_RATE = 24000
    FRAMES_PER_SECOND = 25.0

    # Sampling: render_v2.py's, which is what the audition measured. NB Higgs
    # v3's own deploy defaults are different (temperature 1.0, top_p 0.95,
    # top_k 50) and Owen asked the delivered v3 render to use them; that is a
    # v3 fact and does not move v2's audition numbers.
    TEMPERATURE = 0.3
    TOP_P = 0.95
    TOP_K = 50

    # The audio cap, in FRAMES (max_new_tokens counts LM steps):
    #   cap = int(chars / CHARS_PER_SEC * FPS * CAP_SLACK) + CAP_SLACK_FRAMES
    # render_v2.py's formula exactly. Measured: never reached in 9/9 chunks of
    # 132-898 chars (898 chars -> cap 2794, EOS at frame 1468).
    CHARS_PER_SEC = 15.0        # the narration rate the cap is sized against
    CAP_SLACK = 1.8
    CAP_SLACK_FRAMES = 100

    # The packer's chunk size. 900 is the audition's proven ceiling: the
    # 898-char paragraph was the MOST accurate chunk of the nine (duration ratio
    # 0.98) and nothing in the 132-898 range degraded. Higgs v3 is a different
    # number - 600, because 900 drops the tail reproducibly there and cloning
    # does not fix it - which is why this is a per-version constant and not a
    # constant of "Higgs".
    MAX_CHARS = 900

    # The truncation guard's rate, characters of text per second of audio.
    # ADVISORY for Higgs: the audition measured 11.9-18.0 ch/s across nine
    # chunks, and 20.0 is that ceiling with headroom. It is NOT the real
    # coverage gate - see StopPolicy.coverage_check below - because a duration
    # ratio is not a coverage proxy on this family: a v3 chunk measured 0.99
    # while dropping 22 % of its text and inserting filler.
    MAX_CHARS_PER_SEC = 20.0

    # The fused context window, in positions. 8,192 is the number the plan and
    # the v3 model card give; v2's Llama-3.2-3B backbone nominally carries more,
    # so this is the CONSERVATIVE window and it has not been probed on v2. A
    # prompt that already fills it is refused rather than truncated.
    CONTEXT_TOKENS = 8192

    # Milliseconds of fade the ASSEMBLER applies at each chunk edge. After the
    # codec's sentinel trim an edge still sits near -30 dB and clicks on a join;
    # 10 in / 25 out takes it to -45..-48 dB. Asymmetric: a chunk ends on a
    # decay the ear does not expect, so the tail needs the wider window. Same
    # codec family as v3, so the same numbers.
    EDGE_FADE_IN_MS = 10.0
    EDGE_FADE_OUT_MS = 25.0

    # Total reference audio the engine will accept, across all clips. None on v2
    # (28.5 s of reference was fine; the real limit there is context). Higgs v3
    # caps it at 30 s - 42 s is an HTTP 400 - so v3's config sets it.
    MAX_REFERENCE_SECONDS = None

    # Inline control tokens this version understands. v2: NONE. An unknown
    # control token is not ignored, it is READ ALOUD AS WORDS, so this is an
    # allowlist and not documentation. (v3 adds emotion/style/prosody/sfx;
    # `<|scene_desc_*|>` is a v2 chat ROLE, not a control token, and is applied
    # through ClipsVoice.scene.)
    ALLOWED_CONTROLS = ()


@dataclass
class HiggsConfig:
    """Everything one HiggsEngine instance is built from.

    `voice` is a ClipsVoice, not a string: see the module docstring. `model_id`
    is the HF repo or a local snapshot dir. `sentences_dir` / `process_dir` are
    the SessionStore's and are only touched by the file-writing half
    (`convert`); an in-memory caller (the streaming worker) leaves them None,
    exactly as EngineConfig does for Orpheus.
    """
    voice: ClipsVoice
    model_id: str = MODEL_ID
    sample_rate: int = HiggsDefaults.SAMPLE_RATE
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    system_prompt: Optional[str] = None
    scene: Optional[str] = None
    sentences_dir: Optional[str] = None
    process_dir: Optional[str] = None
    audio_format: str = 'flac'
    seed: Optional[int] = 1234
    temperature: float = HiggsDefaults.TEMPERATURE
    top_p: float = HiggsDefaults.TOP_P
    top_k: int = HiggsDefaults.TOP_K
    max_chars: int = HiggsDefaults.MAX_CHARS
    max_chars_per_sec: float = HiggsDefaults.MAX_CHARS_PER_SEC
    context_tokens: int = HiggsDefaults.CONTEXT_TOKENS

    def __post_init__(self):
        if not isinstance(self.voice, ClipsVoice):
            raise ValueError(
                'HiggsConfig(voice=...) takes a ClipsVoice - reference clips with '
                f'book-exact transcripts. Got {type(self.voice).__name__}. Higgs has '
                'no fine-tuned voice tokens; a voice IS its clips.')
        if not (self.voice.name or '').strip():
            raise ValueError(
                'HiggsConfig(voice=...) needs a NAMED voice. The name is what a load '
                'message, a manifest and a log line call it; an anonymous voice is one '
                'nobody can ask for again.')
        if self.voice.max_reference_seconds is not None:
            known = [c.seconds for c in self.voice.clips if c.seconds is not None]
            total = sum(known)
            if total > self.voice.max_reference_seconds:
                raise ValueError(
                    f"Higgs voice '{self.voice.name}' carries {total:.1f} s of "
                    f'reference audio across {len(self.voice.clips)} clips, over this '
                    f'version\'s {self.voice.max_reference_seconds:.0f} s limit. Drop '
                    'a clip or shorten one - the server rejects the request outright.')


# ---------------------------------------------------------------------------
# Budget and stop policy
# ---------------------------------------------------------------------------


class HiggsBudget:
    """`narrator.engine.protocol.Budget` for Higgs.

    Everything comes off the config so a v3 profile is a different config, not
    a different class.
    """

    def __init__(self, config: HiggsConfig):
        self._config = config

    def max_chars(self, voice=None) -> int:
        """The VOICE's chunk size when the document declared one, else the
        engine placeholder (900 - the audition's proven ceiling).

        A FINE-TUNE WITH NO `maxChars` IS NO LONGER REFUSED (Owen, 2026-09-19).
        It used to be, here and at load, and the reasoning - a fine-tune's safe
        chunk length is measured on THAT model - has not changed; what changed
        is whose number it is. `maxChars` is the CLIENT's packing size, and the
        screening render of an unmeasured checkpoint is what produces it. See
        `HiggsV3Budget.max_chars`, where the ruling is written out in full;
        this is the v2 scaffold's copy of the same rule, kept identical so the
        two cannot drift.
        """
        ref = self._config.voice
        if voice is not None and voice != ref.name:
            raise ValueError(
                f"HiggsBudget: this engine serves '{ref.name}', not '{voice}'.")
        if ref.max_chars is None:
            return int(self._config.max_chars)
        return int(ref.max_chars)

    def max_chars_per_sec(self, voice=None) -> float:
        return float(self._config.max_chars_per_sec)

    def max_total_tokens(self, prompt_tokens: int, voice=None) -> int:
        """The context window - a CEILING the prompt eats into.

        `prompt_tokens` is the whole prompt's position count: the text stream
        plus the reference clips' audio frames (the measured 2-clip, 130-char
        case was 900 text positions and 732 audio frames). The generation budget
        is therefore `max_total_tokens(p) - p`, and a prompt that already fills
        the window is REFUSED here rather than silently leaving no room to
        speak. Contrast Orpheus, whose audio-token cap is independent of the
        prompt.
        """
        prompt_tokens = int(prompt_tokens)
        if prompt_tokens < 0:
            raise ValueError(f'prompt_tokens must be >= 0, got {prompt_tokens}')
        window = int(self._config.context_tokens)
        if prompt_tokens >= window:
            raise ValueError(
                f'Higgs prompt is {prompt_tokens} positions against a {window}-position '
                'window, leaving nothing to generate. Shorten the chunk or use fewer / '
                'shorter reference clips - reference length is paid for in context.')
        return window

    # -- the per-chunk cap ---------------------------------------------------

    def cap_frames(self, text: str) -> int:
        """`max_new_tokens` for one chunk, in FRAMES, by render_v2.py's formula:
        `int(len(text) / 15.0 * 25 * 1.8) + 100`.

        1.8x the expected frame count plus 100 frames of slack. Measured never
        to be reached across the nine-chunk audition, so it is a backstop and
        not a working limit."""
        chars = len(text or '')
        expected_seconds = chars / HiggsDefaults.CHARS_PER_SEC
        return int(expected_seconds * HiggsDefaults.FRAMES_PER_SECOND
                   * HiggsDefaults.CAP_SLACK) + HiggsDefaults.CAP_SLACK_FRAMES


def higgs_stop_policy(config: HiggsConfig) -> StopPolicy:
    """Higgs stops on its own. Measured: EOS 9/9 across 132-898 char chunks,
    every time well inside the cap, zero runaways, zero cap hits - with no
    boost, no logit bias and no stop-string hack. So there is no ladder to
    re-render a capped chunk with (`resplit_on_cap` False), and `max_new_tokens`
    here is the cap for the LARGEST permitted chunk; the per-chunk cap comes
    from `HiggsBudget.cap_frames(text)`.

    `coverage_check` is 'asr' and is a HOOK, not an implementation: duration is
    not a coverage proxy on this family (a v3 chunk measured a 0.99 ratio while
    dropping 22 % of its text), so the render layer must eventually ASR the
    chunk against its text. Nothing gates on it yet.
    """
    return StopPolicy(
        max_new_tokens=HiggsBudget(config).cap_frames('x' * int(config.max_chars)),
        eos_reliable=True,
        resplit_on_cap=False,
        max_chars_per_sec=float(config.max_chars_per_sec),
        levers={'temperature': float(config.temperature),
                'topP': float(config.top_p),
                'topK': float(config.top_k)},
        coverage_check='asr',
    )


# ---------------------------------------------------------------------------
# Voices from disk
# ---------------------------------------------------------------------------


def voices_path() -> str:
    """The voice document's path, or a refusal naming the variable."""
    path = (os.environ.get(VOICES_ENV) or '').strip()
    if not path:
        raise ValueError(
            f'{VOICES_ENV} is not set. A Higgs voice is reference clips plus their '
            'book-exact transcripts, which cannot be passed as a voice name; point '
            f'{VOICES_ENV} at the JSON document that defines them (see '
            'narrator/engine/higgs/config.py for its shape).')
    if not os.path.isfile(path):
        raise ValueError(f'{VOICES_ENV} points at {path}, which does not exist.')
    return path


def load_voices(path: str = None, *, allowed_controls=None,
                max_reference_seconds=..., placeholder_max_chars=None
                ) -> Mapping[str, ClipsVoice]:
    """Read the voice document. Every clip must exist and carry a transcript.

    THE DOCUMENT IS PER-ENGINE TUNING, not just a list of files. Each entry may
    carry, besides `clips`:

        scene                 v2 only (v3 has no scene-description mechanism)
        checkpointDir         a MERGED fine-tune directory; makes this a
                              CHECKPOINT voice
        kind                  'checkpoint' (implied by `checkpointDir`),
                              'default', or 'clips' (implied by `clips`)
        maxCharsSource        where maxChars came from: catalog | placeholder |
                              length-sweep

    THREE DOCUMENT SHAPES:

        {"kind": "checkpoint",                 THE PRODUCTION SHAPE. A
         "checkpointDir": ...,                 fine-tuned voice: its WEIGHTS are
         "maxChars": N}                        the voice, the server runs ON
                                               that directory, and the prompt is
                                               text-only. Owen, 2026-09-04:
                                               production is fine-tuned voices
                                               only.

    THE REQUIRED FILES OF A `checkpointDir`, checked by the v3 engines rather
    than here (this module is shared with the v2 scaffold, and the check belongs
    where the directory is resolved - `v3_served.checkpoint_serve_target`):
    the weights, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    `chat_template.jinja` AND **`generation_config.json`**. That last one is the
    SAMPLING the model is served at - `vllm-omni serve` resolves sampling from
    the model directory (`--generation-config auto`), and a dir without it falls
    back to a bare SamplingParams (top_p 1.0, top_k disabled) which derails long
    chunks into babble. `v3_served.require_generation_config` refuses a
    checkpoint voice without it BY NAME; on the Mac the MLX backend reads the
    same file for its own sampler, because mlx-audio does not.

    THAT LAST FILE IS DEMANDED OF A MERGE ONLY (2026-09-15). A `clips` voice
    may name a `checkpointDir` too, and what it names is the BASE snapshot at
    a pinned revision - Crucible writes it so a clone renders on the bytes the
    pin names rather than on whatever the HF cache holds. The published base
    (`bosonai/higgs-tts-3-4b` at 239f63fb: thirteen files) has never carried a
    `generation_config.json`, which is exactly why narrator states base
    sampling explicitly on both arms. So the KIND BESIDE THE KEY decides what
    the directory is, and this loader reads a `clips` voice's into
    `ClipsVoice.base_dir` - a different field, checked by
    `v3_served.require_base_weights_dir` and asked for no such file.
        {"kind": "default"}                    the model's OWN voice, no
                                               conditioning at all - a smoke
                                               test or a demo. 12 % of the
                                               narrator ceiling.
        {"clips": [...]}                       DIAGNOSTIC: zero-shot cloning,
                                               how a voice is auditioned before
                                               anyone trains it. Kept working
                                               and tested; no book ships on it.

    THERE IS NO 'adapter' SHAPE, and `adapterDir` is refused by name. vllm-omni
    cannot load a LoRA at runtime (no adapter flags; the talker does not
    implement `SupportsLoRA`), so there is nothing for an adapter directory to
    be loaded INTO. The LoRA is the archival artifact; what narrator serves is
    the merged checkpoint.

    A `clips` key that is present must hold at least one usable clip -
    `ClipsVoice` enforces that - so a clone whose references went missing is
    still an error and never a silent downgrade to the default voice.
        maxChars              THE PACKER'S CHUNK SIZE FOR THIS VOICE
        allowedControls       overrides the engine's control-token allowlist
        maxReferenceSeconds   overrides the engine's reference cap

    `allowed_controls` / `max_reference_seconds` / `placeholder_max_chars` are
    the ENGINE'S defaults for entries that declare none - v2 and v3 pass
    different ones, which is why they are arguments and not constants here.

    AN ADAPTER VOICE WITHOUT `maxChars` IS ACCEPTED (Owen, 2026-09-19), and
    packs at the engine's placeholder, saying so through `max_chars_source`.
    Until then a `kind: checkpoint` entry with none was refused here, on the
    ground that a fine-tune's safe chunk length is a measured property of THAT
    model and the base model's placeholder is not it. Both halves of that are
    still true - and it is still worth measuring and declaring - but the
    refusal was in the wrong place: `maxChars` IS THE CLIENT'S PACKING SIZE,
    the client packs the book and sends narrator the chunks, and the render
    that MEASURES a new checkpoint is precisely the render this refusal made
    impossible. Nothing narrator does per chunk reads the number: the frame
    ceiling is `cap_frames(text)` over the text it was actually sent, and the
    stop policy's `max_new_tokens` is the ENGINE's own `config.max_chars`.
    """
    path = path or voices_path()
    if max_reference_seconds is ...:
        max_reference_seconds = HiggsDefaults.MAX_REFERENCE_SECONDS
    if allowed_controls is None:
        allowed_controls = HiggsDefaults.ALLOWED_CONTROLS
    with open(path, 'r', encoding='utf-8') as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not document:
        raise ValueError(
            f'{path} must be a JSON object of voice name -> {{"clips": [...]}}; got '
            f'{type(document).__name__}'
            + ('' if isinstance(document, dict) else '') + '.')
    voices = {}
    for name, entry in document.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: voice '{name}' must be an object, got "
                f'{type(entry).__name__}.')
        if 'adapterDir' in entry:
            raise ValueError(
                f"{path}: voice '{name}' names 'adapterDir'. There is no runtime "
                'LoRA on this stack - vllm-omni has no adapter flags and its talker '
                'does not implement SupportsLoRA - so a fine-tuned voice is a MERGED '
                "checkpoint the server runs on. Use \"kind\": \"checkpoint\" with "
                '"checkpointDir".')
        entry_kind = entry.get(
            'kind', 'checkpoint' if entry.get('checkpointDir') else
            ('clips' if 'clips' in entry else None))
        if entry_kind is None:
            raise ValueError(
                f"{path}: voice '{name}' has no 'clips', no 'checkpointDir' and no "
                "'kind'. Say what it is: a fine-tune ('checkpoint', the production "
                "shape), the model's own voice ('default') - which sits at 12 % of "
                "the narrator ceiling and is never what a book wants by accident - "
                "or a diagnostic zero-shot clone ('clips').")
        clips = []
        for row in entry.get('clips', []):
            for key in ('path', 'transcript'):
                if key not in row:
                    raise ValueError(
                        f"{path}: voice '{name}' has a clip with no '{key}'. The "
                        'transcript is the book-exact text spoken in the clip - the '
                        'corpus row or the narration copy, never a transcription.')
            if not os.path.isfile(row['path']):
                raise ValueError(
                    f"{path}: voice '{name}' names a clip that does not exist: "
                    f"{row['path']}")
            clips.append(ReferenceClip(path=row['path'],
                                       transcript=row['transcript'],
                                       seconds=row.get('seconds')))
        checkpoint_dir = entry.get('checkpointDir')
        kind = entry_kind
        if kind not in ('clips', 'checkpoint', 'default'):
            raise ValueError(
                f"{path}: voice '{name}' has kind {kind!r}; expected 'checkpoint' (a "
                "fine-tune, the production shape), 'default' (the model's own voice) "
                "or 'clips' (a diagnostic zero-shot clone).")
        if kind == 'checkpoint' and not checkpoint_dir:
            raise ValueError(
                f"{path}: voice '{name}' is kind 'checkpoint' with no "
                "'checkpointDir'. The checkpoint IS the voice - there is nothing to "
                'serve without it.')
        base_dir = None
        if kind == 'clips' and checkpoint_dir:
            # A `clips` VOICE'S DIRECTORY IS THE BASE MODEL, and it is read
            # into a different field for that reason (2026-09-15).
            #
            # The key is spelled `checkpointDir` because that is the key
            # narrator has always read a directory from and the one Crucible
            # writes; what it MEANS is decided by the kind beside it. For
            # kind 'checkpoint' (and for a 'default' voice, whose weights are
            # the voice) it is a merged fine-tune, whose own
            # generation_config.json is the sampling the server reads. For
            # kind 'clips' the voice is in the REFERENCE and the directory is
            # the base model the reference conditions - pinned by whoever
            # wrote this document so the clone renders on known bytes. Base
            # weights carry no generation_config.json, have never carried
            # one, and are not asked for one; narrator states base sampling
            # itself on both arms.
            #
            # No real document is re-read by this: every `clips` entry
            # BookForge ships (the four zeroshot-* voices) names no directory
            # at all, and the only writer that ever paired the two is
            # Crucible, where it has always meant the base weights it pulled
            # at the manifest's pin.
            base_dir, checkpoint_dir = checkpoint_dir, None
        if kind == 'clips' and not clips:
            raise ValueError(
                f"{path}: voice '{name}' is a reference clone with no clips. A "
                "zero-shot clone with no reference is the model's own voice, which "
                "is a different thing - say kind 'default' if that is what you "
                'mean.')
        declared = entry.get('maxChars')
        if declared is None:
            # A fine-tune with no `maxChars` lands here too, since 2026-09-19 -
            # see this function's docstring. `source` is what keeps that
            # honest: 'placeholder' says the number is the ENGINE's and not
            # this model's, and it reaches the prep log and the manifest.
            max_chars, source = placeholder_max_chars, 'placeholder'
        else:
            max_chars, source = int(declared), entry.get('maxCharsSource', 'catalog')
        #: THE MODEL'S STATED LIMIT, or None when this document states none -
        #: which is a different thing from the placeholder narrator packs at.
        #: The two cross-checks below compare against the STATED one: since the
        #: no-maxChars refusal retired (2026-09-19) a fine-tune may load with
        #: none, and refusing its measured safe band for exceeding a
        #: placeholder the voice never claimed would be refusing it against
        #: another model's number.
        stated_max = None if declared is None else int(declared)
        target = entry.get('targetChars')
        if target is not None:
            if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                raise ValueError(
                    f"{path}: voice '{name}' declares targetChars {target!r}, which "
                    'is not a positive whole number of characters.')
            if stated_max is not None and target > stated_max:
                raise ValueError(
                    f"{path}: voice '{name}' declares targetChars {target} above its "
                    f'maxChars {stated_max}. The cap is the MEASURED safe chunk '
                    'length; a target above it asks for chunks the length sweep '
                    'refused. Lower the target or re-certify the cap.')
        safe_min, safe_max = _safe_band(path, name, entry, stated_max)
        band = _length_band(path, name, entry)
        sampling = _voice_sampling(path, name, entry)
        if not clips:
            # No reference audio in the request at all: a fine-tune whose
            # weights ARE the voice (the production shape), or the model's own.
            voices[name] = DefaultVoice(
                name=name, checkpoint_dir=checkpoint_dir, max_chars=max_chars,
                max_chars_source=None if max_chars is None else source,
                target_chars=target, safe_min_chars=safe_min,
                safe_max_chars=safe_max, max_chars_per_sec=band[0],
                min_chars_per_sec=band[1], pace_chars_per_sec=band[2],
                sampling=sampling)
            continue
        voices[name] = ClipsVoice(
            clips=tuple(clips),
            name=name,
            scene=entry.get('scene'),
            checkpoint_dir=checkpoint_dir,
            base_dir=base_dir,
            allowed_controls=tuple(entry.get('allowedControls', allowed_controls)),
            max_reference_seconds=entry.get('maxReferenceSeconds',
                                            max_reference_seconds),
            max_chars=max_chars,
            max_chars_source=None if max_chars is None else source,
            target_chars=target,
            safe_min_chars=safe_min,
            safe_max_chars=safe_max,
            max_chars_per_sec=band[0],
            min_chars_per_sec=band[1],
            pace_chars_per_sec=band[2],
            sampling=sampling,
        )
    return voices


def _safe_band(path, name, entry, max_chars):
    """`safeMinChars` / `safeMaxChars` - the band the prep packs between.

    Both optional and independent: a voice may declare a floor without a cap.
    A floor at or above the cap is refused, because the merge rule would then
    never terminate a group on the floor. The cap may not exceed `maxChars`,
    the model's stated limit, for the same reason `targetChars` may not.
    """
    def _one(key):
        v = entry.get(key)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(
                f"{path}: voice '{name}' declares {key} {v!r}, which is not a "
                'positive whole number of characters.')
        return int(v)
    lo, hi = _one('safeMinChars'), _one('safeMaxChars')
    if hi is not None and max_chars is not None and hi > int(max_chars):
        raise ValueError(
            f"{path}: voice '{name}' declares safeMaxChars {hi} above its maxChars "
            f"{max_chars}. The cap is the model's stated limit; a safe band may "
            'sit inside it, never past it.')
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(
            f"{path}: voice '{name}' declares safeMinChars {lo} at or above "
            f'safeMaxChars {hi}, which is not a band.')
    return lo, hi


#: The document's sampling keys (the catalog's camelCase) -> the engine's.
#:
#: `repetitionPenalty` is SERVED-ARM ONLY and reaches this map from a per-backend
#: block, never from the catalog's engine-level `sampling`. mlx-audio's
#: `higgs_audio_v3` has no repetition penalty (PORT_NOTES 13.11), so
#: `HiggsV3MlxConfig.__post_init__` refuses the key by name: a document carrying
#: it on the Mac arm is a loud refusal, not a lever that looks applied and does
#: nothing. It is listed here because the SERVED arm must be able to state it -
#: `sgl_served.SAMPLING_KEYS` already accepts it on the wire.
_SAMPLING_KEYS = {'temperature': 'temperature', 'topP': 'top_p', 'topK': 'top_k',
                  'repetitionPenalty': 'repetition_penalty'}


def _voice_sampling(path: str, name: str, entry: dict):
    """The voice's own `sampling` block, or None when the document carries
    none (the engine then renders at the checkpoint's own generation_config).
    Any subset of temperature / topP / topK / repetitionPenalty; each a positive
    number, top_k a whole one; an unknown key is refused by name - a misspelled
    lever would otherwise render at the file's value while the catalog states
    another.

    A PARTIAL BLOCK IS A TRAP AND THE CALLER OWNS IT: a per-backend block
    REPLACES the engine-level sampling rather than merging with it
    (`higgs-models.ts:higgsVoiceCapsForModel`), so a block stating only
    `repetitionPenalty` would leave temperature to fall back to the checkpoint's
    own generation_config.json (1.0, which nobody chose) instead of the
    catalog's 0.7. A deviating block restates every value it means to keep."""
    block = entry.get('sampling')
    if block is None:
        return None
    if not isinstance(block, dict) or not block:
        raise ValueError(
            f"{path}: voice '{name}' declares sampling {block!r}; expected an object "
            "with any of temperature / topP / topK.")
    unknown = sorted(set(block) - set(_SAMPLING_KEYS))
    if unknown:
        raise ValueError(
            f"{path}: voice '{name}' declares sampling keys {unknown}; the levers are "
            f"{sorted(_SAMPLING_KEYS)}.")
    out = {}
    for key, value in block.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"{path}: voice '{name}' declares sampling {key} {value!r}, which is "
                'not a positive number.')
        if key == 'topK' and int(value) != value:
            raise ValueError(
                f"{path}: voice '{name}' declares sampling topK {value!r}; top_k is a "
                'whole number of candidates.')
        out[_SAMPLING_KEYS[key]] = int(value) if key == 'topK' else float(value)
    return out


def _length_band(path: str, name: str, entry: dict):
    """The voice's `maxCharsPerSec` / `minCharsPerSec` / `paceCharsPerSec` (the
    length guard's seed band and the recorded pace it was derived from), as a
    (max, min, pace) triple of floats or Nones. ALL THREE OR NONE: a band with
    an edge missing, or a band with no pace to re-centre from, is a pace
    nobody finished writing, and is refused by name. Positive, and
    min < pace < max.

    NOTHING IN NARRATOR BUILDS A PACE TRACKER FROM THESE (Owen, 2026-09-19).
    They are still parsed, still checked and still reach `ClipsVoice`, because
    they are what the CLIENT reads off the voice row and sends back as a
    batch's `band` - the band has one owner and it is the caller. What was
    removed is the other reader: `truncation.tracker_for` used to take a voice
    and fall back to the engine's default band when the entry carried none,
    which judged a 17.2 chars/s book against a 15.0 centre."""
    hi = entry.get('maxCharsPerSec')
    lo = entry.get('minCharsPerSec')
    pace = entry.get('paceCharsPerSec')
    if hi is None and lo is None and pace is None:
        return (None, None, None)
    if hi is None or lo is None or pace is None:
        raise ValueError(
            f"{path}: voice '{name}' declares only part of its length band "
            f'(maxCharsPerSec={hi!r}, minCharsPerSec={lo!r}, paceCharsPerSec={pace!r}). '
            'The band comes from the measured pace as a triple - the pace and the '
            'two edges derived from it; write all three or none.')
    for label, value in (('maxCharsPerSec', hi), ('minCharsPerSec', lo),
                         ('paceCharsPerSec', pace)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"{path}: voice '{name}' declares {label} {value!r}, which is not a "
                'positive number of characters per second.')
    if not (float(lo) < float(pace) < float(hi)):
        raise ValueError(
            f"{path}: voice '{name}' declares minCharsPerSec {lo}, paceCharsPerSec "
            f'{pace}, maxCharsPerSec {hi} out of order; the band is min < pace < max.')
    return (float(hi), float(lo), float(pace))


def load_voice(name: str, path: str = None, **engine_defaults) -> ClipsVoice:
    """One voice by name, or a refusal listing the ones the document has.

    `engine_defaults` are forwarded to load_voices - each engine passes its own
    control-token allowlist, reference cap and placeholder chunk size."""
    path = path or voices_path()
    voices = load_voices(path, **engine_defaults)
    if name not in voices:
        raise ValueError(
            f"Higgs voice '{name}' is not in {path}. It carries: "
            f"{', '.join(sorted(voices))}. Refusing to render in the model's own "
            'default narrator.')
    return voices[name]


def higgs_config_from_worker_kwargs(voice=None, model_dir=None, base_dir=None,
                                    adapter_dir=None, caps=None, **extra):
    """Build a HiggsConfig from the keywords `narrator.serve` hands an engine.

    The serve worker's load message is Orpheus-shaped (voice / modelDir /
    baseDir / adapterDir / caps). This maps what Higgs understands and REFUSES
    the rest by name rather than ignoring it - a baseDir that quietly does
    nothing is a load that appears to have applied something it did not.

      voice       the voice NAME, looked up in the NARRATOR_HIGGS_VOICES document
      model_dir   the Higgs snapshot dir or repo id (default: MODEL_ID)
      adapter_dir REFUSED, as on v3: a fine-tuned voice is a merged CHECKPOINT
                  declared in the voice document, never an adapter directory
                  arriving on a load message
      base_dir    REFUSED - Higgs has no base+adapter split of Orpheus's kind
      caps        REFUSED - the Orpheus cap names (eosBoost, eosFloor, ...) have
                  no meaning here, and accepting them would suggest they do
    """
    if base_dir:
        raise ValueError(
            f'Higgs load carried baseDir={base_dir!r}. Higgs has no shared-base + '
            'per-voice-adapter split: the voice is reference clips, and a fine-tune '
            'is a PEFT adapter over the one model. Pass adapterDir alone.')
    if caps:
        raise ValueError(
            f'Higgs load carried caps={sorted(caps)}. Those are ORPHEUS tuning caps '
            '(eosBoost / eosFloor / maxCharsPerSec ...) and Higgs implements none of '
            'them - it needs no EOS help at all. Refusing a payload that would look '
            'applied and do nothing.')
    if extra:
        raise ValueError(
            f'Higgs load carried unknown keys: {sorted(extra)}.')
    if not (voice or '').strip():
        raise ValueError(
            'Higgs load has no voice name. The name selects an entry in the '
            f'{VOICES_ENV} document; there is no default.')

    if adapter_dir:
        raise ValueError(
            f'Higgs load carried adapterDir={adapter_dir!r}. A fine-tuned voice is '
            'a merged checkpoint declared in the NARRATOR_HIGGS_VOICES document '
            '({"kind": "checkpoint", "checkpointDir": ...}), not an adapter '
            'directory on a load message.')
    clips_voice = load_voice(voice.strip(),
                             allowed_controls=HiggsDefaults.ALLOWED_CONTROLS,
                             max_reference_seconds=HiggsDefaults.MAX_REFERENCE_SECONDS,
                             placeholder_max_chars=HiggsDefaults.MAX_CHARS)
    return HiggsConfig(voice=clips_voice, model_id=model_dir or MODEL_ID)
