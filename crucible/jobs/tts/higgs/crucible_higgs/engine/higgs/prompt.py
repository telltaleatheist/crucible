"""The Higgs chat history: reference clips ARE the voice.

Ported from the measured harness
`E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\render_v2.py`
(`build_conv`, 2026-09-04) - the run that produced the deathstalker audition,
not the model card. The template it renders is recorded verbatim in that
campaign's narrator's Higgs notes section A and is reproduced in this module's
constants so a test can pin it without the model.

WHAT THE CONVERSATION LOOKS LIKE (2 reference clips, one target chunk):

    system     "Generate audio following instruction."
    scene      "Audio is recorded from a quiet room."
    user       <REF-1 TRANSCRIPT>
    assistant  {"type": "audio", "url": <REF-1 WAV>}
    user       <REF-2 TRANSCRIPT>
    assistant  {"type": "audio", "url": <REF-2 WAV>}
    user       <TARGET TEXT>
    (+ add_generation_prompt -> a trailing assistant turn opened with
       <|audio_out_bos|>)

`scene` is a FIRST-CLASS ROLE in Higgs v2, not a control token: the processor
renders it into the system block between `<|scene_desc_start|>` (128018) and
`<|scene_desc_end|>` (128019). Higgs v3 has NO scene-description token at all,
which is why the scene lives on the voice and not on this module.

THE REFERENCE AUDIO IS NOT TEXT. `<|AUDIO_OUT|>` (128016) is a placeholder in
the text stream marking where the processor splices in a PARALLEL audio-token
stream; the clip is encoded by the audio tokenizer into `audio_input_ids`
(shape (batch, frames, 8)). 732 frames == 28.5 s of reference in the measured
run, so reference length is paid for in context.

Nothing here imports torch or transformers: it builds the plain list of dicts
`processor.apply_chat_template` takes.
"""
from ...text.sml import SML_UNSPOKEN_PATTERN
from ..protocol import ClipsVoice

# render_v2.py's system and scene text, verbatim. They are DEFAULTS, not law -
# a voice may carry its own scene - but they are what the audition measured.
DEFAULT_SYSTEM_PROMPT = 'Generate audio following instruction.'
DEFAULT_SCENE = 'Audio is recorded from a quiet room.'

# Special ids, measured from the v2 tokenizer (narrator's Higgs notes section A). Kept
# here so a prompt test can assert the framing without loading a tokenizer.
SPECIAL_TOKEN_IDS = {
    '<|begin_of_text|>': 128000,
    '<|end_of_text|>': 128001,
    '<|eot_id|>': 128009,
    '<|audio_bos|>': 128011,
    '<|audio_eos|>': 128012,
    '<|audio_out_bos|>': 128013,
    '<|AUDIO|>': 128015,
    '<|AUDIO_OUT|>': 128016,
    '<|scene_desc_start|>': 128018,
    '<|scene_desc_end|>': 128019,
}

# The per-codebook stream sentinels. These live in the AUDIO vocabulary (1026
# wide: 1024 codes + these two), not the text one, and are what the codec's
# content trim looks for. Same in v2 and v3.
AUDIO_STREAM_BOS_ID = 1024
AUDIO_STREAM_EOS_ID = 1025
NUM_REAL_CODES = 1024
NUM_CODEBOOKS = 8


def build_conversation(text: str, voice: ClipsVoice,
                       system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                       scene: str = DEFAULT_SCENE) -> list:
    """The chat history for one chunk, exactly as render_v2.py built it.

    `voice.scene` beats the `scene` argument when it is set, so a voice can
    describe its own recording space; passing scene=None (and a voice with
    none) drops the scene turn entirely.

    Raises rather than rendering something plausible: an empty chunk, a voice
    that is not a ClipsVoice, or a clip without a transcript are all conditions
    where continuing produces audio that SOUNDS fine and is wrong.
    """
    if not (text or '').strip():
        raise ValueError('Higgs build_conversation: the chunk has no text')
    if not isinstance(voice, ClipsVoice):
        raise ValueError(
            'Higgs renders a voice from reference clips: build_conversation needs '
            f'a ClipsVoice, got {type(voice).__name__}. (A voice TOKEN is Orpheus; '
            'Higgs has no fine-tuned token vocabulary.)')

    conversation = [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
    ]
    scene_text = voice.scene if voice.scene is not None else scene
    if scene_text:
        conversation.append(
            {'role': 'scene', 'content': [{'type': 'text', 'text': scene_text}]})
    for clip in voice.clips:
        # The clip's transcript is BOOK-EXACT text (ReferenceClip enforces that
        # it exists; the doctrine that it is never an ASR guess is in
        # protocol.ReferenceClip's docstring).
        conversation.append(
            {'role': 'user', 'content': [{'type': 'text', 'text': clip.transcript}]})
        conversation.append(
            {'role': 'assistant', 'content': [{'type': 'audio', 'url': clip.path}]})
    conversation.append({'role': 'user', 'content': [{'type': 'text', 'text': text}]})
    return conversation


def clean_text(sentence: str) -> str:
    """Strip the SML markers no TTS model speaks, and nothing else.

    `[break]` / `[pause]` / `[heading]` / `[item]` / `[music]` / `[sfx]` /
    `[silence]` are narrator's OWN markup - they ride in the manifest's chunk
    text, drive the packer's boundaries, and are stripped at the model boundary.
    Same rule for every engine, so this is the same regex Orpheus uses
    (`SML_UNSPOKEN_PATTERN`, ported from e2a conf_models.py). It lives in the
    Orpheus port's text module today because that is where it landed; it belongs
    in `narrator/text/` once that exists, and this import is the one place it
    would have to change.

    THE ENGINE READS THE TEXT AS PRINTED otherwise: no number expansion, no
    scripture rewriting. Number normalization is a BookForge model pass over the
    narration copy and is not repeated here (Owen, 2026-09-02, permanently).
    Higgs v3 additionally REFUSES an unknown `<|...|>` control token before it
    reaches the model - see v3_served.validate_control_tokens - because an
    unknown one is read aloud rather than ignored.
    """
    sentence = (sentence or '').strip()
    return SML_UNSPOKEN_PATTERN.sub('', sentence).strip()
