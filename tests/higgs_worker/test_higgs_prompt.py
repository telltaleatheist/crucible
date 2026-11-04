"""The Higgs chat history, pinned against the template that was MEASURED.

`E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\HIGGS_NOTES.md`
section A records the string the processor rendered for the deathstalker
audition, verbatim, together with the special-token ids. This module asserts
that `crucible_higgs.engine.higgs.prompt` builds the conversation that renders to it -
without a tokenizer, a processor or a model, so it runs on the Windows test
interpreter.

WHY THIS IS WORTH A TEST AT ALL. A voice clone is conditioned on the reference
turns; get the ROLE ORDER wrong (transcript and clip swapped, target text before
the references, the scene as a user turn) and the model still renders fluent
audio - in a voice that is not the one asked for, at a quality nobody can
attribute to a cause. There is no exception and no warning. The template is the
only thing standing between a correct clone and a plausible one.
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs.prompt import (AUDIO_STREAM_BOS_ID,       # noqa: E402
                                          AUDIO_STREAM_EOS_ID,
                                          DEFAULT_SCENE,
                                          DEFAULT_SYSTEM_PROMPT,
                                          NUM_CODEBOOKS, NUM_REAL_CODES,
                                          SPECIAL_TOKEN_IDS,
                                          build_conversation)
from crucible_higgs.engine.protocol import (ClipsVoice, ReferenceClip,     # noqa: E402
                                      TokenVoice)

# The two deathstalker clips render_v2.py used, with their book-exact
# transcripts (from metadata_train.csv of the ds_ad4s corpus).
REF1_TEXT = ('The purpose here is not sectarian, but genuinely ecumenical, since '
             'important insights ought never to be limited to the group from which '
             'they arise. What we are given, accordingly, is an example of the '
             'Catholicity of sharing.')
REF2_TEXT = ('They experienced Jesus as the defining reality of their lives. They '
             'possessed a flaming vision of God that blinded them to all competing '
             'loyalties. They experienced life built on the Rock.')
TARGET = ('It was a Saturday morning. I must admit that though I am usually an early '
          'riser, I am not what one would consider a morning person.')


def deathstalker(**kwargs):
    return ClipsVoice(
        clips=(ReferenceClip('/home/<user>/xtts_ft/ds_ad4s/wavs/cd_cd_00000008.wav',
                             REF1_TEXT, seconds=14.02),
               ReferenceClip('/home/<user>/xtts_ft/ds_ad4s/wavs/cd_cd_00000035.wav',
                             REF2_TEXT, seconds=14.49)),
        name='deathstalker', **kwargs)


class ChatHistoryTest(unittest.TestCase):

    def test_the_measured_two_clip_conversation(self):
        """render_v2.py's build_conv, turn for turn."""
        conversation = build_conversation(TARGET, deathstalker())
        self.assertEqual([turn['role'] for turn in conversation],
                         ['system', 'scene',
                          'user', 'assistant',
                          'user', 'assistant',
                          'user'])
        self.assertEqual(conversation[0]['content'],
                         [{'type': 'text', 'text': DEFAULT_SYSTEM_PROMPT}])
        self.assertEqual(conversation[1]['content'],
                         [{'type': 'text', 'text': DEFAULT_SCENE}])
        self.assertEqual(conversation[2]['content'],
                         [{'type': 'text', 'text': REF1_TEXT}])
        self.assertEqual(conversation[3]['content'],
                         [{'type': 'audio',
                           'url': '/home/<user>/xtts_ft/ds_ad4s/wavs/'
                                  'cd_cd_00000008.wav'}])
        self.assertEqual(conversation[4]['content'],
                         [{'type': 'text', 'text': REF2_TEXT}])
        self.assertEqual(conversation[6]['content'],
                         [{'type': 'text', 'text': TARGET}])

    def test_the_target_text_is_always_the_last_turn(self):
        """add_generation_prompt opens the assistant turn AFTER it; a target
        placed anywhere else is conditioning, not a request."""
        for clips in (1, 2, 3):
            voice = ClipsVoice(clips=tuple(
                ReferenceClip(f'/tmp/ref{i}.wav', f'Transcript {i}.')
                for i in range(clips)), name='v')
            with self.subTest(clips=clips):
                conversation = build_conversation(TARGET, voice)
                self.assertEqual(conversation[-1]['role'], 'user')
                self.assertEqual(conversation[-1]['content'][0]['text'], TARGET)
                self.assertEqual(len(conversation), 2 + 2 * clips + 1)

    def test_the_transcript_precedes_its_clip(self):
        """user: what was said, THEN assistant: the audio of it. Swapped, the
        model is being taught that the audio came first."""
        conversation = build_conversation(TARGET, deathstalker())
        pairs = [(conversation[i], conversation[i + 1]) for i in (2, 4)]
        for text_turn, audio_turn in pairs:
            self.assertEqual(text_turn['role'], 'user')
            self.assertEqual(text_turn['content'][0]['type'], 'text')
            self.assertEqual(audio_turn['role'], 'assistant')
            self.assertEqual(audio_turn['content'][0]['type'], 'audio')

    def test_a_voice_carries_its_own_scene(self):
        conversation = build_conversation(
            TARGET, deathstalker(scene='Recorded in a small studio.'))
        self.assertEqual(conversation[1]['role'], 'scene')
        self.assertEqual(conversation[1]['content'][0]['text'],
                         'Recorded in a small studio.')

    def test_no_scene_means_no_scene_turn(self):
        conversation = build_conversation(TARGET, deathstalker(), scene=None)
        self.assertEqual([turn['role'] for turn in conversation][:2],
                         ['system', 'user'])

    def test_scene_is_a_ROLE_not_a_control_token(self):
        """`<|scene_desc_start|>` is a v2 CHAT ROLE the processor renders into
        the system block. It is not something to paste into the text - and it
        does not exist at all in v3, where an unknown control token is read
        aloud as words."""
        conversation = build_conversation(TARGET, deathstalker())
        rendered = repr(conversation)
        self.assertNotIn('<|scene_desc_start|>', rendered)
        self.assertIn("'role': 'scene'", rendered)


class RefusalTest(unittest.TestCase):

    def test_an_empty_chunk_is_refused(self):
        for text in ('', '   ', None):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    build_conversation(text, deathstalker())

    def test_a_voice_token_is_not_a_higgs_voice(self):
        with self.assertRaises(ValueError) as caught:
            build_conversation(TARGET, TokenVoice('deathstalker'))
        self.assertIn('reference clips', str(caught.exception))

    def test_a_clip_without_a_transcript_is_refused_at_construction(self):
        """The law from protocol.ReferenceClip: a clone conditioned on a missing
        or invented transcript is a whole book in a subtly wrong voice, reported
        as success."""
        for transcript in ('', '   ', None):
            with self.subTest(transcript=transcript):
                with self.assertRaises(ValueError) as caught:
                    ReferenceClip('/tmp/ref.wav', transcript)
                self.assertIn('transcript', str(caught.exception))

    def test_a_voice_with_no_clips_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            ClipsVoice(clips=(), name='nobody')
        self.assertIn('at least one reference clip', str(caught.exception))

    def test_a_bare_path_is_not_a_reference_clip(self):
        with self.assertRaises(ValueError) as caught:
            ClipsVoice(clips=('/tmp/ref.wav',), name='v')
        self.assertIn('transcript is part of the voice', str(caught.exception))


class TokenIdTest(unittest.TestCase):
    """The ids as probed off the v2 tokenizer (HIGGS_NOTES.md section A). They
    are here so a v3 port, a fine-tune's data builder or a vLLM path can be
    diffed against ONE recorded set instead of re-probing a model."""

    def test_the_text_stream_specials(self):
        self.assertEqual(SPECIAL_TOKEN_IDS, {
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
        })

    def test_the_audio_stream_sentinels_are_a_separate_vocabulary(self):
        """1024/1025 are PER-CODEBOOK values in the audio stream, not text-vocab
        ids - which is why the codec's trim looks for a row of eight of them and
        not for a scalar EOS token."""
        self.assertEqual((AUDIO_STREAM_BOS_ID, AUDIO_STREAM_EOS_ID), (1024, 1025))
        self.assertEqual(NUM_REAL_CODES, 1024)
        self.assertEqual(NUM_CODEBOOKS, 8)
        self.assertLess(AUDIO_STREAM_EOS_ID, min(SPECIAL_TOKEN_IDS.values()))


if __name__ == '__main__':
    unittest.main()
