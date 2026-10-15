# Qwen3-ASR (0.6B, 1.7B): how to use it

Crucible ids `qwen3-asr-1.7b`, `qwen3-asr-0.6b` (vLLM on cuda-linux; Qwen's `qwen_asr` on MPS on the Mac), and `qwen3-asr-1.7b-mlx`, `qwen3-asr-0.6b-mlx` (mlx-audio port, Mac only, same weights). Upstream `Qwen/Qwen3-ASR-1.7B` @ `7278e1e70fe206f11671096ffdd38061171dd6e5` and `Qwen/Qwen3-ASR-0.6B` @ `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. Job type `asr`; word times come from `qwen3-aligner`.

## Sources

- [S1] OFFICIAL. Model cards at the pinned revisions. The two are identical apart from the model name. https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md (read 2026-10-15)
- [S2] OFFICIAL. `chat_template.json` at the same revision
- [S3] OFFICIAL. `qwen_asr` package, https://github.com/QwenLM/Qwen3-ASR @ `7c6daf77a2421100f5fb066495372c00129d39ff`: `qwen_asr/inference/qwen3_asr.py` (`_build_messages`, `_build_text_prompt`, `transcribe`) and `qwen_asr/inference/utils.py` (constants, `detect_and_fix_repetitions`)
- [S4] OFFICIAL. Qwen3-ASR Technical Report, arXiv:2601.21337v1 (2026-01-29), https://arxiv.org/html/2601.21337v1
- [C1] COMMUNITY. vLLM PR #43430, "Add hotwords support for model Qwen3-ASR" (closed unmerged, June 2026), https://github.com/vllm-project/vllm/pull/43430
- [C2] COMMUNITY. veltrea, "Qwen3-ASR Context Biasing #1" (2026-06-09), https://note.com/veltrea/n/n7dd0b7ffffe9
- [C3] COMMUNITY. Qwen3-ASR issue #157 (closed as not planned, 2026-07-05), https://github.com/QwenLM/Qwen3-ASR/issues/157
- [K1] Crucible @ `5a579e7`: `docs/API.md` "asr", `docs/internals/asr-and-align.md`, `crucible/jobs/asr/qwen.py`, `crucible/asr/*.toml`, `docs/CHANGELOG.md`
- [M1] BookForge memory `asr-verbatim-whisper-bad-qwen3-asr.md` (Owen's ruling 2026-09-24; ContentStudio measurements)

## What the model expects

- **Prompt** [S2, S3]: a system turn holding the *context* text (empty if none), a user turn holding only the audio, then the assistant turn. The output is `language {Name}<asr_text>{transcript}`. If you pass a language, `qwen_asr` appends `language {Name}<asr_text>` to the prompt so that only the text is generated [S3]. Over audio with no speech, the model emits `language None<asr_text>` [S4].
- **What context is**: "the model learns to utilize the context tokens inside the system prompt as background knowledge, allowing user to obtain customized ASR results" [S4].
- **Not an instruction channel**: "we train the model to be an ASR-only model that does not follow natural-language instructions in the prompt" [S4].
- **Languages**: 30 languages plus 22 Chinese dialects, with language ID. The audio can be speech, singing, or songs with background music [S1].
- **Length**: `MAX_ASR_INPUT_SECONDS = 1200` (20 min) per inference. With timestamps, `qwen_asr` splits at `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`. Minimum input is 0.5 s [S3]. The card's `max_new_tokens` defaults to 256 with "Set a larger value for long audio input", and to 4096 in the vLLM example [S1].
- The official evaluation used greedy decoding, bf16, `max_new_tokens=1024`, and "none of the tests specified a language parameter" [S1].

## Writing the context

- DO put **vocabulary** in `context`: names, jargon, the correct spellings of terms you expect. A plain space-separated list works ("hotword1 hotword2 hotword3" is the vLLM PR's format [C1]). Background prose also counts as context [S4].
- DON'T rely on `context` as an instruction ("transcribe verbatim", "keep the ums"). The report says the model was trained not to follow instructions [S4]. See Conflicts for what Crucible measured.
- DON'T send a huge glossary without testing. A ~300-term glossary fixed many technical homophones, but sometimes the model "created sentences that were not in the input", pulled by the glossary [C2].
- DON'T send context with long stretches of silence. Over near-silence the model can recite the context word for word. Crucible detects this as `context_echo` and drops the piece [K1].

## Settings that matter (Crucible params)

| Crucible param | Model fact |
| --- | --- |
| `context` (≤ 8192 chars, ≤ 1,024 tokens, control tokens refused) | The system-turn text [S2, S3] |
| `language` (required; en de fr es it pt ru ja ko zh yue) | Forces `language {Name}<asr_text>`. Crucible limits it to the aligner's 11 languages and refuses `auto` [K1]. The model itself does 30 languages plus LID [S1] |
| `piece_s` 5–180 (default 30), `overlap_s` 0–5 | Crucible's own cut. Official cuts at 1200 s, or 180 s with timestamps [S3] |
| `speech_only` (on by default) | Drops silence first. This counters the `language None` output and context echo |
| `word_timestamps` | Runs Qwen3-ForcedAligner, which officially handles up to 5 minutes and 11 languages [S1] |
| (fixed) greedy decoding, `max_new_tokens` 4096 per 180 s scaled | Matches official greedy decoding [S1]. Crucible does not apply `detect_and_fix_repetitions`. Its loop guard re-cuts instead [K1] |

## Community findings

- Hotwords go in the system prompt as a space-separated list. The upstream vLLM PR adding this was closed unmerged [C1]. In issue #157 a community reply points to `/v1/chat/completions`, with no format given [C3].
- Glossary biasing fixes homophones but cannot overcome very close pronunciations, and it occasionally hallucinates glossary-driven sentences [C2].

## Measured in Crucible

- **Piece length.** At 30 s, the first word at sample zero was skipped on 327 of 6,532 cues; 1.5 s of lead-in fixed it. 120 s pieces repeated about 20% and 180 s pieces looped (antirez's port). One uncut 600 s decode found zero fillers [K1, M1].
- **Context echo.** 55 s at −41.7 dB decoded as the filler instruction word for word (ContentStudio job 7f3459cf) [K1].
- **Filler context.** ContentStudio's filler prompt took a 10-minute window from 9 fillers to 19 [K1]. Filler counts are fragile: ±30% from sample rounding or cut points [M1].
- **Loops.** A 180 s piece repeated one line about 60 times, and the aligner stamped 2,388 words at one instant [K1, M1].

## Conflicts and open questions

- **Instruction vs background knowledge.** Crucible's docs call `context` "the instruction and vocabulary", and the filler prompt it measured is an instruction ("Verbatim transcript. Transcribe every disfluency…") [K1]. The report says the model does not follow natural-language instructions [S4]. The 9→19 gain may come from the prompt *priming* filler-style text rather than being obeyed, or it may be within the ±30% noise [M1]. To test, compare against a context that is only a list of filler words ("um uh er hmm like") on the same pieces.
- **Language is required even without timestamps** [K1], while the model does its own language ID across 30 languages [S1]. This was noted as an open question at build time [M1].
- **Piece length.** Crucible's 30 s default sits far below the official 1200 s ceiling [S3]. That is deliberate and measured, but it means each piece hears much less context than the model was built to use.
