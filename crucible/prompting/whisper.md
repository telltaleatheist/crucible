# Whisper (large-v3-turbo, tiny): how to use it

Crucible ids `whisper-large-v3-turbo` (cuda-linux: `dropbox-dash/faster-whisper-large-v3-turbo` @ `0a363e91`; Mac: `mlx-community/whisper-large-v3-turbo` @ `a4aaeec0`) and `whisper-tiny` (`Systran/faster-whisper-tiny` @ `d90ca5fe`; `mlx-community/whisper-tiny-mlx` @ `6caf9c55`). The original weights are `openai/whisper-large-v3-turbo` @ `41f01f3f` and `openai/whisper-tiny` @ `169d4a43`. Job type `asr`.

> **Being retired.** Owen ruled on 2026-09-24 to switch transcription fully to Qwen3-ASR ("it tries to remove the ums and uhs, and those are important sometimes"). Whisper manifests stay until the apps switch [M1]. New work should use `qwen3-asr.md`.

## Sources

- [S1] OFFICIAL. OpenAI Cookbook, "Whisper prompting guide", https://developers.openai.com/cookbook/examples/whisper_prompting_guide (marked archived; undated; read 2026-10-15)
- [S2] OFFICIAL. Model card `openai/whisper-large-v3-turbo` @ `41f01f3f` (read 2026-10-15)
- [S3] OFFICIAL. Turbo announcement, https://github.com/openai/whisper/discussions/2363 (2024-10-01)
- [S4] OFFICIAL (engine). faster-whisper v1.2.1 `faster_whisper/transcribe.py` docstrings and defaults, https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py
- [C1] COMMUNITY. "Prompt sometimes appearing in the transcript", https://github.com/openai/whisper/discussions/1150 (2023-03-26 onward, unanswered)
- [C2] COMMUNITY. "Repetitions and Hallucinations when using prompt feature", https://github.com/openai/whisper/discussions/1992 (2024-02-01 onward, unanswered)
- [C3] COMMUNITY. "Hallucination on audio with no speech", https://github.com/openai/whisper/discussions/1606
- [K1] Crucible @ `5a579e7`: `docs/API.md` "asr", `docs/internals/asr-and-align.md` "Whisper runs", `crucible/jobs/asr/worker.py`, `mlx_worker.py`
- [M1] BookForge memory `asr-verbatim-whisper-bad-qwen3-asr.md`

## What the model expects

- The prompt is treated as **the transcript so far**. Whisper copies its *style* and does not follow instructions in it: "it follows the style of the prompt, rather than any instructions contained within" [S1].
- "If the prompt is longer than 224 tokens, only the final 224 tokens of the prompt will be considered" [S1]. Crucible counts tokens and refuses a prompt over the cap, rather than letting the head be dropped silently [K1].
- "The prompt will not override the model's comprehension of the audio" [S1].
- In faster-whisper the `initial_prompt` applies to the **first 30 s window**. After that, `condition_on_previous_text=True` (the default) feeds the model's own output forward, and the prompt is reset when the temperature fallback goes above 0.5 [S4].
- The turbo model was fine-tuned on transcription only, "excluding translation data, on which we don't expect turbo to perform well" [S3]. It is close to large-v2 overall, with "larger degradation on some languages like Thai and Cantonese" [S3].
- Language is auto-detected unless given. Give it when it is known in advance [S2].

## Writing the prompt

- DO put correctly spelled names, products and jargon in the prompt [S1].
- DO write it in the punctuation and capitalisation you want back, as natural text. Long prompts steer style more reliably than short ones, and rare or odd styles are followed less often [S1].
- DON'T write instructions ("don't hallucinate", "include ums"). They are ignored as instructions [S1], and their text can leak into the output [C1].
- DON'T expect a prompt to fix silence hallucination. A prompt adds text the decoder can repeat [C2, C3]. Remove silence instead: Crucible's `speech_only` or faster-whisper's `vad_filter` [K1].
- DON'T use turbo for translation [S3]. Crucible does not expose `task` anyway.

## Settings that matter (Crucible params)

| Crucible param | Engine fact |
| --- | --- |
| `initial_prompt` (Whisper only; not blank; within the 224-token cap) | Sent to every 900 s window, and inside each window it primes only the first 30 s [K1, S4] |
| `language` (an ISO code, or `auto`) | `auto` takes the first window's detection as the document's language [K1] |
| `vad_filter` (faster-whisper only) / `speech_only` | Remove silence before decoding |
| `word_timestamps` | Whisper's own word timing |
| (not exposed) `beam_size` 5, temperature fallback 0.0–1.0, `condition_on_previous_text` True, `hallucination_silence_threshold` None | faster-whisper defaults [S4] |

## Community findings

- The prompt sometimes appears verbatim at the start of the transcript, replacing the first 30 s. Suggested mitigations are a later bugfix (PR #1903) and `hallucination_silence_threshold=2` [C1].
- Prompts with technical terms caused repetition and hallucination for several users. Mitigations were 2–3 minute chunks, post-filtering repeats, and some users going back to large-v2 [C2].
- Silence makes Whisper produce subtitle-like phantom text. Don't send silent chunks, and filter on `no_speech_prob` / `avg_logprob` / `compression_ratio` [C3]. Crucible does not publish those diagnostics [K1].

## Measured in Crucible

- Decoding through PyAV truncated an 18-hour m4b to 6 hours, which is why Crucible decodes with ffmpeg [K1].
- Whisper-turbo caught 76 fillers in total on a 3.9 h stream, against about 104 ums and uhs from Qwen3-ASR [M1].

## Conflicts and open questions

- **Crucible's docs say the prompt "goes to every window"**, which is true per 900 s call [K1]. Per [S4], though, it shapes only the first 30 s of each call, then scrolls out. The effect on an hour of audio is therefore about 4 × 30 s of priming. That is a property of the model, not a Crucible bug, but apps should not expect consistent spelling throughout.
- Crucible exposes none of the hallucination knobs the community leans on (`condition_on_previous_text=False`, `hallucination_silence_threshold`) [S4, C1]. With retirement under way, this is probably not worth adding (inference).
