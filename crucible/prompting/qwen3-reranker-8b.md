# Qwen3 Reranker 8B: how to use it

Crucible id `qwen3-reranker-8b` · upstream `Qwen/Qwen3-Reranker-8B` @ `77d193c791ed757ca307ee72715aa132723da912` (PC serves Mungert's bf16 GGUF @ `fb4e6db0`) · verb `rerank` (`POST /v1/rerank`, `POST /v1/openai/rerank`).

## Sources

- [S1] OFFICIAL. Model card, https://huggingface.co/Qwen/Qwen3-Reranker-8B/blob/77d193c791ed757ca307ee72715aa132723da912/README.md (read 2026-10-15)
- [S2] OFFICIAL. Eval code, https://github.com/QwenLM/Qwen3-Embedding @ `44548aa5`: `evaluation/qwen3_reranker_model.py` (read 2026-10-15)
- [S3] OFFICIAL. Per-task instructions, same repo, `evaluation/task_prompts.json` (shared with the embedding eval)
- [C1] COMMUNITY. VooDisss gist on llama-server and Qwen3-Reranker GGUFs, https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee (2026-03-09)
- [K1] Crucible @ `5a579e7`: `crucible/models/qwen3-reranker-8b.toml`, `crucible/rerank.py`, `docs/RETRIEVAL.md`

## What the model expects

The prompt is fixed, and only the instruction, query and document vary [S1]:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

- The score is P("yes") over {yes, no} at the last position (log_softmax, then exp) [S1]. Crucible writes these exact bytes and scores them the same way [K1].
- Default instruction: "Given a web search query, retrieve relevant passages that answer the query" [S1]. Crucible uses this default when `instruction` is omitted [K1].
- The card's sentence-transformers example gives a custom-instruction pattern: "Classify whether the document matches the query topic" [S1].

## Writing the instruction / input

- DO set `instruction` to say what *relevant* means for your task. The card's tip claims 1–5% retrieval gain over leaving it off and asks developers to customise it per scenario, task and language [S1].
- DO write it in English. The card's note on multilingual use applies to both the embedding and reranking models [S1].
- DO phrase it the way the official retrieval instructions are phrased: *Given a {query kind}, retrieve {document kind} that {relation}* [S3]. One example is "Given a claim, find documents that refute the claim" (ArguAna). The relation you name is what the yes/no judgment checks against.
- DO keep the query as the user's own words. The instruction carries the task and the query carries the need [S1 format].
- DON'T put the task description into the query or the document. There is a slot for it, and the model was trained with that layout [S1].
- DON'T compare scores across models, or carry a cutoff from one model to another [K1].

## Settings that matter (Crucible params)

| Crucible param | Model fact |
| --- | --- |
| `query` | `<Query>:` slot |
| `documents` (≤ 256) | One `<Document>:` per pass. Each is judged alone, so several can score high [S1, K1] |
| `instruction` | `<Instruct>:` slot. Defaults to the card's web-search instruction |
| `model` | Naming a decide model (for example `qwen3.5-9b`) switches to Crucible's general template, not this model's [K1] |

The card's code uses 8192 as max_length [S1]. Crucible allows 8192 tokens for the query plus one document [K1].

## Community findings

- If you use llama-server's own `/v1/rerank`, community GGUFs that lack the `cls.output.weight` tensor, `pooling_type=RANK` and the rerank template give near-zero scores [C1]. Crucible does not use that route. It scores yes/no logits on its own rendered prompt [K1], so this failure does not apply.

## Measured in Crucible

- **Tags against a long description is a poor fit** (Content Studio, Mac, 2026-10-10). Good tags averaged 0.68 and junk 0.64. A decide model (`qwen3.5-9b`) through the same route separated them: good 0.87, junk 0.50 [K1 `docs/RETRIEVAL.md`].
- Cost: on the Mac, a ~2,500-token query with 1 document took 3.8 s and with 20 documents 6.9 s. An unheld call on the PC waited 27–29 s for a reload [K1].

## Conflicts and open questions

- **The official sources disagree on separators.** The card joins `<Instruct>`, `<Query>` and `<Document>` with one `\n` [S1]. The vLLM eval code joins them with `\n\n` inside chat messages [S2]. Crucible follows the card. Which one the published scores used is not stated.
- **The default instructions differ too.** The eval class defaults to "Given the user query, retrieval the relevant passages" [S2], while the card uses the web-search one [S1].
- Open: would a classification-style instruction (the card's "Classify whether the document matches the query topic" pattern [S1]) have rescued the tag use case Crucible measured? That was not tested, and RETRIEVAL.md's "a phrase per label" advice is also unmeasured [K1].
