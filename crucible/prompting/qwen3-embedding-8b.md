# Qwen3 Embedding 8B: how to use it

Crucible id `qwen3-embedding-8b` · upstream `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (PC serves CompendiumLabs' bf16 GGUF of it @ `ac37281f`) · verb `embed` (`POST /v1/embed`, `POST /v1/openai/embeddings`).

## Sources

- [S1] OFFICIAL. Model card, https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/README.md (read 2026-10-15)
- [S2] OFFICIAL. `config_sentence_transformers.json` at the same revision (read 2026-10-15)
- [S3] OFFICIAL. Eval code and per-task instructions, https://github.com/QwenLM/Qwen3-Embedding @ `44548aa5f0a0aed1c76d64e19afe47727a325b8f`: `evaluation/task_prompts.json` (251 tasks), `evaluation/qwen3_embedding_model.py` (`get_instruction`), `evaluation/run_mteb.sh` (read 2026-10-15)
- [S4] OFFICIAL. Paper, "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models", arXiv:2506.05176 (cited by S1; not re-read in full)
- [C1] COMMUNITY. HF discussion "how do I pass instructions to it in llama.cpp?", https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/42 (answer by a community member, 2025-10-27)
- [C2] COMMUNITY. Issue trackers of apps that dropped the query prefix: https://github.com/garrytan/gbrain/issues/5543, https://github.com/tobi/qmd/issues/765 (read 2026-10-15)
- [C3] COMMUNITY. "Qwen3-Embedding-8B produces irrelevant results", https://github.com/QwenLM/Qwen3-Embedding/issues/97 (opened 2025-07-08, still open, no maintainer answer)
- [K1] Crucible @ `5a579e7` (2026-10-13): `crucible/models/qwen3-embedding-8b.toml`, `crucible/embed.py`, `docs/RETRIEVAL.md`, `docs/CHANGELOG.md`

## What the model expects

- **Queries** use `Instruct: {task}\nQuery:{text}`. There is no space after `Query:` in the card's `get_detailed_instruct` [S1], in the sentence-transformers `query` prompt [S2], or in the eval template `Instruct: {}\nQuery:` [S3]. Crucible writes exactly this, plus `<|endoftext|>`, and takes the pooled last token [K1].
- **Retrieval documents take no instruction.** The card says "No need to add instruction for retrieval documents" [S1], and the ST `document` prompt is `""` [S2].
- **Symmetric tasks put the instruction on every text.** In the official eval code, STS and pair-classification tasks use "Retrieve semantically similar text", bitext mining uses "Retrieve parallel sentences", and classification and clustering use their task instruction. These go through the same template on every input, not just one side [S3 `get_instruction`]. Only asymmetric retrieval tasks give the document side an empty instruction. ArguAna gives the same instruction to both queries and passages [S3].
- The card describes the instruction as "a one-sentence instruction that describes the task" [S1].
- Default instruction: "Given a web search query, retrieve relevant passages that answer the query" [S1, S2]. Crucible uses the same default [K1].
- MRL: output dimensions are 32 to 4096 [S1]. Context is 32k [S1]. Crucible serves 8192 tokens per input [K1].

## Writing the instruction / input

- DO write a task-specific instruction for queries. The card puts the gain at 1–5% on most downstream tasks and recommends tailoring it to "specific scenarios, tasks, and languages" [S1].
- DO write the instruction in **English**, even for non-English text. Qwen: "we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English" [S1].
- DO follow the shape of the official instructions: *Given a {kind of query}, retrieve {kind of passage} that {relation}*. Official examples [S3]:
  - "Given a question, retrieve Wikipedia passages that answer the question" (NQ)
  - "Given a claim, retrieve documents that support or refute the claim" (FEVER)
  - "Given a financial question, retrieve user replies that best answer the question" (FiQA)
  - "Given a question, retrieve questions that are semantically equivalent to the given question" (Quora)
  - "Given a online banking query, find the corresponding intents" (Banking77, classification)
  - "Identify the topic or theme of the given news articles" (clustering)
  - "Retrieve semantically similar text" (STS)
- DO, for symmetric uses in Crucible (dedup, clustering, classification, similar-passage lookup), embed **both sides** as `input_type: "query"` with the same `instruction`. This matches what the official eval does [S3]. Crucible refuses an instruction on a `document` (`instruction_not_taken`) [K1], so `query` on both sides is how to do it.
- DON'T write `Instruct:`/`Query:` prefixes into the text yourself. Crucible adds them [K1]. The common failure in the community is the opposite one: the prefix is silently missing from queries [C2].
- DON'T mix `dimensions` between a corpus and its queries [K1].

## Settings that matter (Crucible params)

| Crucible param | Model fact |
| --- | --- |
| `input_type` (`query` / `document`, required) | Selects the query template or the bare document template [S1, S2] |
| `instruction` (query only) | The `{task}` slot. Defaults to the web-search instruction [S1] |
| `dimensions` 32–4096 | MRL truncation, then re-normalised [S1, K1] |
| `fingerprint` | Crucible-only. PC (GGUF on llama-server) and Mac (safetensors on mlx-lm) write vectors that cannot be compared [K1] |

## Community findings

- On llama.cpp, users report that you need `--embedding --pooling last` and the `Instruct: …\nQuery: …` form on queries only [C1]. Crucible derives both flags from the manifest and renders the tokens itself [K1].
- Several apps that dropped the query prefix saw retrieval worse than smaller models until it was put back [C2].
- One open issue reports irrelevant 8B results through vLLM and LangChain, even with an instruction. No cause was found [C3]. It is not reproduced here.

## Measured in Crucible

- The same text in a different batch on the Mac differs by up to about 1.4e-3 (cosine 0.99989). A model held by nothing reloads about 3 s on the Mac [K1 `docs/RETRIEVAL.md`, CHANGELOG].
- No retrieval-quality measurement of instruction wording exists in Crucible's docs.

## Conflicts and open questions

- **Official sources disagree on one space.** The card's TEI curl example writes `Query: What is…` with a space [S1]. The Python examples, the ST config and the eval template have none [S1–S3]. Crucible follows the no-space form, which is the one used in eval.
- **RETRIEVAL.md says "a document takes none" without mentioning symmetric tasks.** For clustering, dedup or STS-like use, the official practice is an instruction on both sides [S3]. That is possible today by sending both sides as `query`, but the docs do not say so (inference).
- Whether an instruction in the text's own language would beat an English one for non-English corpora has not been measured. The card says to use English [S1].
