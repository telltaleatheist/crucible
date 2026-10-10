# Embed and rerank: for app authors

Two verbs for search by meaning. **`embed`** turns texts into vectors you store and compare
yourself; **`rerank`** scores a shortlist of passages against one query, with a model that
reads each pair together. The usual shape is both:

1. embed your corpus once and store the vectors (with the fingerprint, below);
2. embed each query and take the nearest ~30–50 by cosine (milliseconds, in your app);
3. rerank that shortlist and keep what scores above your cutoff.

Crucible serves the models and returns numbers. Storing vectors, the nearest-neighbour
search and the cutoff are the app's. The full reference (every field, every refusal) is
[docs/internals/api.md](internals/api.md) "Embed" and "Rerank", and every server serves its
own at `GET /docs`.

## Install the models

The two models are the optional **retrieval** package: Qwen3-Embedding-8B and
Qwen3-Reranker-8B, bf16, about 16 GB each.

```sh
crucible install retrieval     # on the server; refuses by name if the card cannot hold them
```

Where it is not installed, `embed` and a rerank naming no model answer
`409 package_not_installed` with the command to run. A rerank through a decide model you
name (below) needs no package. `GET /v1/models` rows carry `verbs`, `package` and
`package_installed`, and `GET /v1/info` carries `verbs`, so a picker can list what a
server offers.

Every example below sends the protected-route headers: `Authorization: Bearer <token>`,
`X-Crucible-Api: 1`, and your app's client name (`X-Crucible-Client`), which is also how
queue sessions know your requests (docs/QUEUE.md).

## Hold a session for anything more than one call

**When nothing holds a model, the server unloads it after each call.** The next call loads
it again before it runs, and that load is reported inside `timing_ms.queued`. Measured on
the Mac (1.0.133, 2026-10-10): every unheld embed or rerank waited about 3 s (the 16 GB
reloaded from the page cache) and then ran in 0.1–0.3 s; the server log says
`unloaded qwen3-reranker-8b (the resident llm): nothing holds it — the last chat completion
finished` after each one.

So a run of calls (indexing, a search page, a backfill) opens a **queue session** on the
model first; the model stays loaded between its items and nothing else runs in between:

```sh
curl -s -X POST http://127.0.0.1:7100/v1/queue/sessions \
  -H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1" -H "X-Crucible-Client: myapp@host" \
  -H "Content-Type: application/json" \
  -d '{"act": "embed", "model": "qwen3-embedding-8b", "idle_s": 300}'
# {"session_id": "ses-…", "status": "open", "position": null}
# then send each call with -H "X-Crucible-Session: ses-…", and when done:
curl -s -X DELETE http://127.0.0.1:7100/v1/queue/sessions/ses-… -H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1"
```

```ts
const session = await crucible.session({ act: 'embed', model: 'qwen3-embedding-8b' });
try {
  for (const batch of batches) await session.embed({ inputs: batch, inputType: 'document' });
} finally {
  await session.close();
}
```

The two models do not stay loaded together on the PC's 24 GB card, so a client that
alternates embed and rerank reloads each time: batch by verb, one session per verb.

## Embed

```sh
curl -s http://127.0.0.1:7100/v1/embed \
  -H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1" -H "X-Crucible-Client: myapp@host" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["We brought the wheat in early this year.", "The episode opens with the news."],
       "input_type": "document", "dimensions": 1024, "encoding_format": "base64_float16"}'
```

- **`input_type` is required.** `document` for what you store; `query` for what a person
  searches with. A query gets the model's instruction prefix, and you may say the task in
  `instruction` ("Given a podcast question, find the transcript passages that answer it");
  a document takes none (refused `instruction_not_taken`). Crucible writes each model's own
  format; never write prefixes into your text.
- **`dimensions`** (32–4096 for the 8B) keeps the first N numbers and re-normalises
  (Matryoshka): 1024 is a quarter of the storage for a small loss. Use the same
  `dimensions` for a corpus and its queries.
- **`encoding_format`**: `float` (JSON numbers), `base64` (float32, little-endian, OpenAI's),
  or `base64_float16` (half the bytes again; plenty for cosine search). Every vector is unit
  length, so cosine similarity is the dot product.
- **Limits**: 256 inputs a request; each input at most the served context (8192 tokens),
  refused `400 embed_input_too_long` naming it.

Decoding a base64 vector:

```ts
import { decodeEmbedding } from '@crucible/client';
const answer = await crucible.embed({ inputs: texts, inputType: 'document', encodingFormat: 'base64_float16' });
const vectors = answer.embeddings.map((v) => decodeEmbedding(v, answer.encodingFormat)); // Float32Array[]
```

```python
import base64, numpy as np
vec = np.frombuffer(base64.b64decode(item), dtype="<f2").astype(np.float32)  # "<f4" for base64
```

### Store the fingerprint, and send it back

**Vectors from different models, precisions or engines cannot be compared**, and the PC
(llama-server, a GGUF) and the Mac (mlx-lm, the safetensors) write different ones for the
same model id. Every answer names what wrote its vectors in `model.fingerprint`
(`qwen3-embedding-8b@1d8ad4ca9b3d:repo:mlx-lm-0.31.3+mlx-0.32.2:e1` on the Mac). Store it
with the corpus and send it on every later call for that corpus:

```json
{"inputs": ["what did they say about the harvest"], "input_type": "query",
 "dimensions": 1024, "fingerprint": "qwen3-embedding-8b@1d8ad4ca9b3d:repo:mlx-lm-0.31.3+mlx-0.32.2:e1"}
```

A server that would write anything else answers `409 fingerprint_mismatch` before it embeds,
so a corpus can never silently mix vectors. When the fingerprint changes (a new engine
build, another machine), re-embed the corpus.

One measured caveat: **the same text is not bit-identical across batches on the Mac.**
Embedded beside another text it read up to ~1.4e-3 off the same text alone (bf16 over padded
rows, 2026-10-10); alone twice it was identical. That is far below anything cosine search
notices (the cosine stays above 0.9999), and the fingerprint is the same: it names the
model and the engine, not the batch.

### OpenAI-compatible

`POST /v1/openai/embeddings` takes OpenAI's body and answers OpenAI's shape, with what wrote
the vectors under `crucible`. `input_type` defaults to `document`; send
`"input_type": "query"` for a search query.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:7100/v1/openai", api_key=TOKEN,
                default_headers={"X-Crucible-Api": "1", "X-Crucible-Client": "myapp@host"})
r = client.embeddings.create(model="qwen3-embedding-8b", input=["a passage"], dimensions=1024)
```

## Rerank

```sh
curl -s http://127.0.0.1:7100/v1/rerank \
  -H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1" -H "X-Crucible-Client: myapp@host" \
  -H "Content-Type: application/json" \
  -d '{"query": "how to keep texas from turning democrat",
       "documents": ["…stop Texas going blue…", "…the weather in Austin…"],
       "instruction": "Given a search, find the transcript passages about it"}'
```

```json
{"scores": [0.93, 0.002],
 "results": [{"index": 0, "relevance_score": 0.93}, {"index": 1, "relevance_score": 0.002}],
 "tokens": {"per_document": [62, 58], "total": 240, "cached": 172}, "…": "…"}
```

```ts
const ranked = await crucible.rerank({ query, documents: shortlist, instruction: 'Find passages about it' });
const keep = ranked.results.filter((r) => r.relevanceScore >= cutoff).map((r) => shortlist[r.index]);
```

- **A score is a probability of relevance** (P(yes) against P(no), as the model card
  defines it), each document judged on its own: several can be high at once. `scores` are in
  document order, `results` most relevant first.
- **A fixed cutoff means the same thing across calls for one model**, so you can keep
  "everything above 0.5". Set it by measurement: label a few dozen real queries and pick the
  cutoff that agrees with you. Scores from different models are on different scales; never
  carry a cutoff from one to another.
- **The query is read once**: on llama-server and the Mac the instruction and query are
  shared by every document, so a shortlist of 30 costs about 30 short tails, not 30 queries.
  Measured on the Mac: a ~2,500-token query with 1 document ran 3.8 s, with 20 documents
  6.9 s. `tokens.total` counts each document's prompt with its query (as llama-server is sent
  it); `tokens.cached` is the part that was not read again, so `total - cached` is what the
  engine actually read. vLLM re-reads the query for every document (prompt log-probabilities
  never read its prefix cache).
- **Limits**: 256 documents; the query and one document together at most 8192 tokens.
- **Short labels** (picking tags from a fixed list) work as documents. The model was trained
  on passages, so a phrase per label ("lo-fi: dusty, low-fidelity, relaxed beats") should
  score more reliably than one word (not yet measured here).

### Rerank with a decide model

Any decide model reranks when you name it, with Crucible's general yes/no template, and needs
no package:

```json
{"query": "…", "documents": ["…"], "model": "qwen3.5-9b"}
```

`model.template` in the answer is `crucible-general-1` (the dedicated reranker's is `model`).
The automatic pick is always the dedicated reranker: a decide model's scores are on another
scale, so it is used only when named (or chosen in Settings).

### Cohere/Jina-compatible

`POST /v1/openai/rerank` takes `query`, `documents` (strings or `{"text"}`), `top_n` and
`return_documents`, and answers `{id, model, results: [{index, relevance_score, document?}],
usage}`, with every score in document order under `crucible`.
