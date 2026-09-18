/**
 * `GET /v1/info`'s `pages_engine`, from the client's side.
 *
 * PHASE15-HOST.md 3.10 fact 7. The server publishes what a page request IS —
 * the prompt, the dpi, the pixel budget, the ceiling, the dialect — precisely
 * so a client stops pinning its own copy, and until this file existed the SDK
 * had nothing to read it with: an app that wanted the contract had to hand-roll
 * a fetch and its own parse, which is the third owner the block was added to
 * delete.
 *
 * THE FIELD NAMES ARE THE PYTHON'S, and `tests/test_pages_request.py` is what holds
 * them to it — this file proves the client READS the document, that one proves
 * the two spellings are one spelling.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleProtocolError } from '../src/index.js';

// ------------------------------------------------------------------ fixture

let reply: unknown = {};
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-pages' });
}

function json(response: ServerResponse, body: unknown): void {
  response.writeHead(200, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((_request, response) => json(response, reply));
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

/**
 * A `/v1/info` body captured from `crucible serve` on the cuda-linux backend,
 * `pages_engine` verbatim from `crucible/pages.py`'s `engine_block` and
 * `request_shape`. The prompt is elided to its first line HERE and nowhere in
 * the code: this file is about the field names and the types, and the bytes of
 * the prompt are `tests/test_pages_request.py`'s to compare.
 */
function document(pagesEngine: unknown): Record<string, unknown> {
  const body: Record<string, unknown> = {
    server: { name: 'crucible@owens-pc-wsl', version: '1.0.2', api_version: 1 },
    role: 'engine',
    managed_by: null,
    host: {
      platform: 'linux',
      arch: 'x86_64',
      backend: 'cuda-linux',
      gpu: { vendor: 'nvidia', name: '3090 Ti', vram_bytes: 25757220864 },
    },
    job_types: ['llm', 'tts'],
    capabilities: [],
  };
  if (pagesEngine !== undefined) body['pages_engine'] = pagesEngine;
  return body;
}

function servedBlock(): Record<string, unknown> {
  return {
    engine: 'vllm',
    installed: true,
    detail: 'vllm serves dots-ocr from /home/telltale/.crucible/weights/dots-ocr (4.14 GB)',
    request: {
      model: 'dots-ocr',
      dpi: 200,
      max_pixels: 11289600,
      max_tokens: 8192,
      temperature: 0.0,
      prompt: 'Please output the layout information from the PDF image, …',
      dialect: 'dots-json',
      truncated_finish_reason: 'length',
    },
  };
}

// ------------------------------------------------------------- the contract

test('a server that reads pages hands back the whole request contract', async () => {
  reply = document(servedBlock());
  const info = await client().info();
  assert.notEqual(info.pagesEngine, null, 'the block is on the wire and must survive the read');
  assert.equal(info.pagesEngine!.engine, 'vllm');
  assert.equal(info.pagesEngine!.installed, true);
  assert.match(info.pagesEngine!.detail, /dots-ocr/);
  const request = info.pagesEngine!.request;
  assert.equal(request.model, 'dots-ocr');
  assert.equal(request.dpi, 200);
  assert.equal(request.maxPixels, 11289600);
  assert.equal(request.maxTokens, 8192);
  assert.equal(request.temperature, 0);
  assert.equal(request.dialect, 'dots-json');
  assert.equal(request.truncatedFinishReason, 'length');
  assert.match(request.prompt, /^Please output the layout information/);
});

test('a host that serves no pages still publishes the request, and says which engine is null', async () => {
  // `engine: null` is a complete answer — this machine reads no pages — and the
  // request block is still there, because what a page request IS does not
  // depend on whether this host happens to be able to answer one.
  reply = document({
    engine: null,
    installed: false,
    detail: 'dots-ocr.toml has no mlx-darwin block, so this host reads no pages',
    request: servedBlock()['request'],
  });
  const info = await client().info();
  assert.equal(info.pagesEngine!.engine, null);
  assert.equal(info.pagesEngine!.installed, false);
  assert.equal(info.pagesEngine!.request.dialect, 'dots-json');
});

// ---------------------------------------------------------------- the vintage

test('a server whose document predates the block reads as null, not as an empty contract', async () => {
  // PHASE15 3.3's all-or-nothing rule, the same one `role` gets: an absent
  // block is a statement about the server's VINTAGE, and `null` is how a
  // client sees it. Nothing here invents a prompt or a budget — a caller that
  // needs the contract refuses by name against this null.
  reply = document(undefined);
  const info = await client().info();
  assert.equal(info.pagesEngine, null);
});

// ----------------------------------------------------------- a half-document

test('a block missing a field the contract promises is refused by name', async () => {
  const block = servedBlock();
  const request = { ...(block['request'] as Record<string, unknown>) };
  delete request['max_pixels'];
  block['request'] = request;
  reply = document(block);
  await assert.rejects(
    client().info(),
    (error: unknown) =>
      error instanceof CrucibleProtocolError
      && /info\.pages_engine\.request has no field "max_pixels"/.test((error as Error).message),
    'a half-new document is the one thing a vintage rule cannot read',
  );
});

test('a block with no request at all is refused by name', async () => {
  const block = servedBlock();
  delete block['request'];
  reply = document(block);
  await assert.rejects(
    client().info(),
    (error: unknown) =>
      error instanceof CrucibleProtocolError
      && /info\.pages_engine has no field "request"/.test((error as Error).message),
  );
});
