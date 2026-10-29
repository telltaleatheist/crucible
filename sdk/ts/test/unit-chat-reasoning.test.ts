/**
 * Unit tests for a chat's thinking: `message.reasoning` read as `ChatResponse.reasoning`, Claude's
 * `reasoning_blocks` read as `reasoningBlocks` and sent back unchanged on an assistant turn, a
 * stream's `delta.reasoning` handed to `onReasoning` apart from the content it yields, and the
 * finish chunk's blocks handed to `onReasoningBlocks`.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  type ReasoningBlock,
} from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-chat-reasoning' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function frame(delta: unknown, finishReason: string | null = null): string {
  const chunk = {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1757640000,
    model: 'anthropic/claude-opus-5-5',
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function completion(message: unknown, finishReason: string): unknown {
  return {
    id: 'chatcmpl-reasoning',
    object: 'chat.completion',
    created: 1757640000,
    model: 'anthropic/claude-opus-5-5',
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
  };
}

const BLOCKS: ReasoningBlock[] = [
  { type: 'thinking', thinking: 'Weighing it.', signature: 'EqQBCkYIBxgCKkDsig==' },
  { type: 'redacted_thinking', data: 'opaque' },
];

before(async () => {
  server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      lastBody = Buffer.concat(chunks).toString('utf8');
      handle(request, response, lastBody);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

test('a reply reads its reasoning and its blocks apart from the content', async () => {
  handle = (_request, response) =>
    json(response, 200, completion(
      { role: 'assistant', content: 'Paris.', reasoning: 'Weighing it.', reasoning_blocks: BLOCKS },
      'stop',
    ));
  const answer = await client().chat({
    model: 'anthropic/claude-opus-5-5',
    messages: [{ role: 'user', content: 'Capital of France?' }],
  });
  assert.equal(answer.content, 'Paris.');
  assert.equal(answer.reasoning, 'Weighing it.');
  assert.deepEqual(answer.reasoningBlocks, BLOCKS);
});

test('a reply without thinking reads reasoning null and no blocks', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: 'Paris.', reasoning: '' }, 'stop'));
  const answer = await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Capital of France?' }],
  });
  assert.equal(answer.reasoning, null);
  assert.deepEqual(answer.reasoningBlocks, []);
});

test('a turn that thought and only called tools is an answer, not a refusal', async () => {
  handle = (_request, response) =>
    json(response, 200, completion(
      {
        role: 'assistant',
        content: null,
        reasoning: 'I should look it up.',
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }],
      },
      'tool_calls',
    ));
  const answer = await client().chat({
    model: 'qwen3.8-27b-4bit',
    messages: [{ role: 'user', content: 'Look it up.' }],
  });
  assert.equal(answer.content, null);
  assert.equal(answer.reasoning, 'I should look it up.');
  assert.equal(answer.toolCalls.length, 1);
});

test('thinking that ran out before the answer is still refused by name', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: null, reasoning: 'Hmm…' }, 'length'));
  await assert.rejects(
    client().chat({ model: 'qwen3.8-27b-4bit', messages: [{ role: 'user', content: 'Think.' }] }),
    (error: unknown) => error instanceof CrucibleProtocolError && /token ceiling/.test(error.message),
  );
});

test('an assistant turn sends its reasoningBlocks back unchanged as reasoning_blocks', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: 'Done.' }, 'stop'));
  await client().chat({
    model: 'anthropic/claude-opus-5-5',
    messages: [
      { role: 'user', content: 'Look it up.' },
      {
        role: 'assistant',
        content: null,
        reasoning: 'Weighing it.',
        reasoningBlocks: BLOCKS,
        toolCalls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }],
      },
      { role: 'tool', toolCallId: 'call_1', content: 'found' },
    ],
  });
  const sent = JSON.parse(lastBody);
  assert.equal(JSON.stringify(sent.messages[1].reasoning_blocks), JSON.stringify(BLOCKS));
  assert.equal(sent.messages[1].reasoning, 'Weighing it.');
  assert.equal('reasoningBlocks' in sent.messages[1], false);
});

test('reasoningBlocks that are not thinking blocks are refused before anything is sent', async () => {
  let asked = 0;
  handle = (_request, response) => {
    asked += 1;
    json(response, 200, completion({ role: 'assistant', content: 'x' }, 'stop'));
  };
  await assert.rejects(
    client().chat({
      model: 'anthropic/claude-opus-5-5',
      messages: [
        { role: 'user', content: 'hi' },
        {
          role: 'assistant',
          content: 'said',
          reasoningBlocks: [{ type: 'text', text: 'x' } as unknown as ReasoningBlock],
        },
      ],
    }),
    (error: unknown) =>
      error instanceof CrucibleConfigError && /reasoningBlocks\[0\]/.test(error.message),
  );
  assert.equal(asked, 0);
});

test('chat() refuses the stream-only reasoning callbacks', async () => {
  for (const option of [{ onReasoning: () => undefined }, { onReasoningBlocks: () => undefined }]) {
    await assert.rejects(
      client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }], ...option }),
      CrucibleConfigError,
    );
  }
});

test('chatStream hands delta.reasoning to onReasoning and the finish blocks to onReasoningBlocks', async () => {
  handle = (_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write(frame({ role: 'assistant' }));
    response.write(frame({ reasoning: 'Weighing ' }));
    response.write(frame({ reasoning: 'it.' }));
    response.write(frame({ content: 'Paris.' }));
    response.write(frame({ reasoning_blocks: BLOCKS }, 'stop'));
    response.write('data: [DONE]\n\n');
    response.end();
  };
  const content: string[] = [];
  const thinking: string[] = [];
  let blocks: readonly ReasoningBlock[] | null = null;
  let handed = 0;
  for await (const delta of client().chatStream({
    model: 'anthropic/claude-opus-5-5',
    messages: [{ role: 'user', content: 'Capital of France?' }],
    onReasoning: (piece) => thinking.push(piece),
    onReasoningBlocks: (given) => {
      handed += 1;
      blocks = given;
    },
  })) {
    content.push(delta);
  }
  assert.deepEqual(content, ['Paris.']);
  assert.deepEqual(thinking, ['Weighing ', 'it.']);
  assert.equal(handed, 1);
  assert.deepEqual(blocks, BLOCKS);
});

test('chatStream with no blocks hands onReasoningBlocks an empty array', async () => {
  handle = (_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write(frame({ content: 'Paris.' }, 'stop'));
    response.write('data: [DONE]\n\n');
    response.end();
  };
  let blocks: readonly ReasoningBlock[] | null = null;
  for await (const _delta of client().chatStream({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Capital of France?' }],
    onReasoningBlocks: (given) => {
      blocks = given;
    },
  })) {
    // content only
  }
  assert.deepEqual(blocks, []);
});
