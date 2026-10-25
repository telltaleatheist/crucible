/**
 * Unit tests for OpenAI tool calling on `chat` and `chatStream`: `tools` and `tool_choice` on the
 * wire, an assistant turn's `toolCalls` and a tool turn's `toolCallId` sent snake_case, a reply
 * whose model only called tools (`content` null), and a stream's `delta.tool_calls` fragments
 * assembled by `index` into whole calls.
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
  type ChatMessage,
  type ChatTool,
  type ChatToolCall,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-chat-tools' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function openSse(response: ServerResponse): void {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

function frame(delta: unknown, finishReason: string | null = null): string {
  const chunk = {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1757640000,
    model: 'qwen3.5-9b',
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function completion(message: unknown, finishReason: string): unknown {
  return {
    id: 'chatcmpl-tools',
    object: 'chat.completion',
    created: 1757640000,
    model: 'qwen3.5-9b',
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
  };
}

const WEATHER: ChatTool = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'The weather in a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
  },
};

const CALL: ChatToolCall = {
  id: 'call_1',
  type: 'function',
  function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
};

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

// ------------------------------------------------------------------ request

test('tools, tool_choice and the tool turns go on the wire snake_case', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: 'Sunny.' }, 'stop'));

  const messages: ChatMessage[] = [
    { role: 'system', content: 'Use the tools.' },
    { role: 'user', content: 'Weather in Paris?' },
    { role: 'assistant', content: null, toolCalls: [CALL] },
    { role: 'tool', toolCallId: 'call_1', content: '{"sky":"clear"}' },
  ];
  await client().chat({
    model: 'qwen3.5-9b',
    messages,
    tools: [WEATHER],
    toolChoice: { type: 'function', function: { name: 'get_weather' } },
  });

  const sent = JSON.parse(lastBody);
  assert.deepEqual(sent.tools, [WEATHER]);
  assert.deepEqual(sent.tool_choice, { type: 'function', function: { name: 'get_weather' } });
  assert.deepEqual(sent.messages, [
    { role: 'system', content: 'Use the tools.' },
    { role: 'user', content: 'Weather in Paris?' },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call_1', content: '{"sky":"clear"}' },
  ]);
  assert.ok(!('toolCalls' in sent.messages[2]), 'the camelCase name never reaches the wire');
});

test('a chat without tools sends neither tools nor tool_choice', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: 'Hi.' }, 'stop'));
  await client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
  const sent = JSON.parse(lastBody);
  assert.ok(!('tools' in sent));
  assert.ok(!('tool_choice' in sent));
  assert.deepEqual(sent.messages, [{ role: 'user', content: 'hi' }]);
});

test('the string tool choices are sent as given', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: 'Hi.' }, 'stop'));
  for (const toolChoice of ['auto', 'none', 'required'] as const) {
    await client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      tools: [WEATHER],
      toolChoice,
    });
    assert.equal(JSON.parse(lastBody).tool_choice, toolChoice);
  }
});

test('malformed tool options and messages are refused by name before anything is sent', async () => {
  const user: ChatMessage = { role: 'user', content: 'hi' };
  const refused = async (options: unknown, option: string): Promise<void> => {
    await assert.rejects(
      client().chat(options as Parameters<CrucibleClient['chat']>[0]),
      (error: unknown) => error instanceof CrucibleConfigError && error.option === option,
      `expected ${option} to be refused`,
    );
  };
  await refused({ messages: [user], toolChoice: 'auto' }, 'toolChoice');
  await refused({ messages: [user], tools: [WEATHER], toolChoice: 'sometimes' }, 'toolChoice');
  await refused({ messages: [user], tools: [] }, 'tools');
  await refused({ messages: [user], tools: [{ type: 'retrieval' }] }, 'tools[0].type');
  await refused({ messages: [user], tools: [{ type: 'function', function: {} }] }, 'tools[0].function.name');
  await refused({ messages: [{ role: 'assistant', content: null }] }, 'messages[0].content');
  await refused({ messages: [{ role: 'tool', content: 'x' }] }, 'messages[0].toolCallId');
  await refused(
    {
      messages: [
        { role: 'assistant', content: null, toolCalls: [{ ...CALL, function: { name: 'f', arguments: { city: 'Paris' } } }] },
      ],
    },
    'messages[0].toolCalls[0].function.arguments',
  );
  await refused({ messages: [user], onToolCalls: () => undefined }, 'onToolCalls');
});

// ------------------------------------------------------------------- reply

test('a reply that only called tools reads content null, the calls, and finish_reason tool_calls', async () => {
  handle = (_request, response) =>
    json(
      response,
      200,
      completion(
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            CALL,
            { id: 'call_2', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Oslo"}' } },
          ],
        },
        'tool_calls',
      ),
    );
  const answer = await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Paris and Oslo?' }],
    tools: [WEATHER],
  });
  assert.equal(answer.content, null);
  assert.equal(answer.finishReason, 'tool_calls');
  assert.deepEqual(answer.toolCalls, [
    CALL,
    { id: 'call_2', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Oslo"}' } },
  ]);
});

test('a reply with no tool_calls (absent or null) reads an empty toolCalls', async () => {
  for (const message of [
    { role: 'assistant', content: 'Sunny.' },
    { role: 'assistant', content: 'Sunny.', tool_calls: null },
  ]) {
    handle = (_request, response) => json(response, 200, completion(message, 'stop'));
    const answer = await client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
    assert.equal(answer.content, 'Sunny.');
    assert.deepEqual(answer.toolCalls, []);
  }
});

test('a reply with neither content nor tool calls is a protocol error', async () => {
  handle = (_request, response) =>
    json(response, 200, completion({ role: 'assistant', content: null }, 'stop'));
  await assert.rejects(
    client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] }),
    CrucibleProtocolError,
  );
});

test('a two-round conversation: the call, the tool result sent back, the answer', async () => {
  let round = 0;
  handle = (_request, response) => {
    round += 1;
    if (round === 1) {
      json(response, 200, completion({ role: 'assistant', content: null, tool_calls: [CALL] }, 'tool_calls'));
    } else {
      json(response, 200, completion({ role: 'assistant', content: 'Clear skies in Paris.' }, 'stop'));
    }
  };

  const messages: ChatMessage[] = [{ role: 'user', content: 'Weather in Paris?' }];
  const first = await client().chat({ model: 'qwen3.5-9b', messages, tools: [WEATHER] });
  assert.equal(first.finishReason, 'tool_calls');
  messages.push({ role: 'assistant', content: first.content, toolCalls: first.toolCalls });
  for (const call of first.toolCalls) {
    const args = JSON.parse(call.function.arguments) as { city: string };
    messages.push({ role: 'tool', toolCallId: call.id, content: JSON.stringify({ city: args.city, sky: 'clear' }) });
  }
  const second = await client().chat({ model: 'qwen3.5-9b', messages, tools: [WEATHER] });

  assert.equal(second.content, 'Clear skies in Paris.');
  assert.deepEqual(second.toolCalls, []);
  assert.deepEqual(JSON.parse(lastBody).messages, [
    { role: 'user', content: 'Weather in Paris?' },
    { role: 'assistant', content: null, tool_calls: [CALL] },
    { role: 'tool', tool_call_id: 'call_1', content: '{"city":"Paris","sky":"clear"}' },
  ]);
});

// ----------------------------------------------------------------- stream

test('chatStream assembles tool_calls fragments by index and hands them to onToolCalls', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write(frame({ role: 'assistant', content: null }));
    response.write(frame({ content: 'Checking.' }));
    response.write(
      frame({
        tool_calls: [
          { index: 0, id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '' } },
        ],
      }),
    );
    response.write(frame({ tool_calls: [{ index: 0, function: { arguments: '{"ci' } }] }));
    // The second call opens while the first is still being written.
    response.write(
      frame({
        tool_calls: [
          { index: 1, id: 'call_b', type: 'function', function: { name: 'get_weather', arguments: '{"city":' } },
        ],
      }),
    );
    response.write(frame({ tool_calls: [{ index: 0, function: { arguments: 'ty":"Paris"}' } }] }));
    response.write(frame({ tool_calls: [{ index: 1, function: { arguments: '"Oslo"}' } }] }));
    response.write(frame({}, 'tool_calls'));
    response.write('data: [DONE]\n\n');
    response.end();
  };

  const deltas: string[] = [];
  let calls: readonly ChatToolCall[] | null = null;
  let handed = 0;
  for await (const delta of client().chatStream({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Paris and Oslo?' }],
    tools: [WEATHER],
    toolChoice: 'auto',
    onToolCalls: (assembled) => {
      handed += 1;
      calls = assembled;
    },
  })) {
    deltas.push(delta);
  }

  assert.deepEqual(deltas, ['Checking.']);
  assert.equal(handed, 1, 'onToolCalls is called exactly once');
  assert.deepEqual(calls, [
    { id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } },
    { id: 'call_b', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Oslo"}' } },
  ]);
  const sent = JSON.parse(lastBody);
  assert.deepEqual(sent.tools, [WEATHER]);
  assert.equal(sent.tool_choice, 'auto');
  assert.equal(sent.stream, true);
});

test('chatStream with no tool calls hands onToolCalls an empty array', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write(frame({ content: 'Sunny.' }));
    response.write(frame({}, 'stop'));
    response.write('data: [DONE]\n\n');
    response.end();
  };
  let calls: readonly ChatToolCall[] | null = null;
  for await (const _delta of client().chatStream({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    onToolCalls: (assembled) => {
      calls = assembled;
    },
  })) {
    // drained
  }
  assert.deepEqual(calls, []);
});

test('a streamed tool call that never names its function is a protocol error', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write(frame({ tool_calls: [{ index: 0, id: 'call_a', function: { arguments: '{}' } }] }));
    response.write(frame({}, 'tool_calls'));
    response.write('data: [DONE]\n\n');
    response.end();
  };
  await assert.rejects(
    (async () => {
      for await (const _delta of client().chatStream({
        model: 'qwen3.5-9b',
        messages: [{ role: 'user', content: 'hi' }],
        tools: [WEATHER],
        onToolCalls: () => undefined,
      })) {
        // drained
      }
    })(),
    (error: unknown) => error instanceof CrucibleProtocolError && /without a name/.test(error.message),
  );
});
