/**
 * The settings door in the client: `settings`, `putSettings`, `testUpstream`,
 * `readPairingFile`, and `CapabilityRow.route`.
 *
 * PHASE15-HOST.md section 3.8, against a fake server like every other method
 * in this directory. What is proved here is the half a live server cannot:
 * that the snake_case wire becomes the camelCase types exactly, that a patch
 * travels as a PARTIAL document rather than as a filled-in one, that
 * `testUpstream`'s three refusals come back as RESULTS while an auth failure
 * still throws, and that a missing pairing file is `null` while a malformed
 * one is not.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';

import {
  CrucibleAuthError,
  CrucibleClient,
  CrucibleProtocolError,
  CrucibleRefused,
  LEASE_NOT_NEEDED,
  ROUTE_NOT_ROUTABLE,
  UPSTREAM_IN_USE,
  UPSTREAM_TEST_REFUSALS,
  cruciblePairingPath,
  readPairingFile,
  type SettingsDocument,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastPath = '';
let lastMethod = '';
let lastBody = '';

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-settings' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function answer(status: number, body: unknown): void {
  handle = (_request, response) => json(response, status, body);
}

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      lastPath = request.url ?? '';
      lastMethod = request.method ?? '';
      lastBody = body;
      handle(request, response, body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
});

/** Section 3.1's document, verbatim, as the contract prints it. */
const DOCUMENT = {
  routes: {
    clean: { route: 'local', model: 'qwen3.5-9b' },
    translate: { route: 'upstream', model: 'anthropic/claude-sonnet-5' },
    simplify: { route: 'upstream', model: 'anthropic/claude-sonnet-5' },
    analysis: { route: 'local', model: null },
  },
  upstreams: {
    anthropic: { configured: true, key_hint: '…k3A9' },
    openai: { configured: false, key_hint: null },
    ollama: { configured: true, url: 'http://192.168.68.20:11434' },
  },
  desktop_allowance_bytes: 3221225472,
  backend_kind: 'cuda-linux',
};

// ------------------------------------------------------------------ settings

test('settings() reads the contract document into the typed shape', async () => {
  answer(200, DOCUMENT);
  const settings: SettingsDocument = await client().settings();
  assert.equal(lastMethod, 'GET');
  assert.equal(lastPath, '/v1/settings');
  assert.deepEqual(settings.routes.clean, { route: 'local', model: 'qwen3.5-9b' });
  assert.deepEqual(settings.routes.translate, {
    route: 'upstream',
    model: 'anthropic/claude-sonnet-5',
  });
  // `null` is a value here and must survive as one: it is "nothing fits", not
  // "the server did not say".
  assert.equal(settings.routes.analysis?.model, null);
  assert.equal(settings.desktopAllowanceBytes, 3221225472);
  assert.equal(settings.backendKind, 'cuda-linux');
});

test('the key hint arrives with its ellipsis and is rendered verbatim', async () => {
  answer(200, DOCUMENT);
  const settings = await client().settings();
  assert.equal(settings.upstreams.anthropic.keyHint, '…k3A9');
  assert.equal(settings.upstreams.openai.keyHint, null);
  // Ollama has no secret, so it has no hint field at all — and a url instead.
  assert.equal(settings.upstreams.ollama.keyHint, undefined);
  assert.equal(settings.upstreams.ollama.url, 'http://192.168.68.20:11434');
  assert.equal(settings.upstreams.anthropic.url, undefined);
});

test('a document missing a promised field is a protocol error, not undefined', async () => {
  const { backend_kind: _dropped, ...without } = DOCUMENT;
  answer(200, without);
  await assert.rejects(client().settings(), CrucibleProtocolError);
});

test('putSettings sends a PARTIAL patch in the wire spelling', async () => {
  answer(200, DOCUMENT);
  await client().putSettings({
    upstreams: { anthropic: { key: 'sk-ant-secret' } },
    routes: { translate: 'anthropic/claude-sonnet-5', simplify: 'local' },
  });
  assert.equal(lastMethod, 'PUT');
  assert.equal(lastPath, '/v1/settings');
  const sent = JSON.parse(lastBody) as Record<string, unknown>;
  // Only what the caller stated. A patch that filled in the allowance it never
  // mentioned would overwrite a number somebody else set.
  assert.deepEqual(Object.keys(sent).sort(), ['routes', 'upstreams']);
  assert.deepEqual(sent.routes, {
    translate: 'anthropic/claude-sonnet-5',
    simplify: 'local',
  });
});

test('putSettings renames desktopAllowanceBytes for the wire', async () => {
  answer(200, DOCUMENT);
  await client().putSettings({ desktopAllowanceBytes: 1024 });
  assert.deepEqual(JSON.parse(lastBody), { desktop_allowance_bytes: 1024 });
});

test('putSettings answers with the document AFTER the write', async () => {
  answer(200, DOCUMENT);
  const after = await client().putSettings({ routes: { translate: 'local' } });
  // A window draws what it is handed; it never guesses what took.
  assert.equal(after.routes.translate?.model, 'anthropic/claude-sonnet-5');
});

test('an upstream removal travels as a null', async () => {
  answer(200, DOCUMENT);
  await client().putSettings({ upstreams: { anthropic: null } });
  assert.deepEqual(JSON.parse(lastBody), { upstreams: { anthropic: null } });
});

test('a settings refusal carries the dotted field and keeps its name', async () => {
  answer(400, {
    error: {
      code: ROUTE_NOT_ROUTABLE,
      message: 'tts cannot run anywhere but this card',
      details: { field: 'routes.tts', capability: 'tts' },
    },
  });
  await assert.rejects(client().putSettings({ routes: { tts: 'anthropic/x' } }), (error) => {
    assert.ok(error instanceof CrucibleRefused);
    assert.equal(error.code, ROUTE_NOT_ROUTABLE);
    assert.equal((error.details as { field: string }).field, 'routes.tts');
    return true;
  });
});

test('upstream_in_use names the classes that must be re-routed first', async () => {
  answer(409, {
    error: {
      code: UPSTREAM_IN_USE,
      message: 'anthropic cannot be removed while simplify, translate are routed to it',
      details: {
        field: 'upstreams.anthropic',
        upstream: 'anthropic',
        classes: ['simplify', 'translate'],
      },
    },
  });
  await assert.rejects(client().putSettings({ upstreams: { anthropic: null } }), (error) => {
    assert.ok(error instanceof CrucibleRefused);
    assert.deepEqual(
      (error.details as { classes: string[] }).classes,
      ['simplify', 'translate'],
    );
    return true;
  });
});

// -------------------------------------------------------------- testUpstream

test('testUpstream reports what the upstream itself lists', async () => {
  answer(200, { models: ['claude-sonnet-5', 'claude-haiku-5'] });
  const result = await client().testUpstream('anthropic', { key: 'sk-ant-x' });
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/settings/upstreams/anthropic/test');
  assert.deepEqual(JSON.parse(lastBody), { key: 'sk-ant-x' });
  assert.ok(result.ok);
  assert.deepEqual(result.models, ['claude-sonnet-5', 'claude-haiku-5']);
});

test('testUpstream with no probe sends an empty body and uses the stored one', async () => {
  answer(200, { models: [] });
  const result = await client().testUpstream('ollama');
  assert.deepEqual(JSON.parse(lastBody), {});
  assert.ok(result.ok);
});

test('the three test refusals are RESULTS, with the provider own words', async () => {
  for (const [status, code] of [
    [502, 'upstream_unreachable'],
    [401, 'upstream_rejected'],
    [400, 'upstream_unconfigured'],
  ] as const) {
    answer(status, { error: { code, message: `it said: ${code}` } });
    const result = await client().testUpstream('openai', { key: 'sk-x' });
    assert.equal(result.ok, false, code);
    if (!result.ok) {
      assert.equal(result.code, code);
      assert.match(result.message, new RegExp(code));
    }
  }
  // …and the list is the one the package exports, not one this test typed.
  assert.deepEqual([...UPSTREAM_TEST_REFUSALS].sort(), [
    'upstream_rejected',
    'upstream_unconfigured',
    'upstream_unreachable',
  ]);
});

test('a bad bearer still THROWS from testUpstream', async () => {
  // `upstream_rejected` is a 401 on this door, so narrowing on the status
  // would have turned THIS — a wrong Crucible token — into "your Anthropic
  // key is bad". The narrowing is on the code.
  answer(401, { error: { code: 'unauthorized', message: 'bearer token is not this server' } });
  await assert.rejects(client().testUpstream('anthropic', { key: 'x' }), CrucibleAuthError);
});

test('an unrelated refusal from the test door still throws', async () => {
  answer(400, { error: { code: 'unknown_upstream', message: 'claude is not an upstream' } });
  await assert.rejects(
    client().testUpstream('anthropic' as never, { key: 'x' }),
    CrucibleRefused,
  );
});

// ----------------------------------------------------------- capability row

test('every capability row says where its work runs', async () => {
  answer(200, {
    backend_kind: 'cuda-linux',
    total_bytes: 25757220864,
    desktop_allowance_bytes: 3221225472,
    classes: [
      {
        capability: 'clean',
        enabled: true,
        selected: 'qwen3.5-9b',
        reason: 'qwen3.5-9b fits',
        shortfall_bytes: 0,
        route: 'local',
      },
      {
        capability: 'translate',
        enabled: true,
        selected: 'anthropic/claude-sonnet-5',
        reason: 'routed to anthropic; the local answer would be: disabled …',
        shortfall_bytes: 0,
        route: 'upstream',
      },
    ],
    job_types: [],
  });
  const record = await client().capability();
  const [clean, translate] = record.classes;
  assert.ok(clean !== undefined && translate !== undefined);
  assert.equal(clean.route, 'local');
  assert.equal(translate.route, 'upstream');
  // The upstream row's `selected` is exactly the `model` to send as a chat.
  assert.equal(translate.selected, 'anthropic/claude-sonnet-5');
  assert.match(translate.reason, /the local answer would be: /);
});

test('a row with a route this build has never heard of is a protocol error', async () => {
  answer(200, {
    backend_kind: 'cuda-linux',
    total_bytes: 1,
    desktop_allowance_bytes: 0,
    classes: [
      {
        capability: 'clean',
        enabled: true,
        selected: 'x',
        reason: 'y',
        shortfall_bytes: 0,
        route: 'sideways',
      },
    ],
    job_types: [],
  });
  await assert.rejects(client().capability(), CrucibleProtocolError);
});

// ------------------------------------------------------------- pairing file

test('readPairingFile is null when there is no local server', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  try {
    assert.equal(await readPairingFile(home), null);
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
});

test('readPairingFile parses the one line a server left', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  try {
    writeFileSync(
      join(home, 'pairing'),
      'crucible://crucible%40owens-pc@127.0.0.1:7100/#s3cret-t0ken_x\n',
      'utf8',
    );
    assert.deepEqual(await readPairingFile(home), {
      name: 'crucible@owens-pc',
      url: 'http://127.0.0.1:7100',
      token: 's3cret-t0ken_x',
    });
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
});

test('a malformed pairing file REJECTS rather than reading as absent', async () => {
  // A line somebody's installer wrote badly is a broken install. Answering
  // "there is no server here" would send the user to install a second one.
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  try {
    writeFileSync(join(home, 'pairing'), 'http://127.0.0.1:7100\n', 'utf8');
    await assert.rejects(readPairingFile(home), /pairing/i);
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
});

test('cruciblePairingPath honours CRUCIBLE_HOME and the explicit home', async () => {
  const previous = process.env.CRUCIBLE_HOME;
  try {
    process.env.CRUCIBLE_HOME = join(tmpdir(), 'elsewhere');
    assert.equal(
      await cruciblePairingPath(),
      join(tmpdir(), 'elsewhere', 'pairing'),
    );
    assert.equal(await cruciblePairingPath('/srv/c'), join('/srv/c', 'pairing'));
  } finally {
    if (previous === undefined) delete process.env.CRUCIBLE_HOME;
    else process.env.CRUCIBLE_HOME = previous;
  }
});

test('an empty pairing file is a DEFECT, not an absence', async () => {
  // Reversed 2026-09-14, measured by BookForge. `null` means "there is no
  // server on this machine", which sends a connect door to "install one" -
  // over a server that is running and whose installer truncated its own
  // pairing file. The absent case is the file not being there and nothing
  // else; `pairing_file_malformed` is what a file that exists and says
  // nothing gets, the same name a file with two lines gets.
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  try {
    mkdirSync(home, { recursive: true });
    writeFileSync(join(home, 'pairing'), '\n', 'utf8');
    await assert.rejects(readPairingFile(home), (error: unknown) => {
      assert.equal((error as { code?: string }).code, 'pairing_file_malformed');
      assert.match((error as Error).message, /is empty/);
      return true;
    });
    // And the absent case is still null, which is what the reversal is about.
    rmSync(join(home, 'pairing'));
    assert.equal(await readPairingFile(home), null);
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
});

// ------------------------------------------------------------ the constants

test('the refusal names this package exports are the contract spelling', () => {
  assert.equal(LEASE_NOT_NEEDED, 'lease_not_needed');
  assert.equal(ROUTE_NOT_ROUTABLE, 'route_not_routable');
  assert.equal(UPSTREAM_IN_USE, 'upstream_in_use');
});
