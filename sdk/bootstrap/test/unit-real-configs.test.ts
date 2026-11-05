/** What real machines hold: a config.toml a person or a tool wrote, and a host with no token of its own. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { hostToken, parseToml, TomlError } from '../src/index.js';
import { FakeRunner, refusal } from './fake.js';

test('an array over several lines: this PC\'s own config.toml, 2026-11-05', () => {
  const table = parseToml(
    '[server]\nname = "x"\nadvertise = [\n    "a.example",  # the tailnet\n    "b.example",\n]\n'
      + 'cors_origins = [\n  "http://localhost:4250",\n]\nport = 7100\n',
  ) as Record<string, Record<string, unknown>>;
  assert.deepEqual(table['server']?.['advertise'], ['a.example', 'b.example']);
  assert.deepEqual(table['server']?.['cors_origins'], ['http://localhost:4250']);
  assert.equal(table['server']?.['port'], 7100);
});

test('a bracket or a hash inside a string does not open or close anything', () => {
  const table = parseToml('a = ["x]y", "z#w",\n "[", \'q]\']\nb = 1\n');
  assert.deepEqual(table['a'], ['x]y', 'z#w', '[', 'q]']);
  assert.equal(table['b'], 1);
});

test('nested arrays over several lines', () => {
  assert.deepEqual(parseToml('m = [\n  [1, 2],\n  [\n    3,\n  ],\n]\n')['m'], [[1, 2], [3]]);
});

test('inline tables, multi-line strings and dates, which tomllib reads and so must this', () => {
  const table = parseToml([
    's = { temperature = 0.8, top = { k = 50 } }',
    'l = """',
    'first "quoted"',
    'second\\',
    '   joined"""',
    "r = '''",
    'raw \\n # not a comment',
    "'''",
    'd = 2026-11-05T01:02:03Z',
    'day = 2026-11-05',
    'h = 0xff',
    '',
  ].join('\n'));
  assert.deepEqual(table['s'], { temperature: 0.8, top: { k: 50 } });
  assert.equal(table['l'], 'first "quoted"\nsecondjoined');
  assert.equal(table['r'], 'raw \\n # not a comment\n');
  assert.equal(table['d'], '2026-11-05T01:02:03Z');
  assert.equal(table['day'], '2026-11-05');
  assert.equal(table['h'], 255);
});

test('an array that never closes is still refused by name', () => {
  assert.throws(() => parseToml('a = [\n 1,\n'), TomlError);
});

const LOCAL = 'C:\\Users\\owen\\AppData\\Local';
const PAIRING = `${LOCAL}\\Crucible\\pairing`;
const CONFIG = `${LOCAL}\\Crucible\\config.toml`;
const ORCHESTRATOR_ONLY = '[orchestrator]\ndistro = "Ubuntu"\n';

function windows(files: Record<string, string>): FakeRunner {
  return new FakeRunner({ platform: 'win32', env: { LOCALAPPDATA: LOCAL }, files }, []);
}

test('where the engine lives in WSL the door takes the pairing; a config with no token is no fault', () => {
  const runner = windows({
    [CONFIG]: ORCHESTRATOR_ONLY,
    [PAIRING]: 'crucible://crucible%40pc-wsl@127.0.0.1:7100/#pair%2Ftoken\n',
  });
  assert.equal(hostToken(runner), 'pair/token');
});

test('the pairing comes before the config, as the host itself takes them', () => {
  const runner = windows({
    [CONFIG]: '[auth]\ntoken = "config-token"\n',
    [PAIRING]: 'crucible://x@127.0.0.1:7100/#paired\n',
  });
  assert.equal(hostToken(runner), 'paired');
});

test('a pairing with no token goes to the config', () => {
  const runner = windows({
    [CONFIG]: '[auth]\ntoken = "config-token"\n',
    [PAIRING]: 'crucible://x@127.0.0.1:7100/\n',
  });
  assert.equal(hostToken(runner), 'config-token');
});

test('no pairing and a config with no token is host_no_token, saying both', async () => {
  const r = await refusal(Promise.resolve().then(() => hostToken(windows({ [CONFIG]: ORCHESTRATOR_ONLY }))));
  assert.equal(r.code, 'host_no_token');
  assert.match(r.message, /no pairing/);
});

test('a config this reader cannot read does not claim the server would refuse it', async () => {
  const r = await refusal(Promise.resolve().then(() => hostToken(windows({ [CONFIG]: 'a = @nonsense\n' }))));
  assert.equal(r.code, 'config_unreadable');
  assert.doesNotMatch(r.message, /would refuse/);
  assert.match(r.message, /reader's limit/);
});
