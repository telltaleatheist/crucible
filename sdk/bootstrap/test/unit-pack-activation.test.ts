import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { activatePackSh } from '../src/pack.js';

const shell = process.env['CRUCIBLE_TEST_BASH'] ?? (process.platform === 'win32' ? undefined : '/bin/sh');
for (const failure of ['none', 'shutdown', 'activation', 'interrupted'] as const) {
  test(`real pack activation preserves runtime when ${failure}`, { skip: shell === undefined }, () => {
    const root = mkdtempSync(join(tmpdir(), 'crucible-swap-'));
    try {
      mkdirSync(join(root, 'server', 'bin'), { recursive: true });
      mkdirSync(join(root, 'server.partial', 'bin'), { recursive: true });
      writeFileSync(join(root, 'server', 'identity'), 'old');
      writeFileSync(join(root, 'server.partial', 'identity'), 'new');
      writeFileSync(join(root, 'server.partial', 'bin', 'crucible'),
        '#!/bin/sh\n' +
        `if [ "$1" = local ]; then echo stop >> actions; exit ${failure === 'shutdown' ? 1 : 0}; fi\n` +
        `if [ "$1" = --version ]; then exit ${failure === 'activation' ? 1 : 0}; fi\nexit 9\n`, { mode: 0o755 });
      if (failure === 'interrupted') mkdirSync(join(root, 'server.previous'));
      const result = spawnSync(shell!, ['-c', activatePackSh('server', 'server.partial')], { cwd: root, encoding: 'utf8' });
      assert.equal(result.status, failure === 'none' ? 0 : 1, result.stderr);
      assert.equal(readFileSync(join(root, 'server', 'identity'), 'utf8'), failure === 'none' ? 'new' : 'old');
      if (failure !== 'interrupted') assert.equal(readFileSync(join(root, 'actions'), 'utf8'), 'stop\n');
      else assert.equal(existsSync(join(root, 'actions')), false);
      if (failure === 'activation') assert.equal(readFileSync(join(root, 'server.partial', 'identity'), 'utf8'), 'new');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
}
