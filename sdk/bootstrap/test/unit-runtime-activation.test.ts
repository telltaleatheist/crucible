/**
 * The swap, run against a real shell and a real filesystem.
 *
 * The one moment an install can leave a machine with NO Crucible on it: the old
 * tree has been moved aside and the new one has not landed. Every failure below
 * must end with the old tree back at `server/`, which is what these assert — by
 * running the shell this package generates, not by reading it.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { activateRuntimeSh } from '../src/runtime.js';

const shell = process.env['CRUCIBLE_TEST_BASH'] ?? (process.platform === 'win32' ? undefined : '/bin/sh');
for (const failure of ['none', 'shutdown', 'activation', 'interrupted'] as const) {
  test(`real runtime activation preserves the previous tree when ${failure}`, { skip: shell === undefined }, () => {
    const root = mkdtempSync(join(tmpdir(), 'crucible-swap-'));
    try {
      mkdirSync(join(root, 'server', 'bin'), { recursive: true });
      mkdirSync(join(root, 'staged', 'bin'), { recursive: true });
      writeFileSync(join(root, 'server', 'identity'), 'old');
      writeFileSync(join(root, 'staged', 'identity'), 'new');
      // THE OLD TREE'S `crucible` is what stops the server — it is the one
      // running — and the NEW tree's `python3` is what proves the swap landed.
      // A freshly unpacked interpreter has no `crucible` in it at all: the
      // wheel goes in afterwards, which is the order PHASE20 section 3 sets.
      writeFileSync(join(root, 'server', 'bin', 'crucible'),
        '#!/bin/sh\n'
        + `if [ "$1" = local ]; then echo stop >> actions; exit ${failure === 'shutdown' ? 1 : 0}; fi\nexit 9\n`,
        { mode: 0o755 });
      writeFileSync(join(root, 'staged', 'bin', 'python3'),
        `#!/bin/sh\nexit ${failure === 'activation' ? 1 : 0}\n`, { mode: 0o755 });
      if (failure === 'interrupted') mkdirSync(join(root, 'server.previous'));
      const result = spawnSync(shell!, ['-c', activateRuntimeSh('server', 'staged')], { cwd: root, encoding: 'utf8' });
      assert.equal(result.status, failure === 'none' ? 0 : 1, result.stderr);
      assert.equal(readFileSync(join(root, 'server', 'identity'), 'utf8'), failure === 'none' ? 'new' : 'old');
      if (failure !== 'interrupted') assert.equal(readFileSync(join(root, 'actions'), 'utf8'), 'stop\n');
      else assert.equal(existsSync(join(root, 'actions')), false);
      if (failure === 'activation') assert.equal(readFileSync(join(root, 'staged', 'identity'), 'utf8'), 'new');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
}
