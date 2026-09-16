import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { installSteps } from '../src/steps.js';

const shell = process.env['CRUCIBLE_TEST_BASH'] ?? (process.platform === 'win32' ? undefined : '/bin/sh');
for (const code of [0, 1]) {
  test(`shell installer readiness ${code === 0 ? 'finishes before success' : 'failure prevents success'}`, { skip: shell === undefined }, () => {
    const steps = installSteps({ enableFlags: [], installs: [], bind: [], linger: false });
    assert.equal(steps.at(-2)?.name, 'capability-write');
    assert.equal(steps.at(-1)?.name, 'local-start');
    const program = `set -eu
CRUCIBLE=crucible
die() { printf '%s\\n' "$*" >&2; exit 1; }
crucible() {
  test "$*" = 'local start --json' || exit 91
  printf '%s\\n' readiness-called
  return ${code}
}
` + steps.at(-1)!.sh + `printf '%s\\n' installed\n`;
    const result = spawnSync(shell!, ['-c', program], { encoding: 'utf8' });
    assert.equal(result.status, code);
    assert.equal(result.stdout, code === 0 ? 'readiness-called\ninstalled\n' : 'readiness-called\n');
    assert.equal(result.stderr, code === 0 ? '' : 'step_failed: local-start\n');
  });
}
