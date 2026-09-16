import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readLocalInstallation, localStatus, startLocal, localUninstallCommand } from '../src/local.js';
import { processRunner, type Runner } from '../src/runner.js';

test('missing, malformed and orphaned installations are distinct', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-local-'));
  try {
    assert.equal(readLocalInstallation({ home }), null);
    assert.equal((await localStatus({ home })).state, 'absent');
    writeFileSync(join(home, 'pairing'), 'old installation');
    assert.equal((await localStatus({ home })).state, 'broken');
    writeFileSync(join(home, 'installation.json'), '{');
    assert.throws(() => readLocalInstallation({ home }), /not JSON/);
    assert.equal((await localStatus({ home })).state, 'broken');
  } finally { rmSync(home, { recursive: true }); }
});

test('start uses published command, home and cwd and waits for structured result', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-local-'));
  const record = { schema_version: 1, platform: process.platform, release: 'test', home,
    control: { command: process.execPath, args: ['control'], cwd: home } };
  writeFileSync(join(home, 'installation.json'), JSON.stringify(record));
  const runner: Runner = { ...processRunner(), run: async (argv, options) => {
    assert.deepEqual(argv, [process.execPath, 'control', 'start', '--json']);
    assert.equal(options.cwd, home);
    assert.equal(options.env?.['CRUCIBLE_HOME'], home);
    return { code: 0, failure: null, stderr: '', stdout: JSON.stringify({ schema_version: 1,
      state: 'running', name: 'expected', url: 'http://127.0.0.1:7100', detail: 'ready' }) };
  } };
  try { assert.equal((await startLocal({ home }, runner)).state, 'running'); }
  finally { rmSync(home, { recursive: true }); }
});

test('a launch error and invalid successful output never report running', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-local-'));
  writeFileSync(join(home, 'installation.json'), JSON.stringify({ schema_version: 1, platform: process.platform,
    release: 'test', home, control: { command: process.execPath, args: [], cwd: home } }));
  try {
    const failed: Runner = { ...processRunner(), run: async () => ({code:null,stdout:'',stderr:'',failure:'spawn failed'}) };
    await assert.rejects(startLocal({ home }, failed), /spawn failed/);
    const invalid: Runner = { ...processRunner(), run: async () => ({code:0,stdout:'{}',stderr:'',failure:null}) };
    await assert.rejects(startLocal({ home }, invalid), /incompatible status/);
  } finally { rmSync(home, { recursive: true }); }
});

test('uninstall uses the published runtime/home/cwd and preserves data unless explicitly selected', () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-uninstall-'));
  writeFileSync(join(home, 'installation.json'), JSON.stringify({ schema_version: 1, platform: process.platform,
    release: 'test', home, control: { command: process.execPath, args: ['-m', 'crucible', 'local'], cwd: home } }));
  try {
    const preview = localUninstallCommand({ dryRun: true }, { home });
    assert.deepEqual(preview.argv, [process.execPath, '-m', 'crucible', 'uninstall', '--json', '--dry-run']);
    assert.equal(preview.env['CRUCIBLE_HOME'], home);
    assert.equal(preview.cwd, home);
    const purge = localUninstallCommand({ dryRun: false, purgeWeights: true }, { home });
    assert.equal(purge.argv.includes('--purge-weights'), true);
    assert.equal(purge.argv.includes('--dry-run'), false);
    assert.equal(purge.argv.includes('--wsl-too'), false);
  } finally { rmSync(home, { recursive: true }); }
});
