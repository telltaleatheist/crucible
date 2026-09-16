import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { serverPackSh } from '../src/steps.js';

const shell = process.env['CRUCIBLE_TEST_BASH'] ?? (process.platform === 'win32' ? undefined : '/bin/sh');
const manifest = readFileSync(new URL('../../test/fixtures/release-0.6.1-envpacks.json', import.meta.url), 'utf8');
for (const backend of ['mlx-darwin', 'cuda-linux']) {
  for (const ending of ['LF', 'CRLF']) {
    test(`shell download reads actual pretty release manifest: ${backend} ${ending}`, { skip: shell === undefined }, () => {
      const root = mkdtempSync(join(tmpdir(), 'crucible-manifest-'));
      try {
        const normalized = manifest.replace(/\r\n/g, '\n');
        writeFileSync(join(root, 'manifest.json'), ending === 'LF' ? normalized : normalized.replace(/\n/g, '\r\n'));
        const program = `set -eu
CRUCIBLE_HOME="$(pwd)/home"
BACKEND='${backend}'
RELEASE=0.6.1
stamp_sha=''
free_kib=999999999
SHA_TOOL=sha256sum
say() { :; }
die() { printf '%s\\n' "$*" >&2; exit 1; }
# Git Bash awk translates CRLF in text mode, unlike macOS/Linux awk.
# BINMODE preserves the published bytes so Windows can reproduce that defect.
awk() { command awk -v BINMODE=3 "$@"; }
curl() {
  case "$*" in
    *envpacks.json) cat manifest.json;;
    *) printf '%s\\0' "$@" > download-argv; return 99;;
  esac
}
` + serverPackSh();
        const result = spawnSync(shell!, ['-c', program], { cwd: root, encoding: 'utf8' });
        assert.equal(result.status, 1, result.stderr);
        assert.match(result.stderr, /pack_download_failed/);
        const args = readFileSync(join(root, 'download-argv'), 'utf8').split('\0').filter(Boolean);
        assert.equal(args.at(-1), `https://github.com/telltaleatheist/crucible/releases/download/v0.6.1/crucible-env-server-${backend}-0.6.1.tar.zst.part00`);
        assert.ok(args.every(arg => !/[\r\n]/.test(arg)), 'no carriage return or newline may leak into a path or URL');
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });
  }
}
