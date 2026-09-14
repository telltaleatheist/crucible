/**
 * One release, one version: this package, its literal, the `@crucible/client`
 * it is pinned to, and the SDK checkout beside it all say the same number.
 * `scripts/release.sh` checks the same thing against `crucible/__init__.py`.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { SDK_VERSION } from '@crucible/client';

import { BOOTSTRAP_VERSION } from '../src/index.js';

const here = new URL('../../package.json', import.meta.url);
const pkg = JSON.parse(readFileSync(here, 'utf8')) as { version: string; peerDependencies: Record<string, string>; dependencies?: Record<string, string> };

test('BOOTSTRAP_VERSION matches package.json', () => {
  assert.equal(BOOTSTRAP_VERSION, pkg.version);
});

test('the @crucible/client peer pin is exact and is this version', () => {
  assert.equal(pkg.peerDependencies['@crucible/client'], BOOTSTRAP_VERSION);
});

test('the SDK this was built against is the same version', () => {
  assert.equal(SDK_VERSION, BOOTSTRAP_VERSION);
});

test('zero runtime dependencies: nothing but the peer', () => {
  assert.equal(pkg.dependencies, undefined);
});
