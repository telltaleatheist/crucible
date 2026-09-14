/**
 * `parsePairing` — the inverse of `crucible/pairing.py`, tested against the
 * same literal from the other side.
 *
 * PHASE13-OPERATOR.md section 2.1 is the rule, and it has two implementations
 * because one of them has to run in TypeScript. The seam is therefore a
 * LITERAL LINE, written out in both suites: the Python producer's test asserts
 * it is what `pairing_line` emits, and this asserts it is what `parsePairing`
 * reads. Neither test is written against the other's code, so a change to the
 * rule fails on both sides with the old and new spelling visible.
 *
 * The refusals matter as much as the success. A name contains an `@`, so a
 * line that did not encode it has two, and "take the last one" is the guess
 * ARCHITECTURE.md R3 forbids — the failures below are where that is held to.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  CruciblePairingError,
  INVALID_PAIRING,
  PAIRING_SCHEME,
  parsePairing,
} from '../src/index.js';

/**
 * THE SEAM. Identical to `PAIRING_LITERAL` in `tests/test_setup_route.py`.
 * A server called `crucible@mac-studio`, on a LAN address, with a token made
 * of exactly the characters `secrets.token_urlsafe` produces.
 */
const LINE = 'crucible://crucible%40mac-studio@192.168.68.20:7100/#s3cret-t0ken_x';

// ------------------------------------------------------------------ reading


test('the seam line reads back as its three parts', () => {
  assert.deepEqual(parsePairing(LINE), {
    name: 'crucible@mac-studio',
    url: 'http://192.168.68.20:7100',
    token: 's3cret-t0ken_x',
  });
});

test('the url is exactly what CrucibleClient takes: no /v1, no trailing slash', () => {
  const { url } = parsePairing(LINE);
  assert.equal(url.endsWith('/'), false);
  assert.equal(url.includes('/v1'), false);
});

test('a percent-encoded name comes back whole, spaces and slashes included', () => {
  const parsed = parsePairing('crucible://owen%27s%20box%2F2@10.0.0.5:7100/#t');
  assert.equal(parsed.name, "owen's box/2");
});

test('an IPv6 host keeps its brackets', () => {
  const parsed = parsePairing('crucible://n@[fd00::1]:7100/#t');
  assert.equal(parsed.url, 'http://[fd00::1]:7100');
});

test('surrounding whitespace from a paste is trimmed', () => {
  assert.deepEqual(parsePairing(`\n  ${LINE}\t `), parsePairing(LINE));
});

test('a token with reserved characters survives the round trip', () => {
  const parsed = parsePairing('crucible://n@h:1/#a%2Fb%3Fc');
  assert.equal(parsed.token, 'a/b?c');
});

// ----------------------------------------------------------------- refusing


function refused(line: unknown, because: RegExp): CruciblePairingError {
  let caught: unknown;
  try {
    parsePairing(line as string);
  } catch (error) {
    caught = error;
  }
  assert.ok(
    caught instanceof CruciblePairingError,
    `expected invalid_pairing, got ${String(caught)}`,
  );
  assert.equal(caught.code, INVALID_PAIRING);
  assert.match(caught.detail, because);
  return caught;
}

test('a second literal @ is refused rather than guessed at', () => {
  // The line a naive producer would write: the name's own `@` left raw.
  refused('crucible://crucible@mac-studio@192.168.68.20:7100/#t', /two @/);
});

test('another scheme is refused', () => {
  refused('http://crucible%40x@10.0.0.5:7100/#t', new RegExp(PAIRING_SCHEME));
});

test('a line with no token is refused', () => {
  refused('crucible://n@10.0.0.5:7100/', /#fragment/);
});

test('an empty token is refused, not accepted as a blank one', () => {
  refused('crucible://n@10.0.0.5:7100/#', /token is empty/);
});

test('an empty name is refused', () => {
  refused('crucible://@10.0.0.5:7100/#t', /server name is empty/);
});

test('the / before the # is required', () => {
  refused('crucible://n@10.0.0.5:7100#t', /ends its address with/);
});

test('a path is refused: a pairing line has none', () => {
  refused('crucible://n@10.0.0.5:7100/v1/#t', /no path/);
});

test('a missing port is refused rather than defaulted to 7100', () => {
  refused('crucible://n@10.0.0.5/#t', /<host>:<port>/);
});

test('a port out of range is refused', () => {
  refused('crucible://n@10.0.0.5:99999/#t', /<host>:<port>|is not a port/);
});

test('broken percent-encoding is refused, never half-decoded', () => {
  refused('crucible://a%ZZb@10.0.0.5:7100/#t', /percent-encoding/);
});

test('an empty string and a non-string are refused before anything is parsed', () => {
  refused('', /empty/);
  refused('   ', /empty/);
  refused(undefined, /is a string/);
  refused(42, /is a string/);
});

// ------------------------------------------------------- the token stays put

test('a refusal never carries the token, because a refusal gets logged', () => {
  const caught = refused('crucible://a@b@10.0.0.5:7100/#s3cret-t0ken_x', /two @/);
  assert.equal(caught.line.includes('s3cret-t0ken_x'), false);
  assert.equal(caught.message.includes('s3cret-t0ken_x'), false);
  assert.match(caught.line, /#…$/);
});

test('a very long line is elided rather than pasted whole into a message', () => {
  // A host of 400 legal characters IS a legal host, so the refusal has to come
  // from something else — here the missing port — and what is asserted is the
  // elision, not the reason.
  const caught = refused(`crucible://n@${'x'.repeat(400)}/#t`, /<host>:<port>/);
  assert.ok(caught.line.length <= 120, caught.line.length.toString());
});
