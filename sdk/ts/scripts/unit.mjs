// Run every compiled unit test, found rather than listed.
//
// `test:unit` used to name its files one by one in package.json. That works
// until somebody adds a file — and then it does the worst possible thing, which
// is nothing: the suite stays green, the count goes up by zero, and a test
// nobody notices is missing is indistinguishable from a test that passes. It
// happened the night `unit-unknown-event.test.ts` was written, which is a test
// about a client losing a whole event stream.
//
// So the list is derived from the directory. `unit-*.test.js` and `unit.test.js`
// are the unit tests; `e2e*.test.js` are deliberately excluded, because those
// need a live server and have their own scripts.
//
// A glob in the npm script itself would not do: npm runs scripts through cmd.exe
// on Windows, which does not expand one, and Node's own glob support for
// `--test` landed after the version this SDK supports.

import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

const DIR = join('build', 'test');

let names;
try {
  names = readdirSync(DIR);
} catch (error) {
  console.error(
    `unit: no compiled tests at ${DIR} (${error.code}). ` +
      'Run `npm run build:test` first, or use `npm run test:unit`, which does.',
  );
  process.exit(2);
}

const files = names
  .filter((name) => name.endsWith('.test.js'))
  .filter((name) => name === 'unit.test.js' || name.startsWith('unit-'))
  .sort()
  .map((name) => join(DIR, name));

if (files.length === 0) {
  // Never "0 tests, all passing". An empty run is a broken build, not a clean one.
  console.error(`unit: found no unit tests in ${DIR}; this is a failure, not an empty pass`);
  process.exit(2);
}

console.log(`unit: ${files.length} file(s): ${files.map((f) => f.replace(/\\/g, '/')).join(' ')}`);
const result = spawnSync(process.execPath, ['--test', ...files], { stdio: 'inherit' });
process.exit(result.status ?? 1);
