// Tell Node which module system each build tree is.
//
// The package itself is "type": "module", so dist/esm needs no marker — but it
// gets one anyway, because a tree that says what it is cannot be misread after
// a refactor. dist/cjs must say "commonjs" or Node would load tsc's `require`
// output as ESM and fail on the first line.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const trees = { esm: 'module', cjs: 'commonjs' };

for (const [tree, type] of Object.entries(trees)) {
  const directory = join(packageRoot, 'dist', tree);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, 'package.json'), `${JSON.stringify({ type }, null, 2)}\n`, 'utf8');
}
