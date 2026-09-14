// Remove the two build trees so a rebuild can never serve a stale file.
import { rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
for (const tree of ['dist', 'build']) {
  await rm(join(packageRoot, tree), { recursive: true, force: true });
}
