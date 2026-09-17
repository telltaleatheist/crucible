/**
 * The SDK's own build version, as it appears in `User-Agent`.
 *
 * It is a literal rather than a read of `package.json` so that the ESM build,
 * the CJS build and a bundler all agree without an import assertion or a
 * filesystem read. `test/unit.test.ts` fails if it drifts from `package.json`,
 * and `scripts/release.sh` refuses to cut a release if either drifts from
 * `crucible/__init__.py`.
 */
export const SDK_VERSION = '0.6.7';
