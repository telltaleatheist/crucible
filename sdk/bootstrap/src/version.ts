/**
 * This package's own build version.
 *
 * A literal rather than a read of `package.json`, for the reason `@crucible/client`
 * gives: the ESM build, the CJS build and a bundler all agree without an import
 * assertion or a filesystem read. `test/unit-version.test.ts` fails if it drifts
 * from `package.json` or from the `@crucible/client` peer pin, and
 * `scripts/release.sh` refuses to cut a release if any of them drifts from
 * `crucible/__init__.py` — a bootstrapper is never paired with a server nobody
 * tested it against (PHASE5-APPS.md section 6.0).
 */
export const BOOTSTRAP_VERSION = '0.6.0';
