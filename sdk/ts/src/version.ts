/**
 * The SDK's own build version, as it appears in `User-Agent`.
 *
 * It is a literal rather than a read of `package.json` so that the ESM build,
 * the CJS build and a bundler all agree without an import assertion or a
 * filesystem read. `test/unit.test.ts` fails if it drifts from `package.json`,
 * and `scripts/release.sh` refuses to cut a release if either drifts from
 * `crucible/__init__.py`.
 *
 * **IT IS A BUILD LABEL AND NEVER A COMPATIBILITY STATEMENT**
 * (docs/INSTALL-UNINSTALL.md §6.5.5). What decides whether this client can talk
 * to a server is `API_VERSION` — the `X-Crucible-Api` header the server refuses
 * a request without, and the `api_version` `connect.ts` refuses a mismatch on by
 * name. This number says which BUILD is speaking, which is what a User-Agent is
 * for and what a support question needs.
 *
 * So an app does not have to run a client cut at the same version as the server
 * it talks to, and it never had to: BookForge's `package.json` claimed "the
 * service installer and SDK must name the same release" and that sentence is
 * retired. `release.sh` holds the seven places to one number because one RELEASE
 * is cut at one version, not because two different versions could not
 * interoperate.
 */
export const SDK_VERSION = '1.0.40';
