# Crucible 0.6.2 candidate verification

Published candidate: [v0.6.2](https://github.com/telltaleatheist/crucible/releases/tag/v0.6.2).
Its immutable source tag names `d363eaf27cefe26e1487813a7528ce332d23dde5`.
It is a prerelease, not latest; v0.6.0 remained latest after publication.
Fresh published installation acceptance is a separate gate before promotion.

## Artifacts

All 29 assets were uploaded into a draft and checked against their local SHA256
digests and byte counts before public visibility. Total: 17,348,272,397 bytes.
The complete 13-pack `envpacks.json` was uploaded last. The public installers,
manifest, and both SDK archives were downloaded again and matched their expected
bytes. `scripts/promote_release.py --tag v0.6.2` passed in read-only mode.

Three core runtimes were rebuilt from the exact tag source. The Windows build
used an isolated git archive; Linux and Mac used
[artifact run 35062685950](https://github.com/telltaleatheist/crucible/actions/runs/35062685950).
The wheel's 148 packaged source/data files matched that canonical source
byte-for-byte. Both SDK archives were built once from the same isolated source;
the exact published archives were supplied to BookForge and Foundry.

Ten inference archives were reused only after rechecking the source recipes,
standalone interpreter versions, archive sizes, joined archive SHA256 values,
and individual part digests. Original 0.6.0/0.6.1 part filenames are retained in
the new manifest. The rootfs contains no Crucible runtime; its builder was
unchanged, and its verified existing bytes were reused under the 0.6.2 filename.
The published `release-provenance.json` and `SHA256SUMS` record the artifacts.

## Evidence and limits

- Client: 315 tests passed; bootstrap: 273 passed, including real-shell LF/CRLF
  manifest parsing. Focused packaging: 56 passed.
- Migration/lifecycle checks before the version bump: 216 passed, one platform
  skip. All three relocated core runtime builds passed.
- Mac and Windows packaged runtimes both completed actual fresh-home
  initialization. Linux packaging used a CPU runner and does not claim NVIDIA
  initialization or model inference acceptance.
- Full Linux Python 3.11 CI passed at the release commit. Python 3.12 CI lacked
  setuptools for its no-isolation wheel fixture; the separate wheel build
  passed, and workflow-only commit `4948204` supplies that missing prerequisite.
  A full passing native Windows suite is not claimed: some existing fixtures
  require POSIX process behavior, executable scripts, or permissions.
- No local live Crucible, WSL, or model operations were performed during this
  release assembly. Signing and complete app acceptance are separate checks.

The automatic tag trigger was removed from `envpacks.yml`. Full pack publication
requires explicit dispatch; `scripts/release.sh` dispatches it when using that
full-build path. This candidate used artifact assembly instead. Tag publication
did not start a duplicate pack builder or overwrite any release assets.

The broken 0.6.1 candidate remains unchanged, with a prominent warning in its
release notes. Neither its tag nor its assets were replaced.
