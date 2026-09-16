# Patch release readiness — 2026-09-16

The corrected 0.6.2 candidate is now published separately; see
[its artifact verification record](RELEASE-0.6.2-VERIFICATION.md).
The 0.6.1 history below is retained as evidence of the failed acceptance checks.

The lifecycle changes are not in the published 0.6.0 runtime packs. Source,
client, bootstrap, generated installers, module manifests and both app vendor
dependencies identify this candidate as 0.6.1. Its complete binary prerelease is
now published from source commit `407886bace61fa66bf273c306b8f67f6f5e3f343`.
It remains separate from the recommended latest release pending fresh installation
validation. Development checkout tests alone do not prove that release path.

**Installation validation found a blocker: do not promote 0.6.1.** The Mac's actual
fresh installation read the valid CRLF-formatted manifest through a shell parser
that removed LF but retained CR. The resulting part URL contained carriage returns
before and after its filename, so curl refused it. The acceptance procedure restored
the previous Mac service. Published binaries/installers and the tag remain unchanged;
the corrected installer must ship in the next coordinated patch release.

The source fix normalizes CR and LF before extracting fields and removes JSON
whitespace from the parts list. A fixture copied from the actual published 13-pack
manifest reproduces the exact malformed URL before the fix. Real shell tests cover
Mac/Linux pack selection and LF/CRLF input; Git Bash awk uses binary input mode in
the test because its default Windows text translation otherwise hides this POSIX
failure. All 273 bootstrap tests pass after the fix. Artifact checksum/relocation
checks had passed for 0.6.1; they do not replace a real fresh-install acceptance test.

**A second blocker appeared after testing the corrected parser:** the published
Mac core pack does not contain MLX, but `crucible init` imports `mlx.core` in the
core interpreter before any worker environment exists. The Mac's previous service
was again restored. The core wheel now declares MLX on Darwin arm64, fixing both
pack and pip installs. Relocated Mac and Windows core smoke tests now run actual
`init --backend` in an isolated fresh home and require the config to be written;
they do not start services or fetch models. CUDA initialization still requires a
real NVIDIA acceptance host, so CPU-only pack builders explicitly report that it
was not exercised. Release 0.6.1's public notes now warn about both defects; its
assets and source tag remain unchanged.

The fix is committed in `db9b2b7d961c253212b8e2f24cb93f505940eaf8`.
[Artifact-only run 35061594490](https://github.com/telltaleatheist/crucible/actions/runs/35061594490)
rebuilt both POSIX core packs from that exact commit and passed. The relocated
Mac pack explicitly reported `fresh-home init with real mlx-darwin detection ok`;
the Linux runner explicitly reported that NVIDIA initialization was not exercised.
These are private build artifacts, not replacements for published 0.6.1 assets.
Focused regression results: 273 bootstrap tests passed with zero skips, including
the real-shell manifest cases; 86 Python packaging tests passed with 14 existing
platform/archive-tool skips.

## Measured release inputs

Read the public 0.6.0 release asset list and `envpacks.json`, then compared every
entry against the current checkout's line-ending-normalized recipe digest and
standalone Python version. No assets were uploaded or replaced.

| Inputs | Required action |
|---|---|
| Windows host; Linux server; macOS server | Rebuild from the corrected source. Old archives total 134,390,441 bytes. A matching `pyproject.toml` digest does **not** prove matching Python source. |
| CUDA Linux and macOS llm/asr/align/rvc (8 archives) | Eligible for verified reuse: recipes and Python pins match. Old archives total 11,654,094,328 bytes. |
| CUDA Higgs TTS | Rebuild: recipe changed from vLLM-Omni/Python 3.11 to SGLang-Omni/Python 3.12. Fixed the builder to select the verified Python 3.12.14 standalone pin for this recipe only. Full dependency build remains untested in this pass. |
| macOS TTS | Rebuild: recipe changed (including narrator source pin). |
| WSL rootfs and its SHA256 asset | Build under the new release's name with the existing rootfs job. This is separate from environment-pack completeness. |

The old Higgs archive must never be relabeled to bypass the recipe mismatch.
The release workflow's complete-manifest check must remain in force.

## Read-only preflight and local staging

From the repository root:

```sh
python scripts/release_packs.py \
  --source https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/envpacks.json \
  --version 0.6.1 --check
```

This reads the planned 0.6.1 inputs; it does not change any version. `--check`
exits nonzero if a declared recipe needs an interpreter the builder cannot select.

The helper can prepare one unchanged inference archive locally:

```sh
python scripts/release_packs.py \
  --source https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/envpacks.json \
  --version 0.6.1 --stage llm mlx-darwin --out release-staging/llm-mlx-darwin
```

The output directory must be new. It verifies the concatenated archive size and
SHA256 before emitting a target-version manifest fragment; failed verification
leaves no usable output. It refuses runtime packs, changed recipes, incompatible
Python versions, unexpected filenames, and same-version replacement. CPU-only
tests cover these refusal and success paths.

Schema 1 resolves every part beside its manifest. Reuse therefore requires
uploading the verified parts to the **new** release with their original filenames;
changing only the manifest version does not make old-release parts available
there. The new fragment intentionally retains the original part names and hashes.
No cross-release URL convention or install-time fallback is introduced.

Review changes to pack construction and interpreter pins as well as recipe
digests before reusing environments. The manifest records a Python version but
does not record the interpreter archive digest or every build-tool revision;
the helper is not a substitute for that release review.

## Coordinated release sequence

1. Target-specific interpreter selection is fixed using upstream Python 3.12.14
   and its verified SHA256SUMS digest. Next build/smoke-test both
   changed TTS packs on their target platforms. A pack build requires no model
   weights or running inference, but disk and dependency resolution still need
   verification.
2. The 0.6.1 version is coordinated across `crucible/__init__.py`,
   `pyproject.toml`, both SDK package versions, client and bootstrap version
   literals, bootstrap's exact client peer dependency, and affected lockfiles.
   Both installers and app module manifests are regenerated; consumer vendor
   packages and locks use the corresponding new SDK tarballs.
3. Build the wheel/sdist/client/bootstrap and all three runtime packs from the
   same committed source. Smoke-test relocated packs with `local --help`,
   `local register --help`, `local install-cli --help`, and
   `local install-desktop --help`, in addition to the existing version/catalog
   checks. Test a fresh native Windows install from staged artifacts.
4. Stage/upload the eight verified reusable inference packs and five rebuilt
   packs, plus rootfs assets, before making the release the recommended version.
   Merge fragments using `envpack.merge_manifests` and verify every declared pack
   and every named part is actually present. Upload `envpacks.json` last.
5. Promote only the complete, tested release. `scripts/release.sh` now creates a
   prerelease with `--latest=false`. The tag workflow still rebuilds every pack;
   an opt-in reuse matrix has **not** been added. The staging helper supports
   deliberate, verified reuse if an operator chooses to assemble the release
   manually instead of letting the complete build run.
   A prerelease supports normal release-asset URLs; a draft does not provide the
   unauthenticated download path installers need for candidate validation.
   `python scripts/promote_release.py --tag v0.6.1` checks completeness and sizes
   of every declared archive part, core/SDK/install assets, rootfs and its digest
   companion, plus recipe/interpreter agreement with this checkout. It checks
   metadata, not the archive contents; those hashes are verified during pack
   build/reuse and fresh installation. Promotion additionally requires
   `--publish --confirmed-install-smoke`, an explicit operator attestation.
6. Test consumer fresh-install URLs against the published candidate before
   promotion. Until that passes, report native installation as source-validated,
   with release installation still pending. Never replace 0.6.0 assets with new
   source carrying the old version.

## Actual artifact preflight, 2026-09-16

The artifact-only `release-preflight.yml` workflow was added and run against exact
commit `a8ccb077e2cbf3e69965cc8466e41ad0401220fa`. Its token has `contents: read`;
it cannot upload release assets or promote a release. All three jobs in
[run 35058903633](https://github.com/telltaleatheist/crucible/actions/runs/35058903633)
passed:

| Artifact | Measured result |
|---|---|
| SGLang Higgs / CUDA Linux | Python 3.12.14; 5,336,358,502 compressed bytes in three parts; 13,324,566,947 unpacked bytes; build and relocated narrator import passed in 404 seconds. SHA256 `daeba1022c2a4eef004cfb5621b01812980db21b3fe8b92da36806902b1ca1f9`. |
| Mac TTS | 164,290,057 compressed bytes; build and relocated narrator import passed in 80 seconds. Downloaded archive and recipe reverified locally. SHA256 `b4b5534ab0771ca46b91e2b2ccbf99592bd14f2189a4523d86fea3bb54042aec`. |
| WSL rootfs | 30,689,880 bytes; archive/ownership-marker checks passed; downloaded checksum verified. SHA256 `f3aafba32e9eb7238c10bd4e902b773be02b9efa7044d091cbbfc301d9d30856`. |

All eight eligible 0.6.0 inference archives were downloaded and their complete
concatenated sizes and SHA256 digests verified before writing 0.6.1 staging
fragments. These remain a repair option if the final full build needs one;
they are not uploaded to a release by the staging helper.

An isolated Windows source archive of the same commit built a 46,420,828-byte
host pack in 71 seconds. Its relocated CLI version and lifecycle parser smoke
checks passed. This is packaging preflight, not the final core: connection and
pairing changes made afterward require all final runtime packs to be built from
the final release commit. Inference recipes did not change during that work.

Artifacts and build logs are staged under
`C:\Users\tellt\Projects\crucible-release-staging\0.6.1`. No local service,
GPU workload, model, or WSL instance was changed during these builds.

## Complete published candidate

The final Windows host, Linux server, and Mac server packs were rebuilt from frozen
commit `407886bace61fa66bf273c306b8f67f6f5e3f343`. Both POSIX packs passed the
artifact-only [core run 35059968205](https://github.com/telltaleatheist/crucible/actions/runs/35059968205).
Windows built locally in an isolated source archive and passed relocated CLI smoke
checks; all 148 wheel package files matched the frozen source and installed host
archive byte-for-byte. No live controller was started for this packaging proof.

The Python wheel/sdist and both SDKs were built from that source. The client archive
used by the apps was compared with the isolated rebuild: seven source files differed
only in CRLF/LF line endings. The exact app-consumer archive was published. Generated
installer drift checks passed.

[v0.6.1](https://github.com/telltaleatheist/crucible/releases/tag/v0.6.1) now carries
29 assets: all 13 packs in 18 parts, Python/SDK/install/rootfs artifacts, complete
manifest, provenance, and checksum inventory. The 18 archive parts total
17,298,249,463 bytes. Every pre-manifest asset's GitHub-reported SHA256 and size
matched the local verified input before `envpacks.json` was uploaded last. The
read-only promotion validator passed the complete published manifest/source check.

The automatic tag build queued despite a brief workflow pause around tag creation.
Only that redundant exact-tag run was cancelled. Its two completed asset uploads
were restored from the verified staging files before the digest checks; the ordinary
pack workflow is enabled. No unrelated workflow run was cancelled.

The release is public, marked prerelease, and not latest. Fresh installation
acceptance and promotion are still separate gates. The published tag points to the
frozen core commit; later workflow/documentation commits do not change its runtime.
