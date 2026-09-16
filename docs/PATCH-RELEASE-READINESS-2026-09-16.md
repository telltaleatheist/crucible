# Patch release readiness — 2026-09-16

The lifecycle changes are not in the published 0.6.0 runtime packs. Source,
client, bootstrap, generated installers, module manifests and both app vendor
dependencies now identify this candidate as 0.6.1. Fresh installs request 0.6.1
and fail explicitly until its runtime packs are published. No binary release
has been deployed. Development checkout tests do not prove that release path.

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

The version bump and small SDK/wheel artifacts are prepared. No release, tag,
service restart, GPU job, WSL change, or multi-GB pack download was performed.
Actual runtime/TTS pack builds, fresh release-based installation, publication and
promotion remain outstanding.
