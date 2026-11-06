# The flip: Crucible 2.0.35

REORG Phase C's last step (`bookforge/docs/REORG-PLAN.md`, C3) on paper. Nothing here
runs until Owen's go. Owen's test pass gates the release, the deploys and the promote.
The GPU go for any rebuild is the bookforge lead's, per named run, one per card.

## A. What goes in

These changes go on top of v2.0.34, in landing order.

| Commits | Change |
|---|---|
| bcbd1e3 | cuda-linux `crucible init` writes Higgs `max_num_seqs = 12` |
| 4de6c08, 742b39f, 812e16c | Crucible owns the Higgs worker (`crucible_higgs`) and its tests |
| ee3be0a, 16d7f84 | built-in Higgs voices take their width from the footprint; with no footprint they are refused at load |
| 7fbe103 | `insufficient_kv_cache` names the term that binds |
| 2a70590 | the bookforge module lists `qwen3-asr-1.7b` |
| 1de4b98 | resumable downloads |
| a58d479 … a515bd3 (9) | the repair task, `doctor --repair` / `--verify-only`, `bootstrap.repair()`, `scripts/repair_e2e.py` |
| 1723fc5 | bootstrap reads real configs |
| 5105a32 | the host door reads every POST's body |
| 3fc9308, 151d245, bcdddf3, 1abc8e7, 4e0820a, e36bba6, 433a97d | CI and test fixes |
| 1ab1fc5, 83a0805 | `edges` on `GET /v1/voices` |
| b66ffa1 | repair names a pin the hub no longer has |
| 71e4d9d | CI runs the Higgs worker's tests |
| e487281, d5ea75d, cab9ab6 | tool downloads wait out a release host's bad moment |
| 97798dd | test_segment_cli stubs ffmpeg |
| b92c050 | the YuE2 causal-check test runs |
| 4515c53, fa3f6c7, 11826dd | C3: the tts recipes drop narrator; the Mac recipe keeps torch |
| 384ca10 | CI server jobs time out at 90 minutes |

All of it is on main at 384ca10.

## B. Before ship.sh

1. Done: BookForge B6 step 8 reached main at 246213f4, and C3 was rebased and landed on
   crucible main at 384ca10, with CI 7/7 green (run 42153151038).
2. Still to do, right before the cut: `git fetch --tags origin`. Release tags are made
   server-side, so a checkout never has them. The last tag is v2.0.34.

## C. The changelog

`scripts/ship.sh` renames `## Unreleased` to `## 2.0.35 — <date>` and moves every line
over verbatim. The section reads, top to bottom:

- CI's server jobs stop after 90 minutes.
- The tts envs no longer install narrator. They pin `num2words` and `docopt`, and the
  Mac keeps `torch`. Each machine needs one `--force` rebuild.
- The YuE2 causal-check test runs wherever torch is installed.
- Tool downloads wait out a release host's bad moment.
- CI runs the Higgs worker's tests.
- The Windows host's door reads every POST's body.
- `@crucible/bootstrap` reads real configs.
- `GET /v1/voices` carries `edges`.
- The repair task.
- Built-in Higgs voices take their width from the footprint.
- `insufficient_kv_cache` names the binding term.
- Resumable downloads.
- The bookforge module lists `qwen3-asr-1.7b`.
- cuda-linux init writes Higgs `max_num_seqs = 12`.
- The Higgs worker is Crucible's own code.

No line needs rewording.

## D. The sequence

1. **Cut.** From `C:\Users\tellt\Projects\crucible` on main, run
   `bash scripts/ship.sh patch`, without `--deploy`, so each machine below goes one at a
   time. It bumps to 2.0.35 (the seven version places, plus the regenerated modules and
   API docs), runs the changed tests in WSL, cuts the release as a prerelease and
   dispatches the env packs.
2. **This PC.** Run `./scripts/deploy.sh --release 2.0.35 --only pc --yes`. Then rebuild
   the tts env inside WSL: `crucible install tts --narrator-engine higgs-v3 --force`.
   Check first that gcc is in WSL: cdifflib and docopt build from source, and the
   doctor check for it is being added. The rebuild takes about 3.5 minutes with a warm
   pip cache. With a cold cache it downloads about 5.3 GB more. Afterwards
   `crucible doctor` should show the tts env installed, and one short Higgs render
   and one stream should come back.
3. **The Mac.** Run `./scripts/deploy.sh --release 2.0.35 --only mac --yes`, then the
   same `--force` rebuild. It takes under 30 seconds, or downloads about 0.9 GB more on
   a cold cache. Run the same checks.
4. **Victoria's laptop, last.** Do it at her desk, with her okay. Run
   `install.ps1 -Release 2.0.35`, then `crucible doctor --verify-only`, which checks her
   install. She has no Higgs and does not need it (Owen), so there is no tts rebuild and
   no Deathstalker check. She gets this update for YuE2 (`audio`), which C3 does not touch.
5. **Promote.** This is Owen's attestation after his test pass:
   `python scripts/promote_release.py --tag v2.0.35 --publish --confirmed-install-smoke`.
6. **The apps.**
   - a. BookForge adopts 2.0.35 with `node tools/adopt-crucible-release.mjs 2.0.35`.
     Owner: bookforge lead.
   - b. BookForge lands `feat/crucible-repair-button` (29578b2d, rebased). Owner:
     bookforge lead.
   - c. BookForge records `edges` in the session, read from the released
     `/v1/voices`, and drops `edgeFadeMs` from `higgs-models.json`. Owner: bookforge pc 2.
   - d. Foundry adopts 2.0.35 with the same `adopt-crucible-release.mjs`. Owner:
     foundry pc 1.

Kylie's PC is off and is not updated in this release (Owen). That overrides "all four
servers current" for 2.0.35.

B-Sides' text moves fully to `claude -p` until Owen retrains his model, so
`qwen3.5-4b-bside` is not uploaded. The Mac's copy stays where it is and the catalog
entry stays as it is.

## E. Downtime per machine

- **The service restart:** seconds.
- **tts:** the only verb that is down, from that machine's deploy until its rebuild.
  The new recipe no longer matches the env's stamp, so tts refuses with "run
  `crucible install tts --force`". That is about 4 minutes on this PC with a warm
  cache, and under a minute on the Mac. Every other verb stays up.
- **Nothing may be rendering at deploy.** `deploy.sh` refuses a busy server without
  `--interrupt`, and the flip does not pass it.

## F. Open before the go

- Owen's test pass.
- B6 step 8 is on BookForge main (246213f4); installing it is the next line's condition.
- **No machine's tts env is rebuilt while a BookForge build that still spawns
  narrator is live on it.** Owen's installed BookForge must be the step-8 build
  before this PC's rebuild in D2, and the same goes for any other machine running
  BookForge before its own rebuild.
- Owen told about the rebuilds in advance.
- The held rulings stay out of this release: 884acdf (small-card LLM fit), the
  reserve value, and the B-Sides picker.
