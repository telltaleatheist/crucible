/**
 * Generate `install.sh`, `install.ps1` and `crucible/host/wsl_states.py` from
 * the step list and the WSL state table — the things PHASE14-ENVPACKS.md 4a
 * and 4c, and PHASE15-HOST.md 4.3, say have ONE owner.
 *
 *   node build/scripts/gen-install-scripts.js            # write them
 *   node build/scripts/gen-install-scripts.js --check    # fail if they drift
 *
 * The `--check` mode is what `test/unit-gen-install.test.ts` asserts, so a
 * change to `src/steps.ts` that is not regenerated fails the suite rather than
 * shipping a hand installer that does something else than the app does.
 *
 * What the scripts deliberately do NOT do (4a): no job types, no weights. A
 * bare Crucible that serves nothing until an app or the operator page asks.
 *
 * THE THIRD OUTPUT IS PYTHON, AND THAT IS WHY IT IS HERE (PHASE15 4.3)
 * --------------------------------------------------------------------
 * `crucible host` drives the same 4c table, and it is Python. Two hand-written
 * copies of a ten-row table with ten sentences in it is exactly the shape
 * ARCHITECTURE.md R1 forbids, so the table's DATA is emitted into
 * `crucible/host/wsl_states.py` from this file, and `--check` refuses a drift
 * in it the same way it does for the two scripts. What is NOT emitted is the
 * `means` predicates, which are code rather than data — those live in
 * `crucible/host/wslstate.py`, one per code, and a pytest asserts the two sets
 * are equal. It is the seam `steps.ts` already uses for its three programs and
 * the one `envpack.SMOKE_IMPORT` uses for `cli.INSTALLABLE_JOB_TYPES`.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CRUCIBLE_DISTRO, WSL_CONF_MARKER, WSL_CONF_TEXT } from '../src/distro.js';
import { ENVPACKS_ASSET, HOST_BACKEND, HOST_PACK, RELEASE_REPO, rootfsAssetName } from '../src/envpacks.js';
import { activatePackSh, CURL_ARGS, DOWNLOADS_SUBDIR, HOST_SUBDIR, PARTIAL_SUFFIX, SERVER_SUBDIR, STAMP_NAME, TAR_ARGS } from '../src/pack.js';
import type { RunResult } from '../src/runner.js';
import { installJobTypesSh, installSteps, uninstallSh, type StepPlan } from '../src/steps.js';
import { BOOTSTRAP_VERSION } from '../src/version.js';
import { probeArgv, wslStates, type ProbeKey, type WslStateDef } from '../src/wsl-states.js';

const HERE = dirname(fileURLToPath(import.meta.url));
/** build/scripts → the package root. Written next to this generator, in `scripts/`. */
const SCRIPTS = join(HERE, '..', '..', 'scripts');
/** build/scripts → sdk/bootstrap → sdk → the repo root, where `crucible/` is. */
const REPO = join(HERE, '..', '..', '..', '..');

/**
 * The typographic characters Crucible's prose uses, and their ASCII.
 *
 * **`install.ps1` MUST BE ASCII, and this is the whole of why.** Windows
 * PowerShell 5.1 reads a `.ps1` with no byte-order mark as the system ANSI
 * code page, not as UTF-8 — and the generator writes UTF-8 with no BOM,
 * because a BOM breaks `irm … | iex` in other ways and because every other
 * file in this repo is BOM-less. So an em dash arrives at the 5.1 parser as
 * three cp1252 characters, one of which is a curly double quote, and MEASURED
 * on 2026-09-15: `[Parser]::ParseFile` on the generated script reported *"The
 * string is missing the terminator"* inside a `Die "…"` message that contained
 * one. Comments got away with it for a phase; a string does not.
 *
 * `irm | iex` is not the route that suffers — `Invoke-RestMethod` decodes the
 * HTTP body's declared UTF-8 — but `.\install.ps1 -Uninstall` IS, and that is
 * the documented way to pass a switch, because a piped script cannot take one.
 *
 * So the prose is transliterated rather than being written twice, and
 * `asciiOnly` REFUSES anything not in this table: a new character silently
 * degrading to `?` is exactly the failure this exists to stop.
 */
const ASCII_FOR: Readonly<Record<string, string>> = {
  '—': ' - ',   // em dash
  '–': '-',     // en dash
  '‘': "'",     // left single quote
  '’': "'",     // right single quote
  '“': '"',     // left double quote
  '”': '"',     // right double quote
  '…': '...',   // ellipsis
  ' ': ' ',     // non-breaking space
  '×': 'x',     // multiplication sign
  '→': '->',    // rightwards arrow
  '·': '-',     // middle dot
};

/** The text with {@link ASCII_FOR} applied, refusing any other non-ASCII. */
export function asciiOnly(text: string, what: string): string {
  const out = text.replace(/[^\x00-\x7f]/g, (character) => {
    const replacement = ASCII_FOR[character];
    if (replacement === undefined) {
      throw new Error(
        `gen-install-scripts: ${what} contains U+${character.codePointAt(0)!.toString(16).toUpperCase().padStart(4, '0')} `
        + `(${JSON.stringify(character)}), which has no ASCII spelling in ASCII_FOR. Windows PowerShell 5.1 reads a `
        + 'BOM-less .ps1 as the ANSI code page, so a character outside ASCII does not arrive as itself - add it to the '
        + 'table with the spelling you mean, or write the sentence in ASCII.',
      );
    }
    return replacement;
  });
  // Belt and braces: the transliteration itself must not smuggle one through.
  const left = out.match(/[^\x00-\x7f]/);
  if (left !== null) throw new Error(`gen-install-scripts: ${what} is still not ASCII after transliteration: ${JSON.stringify(left[0])}`);
  return out;
}

/** Every non-empty line two spaces in. For a block that lands inside an `if`. */
function indent(text: string): string {
  return text
    .split('\n')
    .map((line) => (line === '' ? '' : `  ${line}`))
    .join('\n');
}

const BANNER = (comment: string): string =>
  `${comment} GENERATED FILE — do not edit.\n`
  + `${comment} Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts\n`
  + `${comment} and src/wsl-states.ts, so a hand install and an app-driven install cannot\n`
  + `${comment} differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install\n`;

/**
 * The standalone installer's plan.
 *
 * Still no job types and no weights BY DEFAULT (4a: a bare Crucible that
 * serves nothing until an app or the operator page asks). What changed with
 * the droplet route is that a person at a terminal can now say otherwise, and
 * the two ways they say it are shell variables the generated script fills
 * from its own flags: `$BIND` for `--host`/`--port`, and `$JOB_TYPES` for
 * `--install <type>`. Both are EMPTY on a bare run, so the script a
 * `curl … | sh` produces is byte for byte the same install it was.
 */
const STANDALONE: StepPlan = {
  enableFlags: [],
  installs: [],
  bind: [{ sh: '$BIND' }],
  linger: true,
};

/**
 * The flags a hand install takes, and the refusal for a flag it does not.
 *
 * Nothing here has a default that does something: every variable starts empty
 * or `0`, and an empty one means the behaviour the script had before the flag
 * existed. An unknown flag is refused by name rather than ignored — a typo in
 * `--purge-weights` on a machine holding 57 GB of voices must not quietly run
 * the version that keeps them and then be believed to have run the other.
 */
function argumentsSh(): string {
  return [
    'usage() {',
    "  cat <<'USAGE'",
    'crucible install.sh — install or remove a Crucible on this machine.',
    '',
    'Install:',
    '  --token <t>          use this bearer token instead of minting one',
    '  --host <addr>        bind address for the server (default 127.0.0.1;',
    '                       a rented box is reached over the network, so it',
    '                       wants 0.0.0.0 — the bearer token is the lock)',
    '  --port <n>           bind port (default 7100)',
    '  --install <type>     also install this job type from its pack. Repeatable.',
    '                       tts names its engine: --install tts=higgs-v3',
    '  --from-source <ref>  build the server from a git ref instead of the',
    '                       published pack (a branch, a tag or a sha)',
    '  --release <version>  install this release rather than the built-in one',
    '  --min-free-gib <n>   refuse unless this much disk is free for the weights',
    '',
    'Remove:',
    '  --uninstall          undo the install, in the inverse order',
    '  --purge-weights      with --uninstall: delete the weights too',
    '  --dry-run            with --uninstall: print every step and touch nothing',
    '',
    'USAGE',
    '}',
    '',
    'UNINSTALL=0',
    'PURGE_WEIGHTS=0',
    'DRY_RUN=0',
    'TOKEN=""',
    'BIND=""',
    'JOB_TYPES=""',
    'FROM_SOURCE=""',
    'MIN_FREE_GIB=""',
    'need() { [ "$1" -ge 2 ] || die "flag_needs_value: $2 takes a value"; }',
    'while [ $# -gt 0 ]; do',
    '  case "$1" in',
    '    --uninstall) UNINSTALL=1 ;;',
    '    --purge-weights) PURGE_WEIGHTS=1 ;;',
    '    --dry-run) DRY_RUN=1 ;;',
    '    --token) need $# "--token"; shift; TOKEN="$1" ;;',
    '    --host) need $# "--host"; shift; BIND="$BIND --host $1" ;;',
    '    --port) need $# "--port"; shift; BIND="$BIND --port $1" ;;',
    '    --install) need $# "--install"; shift; JOB_TYPES="$JOB_TYPES $1" ;;',
    '    --from-source) need $# "--from-source"; shift; FROM_SOURCE="$1" ;;',
    '    --release) need $# "--release"; shift; RELEASE="$1" ;;',
    '    --min-free-gib) need $# "--min-free-gib"; shift; MIN_FREE_GIB="$1" ;;',
    '    -h|--help) usage; exit 0 ;;',
    '    *) die "unknown_flag: $1 is not a flag this installer takes; run with --help" ;;',
    '  esac',
    '  shift',
    'done',
    'if [ "$UNINSTALL" = 0 ] && [ "$PURGE_WEIGHTS" = 1 ]; then',
    '  die "flag_needs_uninstall: --purge-weights deletes weights and only means something with --uninstall"',
    'fi',
  ].join('\n');
}

/**
 * The prerequisites, checked BY NAME on the backend that has them.
 *
 * `cuda-linux` is the droplet's backend and the one with hardware to refuse
 * over. Each check names the thing it could not find and stops; none of them
 * guesses around a missing answer, because every way this can fail produces a
 * server that installs perfectly and then refuses its first job — which is
 * the failure mode `crucible doctor` exists for and which an installer should
 * not be adding to.
 *
 * `ffmpeg` is the one check that is CONDITIONAL, and deliberately: a bare llm
 * droplet does not need it, and refusing one that asked for nothing else
 * would be an installer inventing a requirement. It is required when a job
 * type that decodes audio was asked for, and REPORTED otherwise.
 *
 * Disk is the other conditional. The server pack states its own size and the
 * pack step already refuses `pack_disk` against it; WEIGHTS do not have a
 * size until somebody names a model, so the only honest disk rule here is the
 * one the operator states — `--min-free-gib` — plus the free figure, printed.
 */
function prerequisitesSh(): string {
  return [
    'if [ "$BACKEND" = cuda-linux ]; then',
    '  command -v nvidia-smi >/dev/null 2>&1 || die "no_nvidia_smi: there is no nvidia-smi on PATH. cuda-linux runs vLLM, SGLang and torch on an NVIDIA card; a box without the driver is not this backend"',
    '  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d \'[:space:]\')"',
    '  [ -n "$driver" ] || die "no_nvidia_driver: nvidia-smi is on PATH and named no driver. On a rented GPU box that usually means the image has the CUDA userland and not the kernel module"',
    '  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d \'[:space:].\')"',
    '  [ -n "$cap" ] || die "no_cuda_arch: this driver would not report a compute capability, so nothing here can say whether the engines will build for this card"',
    '  [ "$cap" -ge 70 ] || die "cuda_arch_too_old: this card reports compute capability $cap (7.0 is the floor: vLLM and SGLang ship no kernels below it, and torch\'s wheels drop it too). Rent a card at 7.0 or newer"',
    '  card="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"',
    '  say "prerequisites: $card, driver $driver, compute capability $cap"',
    'fi',
    'case " $JOB_TYPES " in',
    '  *" tts"*|*" asr"*|*" rvc"*|*" align"*|*" denoise"*)',
    '    command -v ffmpeg >/dev/null 2>&1 || die "no_ffmpeg: --install named a job type that decodes audio and there is no ffmpeg on PATH. Install it first; crucible would otherwise install cleanly and refuse its first job" ;;',
    '  *)',
    '    if command -v ffmpeg >/dev/null 2>&1; then',
    '      say "prerequisites: ffmpeg present"',
    '    else',
    '      say "prerequisites: NO ffmpeg on PATH. Nothing asked for today needs it; tts, asr, align, rvc and denoise will refuse until it is there"',
    '    fi ;;',
    'esac',
    'if [ -n "$MIN_FREE_GIB" ]; then',
    '  have_gib=$(( free_kib / 1048576 ))',
    '  [ "$have_gib" -ge "$MIN_FREE_GIB" ] || die "disk_too_small: $CRUCIBLE_HOME has ${have_gib} GiB free and --min-free-gib asked for $MIN_FREE_GIB"',
    'fi',
    'say "prerequisites: $(( free_kib / 1048576 )) GiB free at $CRUCIBLE_HOME. Weights are pulled later and priced then — a 9B model is ~18 GiB, a Higgs voice ~8.5 GiB"',
  ].join('\n');
}

/**
 * `--from-source <ref>`: the server built from a checkout, not from a pack.
 *
 * For the night a branch is what exists — tonight's `feat/phase6-remote-render`
 * is exactly that case — and for a droplet, where there is no app to ask for
 * a release. It is a SEPARATE path and never a fallback: a pack that failed to
 * download is `pack_download_failed` and stays that, because "the release is
 * broken" and "I want this branch" are different sentences and only one of
 * them is an argument.
 *
 * It needs a `python3` on the machine, and says so by name, because this is
 * the one route where the interpreter does not arrive with the code.
 */
function fromSourceSh(): string {
  return [
    `say "server-pack: --from-source $FROM_SOURCE, building instead of downloading"`,
    'command -v git >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs git"',
    'command -v python3 >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs a python3 on this machine to build the venv with. The published pack brings its own interpreter; a source build cannot"',
    `dest="$CRUCIBLE_HOME/${SERVER_SUBDIR}"`,
    'src="$CRUCIBLE_HOME/src"',
    'rm -rf "$src"',
    `git clone --filter=blob:none "https://github.com/${RELEASE_REPO}" "$src" || die "from_source_clone_failed: https://github.com/${RELEASE_REPO}"`,
    'git -C "$src" checkout --detach "$FROM_SOURCE" || die "from_source_ref_unknown: the checkout has no ref called $FROM_SOURCE"',
    'partial="$dest.partial"',
    'rm -rf "$partial"',
    'python3 -m venv "$partial" || die "from_source_venv_failed: python3 -m venv would not make $partial"',
    '"$partial/bin/python" -m pip install --upgrade pip setuptools wheel || die "from_source_install_failed: pip would not update itself in $partial"',
    '"$partial/bin/python" -m pip install "$src" || die "from_source_install_failed: pip would not install $src into $partial"',
    'if [ "$(uname -s)" = Darwin ]; then "$partial/bin/python" -m pip install pystray pillow || die "from_source_install_failed: desktop packages could not be installed"; fi',
    `"$partial/bin/python" -c 'from pathlib import Path; import sys; from crucible.envpack import relocate_console_scripts; relocate_console_scripts(Path(sys.argv[1]))' "$partial" || die "from_source_install_failed: console scripts could not be relocated"`,
    '"$partial/bin/crucible" --version >/dev/null || die "from_source_install_failed: $partial/bin/crucible would not run"',
    activatePackSh('"$dest"', '"$partial"') + ' || die "from_source_install_failed: runtime activation failed; previous runtime preserved"',
    `printf 'sha256=%s\\nrelease=%s\\n' "from-source" "$(git -C "$src" rev-parse HEAD)" > "$dest/${STAMP_NAME}"`,
    'CRUCIBLE="$dest/bin/crucible"',
    'say "server-pack: built $("$CRUCIBLE" --version) from $(git -C "$src" rev-parse --short HEAD)"',
  ].join('\n');
}

export function generateInstallSh(): string {
  const steps = installSteps(STANDALONE);
  const lines: string[] = [
    '#!/bin/sh',
    BANNER('#').trimEnd(),
    '#',
    '# Install a Crucible on this machine (Linux x86_64, macOS arm64, or inside a',
    '# WSL2 distro). Downloads the server pack from the release, initialises it,',
    '# installs the service, and prints the line that pairs an app with it.',
    '#',
    '#   curl -fsSL https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.sh | sh',
    '#',
    '# On a rented Linux box with a GPU, where the server is reached over the',
    '# network and the bearer token is the lock:',
    '#',
    '#   curl -fsSL https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.sh \\',
    '#     | sh -s -- --token "$CRUCIBLE_TOKEN" --host 0.0.0.0 --install llm',
    '#',
    '# And to take it off again, keeping the weights:',
    '#',
    '#   curl -fsSL https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.sh | sh -s -- --uninstall',
    '#',
    '# CRUCIBLE_RELEASE=<version> or --release <version> installs a NAMED release.',
    '# Given neither, this asks GitHub which release is newest and installs that.',
    '# Everything here is idempotent: run it again after a failure.',
    '#',
    '# THERE IS NO BAKED DEFAULT, and that is the point. This file used to carry',
    '# the version it was GENERATED at, which is wrong in the one situation that',
    '# matters: the documented way to get this script is',
    '# `releases/latest/download/install.sh`, so the copy you run is whichever one',
    '# GitHub calls latest. Every release is cut `--prerelease --latest=false` and',
    '# becomes latest only when promote_release.py says so, so on 2026-09-16 that',
    '# URL served the v0.6.0 script, which then installed 0.6.0 and its packs --',
    '# six versions behind, silently, with nothing in the output looking wrong.',
    '# Asking at RUN time cannot drift that way, and a baked value that is only',
    '# right on the day it was written is exactly the kind of default this repo',
    '# does not keep.',
    '',
    'set -eu',
    '',
    "RELEASE=\"${CRUCIBLE_RELEASE:-}\"",
    '',
    "say() { printf 'crucible: %s\\n' \"$*\"; }",
    "die() { printf 'crucible: %s\\n' \"$*\" >&2; exit 1; }",
    '',
    '# --- arguments -----------------------------------------------------------',
    '# The flags a person types. An app never reaches this file: it calls',
    '# `install()`, which walks the SAME step list (PHASE14 4a).',
    argumentsSh(),
    '',
    '# --- backend -------------------------------------------------------------',
    '# Two backends and no third. Windows is never one: on Windows this script',
    '# runs INSIDE the WSL2 distro that install.ps1 imported.',
    'case "$(uname -s)/$(uname -m)" in',
    '  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;',
    '  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;',
    '  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend '
      + '(cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;',
    'esac',
    '# --- which release -------------------------------------------------------',
    '# Asked only when nobody named one, and NOT asked at all for --uninstall,',
    '# which removes what is on this disk and must work with no network.',
    '# The failure is loud: no fallback to a version this script was built beside,',
    '# because installing a silently-wrong release is the defect being fixed.',
    'newest_release() {',
    `  curl -fsSL --retry 3 -H "Accept: application/vnd.github+json" "https://api.github.com/repos/${RELEASE_REPO}/releases?per_page=1" |`,
    `    grep -o '"tag_name": *"[^"]*"' | head -n 1 | cut -d'"' -f4`,
    '}',
    'if [ "$UNINSTALL" = 1 ]; then',
    '  say "backend $BACKEND, uninstalling"',
    'else',
    '  if [ -z "$RELEASE" ]; then',
    '    tag="$(newest_release)" || tag=""',
    '    RELEASE="${tag#v}"',
    '  fi',
    `  [ -n "$RELEASE" ] || die "release_lookup_failed: could not learn the newest release from https://api.github.com/repos/${RELEASE_REPO}/releases -- name one with --release <version> or CRUCIBLE_RELEASE=<version>"`,
    '  say "release $RELEASE, backend $BACKEND"',
    'fi',
    '',
    '# --- uninstall -----------------------------------------------------------',
    '# The inverse, and then this script exits: `crucible uninstall` does the',
    '# nine steps inside CRUCIBLE_HOME and this removes the pack it unpacked.',
    'if [ "$UNINSTALL" = 1 ]; then',
    '  say "uninstall"',
    indent(uninstallSh().trimEnd()),
    '  say "uninstalled."',
    '  exit 0',
    'fi',
    '',
  ];
  for (const step of steps) {
    lines.push(`# --- ${step.name} ${'-'.repeat(Math.max(0, 68 - step.name.length))}`);
    lines.push(`# ${step.what}`);
    lines.push(`say "${step.name}"`);
    if (step.name === 'server-pack') {
      // The pack and the source build are two routes to one artefact, and the
      // `if` is the whole of their relationship: neither is the other's
      // fallback. `serverPackSh()` stays the one owner of the pack route.
      lines.push('if [ -n "$FROM_SOURCE" ]; then');
      lines.push(indent(fromSourceSh()));
      lines.push('else');
      lines.push(indent(step.sh.trimEnd()));
      lines.push('fi');
    } else {
      lines.push(step.sh.trimEnd());
    }
    lines.push('');
    if (step.name === 'host-facts') {
      // AFTER the probe, because it prices the disk the probe measured, and
      // BEFORE anything is downloaded, because a refusal that arrives after
      // eight gigabytes is a refusal that cost something.
      lines.push('# --- prerequisites -------------------------------------------------------');
      lines.push('# Named, and never guessed around. A missing one is a refusal here rather');
      lines.push('# than a job type that refuses its first request a week later.');
      lines.push('say "prerequisites"');
      lines.push(prerequisitesSh());
      lines.push('');
    }
    if (step.name === 'init') {
      // Exactly where `installSteps` puts `install-<type>` for an app.
      lines.push('# --- install-job-types ---------------------------------------------------');
      lines.push('# `--install <type>`, from the published packs. Empty on a bare run, which');
      lines.push('# is 4a: a Crucible that serves nothing until somebody asks.');
      lines.push(installJobTypesSh().trimEnd());
      lines.push('');
    }
  }
  lines.push('# --- done ----------------------------------------------------------------');
  lines.push('say "installed. Pair an app with the line below."');
  lines.push('"$CRUCIBLE" token --url');
  lines.push('');
  return lines.join('\n');
}

/** The one row we need by name, refused loudly if the table is renamed under us. */
function row(states: WslStateDef[], code: string): WslStateDef {
  const found = states.find((state) => state.code === code);
  if (found === undefined) throw new Error(`gen-install-scripts: the WSL state table has no "${code}" row any more`);
  return found;
}

function psQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

/**
 * `/etc/wsl.conf` as ONE bash command, from {@link WSL_CONF_TEXT}. `printf`
 * with a `%s\n` per line and the lines as arguments: wsl.exe halves
 * backslashes before bash exists, so the `\n` is doubled here to arrive whole.
 */
function wslConfPrintf(): string {
  const lines = WSL_CONF_TEXT.split('\n').filter((line) => line !== '');
  const args = lines.map((line) => `'${line.replace(/'/g, `'\\''`)}'`).join(' ');
  return `printf '%s\\\\n' ${args} > /etc/wsl.conf`;
}

export function generateInstallPs1(): string {
  const states = wslStates({ release: BOOTSTRAP_VERSION });
  const enable = row(states, 'wsl_missing');
  const enableAction = enable.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  if (enableAction.kind !== 'run-elevated') {
    throw new Error('gen-install-scripts: the WSL table changed shape; install.ps1 must be re-thought, not re-run');
  }
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$Release`;
  const lines: string[] = [
    BANNER('#').trimEnd(),
    '#',
    '# Install Crucible on Windows: the HOST, and nothing else.',
    '#',
    '# PHASE15-HOST.md 4.4. This script used to walk the WSL states itself and',
    '# then run install.sh inside an imported distribution. It no longer does,',
    '# and that is the point of the phase: `crucible host` owns that sequence',
    '# (4.3), it can carry a reboot across because it starts at login, and the',
    '# page drives it as a task (4.7) — an app that asks for an install talks',
    '# to the SAME implementation through the host loopback door. Two walks of',
    '# one table was the thing being removed.',
    '#',
    '# So: download the host pack for this release, verify it, unpack it to',
    '# %LOCALAPPDATA%\\Crucible\\host\\, register the Startup item, start the',
    '# host, and STOP.',
    '#',
    '#   irm https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.ps1 | iex',
    '#',
    '# And to take it off again, keeping the weights (the -Uninstall branch',
    '# below): download it to a file first, because `irm | iex` has no way to',
    '# pass a switch.',
    '#',
    '#   irm https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.ps1 -OutFile install.ps1',
    '#   .\\install.ps1 -Uninstall            # weights kept',
    '#   .\\install.ps1 -Uninstall -WslToo    # and the guest engine with it',
    '#',
    '# No admin. Everything here is per-user and idempotent: run it again after',
    '# a failure and it resumes from whatever is already on disk.',
    '',
    '[CmdletBinding()]',
    'param(',
    // No baked default -- see the long note in the sh generator above. Empty
    // means "ask GitHub which release is newest", which is the only answer that
    // cannot go stale in a file served from `releases/latest/download/`.
    "  [string]$Release = '',",
    '  [string]$Root = "$env:LOCALAPPDATA\\Crucible",',
    '  # The inverse. `crucible uninstall` does the work inside the home; this',
    '  # script removes the host pack, because this script is what unpacked it.',
    '  [switch]$Uninstall,',
    '  [switch]$PurgeWeights,',
    '  [switch]$DryRun,',
    '  [switch]$WslToo',
    ')',
    '',
    '# Continue, not Stop: every call below is a native program whose exit code',
    '# is checked explicitly, and Windows PowerShell 5.1 turns a native command'
      + ' writing to stderr into a terminating error under Stop.',
    '$ErrorActionPreference = "Continue"',
    '$Root = [System.IO.Path]::GetFullPath($Root)',
    '$env:CRUCIBLE_HOME = $Root',
    `$HostDir = Join-Path $Root ${psQuote(HOST_SUBDIR)}`,
    `$DownloadDir = Join-Path $Root ${psQuote(DOWNLOADS_SUBDIR)}`,
    `$Partial = "$HostDir${PARTIAL_SUFFIX}"`,
    `$Stamp = Join-Path $HostDir ${psQuote(STAMP_NAME)}`,
    '$Cmd = Join-Path $HostDir "crucible.cmd"',
    '$Pythonw = Join-Path $HostDir "pythonw.exe"',
    '',
    'function Say($m) { Write-Host "crucible: $m" }',
    'function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }',
    '$Previous = "$HostDir.previous"',
    'foreach ($target in @($HostDir, $Partial, $Previous, $DownloadDir)) {',
    '  $absolute = [System.IO.Path]::GetFullPath($target)',
    '  if (-not $absolute.StartsWith($Root.TrimEnd("\\") + "\\", [System.StringComparison]::OrdinalIgnoreCase)) { Die "unsafe_install_path: $absolute is outside $Root" }',
    '}',
    '',
    '# --- 0. this machine can hold the host -----------------------------------',
    '# 64-bit x86 only: the pinned interpreter is',
    '# x86_64-pc-windows-msvc-install_only and there is no second pin (4.4).',
    'if ([System.Environment]::Is64BitOperatingSystem -ne $true) {',
    '  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."',
    '}',
    'if (-not $env:LOCALAPPDATA) {',
    '  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."',
    '}',
    '',
    '# --- the inverse, which exits ---------------------------------------------',
    '# `crucible uninstall` stops the tray, removes the Startup item and takes',
    '# %LOCALAPPDATA%\\Crucible apart step by named step — everything except the',
    '# interpreter it is itself running from. THIS script unpacked that, so this',
    '# script removes it, after the verb has returned. Weights are kept unless',
    '# -PurgeWeights; -WslToo runs the guest\'s own uninstall first.',
    'if ($Uninstall) {',
    '  if (-not (Test-Path $Cmd)) {',
    '    Die "not_installed: there is no $Cmd on this machine, so there is no Crucible host here to remove."',
    '  }',
    '  $verb = @("uninstall")',
    '  if ($DryRun) { $verb += "--dry-run" }',
    '  if ($PurgeWeights) { $verb += "--purge-weights" }',
    '  if ($WslToo) { $verb += "--wsl-too" }',
    '  Say "uninstall: $Cmd $($verb -join \' \')"',
    '  & $Cmd @verb',
    '  if ($LASTEXITCODE -ne 0) { Die "step_failed: uninstall (crucible uninstall exited $LASTEXITCODE; nothing of the pack has been removed)" }',
    '  if ($DryRun) {',
    '    Say "host-pack: would remove $HostDir and $DownloadDir"',
    '    Say "home: would remove $Root if it were then empty"',
    '    exit 0',
    '  }',
    '  Say "host-pack"',
    '  foreach ($gone in @($Partial, $DownloadDir, $HostDir)) {',
    '    if (Test-Path $gone) {',
    '      try {',
    '        Remove-Item $gone -Recurse -Force -ErrorAction Stop',
    '      } catch {',
    '        Die "host_pack_locked: $gone could not be removed ($($_.Exception.Message)). Something still holds a file in it — the tray was just ended, so log out and back in, then run this again. It is idempotent."',
    '      }',
    '    }',
    '  }',
    '  Say "host-pack: removed $HostDir"',
    '  $left = @(Get-ChildItem -Force -Path $Root -ErrorAction SilentlyContinue)',
    '  if ($left.Count -eq 0) {',
    '    Remove-Item $Root -Force -Recurse',
    '    Say "home: removed $Root"',
    '  } else {',
    '    Say "home: KEPT $Root — it still holds $($left.Name -join \', \')"',
    '    Say "home: weights are kept unless -PurgeWeights; nothing else there was Crucible\'s to delete"',
    '  }',
    '  Say "uninstalled."',
    '  exit 0',
    '}',
    '',
    '# A pack is a zstd tarball. Windows 10 1803+ and Windows 11 ship bsdtar',
    '# linked with libzstd, so no zstd.exe is needed — MEASURED on 2026-09-14:',
    '# bsdtar 3.8.1 / libarchive 3.8.1 / libzstd 1.5.5. A machine whose tar has',
    '# no zstd would half-unpack in silence, so it is CHECKED, not assumed.',
    '$tarVersion = ""',
    'try { $tarVersion = (& tar.exe --version | Out-String) } catch { $tarVersion = "" }',
    'if ($tarVersion -notmatch "zstd") {',
    '  Die "guest_missing_tool: this machine tar cannot read zstd (tar --version said: $($tarVersion.Trim())). Windows 10 1803+ and Windows 11 ship one that can."',
    '}',
    '',
    '# --- 1. which pack -------------------------------------------------------',
    '# Asked only when nobody named one. -Uninstall returned long before here,',
    '# so taking Crucible off a machine still needs no network.',
    'if (-not $Release) {',
    `  $feed = "https://api.github.com/repos/${RELEASE_REPO}/releases?per_page=1"`,
    '  $feedRaw = & curl.exe -fsSL --retry 3 -H "Accept: application/vnd.github+json" "$feed"',
    '  if ($LASTEXITCODE -ne 0) { Die "release_lookup_failed: could not read $feed -- name one with -Release <version>" }',
    '  try { $feedJson = $feedRaw | Out-String | ConvertFrom-Json } catch { Die "release_lookup_failed: $feed is not JSON" }',
    "  $Release = ($feedJson[0].tag_name) -replace '^v',''",
    '  if (-not $Release) { Die "release_lookup_failed: $feed named no release" }',
    '}',
    'Say "release $Release"',
    `$manifestUrl = "${base}/${ENVPACKS_ASSET}"`,
    '$manifestRaw = & curl.exe -fsSL --retry 3 "$manifestUrl"',
    'if ($LASTEXITCODE -ne 0) { Die "pack_manifest_unreadable: could not fetch $manifestUrl" }',
    'try { $manifest = $manifestRaw | Out-String | ConvertFrom-Json } catch { Die "pack_manifest_unreadable: $manifestUrl is not JSON" }',
    'if ($manifest.schema -ne 1 -and $manifest.schema -ne 2) { Die "pack_manifest_unreadable: $manifestUrl declares schema $($manifest.schema), this installer reads 1 or 2" }',
    '$pack = $null',
    `foreach ($entry in $manifest.packs) { if ($entry.name -eq ${psQuote(HOST_PACK)} -and $entry.backend -eq ${psQuote(HOST_BACKEND)}) { $pack = $entry } }`,
    `if ($null -eq $pack) { Die "pack_not_published: the $Release release publishes no ${HOST_PACK} pack for ${HOST_BACKEND}" }`,
    '',
    '# --- 2. already installed? ------------------------------------------------',
    '# The stamp is the same two lines the guest-side install writes, read the',
    '# same way: a matching sha means these bytes are already unpacked.',
    '$have = ""',
    'if (Test-Path $Stamp) {',
    '  foreach ($line in (Get-Content $Stamp)) { if ($line -match "^sha256=(.+)$") { $have = $Matches[1].Trim() } }',
    '}',
    'if ($have -eq $pack.sha256 -and (Test-Path $Cmd)) {',
    '  Say "host-pack: already installed ($($pack.sha256))"',
    '} else {',
    '  # --- 3. disk ------------------------------------------------------------',
    '  # The same sum pack.ts requiredBytes() uses: unpacked + the whole archive',
    '  # + one part, a part being the archive over the part count.',
    '  $need = $pack.unpacked_bytes + $pack.bytes + [math]::Floor($pack.bytes / $pack.parts.Count)',
    '  $drive = (Get-Item $env:LOCALAPPDATA).PSDrive',
    '  if ($drive.Free -lt $need) {',
    '    Die "pack_disk: the host pack needs $([math]::Round($need/1GB,1)) GiB free on $($drive.Name): and there is $([math]::Round($drive.Free/1GB,1)) GiB. Nothing has been downloaded."',
    '  }',
    '',
    '  # --- 4. download, join, verify -------------------------------------------',
    '  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null',
    '  $archiveName = $pack.parts[0] -replace "\\.part[0-9]+$", ""',
    '  $archive = Join-Path $DownloadDir $archiveName',
    // THE ROW'S RELEASE. An unchanged pack is carried by reference rather
    // than rebuilt or copied, so its parts stay in the release that built them
    // and the row says which that is. Schema 1 has no such field and needs
    // none: it placed every pack on its own release, so `$Release` IS the
    // answer there — recovered, not defaulted.
    '  $packRelease = if ($pack.PSObject.Properties.Name -contains "release") { $pack.release } else { $Release }',
    `  $packBase = "https://github.com/${RELEASE_REPO}/releases/download/v$packRelease"`,
    '  if (Test-Path $archive) { Remove-Item $archive -Force }',
    '  foreach ($part in $pack.parts) {',
    '    Say "host-pack: $part"',
    '    $partPath = Join-Path $DownloadDir $part',
    '    & curl.exe ' + CURL_ARGS.join(' ') + ' -o $partPath "$packBase/$part"',
    '    if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: $packBase/$part" }',
    '    # Byte-for-byte append, then delete: peak extra disk is ONE part and',
    '    # not the whole set. Add-Content would re-encode the bytes as text.',
    '    $in = [System.IO.File]::OpenRead($partPath)',
    '    $out = [System.IO.File]::Open($archive, "Append", "Write")',
    '    try { $in.CopyTo($out) } finally { $out.Close(); $in.Close() }',
    '    Remove-Item $partPath -Force',
    '  }',
    '  $got = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()',
    '  if ($got -ne $pack.sha256) {',
    '    Remove-Item $archive -Force',
    '    Die "pack_sha_mismatch: $archiveName hashes $got, the manifest says $($pack.sha256)"',
    '  }',
    '',
    '  # --- 5. unpack beside, prove it runs, THEN move --------------------------',
    '  if (Test-Path $Partial) { Remove-Item $Partial -Recurse -Force }',
    '  New-Item -ItemType Directory -Force -Path $Partial | Out-Null',
    `  & tar.exe ${TAR_ARGS.join(' ')} $archive -C $Partial`,
    '  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: tar would not open $archive" }',
    '  # The .cmd and not the .exe: pip Scripts\\*.exe launchers bake the build',
    '  # tree interpreter path into the binary and do not survive this move',
    '  # (PHASE15-HOST.md 4.4, and PHASE14 7.2a for the POSIX half of it).',
    '  & (Join-Path $Partial "crucible.cmd") --version | Out-Null',
    '  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: crucible.cmd in $Partial would not run" }',
    '  # Stop with the new staged control code before touching the installed runtime.',
    '  # A shutdown failure leaves both the old runtime and verified staging intact.',
    '  if (Test-Path -LiteralPath $Previous) { Die "upgrade_recovery_required: $Previous exists from an interrupted upgrade; restore or inspect it before retrying" }',
    '  if (Test-Path -LiteralPath $HostDir) {',
    '    & (Join-Path $Partial "crucible.cmd") local shutdown',
    '    if ($LASTEXITCODE -ne 0) { Die "upgrade_stop_failed: the old runtime was kept because Crucible did not stop cleanly" }',
    '    Move-Item -LiteralPath $HostDir -Destination $Previous -ErrorAction Stop',
    '  }',
    '  try {',
    '    Move-Item $Partial $HostDir -ErrorAction Stop',
    '    & $Cmd --version | Out-Null',
    '    if ($LASTEXITCODE -ne 0) { throw "the installed runtime failed its startup check" }',
    '  } catch {',
    '    if ((Test-Path -LiteralPath $Previous) -and (Test-Path -LiteralPath $HostDir) -and -not (Test-Path -LiteralPath $Partial)) { Move-Item -LiteralPath $HostDir -Destination $Partial -ErrorAction Stop }',
    '    if ((Test-Path -LiteralPath $Previous) -and -not (Test-Path -LiteralPath $HostDir)) { Move-Item -LiteralPath $Previous -Destination $HostDir }',
    '    Die "upgrade_swap_failed: $_. The previous runtime is retained at $Previous when present."',
    '  }',
    '  if (Test-Path -LiteralPath $Previous) { Remove-Item -LiteralPath $Previous -Recurse -Force }',
    '  Set-Content -Path $Stamp -Encoding ascii -Value @("sha256=$($pack.sha256)", "release=$Release")',
    '  Remove-Item $archive -Force',
    '  Say "host-pack: unpacked $($pack.parts.Count) part(s) into $HostDir (Python $($pack.python))"',
    '}',
    '',
    '# --- 6. start at login ----------------------------------------------------',
    '# The host OWNS that shortcut (4.1). This script asks for it by verb rather',
    '# than writing a .lnk of its own, so there is one spelling of what it points',
    '# at and one place that changes when it moves.',
    'foreach ($action in @("register", "install-cli", "install-desktop")) {',
    '  & $Cmd local $action',
    '  if ($LASTEXITCODE -ne 0) { Die "local setup failed: $action (exit $LASTEXITCODE)" }',
    '}',
    '',
    '# --- 7. start the host, and stop ------------------------------------------',
    '# pythonw, not the .cmd: a tray program has no console window (4.1).',
    'Say "starting the tray"',
    'Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","host"',
    '& $Cmd local start',
    'if ($LASTEXITCODE -ne 0) { Die "Crucible installed but did not become ready. Run crucible local status for the named failure." }',
    'Say "Crucible is ready in your notification area. The Windows engine works now; the optional Linux engine is available from its console."',
    '',
  ];
  // ASCII, because Windows PowerShell 5.1 reads a BOM-less .ps1 as the ANSI
  // code page. See `asciiOnly`, and the parse error that measured it.
  return asciiOnly(lines.join('\n'), 'install.ps1');
}

// ------------------------------------------------- the table, as Python data

/**
 * Sentinels. Every row's `sentence` is a FUNCTION of the evidence, so the way
 * to get a template out of it is to call it with values that cannot occur and
 * then swap those for `{placeholders}` — and to ASSERT each swap fired, so a
 * reworded row fails this generator instead of shipping a sentence with a
 * sentinel in it. The alternative, a second table of sentences in Python, is
 * the thing this file exists to prevent.
 */
const SAID = 'XXSAIDXX';
const APP_DISTRO = 'XXAPPDISTROXX';
const GUEST_USER = 'XXGUESTUSERXX';
const RELEASE_MARK = '424.242.424';
/** gib(REQUIRED_SENTINEL) is "424242.0 GiB", which occurs in no real sentence. */
const REQUIRED_SENTINEL = 424242 * 1024 ** 3;
/** The `df -Pk` reply the disk row is given: gib() renders it "953.7 GiB". */
const FREE_KIB_SENTINEL = '999999999';

const SUBSTITUTIONS: { from: string; to: string; required: boolean }[] = [
  { from: SAID, to: '{said}', required: false },
  { from: APP_DISTRO, to: '{app_distro}', required: false },
  { from: RELEASE_MARK, to: '{release}', required: false },
  { from: '424242.0 GiB', to: '{required}', required: false },
  { from: '953.7 GiB', to: '{free}', required: false },
];

function sentinelResult(): RunResult {
  // `said()` prefers stderr, so putting the sentinel there is what makes every
  // row that quotes the evidence come back with it.
  return { code: 1, stdout: FREE_KIB_SENTINEL, stderr: SAID, failure: 'ENOENT' };
}

function pyString(value: string): string {
  return JSON.stringify(value);
}

function pyList(values: readonly string[]): string {
  return `(${values.map((value) => `${pyString(value)}, `).join('')})`;
}

/** Every sentinel swapped for its `{placeholder}`, and none left behind. */
function templated(text: string, code: string, fired: Set<string>): string {
  let out = text;
  for (const swap of SUBSTITUTIONS) {
    if (out.includes(swap.from)) {
      fired.add(swap.to);
      out = out.split(swap.from).join(swap.to);
    }
  }
  for (const sentinel of [SAID, APP_DISTRO, GUEST_USER, RELEASE_MARK, FREE_KIB_SENTINEL, '424242']) {
    if (out.includes(sentinel)) {
      throw new Error(
        `gen-install-scripts: the sentinel ${sentinel} survived into ${code}: ${JSON.stringify(out)}. `
        + 'A sentence that reaches a person with a sentinel in it is worse than a generator that refuses.',
      );
    }
  }
  return out;
}

export function generateWslStatesPy(): string {
  const inputs = {
    release: RELEASE_MARK,
    appDistro: APP_DISTRO,
    requiredBytes: REQUIRED_SENTINEL,
    checkNetwork: true,
  };
  const states = wslStates(inputs);
  // The SAME table with nothing the caller measured, so that `optional` is
  // read off the rows themselves rather than written down here: a row that is
  // disabled when nobody asked for a disk figure or a network probe is a row
  // whose probe costs something, and `detect()` must not run it unasked.
  const bare = wslStates({ release: RELEASE_MARK });
  const optionalCodes = new Set(bare.filter((state) => state.enabled === false).map((state) => state.code));
  const seen = { results: {}, distros: [] as { name: string; version: number }[] };
  const result = sentinelResult();
  const fired = new Set<string>();

  const rows = states.map((state: WslStateDef) => {
    const sentence = templated(state.sentence(result, seen), state.code, fired);
    const action = state.action(result, seen);
    const argv = probeArgv(state.probe as ProbeKey, inputs).map((word) => templated(word, `${state.code}.probe_argv`, fired));
    const actionArgv = (action.kind === 'run' || action.kind === 'run-elevated' ? action.argv : [])
      .map((word) => templated(word, `${state.code}.action_argv`, fired));
    const parts = [
      `        code=${pyString(state.code)},`,
      `        probe=${pyString(state.probe)},`,
      `        probe_argv=${pyList(argv)},`,
      `        sentence=${pyString(sentence)},`,
      `        action_kind=${pyString(action.kind)},`,
      `        action_argv=${pyList(actionArgv)},`,
      `        action_text=${pyString(action.kind === 'instruct' ? templated(action.text, `${state.code}.action_text`, fired) : '')},`,
      `        action_url=${pyString(action.kind === 'link' ? templated(action.url, `${state.code}.action_url`, fired) : '')},`,
      `        optional=${optionalCodes.has(state.code) ? 'True' : 'False'},`,
    ];
    return `    WslStateDef(\n${parts.join('\n')}\n    ),`;
  });

  // Every placeholder the table CAN produce must actually have been produced,
  // or the Python side would show a sentence with a hole nobody fills.
  for (const wanted of ['{said}', '{app_distro}', '{release}', '{required}', '{free}']) {
    if (!fired.has(wanted)) {
      throw new Error(
        `gen-install-scripts: nothing in the WSL table produces ${wanted} any more. `
        + 'Either a row was reworded and crucible/host/wslstate.py must be re-thought, or this generator is substituting for a fact that no longer exists.',
      );
    }
  }

  return [
    BANNER('#').trimEnd(),
    '#',
    '# The WSL state table of PHASE14-ENVPACKS.md 4c, as DATA, for `crucible host`',
    '# (PHASE15-HOST.md 4.3). The ORDER is the order they are tried: deepest cause',
    '# first, so "virtualization is off in the firmware" is never reported as "WSL',
    '# is not installed".',
    '#',
    '# `sentence`, `action_text`, `action_url` and `probe_argv` carry `{said}` /',
    '# `{app_distro}` / `{release}` / `{required}` / `{free}`',
    '# where the TypeScript interpolated something the caller measures. The',
    '# PREDICATES are not here: they are code, and they live in',
    '# `crucible/host/wslstate.py`, one per code, tied to this file by a test.',
    '',
    'from __future__ import annotations',
    '',
    'from dataclasses import dataclass',
    '',
    '#: The distro Crucible owns. One name, and its owner is sdk/bootstrap/src/distro.ts.',
    `CRUCIBLE_DISTRO = ${pyString(CRUCIBLE_DISTRO)}`,
    `ROOTFS_ASSET_TEMPLATE = ${pyString(rootfsAssetName('{version}'))}`,
    `RELEASE_REPOSITORY = ${pyString(RELEASE_REPO)}`,
    '',
    '#: The line /etc/wsl.conf carries in the Crucible rootfs and nowhere else.',
    `WSL_CONF_MARKER = ${pyString(WSL_CONF_MARKER)}`,
    '',
    '#: /etc/wsl.conf, exactly as the rootfs ships it and as the repair writes it.',
    `WSL_CONF_TEXT = ${pyString(WSL_CONF_TEXT)}`,
    '',
    '',
    '@dataclass(frozen=True)',
    'class WslStateDef:',
    '    """One row of 4c. `optional` rows are only probed when the caller asks."""',
    '',
    '    code: str',
    '    probe: str',
    '    probe_argv: tuple[str, ...]',
    '    sentence: str',
    '    action_kind: str',
    '    action_argv: tuple[str, ...]',
    '    action_text: str',
    '    action_url: str',
    '    optional: bool',
    '',
    '',
    'WSL_STATES: tuple[WslStateDef, ...] = (',
    ...rows,
    ')',
    '',
    '#: Every code, in table order. `crucible/host/wslstate.py` must have a',
    '#: predicate for each, and no others.',
    'WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)',
    '',
  ].join('\n');
}

export const GENERATED = [
  { path: join(SCRIPTS, 'install.sh'), text: generateInstallSh() },
  { path: join(SCRIPTS, 'install.ps1'), text: generateInstallPs1() },
  { path: join(REPO, 'crucible', 'host', 'wsl_states.py'), text: generateWslStatesPy() },
];

function main(): void {
  const check = process.argv.includes('--check');
  let drifted = 0;
  for (const file of GENERATED) {
    if (check) {
      let onDisk = '';
      try {
        onDisk = readFileSync(file.path, 'utf8');
      } catch {
        onDisk = '';
      }
      if (onDisk.replace(/\r\n/g, '\n') !== file.text) {
        drifted += 1;
        console.error(`gen-install-scripts: ${file.path} is not what src/steps.ts says. Run: npm run gen:install`);
      }
    } else {
      writeFileSync(file.path, file.text, 'utf8');
      console.log(`gen-install-scripts: wrote ${file.path}`);
    }
  }
  if (drifted > 0) process.exitCode = 1;
}

// Only when run as a program; the test imports the two generators directly.
if (process.argv[1] !== undefined && process.argv[1].endsWith('gen-install-scripts.js')) main();
