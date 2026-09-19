/**
 * THE STEP LIST — one owner for what installing a Crucible is.
 *
 * `install()` runs this list. `scripts/gen-install-scripts.ts` reads the same
 * list and writes `install.sh` and `install.ps1` from it. That is the whole
 * reason this module exists: PHASE14-ENVPACKS.md 4a says an app-driven install
 * and a hand install "cannot differ", and the only way to mean that is for both
 * to be spellings of one list rather than two lists somebody keeps in sync.
 *
 * What is shared, exactly:
 *
 * - the step NAMES, their ORDER, and what each is for;
 * - the skip rules (an existing config keeps its token; a matching interpreter
 *   digest is not re-downloaded);
 * - every argv-shaped step, once — `renderArgv` produces the array `install()`
 *   spawns and `renderSh` produces the shell line, from the same words;
 * - the paths, the curl flags, the tar flags, the interpreter pin and the URL
 *   shapes, which live in `runtime.ts`, `interpreter.ts` and `release.ts` and
 *   are imported by both sides.
 *
 * Three steps are PROGRAMS rather than single commands — probing the host,
 * putting the server runtime in place, granting linger. Those carry their shell
 * in `sh`, beside the TypeScript that performs them, and the tests assert that
 * both use the same constants. Their iteration is spelled twice because two
 * languages; the facts they iterate over are spelled once.
 */
import { DESKTOP_PACKAGES, interpreterFor, interpreterUrl } from './interpreter.js';
import { wheelAssetName, wheelShaAssetName, RELEASE_REPO } from './release.js';
import { activateRuntimeSh, CURL_ARGS, DOWNLOADS_SUBDIR, guestProbeScript, PARTIAL_SUFFIX, SERVER_SUBDIR, STAMP_NAME, TAR_ARGS } from './runtime.js';

/**
 * A word in a step's argv: a literal, one of the values the install carries,
 * or — for the generated script only — a raw shell fragment.
 *
 * `{ sh }` exists because the HAND install has two things an app-driven one
 * does not: flags a person typed (`--host 0.0.0.0` on a droplet) and a job
 * type read out of `$1`. Both are shell words that cannot be a literal or a
 * ref, and spelling the command a second time in the generator to hold them
 * would be the second copy this module exists to prevent. `renderArgv`
 * REFUSES one by name: there is no shell on the TypeScript side to expand it,
 * and a `$BIND` reaching `spawn()` as a literal argument would be an argument
 * called `$BIND`.
 */
export type Word = string | { ref: RefName } | { sh: string };

/**
 * The values a step's argv can refer to. `install()` substitutes what it
 * measured; the generated script substitutes its own shell variable, which is
 * why there is a fixed, named set rather than free-form interpolation.
 */
export type RefName = 'crucible' | 'release' | 'backend' | 'home' | 'token' | 'user';

/** How the generated scripts spell each ref. */
export const SHELL_VARIABLE: Readonly<Record<RefName, string>> = {
  crucible: 'CRUCIBLE',
  release: 'RELEASE',
  backend: 'BACKEND',
  home: 'CRUCIBLE_HOME',
  token: 'TOKEN',
  user: 'GUEST_USER',
};

export type SkipRule = 'config-exists' | 'interpreter-stamp-matches' | 'root-only';

export interface StepDef {
  name: string;
  /** One sentence: the script's comment, and the step's `detail` when it is skipped. */
  what: string;
  /** The argv, as the TARGET sees it. Null for the three steps that are programs. */
  words: readonly Word[] | null;
  /** POSIX sh for the generated installer. Present for every step. */
  sh: string;
  /** Why `install()` may not run it. */
  skip: SkipRule | null;
  /** Which of `install()`'s timeouts this step gets. */
  timeout: 'quickMs' | 'runtimeMs' | 'envMs';
}

export interface StepPlan {
  /** `--enable-<type>` flags for `crucible init`, in request order. */
  enableFlags: readonly string[];
  /** `crucible install <type> …` argv, one per installable type. */
  installs: readonly { type: string; argv: readonly string[] }[];
  /**
   * `--host`/`--port` for `crucible init`, already spelled.
   *
   * `Word[]` and not `string[]` since the droplet route (PHASE15's
   * `docs/INSTALL-UNINSTALL.md`): a rented Linux box is reached over the
   * network, so `install.sh` takes `--host 0.0.0.0` from the operator and
   * passes it through as `{ sh: '$BIND' }`. `install()` still pushes plain
   * strings, which is what an app has.
   */
  bind: readonly Word[];
  /** Whether the linger step is part of this install (systemd hosts only). */
  linger: boolean;
}

/** `$VAR` spelled for sh, quoted. */
function v(ref: RefName): string {
  return `"$${SHELL_VARIABLE[ref]}"`;
}

/** One argument, safe in sh. The same rule as `wsl.ts`'s `shellQuote`, for literals. */
function q(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** A step's words as one shell line. Refs become `"$VAR"`; literals are quoted. */
export function renderSh(words: readonly Word[]): string {
  return words
    .map((word) => {
      if (typeof word === 'string') return q(word);
      if ('sh' in word) return word.sh;
      return v(word.ref);
    })
    .join(' ');
}

/** A step's words as an argv, with this install's measured values. */
export function renderArgv(words: readonly Word[], values: Readonly<Partial<Record<RefName, string>>>): string[] {
  return words.map((word) => {
    if (typeof word === 'string') return word;
    if ('sh' in word) {
      throw new Error(
        `renderArgv: {sh: ${JSON.stringify(word.sh)}} is a shell fragment and there is no shell here. `
        + 'It belongs to the generated installer, whose flags a person typed; an app states its values as refs.',
      );
    }
    const value = values[word.ref];
    if (value === undefined) throw new Error(`renderArgv: nothing measured for {ref: "${word.ref}"}`);
    return value;
  });
}

// ------------------------------------------------------- the three programs

/** The host probe, sh side: the SAME script `probeGuest()` runs, with its output eval'd. */
export function hostFactsSh(): string {
  return `crucible_probe() {\n`
    + `  ${guestProbeScript(undefined)}\n`
    + `}\n`
    + `probe_out="$(crucible_probe)"\n`
    + `CRUCIBLE_HOME="$(printf '%s\\n' "$probe_out" | sed -n 's/^home=//p')"\n`
    + `GUEST_USER="$(printf '%s\\n' "$probe_out" | sed -n 's/^user=//p')"\n`
    + `free_kib="$(printf '%s\\n' "$probe_out" | sed -n 's/^free_kib=//p')"\n`
    + `stamp_python_sha="$(printf '%s\\n' "$probe_out" | sed -n 's/^python_sha256=//p')"\n`
    // WHAT IS ALREADY ON THIS DISK, which is what the never-older gate compares
    // against (INSTALL-UNINSTALL.md 6.5.4). Empty on a tree that predates the
    // stamp, and an empty one is not read as "older": a version nobody recorded
    // cannot be compared with one.
    + `stamp_release="$(printf '%s\\n' "$probe_out" | sed -n 's/^release=//p')"\n`
    + `missing="$(printf '%s\\n' "$probe_out" | sed -n 's/^missing=//p' | tr '\\n' ' ')"\n`
    + `if [ -n "$missing" ]; then die "guest_missing_tool: this machine has no $missing; the interpreter is fetched with curl and unpacked with tar"; fi\n`;
}

/**
 * The server runtime, sh side: the interpreter half and then the wheel half.
 *
 * Same pin, same curl flags, same tar flags, same paths, same order as
 * `installRuntime()` — every one of them interpolated from the constants that
 * function uses.
 *
 * TWO HALVES, AND ONLY THE SECOND ONE RUNS ON AN UPGRADE (PHASE20 section 4).
 * The interpreter is a publisher's bytes pinned by digest, so a tree whose
 * stamp names that digest IS those bytes and there is nothing a second download
 * could correct. The wheel always installs: it is the deploy, it is one
 * megabyte, and re-running it is how a half-finished install is repaired.
 *
 * THEY ARE TWO FUNCTIONS BECAUSE `--from-source` REPLACES ONE OF THEM. The
 * generated `install.sh` can take a git ref instead of a release, and what that
 * changes is the wheel and nothing else — the interpreter is the same pinned
 * CPython either way, which is what makes `--from-source` stop needing a
 * `python3` on the machine at all.
 *
 * There is no manifest to parse and no jq to parse it with. The one thing this
 * reads off the network that is not bytes is `<wheel>.sha256`, one line.
 */
export function serverSh(): string {
  return interpreterSh() + wheelSh();
}

/** The interpreter half: fetch, verify, unpack, swap. Once, ever. */
export function interpreterSh(): string {
  // The pin, per backend, as a `case` — the generated script is run on a
  // machine whose backend is `$BACKEND` and cannot be known here. Only the two
  // POSIX backends: the Windows interpreter is `install.ps1`'s, from the same
  // table.
  const cases = (['cuda-linux', 'mlx-darwin'] as const).map((backend) => {
    const pin = interpreterFor(backend);
    return `  ${backend}) py_asset='${pin.asset}'; py_sha='${pin.sha256}'; py_version='${pin.version}'; py_url='${interpreterUrl(pin)}' ;;\n`;
  }).join('');
  return `dest="$CRUCIBLE_HOME/${SERVER_SUBDIR}"\n`
    + `partial="$dest${PARTIAL_SUFFIX}"\n`
    + `downloads="$CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `case "$BACKEND" in\n`
    + cases
    + `  *) die "unsupported_platform: no interpreter is pinned for $BACKEND" ;;\n`
    + `esac\n`
    // NEVER OVER A NEWER RELEASE (INSTALL-UNINSTALL.md 6.5.4). Checked before
    // either half, because the wheel install is the thing that would take this
    // machine back a version. Number by number in awk, because a string
    // comparison puts 1.0.10 before 1.0.2 and this is the one question here
    // that has to get that right. Exit 0 means older.
    + `crucible_older() {\n`
    + `  awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".");\n`
    + `    for (i = 1; i <= 3; i++) { if ((x[i]+0) < (y[i]+0)) exit 0; if ((x[i]+0) > (y[i]+0)) exit 1 } exit 1 }'\n`
    + `}\n`
    + `if [ -n "$stamp_release" ] && crucible_older "$RELEASE" "$stamp_release"; then\n`
    + `  [ "$ROLLBACK_TO" = "$RELEASE" ] || die "install_would_downgrade: $dest is the $stamp_release release and this would install $RELEASE over it. Nothing was downloaded. An operator who means to go back names the version: --rollback-to $RELEASE"\n`
    + `fi\n`
    + `if [ "$stamp_python_sha" = "$py_sha" ] && [ -x "$dest/bin/python3" ]; then\n`
    + `  say "server: python $py_version is already at $dest"\n`
    + `else\n`
    + `  say "server: python $py_version from python-build-standalone"\n`
    + `  mkdir -p "$downloads"; rm -f "$downloads/$py_asset"\n`
    + `  curl ${CURL_ARGS.join(' ')} -o "$downloads/$py_asset" "$py_url" || die "runtime_download_failed: $py_url"\n`
    + `  got_sha="$($SHA_TOOL "$downloads/$py_asset" | awk '{print $1}')"\n`
    + `  if [ "$got_sha" != "$py_sha" ]; then rm -f "$downloads/$py_asset"; die "runtime_sha_mismatch: $py_asset hashes $got_sha and this installer pins $py_sha. The download was deleted"; fi\n`
    + `  rm -rf "$partial" && mkdir -p "$partial"\n`
    + `  tar ${TAR_ARGS.join(' ')} "$downloads/$py_asset" -C "$partial" || die "runtime_unpack_failed: tar would not open $downloads/$py_asset"\n`
    // `install_only` archives carry ONE top-level `python/` directory and that
    // directory IS the interpreter, so the swap moves `python/` rather than the
    // archive's root.
    + `  [ -x "$partial/python/bin/python3" ] || die "runtime_unpack_failed: $py_asset unpacked without a python/bin/python3"\n`
    + `  ${activateRuntimeSh('"$dest"', '"$partial/python"')} || die "runtime_unpack_failed: the previous runtime was preserved"\n`
    + `  rm -rf "$partial" "$downloads/$py_asset"\n`
    + `fi\n`;
}

/**
 * The wheel half: fetch, verify against the release's own `<wheel>.sha256`, pip
 * it into the interpreter, stamp what is now there.
 *
 * THE SERVER IS SHUT DOWN FIRST, because this rewrites its own `site-packages`
 * under it. `local shutdown` is a no-op on a machine where nothing is running,
 * and `service-install` later in the sequence starts it again.
 */
export function wheelSh(): string {
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$RELEASE`;
  return `wheel="${wheelAssetName('$RELEASE')}"\n`
    + `say "server: $wheel"\n`
    + `mkdir -p "$downloads"; rm -f "$downloads/$wheel"\n`
    + `curl ${CURL_ARGS.join(' ')} -o "$downloads/$wheel" "${base}/$wheel" || die "runtime_download_failed: ${base}/$wheel"\n`
    + `want_sha="$(curl -fsSL --retry 3 "${base}/${wheelShaAssetName('$RELEASE')}" | awk '{print $1}')" || die "runtime_download_failed: ${base}/${wheelShaAssetName('$RELEASE')}"\n`
    + `case "$want_sha" in *[!0-9a-f]*|"") die "runtime_download_failed: ${base}/${wheelShaAssetName('$RELEASE')} is not a sha256" ;; esac\n`
    + `got_sha="$($SHA_TOOL "$downloads/$wheel" | awk '{print $1}')"\n`
    + `if [ "$got_sha" != "$want_sha" ]; then rm -f "$downloads/$wheel"; die "runtime_sha_mismatch: $wheel hashes $got_sha, the release says $want_sha. The download was deleted"; fi\n`
    // The server is shut down before its own site-packages is rewritten under
    // it. A no-op where nothing is running; `service-install` starts it again.
    + `if [ -x "$dest/bin/crucible" ]; then "$dest/bin/crucible" local shutdown || true; fi\n`
    + `"$dest/bin/python3" -m pip install --upgrade --no-input "$downloads/$wheel" || die "runtime_install_failed: pip would not install $wheel"\n`
    // The tray's two packages, on the platform that has a desktop. Declared in
    // `interpreter.ts` rather than in `pyproject.toml`, so a headless Linux
    // server never carries a GUI toolkit — see DESKTOP_PACKAGES.
    + `if [ "$(uname -s)" = Darwin ]; then "$dest/bin/python3" -m pip install ${DESKTOP_PACKAGES.join(' ')} || die "runtime_install_failed: the desktop packages would not install"; fi\n`
    + `rm -f "$downloads/$wheel"\n`
    + `printf 'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n' "$py_sha" "$py_version" "$RELEASE" > "$dest/${STAMP_NAME}"\n`
    + `CRUCIBLE="$dest/bin/crucible"\n`
    + `"$CRUCIBLE" --version >/dev/null || die "runtime_install_failed: $CRUCIBLE would not run"\n`;
}

/**
 * Linger, sh side. A systemd USER unit dies with the last session without it.
 * We are usually not root in a hand install, so: try as ourselves, then
 * `sudo -n` (which never prompts), and if that is refused print the one line
 * a person must run — the single hand-over PHASE14 4c leaves standing.
 */
export function lingerSh(): string {
  return `if [ "$MECHANISM" = systemd ]; then\n`
    + `  if loginctl show-user "$GUEST_USER" -p Linger 2>/dev/null | grep -q 'Linger=yes'; then\n`
    + `    say "linger: already on for $GUEST_USER"\n`
    + `  elif [ "$(id -u)" = 0 ] && loginctl enable-linger "$GUEST_USER"; then\n`
    + `    say "linger: granted to $GUEST_USER"\n`
    + `  elif sudo -n loginctl enable-linger "$GUEST_USER" 2>/dev/null; then\n`
    + `    say "linger: granted to $GUEST_USER with sudo"\n`
    + `  else\n`
    + `    say "linger: NOT granted. The Crucible service will stop when you log out."\n`
    + `    say "linger: run this once, by hand:  sudo loginctl enable-linger $GUEST_USER"\n`
    + `  fi\n`
    + `fi\n`;
}

/**
 * `crucible install <type>` over a shell variable — the hand install's
 * `--install <type>` loop.
 *
 * The COMMAND comes from `renderSh` and is therefore the same one
 * `installSteps` gives `install()`; only the iteration is spelled twice,
 * which is this module's standing rule for its programs. `tts` carries its
 * narrator engine as `tts=<engine>`, because cuda-linux names one venv per
 * engine and `crucible install tts` refuses without it — the same refusal an
 * app gets from `planJobTypes`.
 */
export function installJobTypesSh(): string {
  const crucible: Word = { ref: 'crucible' };
  const plain = renderSh([crucible, 'install', { sh: '"$type"' }]);
  const engined = renderSh([
    crucible,
    'install',
    { sh: '"$type"' },
    '--narrator-engine',
    { sh: '"$engine"' },
  ]);
  return `if [ -n "$JOB_TYPES" ]; then\n`
    + `  for entry in $JOB_TYPES; do\n`
    + `    case "$entry" in\n`
    + `      *=*) type="\${entry%%=*}"; engine="\${entry#*=}" ;;\n`
    + `      *)   type="$entry"; engine="" ;;\n`
    + `    esac\n`
    + `    say "install-$type"\n`
    + `    if [ -n "$engine" ]; then\n`
    + `      ${engined} || die "step_failed: install-$type"\n`
    + `    else\n`
    + `      ${plain} || die "step_failed: install-$type"\n`
    + `    fi\n`
    + `  done\n`
    + `fi\n`;
}

/**
 * `install.sh --uninstall`, whole: the CLI verb, and then the runtime.
 *
 * TWO HALVES, AND THE SPLIT IS NOT ARBITRARY. `crucible uninstall`
 * (`crucible/uninstall.py`) stops the server, removes the service and takes
 * `CRUCIBLE_HOME` apart step by named step — everything except the
 * relocatable interpreter it is itself running from, which it cannot unlink
 * without pulling `site-packages` out from under a live process. THIS script
 * unpacked that interpreter, so this script removes it, after the verb has
 * returned. One owner per artefact, and the order is the only one that works.
 *
 * `--dry-run` is passed through and the runtime removal becomes a sentence, so
 * the wrapper's dry run is as complete a description as the verb's.
 */
export function uninstallSh(): string {
  const crucible: Word = { ref: 'crucible' };
  const verb = renderSh([crucible, 'uninstall', { sh: '$UNINSTALL_FLAGS' }]);
  return `CRUCIBLE_HOME="\${CRUCIBLE_HOME:-$HOME/.crucible}"\n`
    + `CRUCIBLE="$CRUCIBLE_HOME/${SERVER_SUBDIR}/bin/crucible"\n`
    + `if [ ! -x "$CRUCIBLE" ]; then\n`
    + `  die "not_installed: there is no $CRUCIBLE on this machine, so there is no Crucible here for this script to remove. \\$CRUCIBLE_HOME names where one would be"\n`
    + `fi\n`
    + `UNINSTALL_FLAGS=""\n`
    + `if [ "$PURGE_WEIGHTS" = 1 ]; then UNINSTALL_FLAGS="$UNINSTALL_FLAGS --purge-weights"; fi\n`
    + `if [ "$DRY_RUN" = 1 ]; then UNINSTALL_FLAGS="$UNINSTALL_FLAGS --dry-run"; fi\n`
    + `${verb} || die "step_failed: uninstall"\n`
    + `# The runtime, which the verb deliberately leaves: it is the interpreter\n`
    + `# that just ran, and this script is what unpacked it.\n`
    + `say "server"\n`
    + `if [ "$DRY_RUN" = 1 ]; then\n`
    + `  say "server: would remove $CRUCIBLE_HOME/${SERVER_SUBDIR} and $CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `  say "home: would remove $CRUCIBLE_HOME if it were then empty"\n`
    + `else\n`
    + `  rm -rf "$CRUCIBLE_HOME/${SERVER_SUBDIR}" "$CRUCIBLE_HOME/${SERVER_SUBDIR}${PARTIAL_SUFFIX}" "$CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `  say "server: removed $CRUCIBLE_HOME/${SERVER_SUBDIR}"\n`
    + `  if rmdir "$CRUCIBLE_HOME" 2>/dev/null; then\n`
    + `    say "home: removed $CRUCIBLE_HOME"\n`
    + `  else\n`
    + `    say "home: KEPT $CRUCIBLE_HOME — it still holds $(ls -A "$CRUCIBLE_HOME" | tr '\\n' ' ')"\n`
    + `    say "home: weights are kept unless --purge-weights; nothing else here was Crucible's to delete"\n`
    + `  fi\n`
    + `fi\n`;
}

// ------------------------------------------------------------- the sequence

/**
 * The sequence, by name. PHASE14 section 4:
 *
 *     host-facts       what this machine is; the conda walk is DELETED
 *     server           the pinned interpreter (once) + this release's wheel, into <CRUCIBLE_HOME>/server
 *     init             <server>/bin/crucible init --token …   (skipped when a config exists)
 *     install-<type>   <server>/bin/crucible install <type>   (none in the standalone installer)
 *     service-install  <server>/bin/crucible service install
 *     linger           systemd hosts
 *     capability-write <server>/bin/crucible capability --write
 */
export function installSteps(plan: StepPlan): StepDef[] {
  const crucible: Word = { ref: 'crucible' };
  const steps: StepDef[] = [
    {
      name: 'host-facts',
      what: 'read this host: CRUCIBLE_HOME, the user, free disk, the tools an install needs',
      words: null,
      sh: hostFactsSh(),
      skip: null,
      timeout: 'quickMs',
    },
    {
      name: 'server',
      what: "download the pinned interpreter (once) and pip-install this release's wheel into it",
      words: null,
      sh: serverSh(),
      skip: 'interpreter-stamp-matches',
      timeout: 'runtimeMs',
    },
    {
      name: 'init',
      what: 'write config.toml with a token this side minted',
      words: [crucible, 'init', '--token', { ref: 'token' }, ...plan.bind, ...plan.enableFlags],
      // The token is minted HERE unless the caller brought one. `--token` on
      // the command line is the droplet's case: an operator who is about to
      // paste a connect code into two apps on two machines would rather state
      // the secret than read it back out of a terminal. An empty `$TOKEN` is
      // the ordinary path and mints, which is what every install before the
      // flag existed did.
      sh: `if [ -f "$CRUCIBLE_HOME/config.toml" ]; then\n`
        + `  say "init: $CRUCIBLE_HOME/config.toml exists; its token is kept"\n`
        + `else\n`
        + `  if [ -z "\${TOKEN:-}" ]; then\n`
        + `    TOKEN="$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=')"\n`
        + `  fi\n`
        + `  ${renderSh([crucible, 'init', '--token', { ref: 'token' }, ...plan.bind, ...plan.enableFlags])} || die "step_failed: init"\n`
        + `fi\n`,
      skip: 'config-exists',
      timeout: 'quickMs',
    },
  ];
  for (const entry of plan.installs) {
    const words: Word[] = [crucible, ...entry.argv];
    steps.push({
      name: `install-${entry.type}`,
      what: `build the ${entry.type} environment from its recipe`,
      words,
      sh: `${renderSh(words)} || die "step_failed: install-${entry.type}"\n`,
      skip: null,
      timeout: 'envMs',
    });
  }
  const serviceWords: Word[] = [crucible, 'service', 'install'];
  steps.push({
    name: 'service-install',
    what: 'write the systemd unit (or the launchd plist) and start it',
    words: serviceWords,
    sh: `${renderSh(serviceWords)} || die "step_failed: service-install"\n`,
    skip: null,
    timeout: 'quickMs',
  });
  for (const action of ['register', 'install-cli', 'install-desktop']) {
    const words: Word[] = [crucible, 'local', action];
    steps.push({
      name: `local-${action}`, what: `publish and configure the local Crucible ${action}`,
      words, sh: `${renderSh(words)} || die "step_failed: local-${action}"\n`,
      skip: null, timeout: 'quickMs',
    });
  }
  if (plan.linger) {
    steps.push({
      name: 'linger',
      what: 'make the service survive a logout',
      words: null,
      sh: lingerSh(),
      skip: 'root-only',
      timeout: 'quickMs',
    });
  }
  const capabilityWords: Word[] = [crucible, 'capability', '--write'];
  steps.push({
    name: 'capability-write',
    what: 'record what this card can hold',
    words: capabilityWords,
    sh: `${renderSh(capabilityWords)} || die "step_failed: capability-write"\n`,
    skip: null,
    timeout: 'quickMs',
  });
  // launchd/systemd accepting a start request does not prove the API is ready.
  // The local lifecycle owner waits for the paired identity and authenticated
  // info response, with a bounded timeout and explicit failures. Both callers
  // of this shared sequence must finish that check before reporting success.
  const readyWords: Word[] = [crucible, 'local', 'start', '--json'];
  steps.push({
    name: 'local-start',
    what: 'wait for the paired engine to answer with authenticated identity',
    words: readyWords,
    sh: `${renderSh(readyWords)} || die "step_failed: local-start"\n`,
    skip: null,
    timeout: 'quickMs',
  });
  return steps;
}
