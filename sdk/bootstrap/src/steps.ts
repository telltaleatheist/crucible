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
 * - the skip rules (an existing config keeps its token; a matching pack stamp
 *   is not re-downloaded);
 * - every argv-shaped step, once — `renderArgv` produces the array `install()`
 *   spawns and `renderSh` produces the shell line, from the same words;
 * - the paths, the curl flags, the tar flags and the URL shapes, which live in
 *   `pack.ts` and `envpacks.ts` and are imported by both sides.
 *
 * Three steps are PROGRAMS rather than single commands — probing the host,
 * fetching the pack, granting linger. Those carry their shell in `sh`, beside
 * the TypeScript that performs them, and the tests assert that both use the
 * same constants. Their iteration is spelled twice because two languages; the
 * facts they iterate over are spelled once.
 */
import { activatePackSh, CURL_ARGS, DOWNLOADS_SUBDIR, guestProbeScript, PARTIAL_SUFFIX, SERVER_SUBDIR, STAMP_NAME, TAR_ARGS } from './pack.js';
import { ENVPACKS_ASSET, RELEASE_REPO } from './envpacks.js';

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

export type SkipRule = 'config-exists' | 'pack-stamp-matches' | 'root-only';

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
  timeout: 'quickMs' | 'packMs' | 'envMs';
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
    + `stamp_sha="$(printf '%s\\n' "$probe_out" | sed -n 's/^sha256=//p')"\n`
    + `missing="$(printf '%s\\n' "$probe_out" | sed -n 's/^missing=//p' | tr '\\n' ' ')"\n`
    + `if [ -n "$missing" ]; then die "guest_missing_tool: this machine has no $missing; a pack is fetched with curl and unpacked with tar --zstd"; fi\n`;
}

/**
 * The server pack, sh side. Same manifest URL, same curl flags, same tar
 * flags, same paths, same order as `installPack()` — every one of them
 * interpolated from the constants that function uses.
 *
 * The manifest is read with awk rather than a JSON parser because a fresh
 * machine has no jq and may have no python. `RS="}"` puts one pack object per
 * record, which is sound because no value in the manifest contains a brace.
 */
export function serverPackSh(): string {
  const manifest = `https://github.com/${RELEASE_REPO}/releases/download/v$RELEASE/${ENVPACKS_ASSET}`;
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$RELEASE`;
  return `dest="$CRUCIBLE_HOME/${SERVER_SUBDIR}"\n`
    + `partial="$dest${PARTIAL_SUFFIX}"\n`
    + `downloads="$CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `manifest_url="${manifest}"\n`
    + `manifest="$(curl -fsSL --retry 3 "$manifest_url")" || die "pack_manifest_unreadable: could not fetch $manifest_url"\n`
    // One pack per awk record (no value in the manifest contains a brace), then
    // flattened to one line so the field reads work whether the JSON is
    // pretty-printed or not. JSON whitespace includes CRLF: stripping LF alone
    // leaves CR around the part names and constructs an invalid download URL.
    // No jq, no python: a fresh machine has neither.
    + `pack="$(printf '%s' "$manifest" | awk -v RS='}' -v b="$BACKEND" '$0 ~ /"name"[[:space:]]*:[[:space:]]*"server"/ && $0 ~ ("\\"backend\\"[[:space:]]*:[[:space:]]*\\"" b "\\"")' | tr -d '\\r\\n')"\n`
    + `[ -n "$pack" ] || die "pack_not_published: the $RELEASE release publishes no server pack for $BACKEND"\n`
    + `want_sha="$(printf '%s' "$pack" | sed -n 's/.*"sha256"[[:space:]]*:[[:space:]]*"\\([0-9a-f]*\\)".*/\\1/p')"\n`
    + `unpacked="$(printf '%s' "$pack" | sed -n 's/.*"unpacked_bytes"[[:space:]]*:[[:space:]]*\\([0-9]*\\).*/\\1/p')"\n`
    + `archive_bytes="$(printf '%s' "$pack" | sed -n 's/.*"bytes"[[:space:]]*:[[:space:]]*\\([0-9]*\\).*/\\1/p')"\n`
    + `parts="$(printf '%s' "$pack" | sed -n 's/.*"parts"[[:space:]]*:[[:space:]]*\\[\\([^]]*\\)\\].*/\\1/p' | tr -d '[:space:]"' | tr ',' ' ')"\n`
    + `[ -n "$want_sha" ] && [ -n "$parts" ] && [ -n "$unpacked" ] && [ -n "$archive_bytes" ] || die "pack_manifest_unreadable: $manifest_url does not describe the server pack"\n`
    + `if [ "$stamp_sha" = "$want_sha" ] && [ -x "$dest/bin/crucible" ]; then\n`
    + `  say "server-pack: already installed ($want_sha)"\n`
    + `else\n`
    // The SAME sum as pack.ts's requiredBytes(): unpacked + the whole archive
    // + one part, a part being the archive over the part count.
    + `  n=0; for part in $parts; do n=$(( n + 1 )); done\n`
    + `  need_kib=$(( (unpacked + archive_bytes + archive_bytes / n) / 1024 ))\n`
    + `  [ "$free_kib" -ge "$need_kib" ] || die "pack_disk: the server pack needs $(( need_kib / 1048576 )) GiB free and there is $(( free_kib / 1048576 )) GiB"\n`
    + `  archive="$downloads/$(printf '%s' "$parts" | awk '{print $1}' | sed 's/\\.part[0-9]*$//')"\n`
    + `  rm -f "$archive"; mkdir -p "$downloads"\n`
    + `  for part in $parts; do\n`
    + `    say "server-pack: $part"\n`
    + `    curl ${CURL_ARGS.join(' ')} -o "$downloads/$part" "${base}/$part" || die "pack_download_failed: ${base}/$part"\n`
    + `    cat "$downloads/$part" >> "$archive" && rm -f "$downloads/$part"\n`
    + `  done\n`
    + `  got_sha="$($SHA_TOOL "$archive" | awk '{print $1}')"\n`
    + `  if [ "$got_sha" != "$want_sha" ]; then rm -f "$archive"; die "pack_sha_mismatch: $archive hashes $got_sha, the manifest says $want_sha"; fi\n`
    + `  rm -rf "$partial" && mkdir -p "$partial"\n`
    + `  tar ${TAR_ARGS.join(' ')} "$archive" -C "$partial" || die "pack_unpack_failed: tar would not open $archive"\n`
    + `  "$partial/bin/crucible" --version >/dev/null || die "pack_unpack_failed: $partial/bin/crucible would not run"\n`
    + `  ${activatePackSh('"$dest"', '"$partial"')} || die "pack_activation_failed: the previous runtime was preserved"\n`
    + `  printf 'sha256=%s\\nrelease=%s\\n' "$want_sha" "$RELEASE" > "$dest/${STAMP_NAME}"\n`
    + `  rm -f "$archive"\n`
    + `fi\n`
    + `CRUCIBLE="$dest/bin/crucible"\n`;
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
 * `install.sh --uninstall`, whole: the CLI verb, and then the pack.
 *
 * TWO HALVES, AND THE SPLIT IS NOT ARBITRARY. `crucible uninstall`
 * (`crucible/uninstall.py`) stops the server, removes the service and takes
 * `CRUCIBLE_HOME` apart step by named step — everything except the
 * relocatable interpreter it is itself running from, which it cannot unlink
 * without pulling `site-packages` out from under a live process. THIS script
 * unpacked that interpreter, so this script removes it, after the verb has
 * returned. One owner per artefact, and the order is the only one that works.
 *
 * `--dry-run` is passed through and the pack removal becomes a sentence, so
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
    + `# The pack, which the verb deliberately leaves: it is the interpreter\n`
    + `# that just ran, and this script is what unpacked it.\n`
    + `say "server-pack"\n`
    + `if [ "$DRY_RUN" = 1 ]; then\n`
    + `  say "server-pack: would remove $CRUCIBLE_HOME/${SERVER_SUBDIR} and $CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `  say "home: would remove $CRUCIBLE_HOME if it were then empty"\n`
    + `else\n`
    + `  rm -rf "$CRUCIBLE_HOME/${SERVER_SUBDIR}" "$CRUCIBLE_HOME/${SERVER_SUBDIR}${PARTIAL_SUFFIX}" "$CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `  say "server-pack: removed $CRUCIBLE_HOME/${SERVER_SUBDIR}"\n`
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
 *     server-pack      download + verify + unpack into <CRUCIBLE_HOME>/server
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
      what: 'read this host: CRUCIBLE_HOME, the user, free disk, the tools a pack needs',
      words: null,
      sh: hostFactsSh(),
      skip: null,
      timeout: 'quickMs',
    },
    {
      name: 'server-pack',
      what: 'download, verify and unpack the server pack — the interpreter comes WITH it',
      words: null,
      sh: serverPackSh(),
      skip: 'pack-stamp-matches',
      timeout: 'packMs',
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
      what: `download the ${entry.type} environment pack`,
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
  return steps;
}
