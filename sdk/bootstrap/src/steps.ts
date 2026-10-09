import { DESKTOP_PACKAGES, interpreterFor, interpreterUrl } from './interpreter.js';
import { wheelAssetName, wheelShaAssetName, RELEASE_REPO } from './release.js';
import { activateRuntimeSh, CURL_ARGS, DOWNLOADS_SUBDIR, guestProbeScript, PARTIAL_SUFFIX, SERVER_SUBDIR, STAMP_NAME, TAR_ARGS } from './runtime.js';

/**
 * A word in a step's argv: a literal, an install value, or a raw shell fragment for the generated
 * script.
 */
export type Word = string | { ref: RefName } | { sh: string };

/** The values a step's argv can refer to. */
export type RefName = 'crucible' | 'release' | 'backend' | 'home' | 'user';

/**
 * The environment variable `crucible init --token-env` reads its token from. Never argv: a
 * process's command line is readable by every user on the box (`ps aux`), its environment
 * only by its own user and root.
 */
export const INIT_TOKEN_ENV = 'CRUCIBLE_INIT_TOKEN';

/** How the generated scripts spell each ref. */
export const SHELL_VARIABLE: Readonly<Record<RefName, string>> = {
  crucible: 'CRUCIBLE',
  release: 'RELEASE',
  backend: 'BACKEND',
  home: 'CRUCIBLE_HOME',
  user: 'GUEST_USER',
};

export type SkipRule = 'config-exists' | 'interpreter-stamp-matches' | 'root-only';

export interface StepDef {
  name: string;
  /** One sentence describing the step, used as a skipped step's `detail`. */
  what: string;
  /** The argv as the target sees it; null for the steps that are programs. */
  words: readonly Word[] | null;
  /** POSIX sh for the generated installer. */
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
  /** `--host`/`--port` words for `crucible init`. */
  bind: readonly Word[];
  /** Whether the linger step is part of this install. */
  linger: boolean;
}

function v(ref: RefName): string {
  return `"$${SHELL_VARIABLE[ref]}"`;
}

function q(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** A step's words as one shell line; refs become `"$VAR"`, literals are quoted. */
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

export function hostFactsSh(): string {
  return `crucible_probe() {\n`
    + `  ${guestProbeScript(undefined)}\n`
    + `}\n`
    + `probe_out="$(crucible_probe)"\n`
    + `CRUCIBLE_HOME="$(printf '%s\\n' "$probe_out" | sed -n 's/^home=//p')"\n`
    + `GUEST_USER="$(printf '%s\\n' "$probe_out" | sed -n 's/^user=//p')"\n`
    + `free_kib="$(printf '%s\\n' "$probe_out" | sed -n 's/^free_kib=//p')"\n`
    + `stamp_python_sha="$(printf '%s\\n' "$probe_out" | sed -n 's/^python_sha256=//p')"\n`
    + `stamp_release="$(printf '%s\\n' "$probe_out" | sed -n 's/^release=//p')"\n`
    + `missing="$(printf '%s\\n' "$probe_out" | sed -n 's/^missing=//p' | tr '\\n' ' ')"\n`
    + `if [ -n "$missing" ]; then die "guest_missing_tool: this machine has no $missing; the interpreter is fetched with curl and unpacked with tar"; fi\n`;
}

export function serverSh(): string {
  return serverPreludeSh() + wheelFetchSh() + interpreterSh() + wheelInstallSh();
}

export const PROGRESS_PREFIX = 'crucible-progress ';

export function progressFetchSh(destination: string, url: string): string {
  return `  total="$(curl -fsSLI -m 20 ${url} | tr -d '\\r' `
    + `| awk 'tolower($1) == "content-length:" { print $2 }' | tail -n 1)"\n`
    + `  case "$total" in ''|*[!0-9]*) total=null ;; esac\n`
    + `  curl ${CURL_ARGS.join(' ')} -o ${destination} ${url} &\n`
    + `  fetch_pid=$!\n`
    + `  while kill -0 "$fetch_pid" 2>/dev/null; do\n`
    + `    got=0\n`
    + `    if [ -f ${destination} ]; then got="$(wc -c < ${destination} | tr -d ' ')"; fi\n`
    + `    printf '${PROGRESS_PREFIX}{"bytes_done": %s, "bytes_total": %s, "file": "%s"}\\n' `
    + `"$got" "$total" "$py_asset"\n`
    + `    sleep 1\n`
    + `  done\n`
    + `  wait "$fetch_pid" || die "runtime_download_failed: $py_url"\n`;
}

export function serverPreludeSh(): string {
  const cases = (['cuda-linux', 'mlx-darwin'] as const).map((backend) => {
    const pin = interpreterFor(backend);
    return `  ${backend}) py_asset='${pin.asset}'; py_sha='${pin.sha256}'; py_version='${pin.version}'; py_url='${interpreterUrl(pin)}' ;;\n`;
  }).join('');
  return `dest="$CRUCIBLE_HOME/${SERVER_SUBDIR}"\n`
    + `partial="$dest${PARTIAL_SUFFIX}"\n`
    + `downloads="$CRUCIBLE_HOME/${DOWNLOADS_SUBDIR}"\n`
    + `wheel=""\n`
    + `case "$BACKEND" in\n`
    + cases
    + `  *) die "unsupported_platform: no interpreter is pinned for $BACKEND" ;;\n`
    + `esac\n`
    + `crucible_older() {\n`
    + `  awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".");\n`
    + `    for (i = 1; i <= 3; i++) { if ((x[i]+0) < (y[i]+0)) exit 0; if ((x[i]+0) > (y[i]+0)) exit 1 } exit 1 }'\n`
    + `}\n`
    + `if [ -n "$stamp_release" ] && crucible_older "$RELEASE" "$stamp_release"; then\n`
    + `  [ "$ROLLBACK_TO" = "$RELEASE" ] || die "install_would_downgrade: $dest is the $stamp_release release and this would install $RELEASE over it. Nothing was downloaded. An operator who means to go back names the version: --rollback-to $RELEASE"\n`
    + `fi\n`
    + quiesceSh();
}

export function quiesceSh(): string {
  return `quiesced=0\n`
    + `crucible_quiesce() {\n`
    + `  if [ ! -x "$dest/bin/crucible" ]; then quiesced=1; return 0; fi\n`
    + `  if [ -n "$wheel" ] && [ -f "$downloads/$wheel" ]; then\n`
    + `    stage="$downloads/stage"\n`
    + `    rm -rf "$stage"\n`
    + `    if "$dest/bin/python3" -m pip install --quiet --no-deps --no-input --target "$stage" "$downloads/$wheel" >/dev/null 2>&1 \\\n`
    + `      && PYTHONPATH="$stage" "$dest/bin/python3" -m crucible.cli local shutdown; then\n`
    + `      rm -rf "$stage"; quiesced=1; say "server: the running server was stopped by $RELEASE's own code"; return 0\n`
    + `    fi\n`
    + `    rm -rf "$stage"\n`
    + `    say "server: $RELEASE's code could not stop the running server; asking the installed release to stop it"\n`
    + `  fi\n`
    + `  "$dest/bin/crucible" local shutdown || return 1\n`
    + `  quiesced=1\n`
    + `}\n`
    + `crucible_quiesce_after() {\n`
    + `  if [ "$quiesced" = 1 ]; then return 0; fi\n`
    + `  say "server: stopping the old server with the release just installed"\n`
    + `  "$dest/bin/crucible" local shutdown || die "upgrade_stop_failed: the Crucible server that was already running could not be stopped, by the old release or by $RELEASE, so the new one cannot take its place yet. $RELEASE is installed and takes over when the old server next stops: when this machine restarts, or on Windows when Crucible next starts"\n`
    + `  quiesced=1\n`
    + `}\n`;
}

export function interpreterSh(): string {
  return `if [ -n "$stamp_release" ] && [ "$stamp_python_sha" = "$py_sha" ] && [ -x "$dest/bin/python3" ]; then\n`
    + `  say "server: python $py_version is already at $dest"\n`
    + `else\n`
    + `  say "server: python $py_version from python-build-standalone"\n`
    + `  mkdir -p "$downloads"; rm -f "$downloads/$py_asset"\n`
    + progressFetchSh('"$downloads/$py_asset"', '"$py_url"')
    + `  got_sha="$($SHA_TOOL "$downloads/$py_asset" | awk '{print $1}')"\n`
    + `  if [ "$got_sha" != "$py_sha" ]; then rm -f "$downloads/$py_asset"; die "runtime_sha_mismatch: $py_asset hashes $got_sha and this installer pins $py_sha. The download was deleted"; fi\n`
    + `  rm -rf "$partial" && mkdir -p "$partial"\n`
    + `  tar ${TAR_ARGS.join(' ')} "$downloads/$py_asset" -C "$partial" || die "runtime_unpack_failed: tar would not open $downloads/$py_asset"\n`
    + `  [ -x "$partial/python/bin/python3" ] || die "runtime_unpack_failed: $py_asset unpacked without a python/bin/python3"\n`
    + `  ${activateRuntimeSh('"$dest"', '"$partial/python"', 'crucible_quiesce || return 1')} || die "runtime_unpack_failed: the previous runtime was preserved"\n`
    + `  rm -rf "$partial" "$downloads/$py_asset"\n`
    + `fi\n`;
}

export function wheelFetchSh(): string {
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$RELEASE`;
  return `wheel="${wheelAssetName('$RELEASE')}"\n`
    + `say "server: $wheel"\n`
    + `mkdir -p "$downloads"; rm -f "$downloads/$wheel"\n`
    + `curl ${CURL_ARGS.join(' ')} -o "$downloads/$wheel" "${base}/$wheel" || die "runtime_download_failed: ${base}/$wheel"\n`
    + `want_sha="$(curl -fsSL --retry 3 "${base}/${wheelShaAssetName('$RELEASE')}" | awk '{print $1}')" || die "runtime_download_failed: ${base}/${wheelShaAssetName('$RELEASE')}"\n`
    + `case "$want_sha" in *[!0-9a-f]*|"") die "runtime_download_failed: ${base}/${wheelShaAssetName('$RELEASE')} is not a sha256" ;; esac\n`
    + `got_sha="$($SHA_TOOL "$downloads/$wheel" | awk '{print $1}')"\n`
    + `if [ "$got_sha" != "$want_sha" ]; then rm -f "$downloads/$wheel"; die "runtime_sha_mismatch: $wheel hashes $got_sha, the release says $want_sha. The download was deleted"; fi\n`;
}

export function wheelInstallSh(): string {
  return `crucible_quiesce || say "server: the running server did not stop; installing $RELEASE and stopping it with that"\n`
    + `"$dest/bin/python3" -m pip install --upgrade --no-input "$downloads/$wheel" || die "runtime_install_failed: pip would not install $wheel"\n`
    + `if [ "$(uname -s)" = Darwin ]; then "$dest/bin/python3" -m pip install ${DESKTOP_PACKAGES.join(' ')} || die "runtime_install_failed: the desktop packages would not install"; fi\n`
    + `rm -f "$downloads/$wheel"\n`
    + `printf 'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n' "$py_sha" "$py_version" "$RELEASE" > "$dest/${STAMP_NAME}"\n`
    + `CRUCIBLE="$dest/bin/crucible"\n`
    + `"$CRUCIBLE" --version >/dev/null || die "runtime_install_failed: $CRUCIBLE would not run"\n`
    + `crucible_quiesce_after\n`;
}

export function lingerSh(): string {
  return `if [ "$MECHANISM" = systemd ] && ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then\n`
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

/** The install sequence, as named steps. */
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
      words: [crucible, 'init', '--token-env', ...plan.bind, ...plan.enableFlags],
      sh: `if [ -f "$CRUCIBLE_HOME/config.toml" ]; then\n`
        + `  say "init: $CRUCIBLE_HOME/config.toml exists; its token is kept"\n`
        + `else\n`
        + `  if [ -z "\${TOKEN:-}" ]; then\n`
        + `    TOKEN="$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=')"\n`
        + `  fi\n`
        + `  ${INIT_TOKEN_ENV}="$TOKEN" ${renderSh([crucible, 'init', '--token-env', ...plan.bind, ...plan.enableFlags])} || die "step_failed: init"\n`
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
  const patchWords: Word[] = [crucible, 'env', 'patch', 'llm'];
  steps.push({
    name: 'env-patch-llm',
    what: "apply the llm environment's site-packages patches before the service starts",
    words: patchWords,
    sh: `${renderSh(patchWords)} || die "step_failed: env-patch-llm"\n`,
    skip: null,
    timeout: 'quickMs',
  });
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
