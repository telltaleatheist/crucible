/** Local installation discovery and lifecycle. The installation owns its paths. */
import * as path from 'node:path';
import { processRunner, type Runner } from './runner.js';

export interface LocalOptions { home?: string }
export interface LocalUninstallFlags { dryRun: boolean; purgeWeights?: boolean; wslToo?: boolean }
export interface LocalCommand {
  argv: string[];
  env: Record<string, string>;
  cwd: string;
  platform: NodeJS.Platform;
}
export interface LocalInstallation {
  schema_version: 1;
  platform: NodeJS.Platform;
  release: string;
  home: string;
  control: { command: string; args: string[]; cwd: string };
}
export interface LocalStatus {
  schema_version: 1;
  state: 'absent' | 'running' | 'stopped' | 'unreachable' | 'wrong_service' | 'unauthorized' | 'unhealthy' | 'broken';
  name: string;
  url: string;
  detail: string;
}
export class LocalInstallationError extends Error {
  constructor(readonly code: string, message: string) { super(message); this.name = 'LocalInstallationError'; }
}

function root(options: LocalOptions, runner: Runner): string {
  const paths = runner.platform === 'win32' ? path.win32 : path.posix;
  const specified = options.home ?? runner.env['CRUCIBLE_HOME'];
  if (specified !== undefined) {
    if (!paths.isAbsolute(specified)) throw new LocalInstallationError('local_home_invalid', 'CRUCIBLE_HOME must be an absolute path');
    return specified;
  }
  if (runner.platform === 'win32') {
    const appData = runner.env['LOCALAPPDATA'];
    if (!appData) throw new LocalInstallationError('local_home_missing', 'LOCALAPPDATA is unset');
    return paths.join(appData, 'Crucible');
  }
  return paths.join(runner.homedir, '.crucible');
}

export function readLocalInstallation(options: LocalOptions = {}, runner: Runner = processRunner()): LocalInstallation | null {
  const paths = runner.platform === 'win32' ? path.win32 : path.posix;
  const home = root(options, runner);
  const file = paths.join(home, 'installation.json');
  let raw: string;
  try { raw = runner.readFile(file); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw new LocalInstallationError('local_record_unreadable', `Cannot read ${file}: ${String(error)}`);
  }
  let value: LocalInstallation;
  try { value = JSON.parse(raw) as LocalInstallation; }
  catch { throw new LocalInstallationError('local_record_invalid', `${file} is not JSON`); }
  if (!value || value.schema_version !== 1 || value.platform !== runner.platform
      || typeof value.release !== 'string' || typeof value.home !== 'string'
      || paths.normalize(value.home) !== paths.normalize(home)
      || !value.control || typeof value.control.command !== 'string'
      || !paths.isAbsolute(value.control.command) || typeof value.control.cwd !== 'string'
      || !paths.isAbsolute(value.control.cwd) || !Array.isArray(value.control.args)
      || !value.control.args.every(arg => typeof arg === 'string')) {
    throw new LocalInstallationError('local_record_invalid', `${file} is incompatible or malformed; repair Crucible's installation`);
  }
  if (!runner.fileExists(value.control.command)) {
    throw new LocalInstallationError('local_runtime_missing', `Crucible's runtime is missing: ${value.control.command}`);
  }
  return value;
}

const STATES = new Set(['absent', 'running', 'stopped', 'unreachable', 'wrong_service', 'unauthorized', 'unhealthy', 'broken']);
function parseStatus(raw: string): LocalStatus {
  let value: LocalStatus;
  try { value = JSON.parse(raw) as LocalStatus; }
  catch { throw new LocalInstallationError('local_protocol_invalid', 'Crucible returned invalid status JSON'); }
  if (!value || value.schema_version !== 1 || !STATES.has(value.state)
      || typeof value.name !== 'string' || typeof value.url !== 'string' || typeof value.detail !== 'string') {
    throw new LocalInstallationError('local_protocol_invalid', 'Crucible returned an incompatible status document');
  }
  return value;
}

async function invoke(action: string, options: LocalOptions, runner: Runner): Promise<LocalStatus> {
  let install: LocalInstallation | null;
  try { install = readLocalInstallation(options, runner); }
  catch (error) {
    if (action !== 'status') throw error;
    return { schema_version: 1, state: 'broken', name: '', url: '', detail: String(error) };
  }
  if (install === null) {
    const home = root(options, runner);
    const paths = runner.platform === 'win32' ? path.win32 : path.posix;
    const legacy = ['config.toml', 'pairing'].some(file => runner.fileExists(paths.join(home, file)));
    if (action !== 'status') throw new LocalInstallationError('local_not_registered', 'Repair or update Crucible to register its local controls');
    return { schema_version: 1, state: legacy ? 'broken' : 'absent', name: '', url: '',
      detail: legacy ? 'An existing Crucible needs an installer update to register its local controls.' : 'Crucible is not installed on this computer.' };
  }
  const result = await runner.run([install.control.command, ...install.control.args, action, '--json'], {
    timeoutMs: action === 'status' ? 15_000 : 120_000,
    env: { CRUCIBLE_HOME: install.home }, cwd: install.control.cwd,
  });
  if (result.code !== 0 || result.failure !== null) {
    const detail = result.failure ?? result.stderr.trim();
    if (action === 'status') return { schema_version: 1, state: 'broken', name: '', url: '', detail };
    throw new LocalInstallationError('local_action_failed', detail);
  }
  return parseStatus(result.stdout);
}
export function localStatus(options: LocalOptions = {}, runner: Runner = processRunner()): Promise<LocalStatus> {
  return invoke('status', options, runner);
}
export function startLocal(options: LocalOptions = {}, runner: Runner = processRunner()): Promise<LocalStatus> {
  return invoke('start', options, runner);
}
export function stopLocal(options: LocalOptions = {}, runner: Runner = processRunner()): Promise<LocalStatus> {
  return invoke('stop', options, runner);
}

/** Build the installed CLI's uninstall invocation; this function never runs it. */
export function localUninstallCommand(
  flags: LocalUninstallFlags,
  options: LocalOptions = {},
  runner: Runner = processRunner(),
): LocalCommand {
  const installed = readLocalInstallation(options, runner);
  if (installed === null) throw new LocalInstallationError('local_not_registered', 'Repair or update Crucible to register its local controls');
  const prefix = installed.control.args;
  if (prefix.at(-1) !== 'local') {
    throw new LocalInstallationError('local_control_invalid', 'This installation does not publish a supported uninstall command; repair Crucible');
  }
  if (flags.wslToo && installed.platform !== 'win32') {
    throw new LocalInstallationError('uninstall_wsl_too_needs_host', 'Only the Windows controller can remove its WSL engine');
  }
  const argv = [installed.control.command, ...prefix.slice(0, -1), 'uninstall', '--json'];
  if (flags.dryRun) argv.push('--dry-run');
  if (flags.purgeWeights) argv.push('--purge-weights');
  if (flags.wslToo) argv.push('--wsl-too');
  return { argv, env: { CRUCIBLE_HOME: installed.home }, cwd: installed.control.cwd, platform: installed.platform };
}
