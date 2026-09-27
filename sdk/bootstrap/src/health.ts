import {
  CrucibleAuthError,
  CrucibleClient,
  CrucibleNotACrucible,
  CrucibleUnreachable,
  CrucibleVersionError,
  type Activity,
} from '@crucible/client';

import { readLocalConfig, type LocalConfigOptions } from './config.js';
import { BootstrapRefusal } from './errors.js';
import { processRunner, type Runner } from './runner.js';
import { BOOTSTRAP_VERSION } from './version.js';

/** What `health()` needs of a client: the one call it makes. */
export interface ActivityClient {
  activity(): Promise<Activity>;
}

export interface HealthOptions extends LocalConfigOptions {
  /** Who is asking, for the server's log. */
  clientName?: string;
  /** Injectable so a test can script the answer without a server. */
  clientFactory?: (options: { url: string; token: string; clientName: string }) => ActivityClient;
}

export async function health(options: HealthOptions = {}, runner: Runner = processRunner()): Promise<Activity> {
  const config = await readLocalConfig(
    {
      ...(options.distro === undefined ? {} : { distro: options.distro }),
      ...(options.exact === undefined ? {} : { exact: options.exact }),
      ...(options.home === undefined ? {} : { home: options.home }),
    },
    runner,
  );
  const clientName = options.clientName ?? `crucible-bootstrap/${BOOTSTRAP_VERSION}`;
  const factory = options.clientFactory ?? ((given) => new CrucibleClient(given));
  const client = factory({ url: config.url, token: config.token, clientName });
  try {
    return await client.activity();
  } catch (err) {
    if (err instanceof CrucibleUnreachable) {
      throw new BootstrapRefusal(
        'unreachable',
        `${config.name} at ${config.url} is not answering: ${err.message}. Its config is at ${config.configPath}; ensureRunning() starts the service.`,
        { command: 'crucible service start', cause: err },
      );
    }
    if (err instanceof CrucibleAuthError) {
      throw new BootstrapRefusal(
        'wrong_token',
        `${config.name} at ${config.url} refused the token its own ${config.configPath} holds (${err.code}). The running server was started against a different config — `
          + 're-run `crucible service install` so the service and the file agree.',
        { command: 'crucible service install', cause: err },
      );
    }
    if (err instanceof CrucibleNotACrucible) {
      throw new BootstrapRefusal(
        'not_a_crucible',
        `${config.url} answered but is not a Crucible: ${err.body.slice(0, 200)}. Something else holds the port ${config.configPath} names.`,
        { cause: err },
      );
    }
    if (err instanceof CrucibleVersionError) {
      throw new BootstrapRefusal(
        'version_mismatch',
        `${config.name} speaks API version ${err.serverApiVersion ?? 'unknown'}; this bootstrap's SDK speaks ${err.clientApiVersion}. Install the matching release: the server wheel and the two tarballs are cut together.`,
        { cause: err },
      );
    }
    throw err;
  }
}
