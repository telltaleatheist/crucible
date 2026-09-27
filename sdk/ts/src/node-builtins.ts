import { CrucibleError } from './errors.js';

export type NodeModule = Record<string, unknown>;

export interface NodeBuiltins {
  readonly fs: NodeModule;
  readonly path: NodeModule;
  readonly os: NodeModule;
}

let builtins: Promise<NodeBuiltins> | null = null;

async function importBuiltins(): Promise<NodeBuiltins> {
  const scheme = 'node:';
  return {
    fs: (await import(/* webpackIgnore: true */ `${scheme}fs/promises`)) as NodeModule,
    path: (await import(/* webpackIgnore: true */ `${scheme}path`)) as NodeModule,
    os: (await import(/* webpackIgnore: true */ `${scheme}os`)) as NodeModule,
  };
}

export async function loadNodeBuiltins(absent: string): Promise<NodeBuiltins> {
  if (builtins === null) builtins = importBuiltins();
  try {
    return await builtins;
  } catch (cause) {
    builtins = null;
    throw new CrucibleError(absent, { cause });
  }
}

export function requireFunctions(
  module: NodeModule | null,
  names: readonly string[],
  missing: (name: string) => string,
): void {
  for (const name of names) {
    if (module === null || typeof module[name] !== 'function') {
      throw new CrucibleError(missing(name));
    }
  }
}
