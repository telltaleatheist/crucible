/**
 * Enough TOML to read a `config.toml` that `crucible/config.py` wrote.
 *
 * Zero runtime dependencies is the package rule, and the one file this package
 * reads is written by `tomli_w` from a dict of strings, integers, booleans and
 * an array of tables. So this reads exactly that dialect — tables, arrays of
 * tables, bare and quoted keys, basic and literal strings, integers, floats,
 * booleans, single-line arrays — and REFUSES by name anything else (inline
 * tables, multi-line strings, dates) rather than guessing at it. A refusal is a
 * `TomlError` naming the line; `config.ts` turns it into `config_unreadable`,
 * which is the same thing the server would do with a file it cannot parse.
 */

export class TomlError extends Error {
  readonly line: number;

  constructor(line: number, message: string) {
    super(`line ${line}: ${message}`);
    this.name = 'TomlError';
    this.line = line;
  }
}

export type TomlValue = string | number | boolean | TomlValue[] | TomlTable;
export interface TomlTable {
  [key: string]: TomlValue;
}

export function parseToml(text: string): TomlTable {
  const root: TomlTable = {};
  let current: TomlTable = root;
  const lines = text.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const lineNo = index + 1;
    const line = stripComment(lines[index] as string).trim();
    if (line.length === 0) continue;

    if (line.startsWith('[[')) {
      if (!line.endsWith(']]')) throw new TomlError(lineNo, 'unterminated array-of-tables header');
      const keys = splitKey(line.slice(2, -2).trim(), lineNo);
      const parent = descend(root, keys.slice(0, -1), lineNo);
      const last = keys[keys.length - 1] as string;
      const existing = parent[last];
      let array: TomlValue[];
      if (existing === undefined) {
        array = [];
        parent[last] = array;
      } else if (Array.isArray(existing)) {
        array = existing;
      } else {
        throw new TomlError(lineNo, `${keys.join('.')} is already defined and is not an array of tables`);
      }
      const table: TomlTable = {};
      array.push(table);
      current = table;
      continue;
    }

    if (line.startsWith('[')) {
      if (!line.endsWith(']')) throw new TomlError(lineNo, 'unterminated table header');
      const keys = splitKey(line.slice(1, -1).trim(), lineNo);
      current = descend(root, keys, lineNo);
      continue;
    }

    const equals = findUnquoted(line, '=');
    if (equals < 0) throw new TomlError(lineNo, `expected key = value, got ${JSON.stringify(line)}`);
    const keys = splitKey(line.slice(0, equals).trim(), lineNo);
    const target = descend(current, keys.slice(0, -1), lineNo);
    const key = keys[keys.length - 1] as string;
    if (key in target) throw new TomlError(lineNo, `${keys.join('.')} is defined twice`);
    const rawValue = line.slice(equals + 1).trim();
    const { value, rest } = parseValue(rawValue, lineNo);
    if (rest.trim().length > 0) throw new TomlError(lineNo, `trailing text after value: ${JSON.stringify(rest)}`);
    target[key] = value;
  }
  return root;
}

function stripComment(line: string): string {
  const hash = findUnquoted(line, '#');
  return hash < 0 ? line : line.slice(0, hash);
}

/** Index of `needle` outside any quoted string, or -1. */
function findUnquoted(line: string, needle: string): number {
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i] as string;
    if (quote === null) {
      if (ch === needle) return i;
      if (ch === '"' || ch === "'") quote = ch;
    } else if (ch === '\\' && quote === '"') {
      i += 1;
    } else if (ch === quote) {
      quote = null;
    }
  }
  return -1;
}

function splitKey(text: string, lineNo: number): string[] {
  if (text.length === 0) throw new TomlError(lineNo, 'empty key');
  const keys: string[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i] as string;
    if (ch === '"' || ch === "'") {
      const end = text.indexOf(ch, i + 1);
      if (end < 0) throw new TomlError(lineNo, 'unterminated quoted key');
      keys.push(text.slice(i + 1, end));
      i = end + 1;
    } else {
      const match = /^[A-Za-z0-9_-]+/.exec(text.slice(i));
      if (match === null) throw new TomlError(lineNo, `bad key ${JSON.stringify(text)}`);
      keys.push(match[0]);
      i += match[0].length;
    }
    while (i < text.length && text[i] === ' ') i += 1;
    if (i < text.length) {
      if (text[i] !== '.') throw new TomlError(lineNo, `bad key ${JSON.stringify(text)}`);
      i += 1;
      while (i < text.length && text[i] === ' ') i += 1;
      if (i >= text.length) throw new TomlError(lineNo, `bad key ${JSON.stringify(text)}`);
    }
  }
  return keys;
}

function descend(from: TomlTable, keys: readonly string[], lineNo: number): TomlTable {
  let table = from;
  for (const key of keys) {
    const next = table[key];
    if (next === undefined) {
      const created: TomlTable = {};
      table[key] = created;
      table = created;
    } else if (Array.isArray(next)) {
      // A dotted header under an array of tables addresses its last element.
      const last = next[next.length - 1];
      if (last === undefined || typeof last !== 'object' || Array.isArray(last)) {
        throw new TomlError(lineNo, `${key} is an array, not a table`);
      }
      table = last;
    } else if (typeof next === 'object') {
      table = next;
    } else {
      throw new TomlError(lineNo, `${key} is a value, not a table`);
    }
  }
  return table;
}

function parseValue(text: string, lineNo: number): { value: TomlValue; rest: string } {
  if (text.length === 0) throw new TomlError(lineNo, 'missing value');
  const first = text[0] as string;

  if (text.startsWith('"""') || text.startsWith("'''")) {
    throw new TomlError(lineNo, 'multi-line strings are not something crucible writes; refusing to guess');
  }
  if (first === '"') return parseBasicString(text, lineNo);
  if (first === "'") {
    const end = text.indexOf("'", 1);
    if (end < 0) throw new TomlError(lineNo, 'unterminated literal string');
    return { value: text.slice(1, end), rest: text.slice(end + 1) };
  }
  if (first === '[') return parseArray(text, lineNo);
  if (first === '{') throw new TomlError(lineNo, 'inline tables are not something crucible writes; refusing to guess');

  const scalar = /^[^,\]\s]+/.exec(text);
  if (scalar === null) throw new TomlError(lineNo, `bad value ${JSON.stringify(text)}`);
  const token = scalar[0];
  const rest = text.slice(token.length);
  if (token === 'true') return { value: true, rest };
  if (token === 'false') return { value: false, rest };
  if (/^[+-]?(0|[1-9](_?[0-9])*)$/.test(token)) return { value: Number(token.replace(/_/g, '')), rest };
  if (/^[+-]?(0|[1-9](_?[0-9])*)(\.[0-9](_?[0-9])*)?([eE][+-]?[0-9]+)?$/.test(token)) {
    return { value: Number(token.replace(/_/g, '')), rest };
  }
  throw new TomlError(lineNo, `unsupported value ${JSON.stringify(token)} (dates and other forms are refused, not guessed)`);
}

function parseBasicString(text: string, lineNo: number): { value: string; rest: string } {
  let out = '';
  let i = 1;
  while (i < text.length) {
    const ch = text[i] as string;
    if (ch === '"') return { value: out, rest: text.slice(i + 1) };
    if (ch === '\\') {
      const next = text[i + 1];
      if (next === undefined) throw new TomlError(lineNo, 'dangling escape');
      switch (next) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case 'u':
        case 'U': {
          const width = next === 'u' ? 4 : 8;
          const hex = text.slice(i + 2, i + 2 + width);
          if (!new RegExp(`^[0-9A-Fa-f]{${width}}$`).test(hex)) throw new TomlError(lineNo, `bad unicode escape \\${next}${hex}`);
          out += String.fromCodePoint(Number.parseInt(hex, 16));
          i += width;
          break;
        }
        default:
          throw new TomlError(lineNo, `unknown escape \\${next}`);
      }
      i += 2;
      continue;
    }
    out += ch;
    i += 1;
  }
  throw new TomlError(lineNo, 'unterminated string');
}

function parseArray(text: string, lineNo: number): { value: TomlValue[]; rest: string } {
  const items: TomlValue[] = [];
  let rest = text.slice(1).trimStart();
  for (;;) {
    if (rest.startsWith(']')) return { value: items, rest: rest.slice(1) };
    if (rest.length === 0) throw new TomlError(lineNo, 'unterminated array (arrays must be on one line)');
    const parsed = parseValue(rest, lineNo);
    items.push(parsed.value);
    rest = parsed.rest.trimStart();
    if (rest.startsWith(',')) rest = rest.slice(1).trimStart();
    else if (!rest.startsWith(']')) throw new TomlError(lineNo, `expected , or ] in array, got ${JSON.stringify(rest)}`);
  }
}
