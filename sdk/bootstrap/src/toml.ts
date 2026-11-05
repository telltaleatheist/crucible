/**
 * The TOML of a Crucible config.toml. The server reads it with Python's tomllib, and a person or
 * another tool can write any valid TOML into it, so a form this reader cannot read says so as this
 * reader's limit and never as the file's fault. What it reads beyond tomli_w's output is what real
 * configs hold: arrays over several lines (this PC's `advertise` and `cors_origins`, 2026-11-05),
 * inline tables, multi-line strings and dates (kept as their text).
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
    let rawValue = line.slice(equals + 1).trim();
    // A value that has not ended on this line carries on over the next ones.
    while (valueIsOpen(rawValue) && index + 1 < lines.length) {
      index += 1;
      const next = lines[index] as string;
      rawValue += '\n' + (insideMultilineString(rawValue) ? next : stripComment(next).trim());
    }
    const { value, rest } = parseValue(rawValue, lineNo);
    if (rest.trim().length > 0) throw new TomlError(lineNo, `trailing text after value: ${JSON.stringify(rest)}`);
    target[key] = value;
  }
  return root;
}

/** Walks a value's text: open brackets and whether it stopped inside a multi-line string. */
function scan(text: string): { depth: number; multiline: boolean } {
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    const ch = text[i] as string;
    const three = text.slice(i, i + 3);
    if (three === '"""' || three === "'''") {
      let end = text.indexOf(three, i + 3);
      while (three === '"""' && end > 0 && escaped(text, end)) end = text.indexOf(three, end + 1);
      if (end < 0) return { depth, multiline: true };
      i = end + 3;
      while (text[i] === three[0]) i += 1;
      continue;
    }
    if (ch === '"' || ch === "'") {
      i += 1;
      while (i < text.length && text[i] !== ch) i += ch === '"' && text[i] === '\\' ? 2 : 1;
    } else if (ch === '[' || ch === '{') {
      depth += 1;
    } else if (ch === ']' || ch === '}') {
      depth -= 1;
    }
    i += 1;
  }
  return { depth, multiline: false };
}

function escaped(text: string, at: number): boolean {
  let slashes = 0;
  for (let i = at - 1; i >= 0 && text[i] === '\\'; i -= 1) slashes += 1;
  return slashes % 2 === 1;
}

function valueIsOpen(text: string): boolean {
  const { depth, multiline } = scan(text);
  return multiline || depth > 0;
}

function insideMultilineString(text: string): boolean {
  return scan(text).multiline;
}

function stripComment(line: string): string {
  const hash = findUnquoted(line, '#');
  return hash < 0 ? line : line.slice(0, hash);
}

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

  if (text.startsWith('"""') || text.startsWith("'''")) return parseMultilineString(text, lineNo);
  if (first === '"') return parseBasicString(text, lineNo);
  if (first === "'") {
    const end = text.indexOf("'", 1);
    if (end < 0) throw new TomlError(lineNo, 'unterminated literal string');
    return { value: text.slice(1, end), rest: text.slice(end + 1) };
  }
  if (first === '[') return parseArray(text, lineNo);
  if (first === '{') return parseInlineTable(text, lineNo);

  // A date and time has a space in it; it is kept as its text, which nothing here reads.
  const date = /^\d{4}-\d{2}-\d{2}(?:[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})?)?|^\d{2}:\d{2}:\d{2}(?:\.\d+)?/.exec(text);
  if (date !== null) return { value: date[0], rest: text.slice(date[0].length) };

  const scalar = /^[^,\]}\s]+/.exec(text);
  if (scalar === null) throw new TomlError(lineNo, `bad value ${JSON.stringify(text)}`);
  const token = scalar[0];
  const rest = text.slice(token.length);
  if (token === 'true') return { value: true, rest };
  if (token === 'false') return { value: false, rest };
  if (/^[+-]?(0|[1-9](_?[0-9])*)$/.test(token)) return { value: Number(token.replace(/_/g, '')), rest };
  if (/^0x[0-9A-Fa-f](_?[0-9A-Fa-f])*$|^0o[0-7](_?[0-7])*$|^0b[01](_?[01])*$/.test(token)) {
    return { value: Number(token.replace(/_/g, '')), rest };
  }
  if (/^[+-]?inf$/.test(token)) return { value: token.startsWith('-') ? -Infinity : Infinity, rest };
  if (/^[+-]?nan$/.test(token)) return { value: Number.NaN, rest };
  if (/^[+-]?(0|[1-9](_?[0-9])*)(\.[0-9](_?[0-9])*)?([eE][+-]?[0-9]+)?$/.test(token)) {
    return { value: Number(token.replace(/_/g, '')), rest };
  }
  throw new TomlError(lineNo, `unsupported value ${JSON.stringify(token)} (dates and other forms are refused, not guessed)`);
}

function parseMultilineString(text: string, lineNo: number): { value: string; rest: string } {
  const quote = text.slice(0, 3);
  let end = text.indexOf(quote, 3);
  while (quote === '"""' && end > 0 && escaped(text, end)) end = text.indexOf(quote, end + 1);
  if (end < 0) throw new TomlError(lineNo, 'unterminated multi-line string');
  // Up to two more quotes may close it: they belong to the string.
  while (text[end + 3] === quote[0]) end += 1;
  let body = text.slice(3, end);
  if (body.startsWith('\r\n')) body = body.slice(2);
  else if (body.startsWith('\n')) body = body.slice(1);
  if (quote === '"""') {
    // A line-ending backslash joins the lines; the rest are the basic string's own escapes.
    body = body.replace(/\\[ \t]*\r?\n\s*/g, '');
    const { value } = parseBasicString(`"${body.replace(/(?<!\\)((?:\\\\)*)"/g, '$1\\"').replace(/\r?\n/g, '\\n')}"`, lineNo);
    return { value, rest: text.slice(end + 3) };
  }
  return { value: body, rest: text.slice(end + 3) };
}

function parseInlineTable(text: string, lineNo: number): { value: TomlTable; rest: string } {
  const table: TomlTable = {};
  let rest = text.slice(1).trimStart();
  for (;;) {
    if (rest.startsWith('}')) return { value: table, rest: rest.slice(1) };
    if (rest.length === 0) throw new TomlError(lineNo, 'unterminated inline table');
    const equals = findUnquoted(rest, '=');
    if (equals < 0) throw new TomlError(lineNo, `expected key = value in an inline table, got ${JSON.stringify(rest)}`);
    const keys = splitKey(rest.slice(0, equals).trim(), lineNo);
    const target = descend(table, keys.slice(0, -1), lineNo);
    const parsed = parseValue(rest.slice(equals + 1).trimStart(), lineNo);
    target[keys[keys.length - 1] as string] = parsed.value;
    rest = parsed.rest.trimStart();
    if (rest.startsWith(',')) rest = rest.slice(1).trimStart();
    else if (!rest.startsWith('}')) throw new TomlError(lineNo, `expected , or } in an inline table, got ${JSON.stringify(rest)}`);
  }
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
    if (rest.length === 0) throw new TomlError(lineNo, 'unterminated array');
    const parsed = parseValue(rest, lineNo);
    items.push(parsed.value);
    rest = parsed.rest.trimStart();
    if (rest.startsWith(',')) rest = rest.slice(1).trimStart();
    else if (!rest.startsWith(']')) throw new TomlError(lineNo, `expected , or ] in array, got ${JSON.stringify(rest)}`);
  }
}
