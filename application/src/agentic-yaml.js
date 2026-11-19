// Set values in a YAML file as written, keeping its comments and layout.
//
// The configuration API takes the whole file as text (PUT /config/{name} {raw,
// version}), and in these files the comments are the documentation. So instead
// of re-serialising the parsed config, each changed setting has its own line
// rewritten in place: same indentation, same trailing comment, nothing else
// touched. The server parses the result and shows every setting that moves
// before anything is saved, so a mistake here is caught at review.
//
// Handles what config/agentic_os.yaml contains: nested mappings, scalars, flow
// lists ([a, b]) on one or more lines, and block scalars (>- / |-).

const BLOCK = /^[|>][+-]?\d*[+-]?$/;
const KEY_LINE = /^( *)([A-Za-z_][\w-]*)\s*:(\s.*|)$/;
const RESERVED = /^(true|false|yes|no|on|off|y|n|null|~)$/i;
const BARE = /^[A-Za-z_][A-Za-z0-9_./-]*(:[A-Za-z0-9_./-]+)*$/;

const indentOf = line => line.match(/^ */)[0].length;

// "rest" is everything after "key:". Returns the leading space, the value as
// written and the tail (spaces and comment), so the comment can be kept.
function splitValue(rest) {
  const lead = rest.match(/^\s*/)[0];
  const body = rest.slice(lead.length);
  let end;
  if (body[0] === '"') {
    let i = 1;
    while (i < body.length && !(body[i] === '"' && body[i - 1] !== '\\')) i++;
    end = Math.min(i + 1, body.length);
  } else if (body[0] === "'") {
    let i = 1;
    while (i < body.length) { if (body[i] === "'") { if (body[i + 1] === "'") { i += 2; continue; } break; } i++; }
    end = Math.min(i + 1, body.length);
  } else if (body[0] === '#') {
    end = 0;
  } else {
    const m = /\s#/.exec(body);
    end = m ? m.index : body.length;
  }
  return { lead, value: body.slice(0, end).trimEnd(), tail: body.slice(end) };
}

const bracketDepth = text => {
  const code = text.replace(/(^|\s)#.*$/, '');
  return (code.match(/\[/g) || []).length - (code.match(/\]/g) || []).length;
};

// Find the line(s) holding a dotted key. Returns {start, end, line parts} or null.
export function locate(lines, path) {
  const stack = [];
  let skipDeeperThan = null; // inside a block scalar
  let flowDepth = 0;         // inside a multi-line flow list
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (flowDepth > 0) { flowDepth += bracketDepth(line); continue; }
    if (!line.trim() || /^\s*#/.test(line)) continue;
    const indent = indentOf(line);
    if (skipDeeperThan !== null) { if (indent > skipDeeperThan) continue; skipDeeperThan = null; }
    const m = KEY_LINE.exec(line);
    if (!m) continue;
    while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
    const key = m[2];
    const parts = splitValue(m[3]);
    const isBlock = BLOCK.test(parts.value);
    const isFlow = parts.value.startsWith('[') && bracketDepth(parts.value) > 0;
    if ([...stack.map(s => s.key), key].join('.') === path) {
      let end = i;
      if (isBlock) {
        let j = i + 1;
        while (j < lines.length && (!lines[j].trim() || indentOf(lines[j]) > indent)) j++;
        while (j - 1 > i && !lines[j - 1].trim()) j--;
        end = j - 1;
      } else if (isFlow) {
        let depth = bracketDepth(parts.value);
        let j = i;
        while (depth > 0 && j + 1 < lines.length) { j++; depth += bracketDepth(lines[j]); }
        end = j;
      }
      return { start: i, end, indent, key, ...parts, isBlock };
    }
    stack.push({ indent, key });
    if (isBlock) skipDeeperThan = indent;
    else if (isFlow) flowDepth = bracketDepth(parts.value);
  }
  return null;
}

export function scalar(value) {
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (value === null || value === undefined) return 'null';
  const s = String(value);
  if (BARE.test(s) && !RESERVED.test(s)) return s;
  return JSON.stringify(s); // a JSON string is a valid YAML double-quoted scalar
}

const flowList = list => `[${list.map(scalar).join(', ')}]`;

function wrap(text, width) {
  const words = text.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    if (current && current.length + 1 + word.length > width) { lines.push(current); current = word; }
    else current = current ? `${current} ${word}` : word;
  }
  if (current) lines.push(current);
  return lines;
}

// Long text as a block scalar under a key at `indent`, or null when a plain
// one-line scalar says it exactly.
function blockLines(text, indent) {
  const pad = ' '.repeat(indent + 2);
  if (text.includes('\n')) {
    const body = text.replace(/\s+$/, '').split('\n').map(l => l.replace(/\s+$/, ''));
    const first = body.find(l => l.trim()) || '';
    const header = /^\s/.test(first) ? '|2-' : '|-';
    return [header, ...body.map(l => (l ? pad + l : ''))];
  }
  // Folding joins wrapped lines with one space, so only single-spaced text folds exactly.
  if (/^\S+( \S+)*$/.test(text) && text.length > 60) return ['>-', ...wrap(text, Math.max(40, 92 - pad.length)).map(l => pad + l)];
  return null;
}

// Set one dotted key to a value. Throws when the key is not in the file.
export function setValue(raw, path, value) {
  const eol = raw.includes('\r\n') ? '\r\n' : '\n';
  const lines = raw.split(/\r?\n/);
  const at = locate(lines, path);
  if (!at) throw new Error(`${path} is not in the file, so it cannot be set here`);
  const original = lines[at.start];
  const prefix = `${' '.repeat(at.indent)}${at.key}:${at.lead || ' '}`;
  const comment = at.tail.trim();
  const withComment = text => {
    if (!comment) return text;
    const column = original.length - at.tail.trimStart().length;
    return text.length < column ? text.padEnd(column) + comment : `${text}  ${comment}`;
  };
  let replacement;
  if (Array.isArray(value)) {
    replacement = [withComment(prefix + flowList(value))];
  } else if (typeof value === 'string' && (at.isBlock || value.includes('\n'))) {
    const block = blockLines(value, at.indent);
    replacement = block ? [withComment(prefix + block[0]), ...block.slice(1)] : [withComment(prefix + scalar(value))];
  } else {
    replacement = [withComment(prefix + scalar(value))];
  }
  lines.splice(at.start, at.end - at.start + 1, ...replacement);
  return lines.join(eol);
}

// Apply {path: value} in order.
export function setValues(raw, changes) {
  return Object.entries(changes).reduce((text, [path, value]) => setValue(text, path, value), raw);
}

// "AOS-2026.11" -> "AOS-2026.12"; anything without a trailing number gets "-r1".
export function nextVersion(version) {
  const m = /^(.*?)(\d+)$/.exec(version || '');
  if (!m) return version ? `${version}-r1` : '';
  return m[1] + String(Number(m[2]) + 1).padStart(m[2].length, '0');
}
