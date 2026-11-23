import React from 'react';

// A small markdown renderer for the release notes. It turns the subset those notes use
// into React elements and nothing else: there is no HTML in, and no dangerouslySetInnerHTML
// out, so whatever a note contains can only ever be text, a link or a table.
//
//   blocks:  # to #### headings, paragraphs, - and 1. lists (nested by indentation),
//            pipe tables with a header row, fenced code
//   inline:  `code`, **bold**, *italic*, [text](url)
//
// Anything else is shown as the plain text it is. A link is kept only when its address is
// http, https or relative; any other scheme (javascript:, data:) is shown as its text.

const FENCE = /^\s*```/;
const HEADING = /^(#{1,4})\s+(.*?)\s*#*\s*$/;
const ITEM = /^(\s*)([-*]|\d+\.)\s+(.*)$/;
const TABLE_SEPARATOR = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

export const safeHref = url => {
  const u = String(url || '').trim();
  if (!u || /[\u0000-\u001f\s]/.test(u)) return null;
  if (/^https?:\/\//i.test(u)) return u;
  if (/^[a-z][a-z0-9+.-]*:/i.test(u) || u.startsWith('//')) return null;
  return u;
};

const INLINE = [
  ['code', /`([^`]+)`/],
  ['link', /\[([^\]]+)\]\(([^)\s]+)\)/],
  ['bold', /\*\*(?!\s)(.+?)(?<!\s)\*\*/],
  ['italic', /\*(?![\s*])([^*]+?)(?<![\s*])\*/],
];

export function renderInline(text, keyPrefix = 'i') {
  const out = [];
  let rest = String(text);
  let n = 0;
  while (rest) {
    let best = null;
    for (const [kind, re] of INLINE) {
      const m = re.exec(rest);
      if (m && (!best || m.index < best.m.index)) best = { kind, m };
    }
    if (!best) { out.push(rest); break; }
    const { kind, m } = best;
    if (m.index) out.push(rest.slice(0, m.index));
    const key = `${keyPrefix}-${n++}`;
    if (kind === 'code') out.push(<code key={key}>{m[1]}</code>);
    else if (kind === 'bold') out.push(<strong key={key}>{renderInline(m[1], key)}</strong>);
    else if (kind === 'italic') out.push(<em key={key}>{renderInline(m[1], key)}</em>);
    else {
      const href = safeHref(m[2]);
      if (!href) out.push(m[1]);
      else if (/^https?:/i.test(href)) out.push(<a key={key} href={href} target="_blank" rel="noopener noreferrer">{renderInline(m[1], key)}</a>);
      else out.push(<a key={key} href={href}>{renderInline(m[1], key)}</a>);
    }
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

const indentOf = s => s.replace(/\t/g, '    ').match(/^ */)[0].length;

// One row of a pipe table; `\|` is a literal bar.
function cells(line) {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map(c => c.replace(/\\\|/g, '|').trim());
}

const isTableStart = (line, next) =>
  line.trim().startsWith('|') && next !== undefined && TABLE_SEPARATOR.test(next) && next.includes('-');

const startsBlock = (line, next) =>
  FENCE.test(line) || HEADING.test(line) || ITEM.test(line) || isTableStart(line, next);

// Items [{indent, ordered, text}] to nested lists, by indentation.
function nest(items, from, indent) {
  const list = { ordered: items[from].ordered, items: [] };
  let i = from;
  while (i < items.length && items[i].indent >= indent) {
    if (items[i].indent > indent && list.items.length) {
      const [child, next] = nest(items, i, items[i].indent);
      list.items[list.items.length - 1].children.push(child);
      i = next;
      continue;
    }
    if (items[i].ordered !== list.ordered && list.items.length) break;
    list.items.push({ text: items[i].text, children: [] });
    i += 1;
    while (i < items.length && items[i].indent > indent) {
      const [child, next] = nest(items, i, items[i].indent);
      list.items[list.items.length - 1].children.push(child);
      i = next;
    }
  }
  return [list, i];
}

export function parseBlocks(source) {
  const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    if (FENCE.test(line)) {
      const body = [];
      i += 1;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      i += 1;
      blocks.push({ type: 'code', text: body.join('\n') });
      continue;
    }

    const h = HEADING.exec(line);
    if (h) { blocks.push({ type: 'heading', level: h[1].length, text: h[2] }); i += 1; continue; }

    if (isTableStart(line, lines[i + 1])) {
      const head = cells(line);
      const align = cells(lines[i + 1]).map(c => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : undefined));
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(cells(lines[i++]));
      blocks.push({ type: 'table', head, align, rows });
      continue;
    }

    if (ITEM.test(line)) {
      const items = [];
      while (i < lines.length) {
        const l = lines[i];
        const m = ITEM.exec(l);
        if (m) { items.push({ indent: indentOf(m[1]), ordered: /\d/.test(m[2]), text: m[3].trim() }); i += 1; continue; }
        if (!l.trim()) {
          // a blank line ends the list unless the list goes on after it
          let j = i;
          while (j < lines.length && !lines[j].trim()) j += 1;
          if (j < lines.length && (ITEM.test(lines[j]) || (indentOf(lines[j]) > 0 && !startsBlock(lines[j], lines[j + 1])))) { i = j; continue; }
          break;
        }
        // an indented line carries on the item above it
        if (indentOf(l) > 0 && !startsBlock(l, lines[i + 1])) { items[items.length - 1].text += ' ' + l.trim(); i += 1; continue; }
        break;
      }
      const base = Math.min(...items.map(it => it.indent));
      let at = 0;
      while (at < items.length) {
        const [list, next] = nest(items, at, base);
        blocks.push({ type: 'list', list });
        at = Math.max(next, at + 1);
      }
      continue;
    }

    const para = [line.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) para.push(lines[i++].trim());
    blocks.push({ type: 'paragraph', text: para.join(' ') });
  }
  return blocks;
}

function List({ list, id }) {
  const Tag = list.ordered ? 'ol' : 'ul';
  return <Tag>{list.items.map((it, k) => (
    <li key={k}>{renderInline(it.text, `${id}-${k}`)}{it.children.map((c, j) => <List key={j} list={c} id={`${id}-${k}-${j}`}/>)}</li>
  ))}</Tag>;
}

export function Markdown({ source, className = '' }) {
  const blocks = parseBlocks(source);
  return <div className={`md ${className}`.trim()}>{blocks.map((b, n) => {
    const id = `b${n}`;
    if (b.type === 'heading') {
      const Tag = `h${Math.min(b.level + 1, 6)}`;
      return <Tag key={n} className={`md-h${b.level}`}>{renderInline(b.text, id)}</Tag>;
    }
    if (b.type === 'code') return <pre key={n}><code>{b.text}</code></pre>;
    if (b.type === 'list') return <List key={n} list={b.list} id={id}/>;
    if (b.type === 'table') return (
      <div key={n} className="md-table-scroll"><table>
        <thead><tr>{b.head.map((c, k) => <th key={k} style={{ textAlign: b.align[k] }}>{renderInline(c, `${id}h${k}`)}</th>)}</tr></thead>
        <tbody>{b.rows.map((r, k) => <tr key={k}>{b.head.map((_, j) => <td key={j} style={{ textAlign: b.align[j] }}>{renderInline(r[j] ?? '', `${id}r${k}c${j}`)}</td>)}</tr>)}</tbody>
      </table></div>
    );
    return <p key={n}>{renderInline(b.text, id)}</p>;
  })}</div>;
}
