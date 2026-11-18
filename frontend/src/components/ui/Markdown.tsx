import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

// Tiny markdown subset renderer: paragraphs, headings, lists, code fences,
// **bold**, *italic*, `code` and [links](https://...). No HTML injection.

const INLINE_RE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\s\n][^*\n]*\*|\[[^\]\n]+\]\([^)\s]+\))/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push(text.slice(last, idx));
    const tok = m[0];
    const key = `${keyPrefix}-${i++}`;
    if (tok.startsWith('`')) {
      out.push(
        <code key={key} className="font-mono text-[0.85em] px-1 py-px rounded bg-bg-primary border border-border">
          {tok.slice(1, -1)}
        </code>,
      );
    } else if (tok.startsWith('**') || tok.startsWith('__')) {
      out.push(
        <strong key={key} className="font-semibold text-text-primary">
          {renderInline(tok.slice(2, -2), key)}
        </strong>,
      );
    } else if (tok.startsWith('[')) {
      const lm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok);
      const href = lm?.[2] ?? '';
      if (lm && /^(https?:\/\/|\/)/.test(href)) {
        out.push(
          <a key={key} href={href} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-2">
            {lm[1]}
          </a>,
        );
      } else {
        out.push(tok);
      }
    } else {
      out.push(<em key={key}>{renderInline(tok.slice(1, -1), key)}</em>);
    }
    last = idx + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

type Block =
  | { kind: 'p'; lines: string[] }
  | { kind: 'h'; level: number; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] };

const LIST_RE = /^\s*([-*+•]|\d+[.)])\s+(.*)$/;

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (line.trim().startsWith('```')) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) buf.push(lines[i++]);
      i++;
      blocks.push({ kind: 'code', text: buf.join('\n') });
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      blocks.push({ kind: 'h', level: h[1].length, text: h[2] });
      i++;
      continue;
    }
    const lm = LIST_RE.exec(line);
    if (lm) {
      const ordered = /\d/.test(lm[1]);
      const items: string[] = [];
      while (i < lines.length) {
        const m = LIST_RE.exec(lines[i]);
        if (m) {
          items.push(m[2]);
          i++;
        } else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1] += ` ${lines[i].trim()}`;
          i++;
        } else break;
      }
      blocks.push({ kind: 'list', ordered, items });
      continue;
    }
    const buf: string[] = [];
    while (i < lines.length && lines[i].trim() && !LIST_RE.test(lines[i]) && !lines[i].trim().startsWith('```') && !/^#{1,6}\s/.test(lines[i])) {
      buf.push(lines[i++]);
    }
    blocks.push({ kind: 'p', lines: buf });
  }
  return blocks;
}

export function Markdown({ text, className }: { text: string; className?: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className={cn('flex flex-col gap-2 break-words', className)}>
      {blocks.map((b, bi) => {
        const key = `b${bi}`;
        switch (b.kind) {
          case 'code':
            return (
              <pre key={key} className="font-mono text-xs bg-bg-primary border border-border rounded-lg p-2 overflow-x-auto whitespace-pre">
                {b.text}
              </pre>
            );
          case 'h':
            return (
              <div key={key} className={cn('font-semibold text-text-primary', b.level <= 2 ? 'text-[15px]' : 'text-sm')}>
                {renderInline(b.text, key)}
              </div>
            );
          case 'list': {
            const Tag = b.ordered ? 'ol' : 'ul';
            return (
              <Tag key={key} className={cn('pl-5 flex flex-col gap-0.5', b.ordered ? 'list-decimal' : 'list-disc')}>
                {b.items.map((it, ii) => (
                  <li key={ii}>{renderInline(it, `${key}-${ii}`)}</li>
                ))}
              </Tag>
            );
          }
          default:
            return (
              <p key={key}>
                {b.lines.map((l, li) => (
                  <span key={li}>
                    {li > 0 && <br />}
                    {renderInline(l, `${key}-${li}`)}
                  </span>
                ))}
              </p>
            );
        }
      })}
    </div>
  );
}
