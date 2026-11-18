import type { PersonRef, Person } from '../types';

export function pct(score: number | null | undefined, digits = 0): string {
  if (score == null || Number.isNaN(score)) return '–';
  return `${(score * 100).toFixed(digits)}%`;
}

export function shortId(id: string): string {
  const raw = id.includes('_') ? id.slice(id.indexOf('_') + 1) : id;
  return raw.slice(0, 6);
}

export function personName(p: Pick<Person, 'id' | 'label'> | PersonRef | null | undefined): string {
  if (!p) return 'Unknown';
  return p.label?.trim() ? p.label : `Unnamed #${shortId(p.id)}`;
}

export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '–';
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${m}:${rest.toFixed(1).padStart(4, '0')}`;
}

export function formatDate(epoch: number | null | undefined): string {
  if (!epoch) return '–';
  return new Date(epoch * 1000).toLocaleString();
}

export function relativeTime(epoch: number | null | undefined): string {
  if (!epoch) return '–';
  const diff = Date.now() / 1000 - epoch;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}

export function formatJson(value: unknown): string {
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
