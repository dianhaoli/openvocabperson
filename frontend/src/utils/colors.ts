const PALETTE = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#a855f7', '#84cc16',
  '#f97316', '#14b8a6', '#eab308', '#3b82f6', '#e879f9', '#10b981', '#fb7185', '#8b5cf6',
];

/** Stable color for a label (FNV-1a hash into a fixed palette). */
export function labelColor(label: string): string {
  let h = 0x811c9dc5;
  const s = label.toLowerCase();
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return PALETTE[(h >>> 0) % PALETTE.length];
}

export function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
