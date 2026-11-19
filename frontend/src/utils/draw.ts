import type { Entity } from '../types';
import { personName } from './format';
/** Canvas tag text: label and score, plus the person when it is named or on the watchlist. */
export function entityCaption(e: Entity): string {
  const base = `${e.label} ${Math.round(e.score * 100)}%`;
  const person = e.person;
  if (!person || (!person.label?.trim() && !person.is_watchlist)) return base;
  return `${base} · ${person.is_watchlist ? '★ ' : ''}${personName(person)}`;
}

/** Draws a label tag at (x, y) top-left of a box, flipping inside when near the top edge. */
export function drawTag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, dpr: number, maxW: number) {
  const fontSize = 11 * dpr;
  ctx.font = `600 ${fontSize}px Outfit, sans-serif`;
  const padX = 4 * dpr;
  const h = fontSize + 5 * dpr;
  const w = Math.min(ctx.measureText(text).width + padX * 2, maxW);
  let ty = y - h;
  if (ty < 0) ty = y;
  let tx = x;
  if (tx + w > maxW) tx = Math.max(0, maxW - w);
  ctx.fillStyle = color;
  ctx.fillRect(tx, ty, w, h);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.save();
  ctx.beginPath();
  ctx.rect(tx, ty, w, h);
  ctx.clip();
  ctx.fillText(text, tx + padX, ty + h / 2 + 0.5);
  ctx.restore();
}

