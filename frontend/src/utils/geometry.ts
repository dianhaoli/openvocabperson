import type { Box, Entity } from '../types';

export interface Fit {
  scale: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

/** object-fit: contain placement of a (w,h) source inside a (cw,ch) container. */
export function containFit(srcW: number, srcH: number, cw: number, ch: number): Fit {
  if (!srcW || !srcH || !cw || !ch) return { scale: 1, offsetX: 0, offsetY: 0, width: cw, height: ch };
  const scale = Math.min(cw / srcW, ch / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return { scale, offsetX: (cw - width) / 2, offsetY: (ch - height) / 2, width, height };
}

export function boxArea(b: Box): number {
  return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}

export function boxContains(b: Box, x: number, y: number): boolean {
  return x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
}

/** Smallest box among `items` containing the point. */
export function pickSmallest<T>(items: T[], getBox: (t: T) => Box | null, x: number, y: number): T | null {
  let best: T | null = null;
  let bestArea = Infinity;
  for (const it of items) {
    const b = getBox(it);
    if (!b || !boxContains(b, x, y)) continue;
    const a = boxArea(b);
    if (a < bestArea) {
      bestArea = a;
      best = it;
    }
  }
  return best;
}

const TIME_TOLERANCE = 0.3;

/** Box of a video track at time t, interpolated along its trajectory; null if not visible. */
export function boxAtTime(entity: Entity, t: number): Box | null {
  const start = entity.t_start ?? null;
  const end = entity.t_end ?? null;
  if (start != null && t < start - TIME_TOLERANCE) return null;
  if (end != null && t > end + TIME_TOLERANCE) return null;
  const traj = entity.trajectory;
  if (!traj || traj.length === 0) {
    return start != null || end != null ? entity.box : null;
  }
  if (t <= traj[0][0]) {
    return t >= traj[0][0] - TIME_TOLERANCE ? [traj[0][1], traj[0][2], traj[0][3], traj[0][4]] : null;
  }
  const last = traj[traj.length - 1];
  if (t >= last[0]) {
    return t <= last[0] + TIME_TOLERANCE ? [last[1], last[2], last[3], last[4]] : null;
  }
  for (let i = 0; i < traj.length - 1; i++) {
    const a = traj[i];
    const b = traj[i + 1];
    if (t >= a[0] && t <= b[0]) {
      const span = b[0] - a[0];
      const k = span > 0 ? (t - a[0]) / span : 0;
      return [
        a[1] + (b[1] - a[1]) * k,
        a[2] + (b[2] - a[2]) * k,
        a[3] + (b[3] - a[3]) * k,
        a[4] + (b[4] - a[4]) * k,
      ];
    }
  }
  return null;
}

export function isVideoEntity(e: Entity): boolean {
  return e.t_start != null || (e.trajectory != null && e.trajectory.length > 0);
}
