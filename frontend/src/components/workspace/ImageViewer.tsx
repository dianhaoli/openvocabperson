import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { useElementSize } from '../../hooks/useElementSize';
import type { Entity, MediaDetail } from '../../types';
import { labelColor, withAlpha } from '../../utils/colors';
import { containFit, pickSmallest } from '../../utils/geometry';
import { drawTag, entityCaption } from '../../utils/draw';
import { Spinner } from '../ui/Spinner';

interface Props {
  media: MediaDetail;
  entities: Entity[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export function ImageViewer({ media, entities, selectedId, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useElementSize(containerRef);
  const [loaded, setLoaded] = useState<{ src: string; img: HTMLImageElement } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const img = new Image();
    let alive = true;
    img.onload = () => {
      if (alive) setLoaded({ src: media.url, img });
    };
    img.onerror = () => {
      if (alive) setFailed(media.url);
    };
    img.src = media.url;
    return () => {
      alive = false;
      img.onload = null;
      img.onerror = null;
    };
  }, [media.url]);

  const img = loaded?.src === media.url ? loaded.img : null;
  const srcW = media.width || img?.naturalWidth || 1;
  const srcH = media.height || img?.naturalHeight || 1;
  const maxH = Math.max(240, Math.round(window.innerHeight * 0.62));
  const fit = containFit(srcW, srcH, size.width, maxH);
  const cssW = Math.floor(Math.min(size.width, fit.width));
  const cssH = Math.floor(fit.height);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img || cssW <= 0 || cssH <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const sx = canvas.width / srcW;
    const sy = canvas.height / srcH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const ordered = [...entities].sort((a, b) => (a.id === selectedId ? 1 : 0) - (b.id === selectedId ? 1 : 0));
    const hasSel = selectedId !== null && entities.some((e) => e.id === selectedId);

    for (const e of ordered) {
      const color = labelColor(e.label);
      const isSel = e.id === selectedId;
      const dim = hasSel && !isSel;
      ctx.globalAlpha = dim ? 0.45 : 1;

      if (e.polygon && e.polygon.length) {
        ctx.beginPath();
        for (const poly of e.polygon) {
          if (!poly.length) continue;
          ctx.moveTo(poly[0][0] * sx, poly[0][1] * sy);
          for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0] * sx, poly[i][1] * sy);
          ctx.closePath();
        }
        ctx.fillStyle = withAlpha(color, isSel ? 0.4 : 0.25);
        ctx.fill('evenodd');
        ctx.strokeStyle = withAlpha(color, 0.9);
        ctx.lineWidth = 1.25 * dpr;
        ctx.stroke();
      }

      const [x1, y1, x2, y2] = e.box;
      const lw = (isSel ? 3 : 2) * dpr;
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      if (e.stage === 'rejected') ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.setLineDash([]);
      if (isSel) {
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth = 1 * dpr;
        ctx.strokeRect(x1 * sx - lw, y1 * sy - lw, (x2 - x1) * sx + 2 * lw, (y2 - y1) * sy + 2 * lw);
      }
      drawTag(ctx, entityCaption(e), x1 * sx - lw / 2, y1 * sy - lw / 2, color, dpr, canvas.width);
    }
    ctx.globalAlpha = 1;
  }, [img, entities, selectedId, cssW, cssH, srcW, srcH]);

  const onClick = (ev: MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * srcW;
    const y = ((ev.clientY - rect.top) / rect.height) * srcH;
    const hit = pickSmallest(entities, (e) => e.box, x, y);
    onSelect(hit ? hit.id : null);
  };

  return (
    <div ref={containerRef} className="w-full flex justify-center">
      {failed === media.url ? (
        <div className="h-60 w-full flex items-center justify-center text-sm text-error border border-border rounded-xl">Could not load image</div>
      ) : !img ? (
        <div className="h-60 w-full flex items-center justify-center text-text-muted gap-2">
          <Spinner /> Loading image…
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          onClick={onClick}
          style={{ width: cssW, height: cssH }}
          className="rounded-xl cursor-crosshair border border-border bg-black"
        />
      )}
    </div>
  );
}
