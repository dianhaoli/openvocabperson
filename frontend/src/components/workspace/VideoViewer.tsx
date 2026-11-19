import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import type { Entity, MediaDetail } from '../../types';
import { labelColor } from '../../utils/colors';
import { cn } from '../../utils/cn';
import { drawTag, entityCaption } from '../../utils/draw';
import { formatTime, personName } from '../../utils/format';
import { boxAtTime, containFit, pickSmallest } from '../../utils/geometry';

/** Height of the native video controls bar, left free so the controls stay usable. */
const CONTROLS_HEIGHT = 56;

interface Props {
  media: MediaDetail;
  entities: Entity[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export function VideoViewer({ media, entities, selectedId, onSelect }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState<number>(media.duration ?? 0);
  const [error, setError] = useState(false);

  // Latest drawing inputs, read by the rAF loop / media event handlers.
  const drawInputs = useRef({ entities, selectedId, media });
  useEffect(() => {
    drawInputs.current = { entities, selectedId, media };
  });

  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const { entities: ents, selectedId: sel, media: m } = drawInputs.current;
    const dpr = window.devicePixelRatio || 1;
    const cw = video.clientWidth;
    const ch = video.clientHeight;
    const pw = Math.round(cw * dpr);
    const ph = Math.round(ch * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, pw, ph);
    const vw = video.videoWidth || m.width;
    const vh = video.videoHeight || m.height;
    if (!vw || !vh) return;
    const fit = containFit(vw, vh, cw, ch);
    // Boxes are in the media's frame coordinates (fall back to the decoded size).
    const frameW = m.width || vw;
    const frameH = m.height || vh;
    const sx = (fit.width / frameW) * dpr;
    const sy = (fit.height / frameH) * dpr;
    const ox = fit.offsetX * dpr;
    const oy = fit.offsetY * dpr;
    const t = video.currentTime;

    const ordered = [...ents].sort((a, b) => (a.id === sel ? 1 : 0) - (b.id === sel ? 1 : 0));
    for (const e of ordered) {
      const box = boxAtTime(e, t);
      if (!box) continue;
      const color = labelColor(e.label);
      const isSel = e.id === sel;
      const lw = (isSel ? 3 : 2) * dpr;
      ctx.globalAlpha = sel && !isSel ? 0.55 : 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      const x = ox + box[0] * sx;
      const y = oy + box[1] * sy;
      ctx.strokeRect(x, y, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
      const caption = e.track_id != null ? `#${e.track_id} ${entityCaption(e)}` : entityCaption(e);
      drawTag(ctx, caption, x - lw / 2, y - lw / 2, color, dpr, pw);
    }
    ctx.globalAlpha = 1;
  }, []);

  // Redraw on input changes.
  useEffect(() => {
    draw();
  }, [draw, entities, selectedId]);

  // Redraw on element resize.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(video);
    return () => ro.disconnect();
  }, [draw]);

  // Animation loop while playing.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let raf = 0;
    const loop = () => {
      draw();
      if (!video.paused && !video.ended) raf = requestAnimationFrame(loop);
    };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };
    video.addEventListener('play', onPlay);
    return () => {
      video.removeEventListener('play', onPlay);
      cancelAnimationFrame(raf);
    };
  }, [draw]);

  // Seek to the selected track when it is not on screen.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !selectedId) return;
    const e = entities.find((x) => x.id === selectedId);
    if (!e || e.t_start == null) return;
    const t = video.currentTime;
    if (t < e.t_start || (e.t_end != null && t > e.t_end)) {
      video.currentTime = e.t_start;
    }
    // Only react to selection changes, not entity list updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  const dur = duration || media.duration || Math.max(1, ...entities.map((e) => e.t_end ?? 0));

  const seek = (t: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = Math.max(0, t);
  };

  /** Click on the frame: select the smallest track box under the cursor, otherwise toggle playback. */
  const onFrameClick = (ev: MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    if (!video) return;
    const rect = video.getBoundingClientRect();
    const fit = containFit(video.videoWidth || media.width, video.videoHeight || media.height, rect.width, rect.height);
    const x = ((ev.clientX - rect.left - fit.offsetX) / fit.width) * (media.width || video.videoWidth);
    const y = ((ev.clientY - rect.top - fit.offsetY) / fit.height) * (media.height || video.videoHeight);
    const hit = pickSmallest(entities, (e) => boxAtTime(e, video.currentTime), x, y);
    if (hit) onSelect(hit.id);
    else if (video.paused) void video.play();
    else video.pause();
  };

  const seekFromLane = (ev: MouseEvent<HTMLDivElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    seek(((ev.clientX - rect.left) / rect.width) * dur);
  };

  const tracks = [...entities].sort((a, b) => (a.t_start ?? 0) - (b.t_start ?? 0));

  return (
    <div className="flex flex-col gap-3">
      <div className="relative w-full bg-black rounded-xl overflow-hidden border border-border">
        {error ? (
          <div className="h-60 flex items-center justify-center text-sm text-error">Could not load video</div>
        ) : (
          <>
            <video
              ref={videoRef}
              src={media.url}
              controls
              preload="metadata"
              playsInline
              className="block w-full max-h-[62vh] object-contain bg-black"
              onLoadedMetadata={(e) => {
                setDuration(e.currentTarget.duration || media.duration || 0);
                draw();
              }}
              onLoadedData={draw}
              onSeeked={(e) => {
                setCurrentTime(e.currentTarget.currentTime);
                draw();
              }}
              onTimeUpdate={(e) => {
                setCurrentTime(e.currentTarget.currentTime);
                if (e.currentTarget.paused) draw();
              }}
              onError={() => setError(true)}
            />
            <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
            <div
              className="absolute inset-x-0 top-0 cursor-crosshair"
              style={{ bottom: CONTROLS_HEIGHT }}
              onClick={onFrameClick}
              title="Click a box to select its track · click elsewhere to play/pause"
            />
          </>
        )}
      </div>

      <div className="rounded-xl border border-border bg-bg-card p-2">
        <div className="flex items-center justify-between text-[11px] text-text-muted mb-1 px-1">
          <span>Tracks ({tracks.length})</span>
          <span className="font-mono">
            {formatTime(currentTime)} / {formatTime(dur)}
          </span>
        </div>
        {tracks.length === 0 ? (
          <div className="text-xs text-text-muted px-1 py-2">No tracks.</div>
        ) : (
          <div className="relative max-h-[200px] overflow-y-auto pr-1">
            {tracks.map((e) => {
              const start = Math.max(0, e.t_start ?? 0);
              const end = Math.max(start, e.t_end ?? start);
              const left = (start / dur) * 100;
              const width = Math.max(0.6, ((end - start) / dur) * 100);
              const color = labelColor(e.label);
              const isSel = e.id === selectedId;
              return (
                <div key={e.id} className="flex items-center gap-2 h-[22px]">
                  <button
                    type="button"
                    onClick={() => onSelect(e.id)}
                    className={cn('w-[110px] shrink-0 text-left text-[10.5px] truncate', isSel ? 'text-text-primary font-medium' : 'text-text-secondary')}
                    title={entityCaption(e)}
                  >
                    {e.track_id != null && <span className="text-text-muted font-mono">#{e.track_id} </span>}
                    {e.person ? personName(e.person) : e.label}
                  </button>
                  <div className="relative flex-1 h-3 rounded bg-bg-primary cursor-pointer" onClick={seekFromLane} title="Click to seek · click a bar to select its track">
                    <button
                      type="button"
                      title={`${e.label}: ${formatTime(start)} – ${formatTime(end)}`}
                      onClick={() => onSelect(e.id)}
                      className={cn('absolute top-0 h-3 rounded transition-opacity', isSel ? 'ring-2 ring-white/80' : 'opacity-75 hover:opacity-100')}
                      style={{ left: `${left}%`, width: `${width}%`, background: color }}
                    />
                    <div className="absolute top-[-3px] bottom-[-3px] w-px bg-white/70 pointer-events-none" style={{ left: `${Math.min(100, (currentTime / dur) * 100)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
