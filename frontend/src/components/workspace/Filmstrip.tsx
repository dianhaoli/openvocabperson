import { useActions, useAppState } from '../../context/hooks';
import { cn } from '../../utils/cn';
import { Img } from '../common/Img';

export function Filmstrip() {
  const { collection, currentMediaId } = useAppState();
  const { openMedia, setCollection } = useActions();
  if (!collection || collection.media.length < 2) return null;
  const idx = collection.media.findIndex((m) => m.id === currentMediaId);
  const go = (d: number) => {
    const n = collection.media.length;
    const next = collection.media[(((idx === -1 ? 0 : idx + d) % n) + n) % n];
    if (next) openMedia(next.id);
  };
  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-bg-secondary/60">
      <div className="flex flex-col shrink-0 w-[92px]">
        <span className="text-[10px] uppercase tracking-wider text-text-muted truncate" title={collection.title}>
          {collection.title}
        </span>
        <span className="text-[11px] text-text-secondary">
          {idx === -1 ? '–' : idx + 1} / {collection.media.length}
        </span>
      </div>
      <button type="button" className="text-text-muted hover:text-text-primary px-1" onClick={() => go(-1)} aria-label="Previous">
        ‹
      </button>
      <div className="flex gap-1.5 overflow-x-auto flex-1 py-0.5">
        {collection.media.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => openMedia(m.id)}
            title={m.name}
            className={cn(
              'relative shrink-0 w-12 h-12 rounded-lg overflow-hidden border-2 transition-colors',
              m.id === currentMediaId ? 'border-accent' : 'border-transparent opacity-70 hover:opacity-100',
            )}
          >
            <Img src={m.thumb_url} alt={m.name} className="w-full h-full object-cover" />
            {m.kind === 'video' && <span className="absolute top-0.5 left-0.5 text-[8px] bg-black/70 rounded px-0.5">▶</span>}
          </button>
        ))}
      </div>
      <button type="button" className="text-text-muted hover:text-text-primary px-1" onClick={() => go(1)} aria-label="Next">
        ›
      </button>
      <button type="button" className="text-text-muted hover:text-text-primary text-xs px-1" onClick={() => setCollection(null)} title="Close filmstrip">
        ✕
      </button>
    </div>
  );
}
