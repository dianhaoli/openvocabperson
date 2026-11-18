import { useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import type { MediaDetail } from '../../types';
import { labelColor } from '../../utils/colors';
import { formatDate, relativeTime } from '../../utils/format';
import { Img } from '../common/Img';
import { Button } from '../ui/Button';
import { Badge, Chip, EmptyState, ErrorState, Loading } from '../ui/Misc';
import { EntityCard } from './EntityCard';
import { Filmstrip } from './Filmstrip';
import { ImageViewer } from './ImageViewer';
import { Legend } from './Legend';
import { SceneQA } from './SceneQA';
import { VideoViewer } from './VideoViewer';

function MediaView({ media, onReload, reloading }: { media: MediaDetail; onReload: () => void; reloading: boolean }) {
  const { selectedEntityId } = useAppState();
  const { selectEntity, closeMedia } = useActions();
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [showRejected, setShowRejected] = useState(false);

  const all = media.entities ?? [];
  const rejectedCount = all.filter((e) => e.stage === 'rejected').length;
  const base = showRejected ? all : all.filter((e) => e.stage !== 'rejected' || e.id === selectedEntityId);
  const visible = base.filter((e) => !hidden.has(e.label) || e.id === selectedEntityId);

  const countMap = new Map<string, number>();
  for (const e of base) countMap.set(e.label, (countMap.get(e.label) ?? 0) + 1);
  const counts = [...countMap.entries()].sort((a, b) => b[1] - a[1]);

  const toggle = (label: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  const showOnly = (label: string | null) => setHidden(label ? new Set(counts.map(([l]) => l).filter((l) => l !== label)) : new Set());

  const cards = [...visible].sort((a, b) => {
    if (a.stage === 'rejected' && b.stage !== 'rejected') return 1;
    if (b.stage === 'rejected' && a.stage !== 'rejected') return -1;
    return (a.t_start ?? 0) - (b.t_start ?? 0) || b.score - a.score;
  });

  return (
    <div className="flex flex-col gap-4 p-4 max-w-[1400px] mx-auto w-full">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <h2 className="text-base font-semibold truncate" title={media.name}>
              {media.name}
            </h2>
            <Badge tone={media.kind === 'video' ? 'accent' : 'neutral'}>{media.kind}</Badge>
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-text-muted mt-0.5">
            <span>
              {media.width}×{media.height}
            </span>
            {media.kind === 'video' && media.duration != null && <span>{media.duration.toFixed(1)}s{media.fps ? ` @ ${media.fps.toFixed(0)} fps` : ''}</span>}
            <span>{all.length} entities</span>
            {media.person_count > 0 && <span>{media.person_count} persons</span>}
            <span>query: {media.query ? `“${media.query}”` : 'default'}</span>
            <span title={formatDate(media.created_at)}>{relativeTime(media.created_at)}</span>
          </div>
          {media.plan && media.plan.prompts.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {media.plan.prompts.slice(0, 20).map((p) => (
                <Chip key={p} color={labelColor(p)} className="!py-0 !text-[10.5px]">
                  {p}
                </Chip>
              ))}
            </div>
          )}
        </div>
        <div className="flex gap-1 shrink-0">
          <Button size="xs" variant="ghost" onClick={onReload} loading={reloading} title="Reload">
            ↻
          </Button>
          <a href={media.url} target="_blank" rel="noreferrer" className="text-[11px] px-2 py-1 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary">
            Original ↗
          </a>
          <Button size="xs" variant="ghost" onClick={closeMedia} title="Close">
            ✕
          </Button>
        </div>
      </div>

      {media.kind === 'video' ? (
        <VideoViewer media={media} entities={visible} selectedId={selectedEntityId} onSelect={selectEntity} />
      ) : (
        <ImageViewer media={media} entities={visible} selectedId={selectedEntityId} onSelect={selectEntity} />
      )}

      <Legend
        counts={counts}
        hidden={hidden}
        onToggle={toggle}
        onShowOnly={showOnly}
        rejectedCount={rejectedCount}
        showRejected={showRejected}
        onToggleRejected={() => setShowRejected((v) => !v)}
      />

      {cards.length === 0 ? (
        <EmptyState icon="◻" title={all.length ? 'All entities are hidden' : 'No entities detected'} hint={all.length ? 'Toggle labels in the legend above.' : 'Try another query, or lower the confidence threshold.'} />
      ) : (
        <div className="grid gap-2.5 grid-cols-[repeat(auto-fill,minmax(150px,1fr))]">
          {cards.map((e) => (
            <EntityCard key={e.id} entity={e} selected={e.id === selectedEntityId} onClick={() => selectEntity(e.id === selectedEntityId ? null : e.id)} />
          ))}
        </div>
      )}

      <SceneQA key={media.id} mediaId={media.id} kind={media.kind} />
    </div>
  );
}

function Home() {
  const { libraryVersion } = useAppState();
  const { openMedia, setTab, setCollection } = useActions();
  const stats = useApi(`stats:${libraryVersion}`, (s) => api.stats(s));
  const recent = useApi(`recent:${libraryVersion}`, (s) => api.listMedia({ limit: 12 }, s));
  const st = stats.data ?? stats.stale;
  const rec = recent.data ?? recent.stale;

  return (
    <div className="p-6 max-w-[900px] mx-auto w-full flex flex-col gap-6">
      <div className="text-center flex flex-col gap-2 pt-6">
        <div className="text-2xl font-semibold">
          Open-vocabulary <span className="text-gradient">vision</span>
        </div>
        <div className="text-sm text-text-secondary">
          Find anything in images and videos — “dogs”, “dangerous things”, “people wearing red” — then explore entities, identities and ask questions.
        </div>
        <div className="flex justify-center gap-2 mt-2">
          <Button onClick={() => setTab('analyze')}>Analyze media</Button>
          <Button variant="secondary" onClick={() => setTab('agent')}>
            Ask the agent
          </Button>
        </div>
      </div>
      {!!stats.error && <ErrorState error={stats.error} onRetry={stats.reload} />}
      {st && (
        <div className="grid grid-cols-6 gap-2">
          {(
            [
              ['Media', st.media],
              ['Images', st.images],
              ['Videos', st.videos],
              ['Entities', st.entities],
              ['Persons', st.persons],
              ['Watchlist', st.watchlist],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="rounded-xl border border-border bg-bg-card px-3 py-2">
              <div className="text-lg font-semibold">{v}</div>
              <div className="text-[10.5px] text-text-muted uppercase tracking-wider">{k}</div>
            </div>
          ))}
        </div>
      )}
      {st && st.labels.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {st.labels.slice(0, 30).map((l) => (
            <Chip key={l.label} color={labelColor(l.label)}>
              {l.label} <span className="text-text-muted">{l.count}</span>
            </Chip>
          ))}
        </div>
      )}
      {rec && rec.media.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="text-[11px] uppercase tracking-wider text-text-muted font-semibold">Recent media</div>
          <div className="grid grid-cols-6 gap-2">
            {rec.media.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => {
                  setCollection({ title: 'Recent', media: rec.media });
                  openMedia(m.id);
                }}
                className="relative aspect-square rounded-lg overflow-hidden border border-border hover:border-accent"
                title={m.name}
              >
                <Img src={m.thumb_url} alt={m.name} className="w-full h-full object-cover" />
                {m.kind === 'video' && <span className="absolute top-1 left-1 text-[9px] bg-black/70 rounded px-1">▶</span>}
                <span className="absolute bottom-0 right-0 text-[9.5px] bg-black/75 px-1 rounded-tl">{m.entity_count}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function Workspace() {
  const { currentMediaId, mediaCache } = useAppState();
  const { cacheMedia, closeMedia, toastError } = useActions();
  const q = useApi(currentMediaId ? `media-detail:${currentMediaId}` : null, async (signal) => {
    try {
      const detail = await api.media(currentMediaId ?? '', signal);
      cacheMedia(detail);
      return detail;
    } catch (err) {
      if (!signal.aborted && !mediaCache[currentMediaId ?? '']) {
        toastError(err, 'Could not load media');
        closeMedia();
      }
      throw err;
    }
  });

  const media = currentMediaId ? mediaCache[currentMediaId] : undefined;

  return (
    <div className="flex flex-col min-h-0 h-full">
      <Filmstrip />
      <div className="flex-1 min-h-0 overflow-y-auto">
        {!currentMediaId && <Home />}
        {currentMediaId && !media && q.loading && <Loading label="Loading media…" />}
        {currentMediaId && !media && !!q.error && <ErrorState error={q.error} onRetry={q.reload} />}
        {media && <MediaView key={media.id} media={media} onReload={q.reload} reloading={q.loading} />}
      </div>
    </div>
  );
}
