import { useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import type { Entity, ReviewAction } from '../../types';
import { personName, pct } from '../../utils/format';
import { Img } from '../common/Img';
import { PersonThumb } from '../common/PersonThumb';
import { Button } from '../ui/Button';
import { EmptyState, ErrorState, Loading } from '../ui/Misc';

function ReviewItem({ entity, onResolved }: { entity: Entity; onResolved: (id: string) => void }) {
  const actions = useActions();
  const [busy, setBusy] = useState<ReviewAction | null>(null);
  const suggested = entity.suggested_person ?? entity.person;

  const act = async (action: ReviewAction) => {
    setBusy(action);
    try {
      const updated = await api.reviewEntity(entity.id, action);
      actions.upsertEntity(updated);
      onResolved(entity.id);
    } catch (err) {
      actions.toastError(err, 'Review failed');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="rounded-xl border border-border bg-bg-card p-2.5 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <button type="button" onClick={() => actions.openMedia(entity.media_id, entity.id)} title="Open in workspace" className="shrink-0">
          <Img src={entity.crop_url} alt="sighting" className="w-[84px] h-[84px] rounded-lg object-cover border border-border hover:border-accent" />
        </button>
        <div className="flex flex-col items-center text-text-muted text-[10px] shrink-0 w-12">
          <span className="text-lg">⇄</span>
          {entity.match_score != null && <span className="font-mono">{pct(entity.match_score)}</span>}
        </div>
        {suggested ? (
          <button type="button" onClick={() => actions.openPerson(suggested.id)} className="flex flex-col items-center gap-1 min-w-0" title="Open person">
            <PersonThumb personId={suggested.id} className="w-[84px] h-[84px] rounded-lg object-cover border border-border hover:border-accent" />
            <span className="text-[10.5px] truncate max-w-[90px]">{personName(suggested)}</span>
          </button>
        ) : (
          <div className="text-xs text-text-muted">No suggestion</div>
        )}
      </div>
      {entity.description && <div className="text-[11px] text-text-muted line-clamp-2">{entity.description}</div>}
      <div className="flex gap-1.5">
        <Button size="xs" className="flex-1" onClick={() => void act('accept')} loading={busy === 'accept'} disabled={!!busy || !suggested}>
          ✓ Accept
        </Button>
        <Button size="xs" variant="secondary" className="flex-1" onClick={() => void act('new')} loading={busy === 'new'} disabled={!!busy}>
          + New person
        </Button>
        <Button size="xs" variant="secondary" className="flex-1" onClick={() => void act('reject')} loading={busy === 'reject'} disabled={!!busy}>
          ✗ Reject
        </Button>
      </div>
    </div>
  );
}

export function ReviewQueue() {
  const { personsVersion } = useAppState();
  const { bumpPersons } = useActions();
  const q = useApi(`review:${personsVersion}`, (s) => api.reviewQueue(s));
  const data = q.data ?? q.stale;

  const resolved = (id: string) => {
    q.mutate((prev) => ({ pending: prev.pending.filter((e) => e.id !== id) }));
    bumpPersons();
  };

  if (!data && q.loading) return <Loading />;
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />;
  if (!data || data.pending.length === 0) return <EmptyState icon="✓" title="Review queue is empty" hint="Uncertain identity matches will appear here." />;
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[11px] text-text-muted">{data.pending.length} sighting(s) waiting for review</div>
      {data.pending.map((e) => (
        <ReviewItem key={e.id} entity={e} onResolved={resolved} />
      ))}
    </div>
  );
}
