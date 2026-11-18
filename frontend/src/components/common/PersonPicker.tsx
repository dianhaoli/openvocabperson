import { useState } from 'react';
import { api } from '../../api/client';
import { useApi } from '../../hooks/useApi';
import { personName } from '../../utils/format';
import { Button } from '../ui/Button';
import { EmptyState, ErrorState, Loading, inputClass } from '../ui/Misc';
import { Modal } from '../ui/Modal';
import { Img } from './Img';

interface Props {
  open: boolean;
  title: string;
  excludeId?: string | null;
  allowNew?: boolean;
  busy?: boolean;
  onPick: (personId: string | null) => void;
  onClose: () => void;
}

export function PersonPicker({ open, title, excludeId, allowNew, busy, onPick, onClose }: Props) {
  const [q, setQ] = useState('');
  const { data, error, loading, reload } = useApi(open ? `picker:${q}` : null, (signal) =>
    api.persons({ q: q || undefined, limit: 200 }, signal),
  );
  const persons = (data?.persons ?? []).filter((p) => p.id !== excludeId);

  return (
    <Modal open={open} onClose={onClose} title={title} className="max-w-[560px]">
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          <input className={inputClass} placeholder="Filter by name…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
          {allowNew && (
            <Button variant="secondary" size="sm" onClick={() => onPick(null)} disabled={busy}>
              + New person
            </Button>
          )}
        </div>
        {loading && <Loading />}
        {!!error && <ErrorState error={error} onRetry={reload} />}
        {data && persons.length === 0 && <EmptyState icon="👤" title="No persons found" />}
        <div className="grid grid-cols-4 gap-2 max-h-[50vh] overflow-y-auto pr-1">
          {persons.map((p) => (
            <button
              key={p.id}
              type="button"
              disabled={busy}
              onClick={() => onPick(p.id)}
              className="flex flex-col items-center gap-1 p-2 rounded-xl border border-border bg-bg-card hover:border-accent transition-colors disabled:opacity-50"
            >
              <Img src={p.thumb_url ?? undefined} alt="" className="w-16 h-16 rounded-lg object-cover" />
              <span className="text-[11px] truncate max-w-full">{personName(p)}</span>
              <span className="text-[10px] text-text-muted">
                {p.sighting_count} sightings{p.is_watchlist ? ' · ★' : ''}
              </span>
            </button>
          ))}
        </div>
      </div>
    </Modal>
  );
}
