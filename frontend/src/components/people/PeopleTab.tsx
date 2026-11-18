import { useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import { useObjectUrl } from '../../hooks/useObjectUrl';
import type { Person } from '../../types';
import { personName, pct } from '../../utils/format';
import { Img } from '../common/Img';
import { Button } from '../ui/Button';
import { Dropzone } from '../ui/Dropzone';
import { EmptyState, ErrorState, Loading, Segmented, inputClass } from '../ui/Misc';
import { ReviewQueue } from './ReviewQueue';

type View = 'all' | 'review' | 'photo';

function PersonCard({ p, extra, onClick }: { p: Person; extra?: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-col rounded-xl overflow-hidden border border-border bg-bg-card hover:border-accent transition-colors text-left min-w-0"
    >
      <div className="relative aspect-square">
        <Img src={p.thumb_url ?? undefined} alt="" className="absolute inset-0 w-full h-full object-cover" />
        {p.is_watchlist && <span className="absolute top-1 right-1 text-[9.5px] bg-error text-white rounded px-1">★ watch</span>}
        {extra && <span className="absolute bottom-1 right-1 text-[10px] font-mono bg-black/75 rounded px-1">{extra}</span>}
      </div>
      <div className="px-1.5 py-1 min-w-0">
        <div className={`text-[11px] truncate ${p.label ? '' : 'text-text-muted italic'}`}>{personName(p)}</div>
        <div className="text-[9.5px] text-text-muted">
          {p.sighting_count} sighting{p.sighting_count === 1 ? '' : 's'} · {p.media_count} media
        </div>
      </div>
    </button>
  );
}

function PhotoSearch() {
  const { openPerson, toastError } = useActions();
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<{ person: Person; similarity: number }[] | null>(null);
  const [loading, setLoading] = useState(false);
  const url = useObjectUrl(file);

  const run = async (f: File) => {
    setFile(f);
    setLoading(true);
    setResults(null);
    try {
      const res = await api.searchPersons(f);
      setResults(res.results);
    } catch (err) {
      toastError(err, 'Person search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <Dropzone accept="image/*" onFiles={(f) => f[0] && void run(f[0])} compact>
        <div className="flex items-center gap-3 justify-center">
          {url && <img src={url} alt="query" className="w-12 h-12 rounded-lg object-cover" />}
          <span className="text-xs text-text-muted">{file ? 'Drop another photo to search again' : 'Drop / pick a photo of a person'}</span>
        </div>
      </Dropzone>
      {loading && <Loading label="Searching…" />}
      {results && results.length === 0 && <EmptyState icon="∅" title="No similar persons" />}
      {results && results.length > 0 && (
        <div className="grid grid-cols-3 gap-2">
          {results.map((r) => (
            <PersonCard key={r.person.id} p={r.person} extra={pct(r.similarity)} onClick={() => openPerson(r.person.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

export function PeopleTab() {
  const { personsVersion } = useAppState();
  const { openPerson, toast, toastError, bumpPersons } = useActions();
  const [view, setView] = useState<View>('all');
  const [q, setQ] = useState('');
  const [watchlist, setWatchlist] = useState(false);
  const [reclustering, setReclustering] = useState(false);

  const list = useApi(view === 'all' ? `persons:${q}:${watchlist}:${personsVersion}` : null, (s) =>
    api.persons({ q: q.trim() || undefined, watchlist, limit: 200 }, s),
  );
  const data = list.data ?? list.stale;

  const recluster = async () => {
    setReclustering(true);
    try {
      const res = await api.recluster();
      toast(`Reclustered: ${res.persons} persons, ${res.changed} changed`, 'success');
      bumpPersons();
    } catch (err) {
      toastError(err, 'Recluster failed');
    } finally {
      setReclustering(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <Segmented
          value={view}
          options={[
            { value: 'all', label: 'People' },
            { value: 'review', label: 'Review queue' },
            { value: 'photo', label: 'By photo', title: 'Find by photo' },
          ]}
          onChange={setView}
        />
        <Button size="xs" variant="secondary" className="ml-auto" onClick={() => void recluster()} loading={reclustering} title="Re-cluster all person sightings">
          ⟲ Recluster
        </Button>
      </div>

      {view === 'all' && (
        <>
          <div className="flex gap-2 items-center">
            <input className={inputClass} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by name / notes…" />
            <label className="flex items-center gap-1.5 text-xs text-text-secondary shrink-0 cursor-pointer">
              <input type="checkbox" checked={watchlist} onChange={(e) => setWatchlist(e.target.checked)} className="accent-[#ef4444]" />
              Watchlist
            </label>
          </div>
          {list.loading && !data && <Loading />}
          {!!list.error && <ErrorState error={list.error} onRetry={list.reload} />}
          {data && (
            <div className="text-[11px] text-text-muted">
              {data.total} person{data.total === 1 ? '' : 's'}
            </div>
          )}
          {data && data.persons.length === 0 && (
            <EmptyState icon="👤" title={q || watchlist ? 'No matching persons' : 'No persons yet'} hint="Persons are created automatically when people are detected and Re-ID is enabled." />
          )}
          {data && data.persons.length > 0 && (
            <div className="grid grid-cols-3 gap-2">
              {data.persons.map((p) => (
                <PersonCard key={p.id} p={p} onClick={() => openPerson(p.id)} />
              ))}
            </div>
          )}
        </>
      )}
      {view === 'review' && <ReviewQueue />}
      {view === 'photo' && <PhotoSearch />}
    </div>
  );
}
