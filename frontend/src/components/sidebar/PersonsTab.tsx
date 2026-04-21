import { useCallback, useEffect, useState } from 'react';
import { Card, Button, Dropzone } from '../ui';
import { cn } from '../../utils/cn';
import {
  checkHealth,
  listPersons,
  searchPersonsByPhoto,
} from '../../api/client';
import type { Person, PersonSearchResult } from '../../types';
import { PersonDetailModal } from '../entity/PersonDetailModal';
import { formatPersonDisplayLabel } from '../../utils/personDisplay';

export function PersonsTab() {
  const [watchlistOnly, setWatchlistOnly] = useState(false);
  const [persons, setPersons] = useState<Person[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailPersonId, setDetailPersonId] = useState<string | null>(null);

  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [searchPreview, setSearchPreview] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<PersonSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [reidReady, setReidReady] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth()
      .then((d) => setReidReady(d.reid_ready === true))
      .catch(() => setReidReady(false));
  }, []);

  const loadPersons = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listPersons(watchlistOnly, 80, 0);
      setPersons(data.persons);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [watchlistOnly]);

  useEffect(() => {
    loadPersons();
  }, [loadPersons]);

  const handleSearchImage = (file: File) => {
    setSearchFile(file);
    setSearchError(null);
    setSearchResults([]);
    const reader = new FileReader();
    reader.onload = (e) => setSearchPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const runPersonSearch = async () => {
    if (!searchFile || !reidReady) return;
    setSearching(true);
    setSearchError(null);
    try {
      const data = await searchPersonsByPhoto(searchFile, 15, 0.2);
      setSearchResults(data.results);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const watchlistPersons = persons.filter((p) => p.is_watchlist);
  const autoPersons = persons.filter((p) => !p.is_watchlist);

  return (
    <>
      <Card title="Person identities">
        <p className="text-xs text-text-muted mb-3">
          Auto-clusters from Re-ID; promote to watchlist to track suspects across
          uploads.
        </p>

        {reidReady === false && (
          <p className="text-xs text-warning mb-3 p-2 rounded-lg bg-warning-soft border border-warning">
            Re-ID model is not available. Person search and matching require a
            working torchreid setup on the server.
          </p>
        )}

        <div className="flex rounded-lg border border-border overflow-hidden mb-3">
          <button
            type="button"
            onClick={() => setWatchlistOnly(false)}
            className={cn(
              'flex-1 py-2 text-xs font-medium transition-colors',
              !watchlistOnly
                ? 'bg-accent text-white'
                : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            )}
          >
            All
          </button>
          <button
            type="button"
            onClick={() => setWatchlistOnly(true)}
            className={cn(
              'flex-1 py-2 text-xs font-medium transition-colors',
              watchlistOnly
                ? 'bg-accent text-white'
                : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            )}
          >
            Watchlist only
          </button>
        </div>

        <Button
          variant="secondary"
          className="w-full mb-4"
          onClick={loadPersons}
          loading={loading}
        >
          Refresh
        </Button>

        {watchlistOnly ? (
          <>
            <h4 className="text-[10px] uppercase text-text-muted mb-2">
              Watchlist
            </h4>
            <div className="flex flex-col gap-2">
              {persons.length === 0 && !loading ? (
                <p className="text-xs text-text-muted py-2">No persons on watchlist</p>
              ) : (
                persons.map((p) => (
                  <PersonListCard
                    key={p.person_id}
                    person={p}
                    onOpen={() => setDetailPersonId(p.person_id)}
                  />
                ))
              )}
            </div>
          </>
        ) : (
          <>
            <h4 className="text-[10px] uppercase text-text-muted mb-2">
              Watchlist
            </h4>
            <div className="flex flex-col gap-2 mb-4">
              {watchlistPersons.length === 0 ? (
                <p className="text-xs text-text-muted py-2">No watchlist entries</p>
              ) : (
                watchlistPersons.map((p) => (
                  <PersonListCard
                    key={p.person_id}
                    person={p}
                    onOpen={() => setDetailPersonId(p.person_id)}
                  />
                ))
              )}
            </div>
            <h4 className="text-[10px] uppercase text-text-muted mb-2">
              Auto-clusters
            </h4>
            <div className="flex flex-col gap-2">
              {autoPersons.length === 0 && !loading ? (
                <p className="text-xs text-text-muted py-2">No auto-clusters yet</p>
              ) : (
                autoPersons.map((p) => (
                  <PersonListCard
                    key={p.person_id}
                    person={p}
                    onOpen={() => setDetailPersonId(p.person_id)}
                    showPromoteHint
                  />
                ))
              )}
            </div>
          </>
        )}
      </Card>

      <Card title="Find person by photo">
        <Dropzone
          compact
          onFileSelect={handleSearchImage}
          preview={searchPreview}
          previewAlt="Query"
        />
        <Button
          className="w-full mt-3"
          onClick={runPersonSearch}
          disabled={!searchFile || !reidReady}
          loading={searching}
        >
          Search persons
        </Button>
        {searchError && (
          <p className="text-error text-xs mt-2">{searchError}</p>
        )}
        {searchResults.length > 0 && (
          <div className="mt-4 flex flex-col gap-2">
            <span className="text-xs text-text-secondary">
              {searchResults.length} match{searchResults.length !== 1 ? 'es' : ''}
            </span>
            {searchResults.map((r) => (
              <button
                key={r.person_id}
                type="button"
                onClick={() => setDetailPersonId(r.person_id)}
                className={cn(
                  'flex gap-3 p-2 rounded-xl border border-border text-left w-full',
                  'hover:border-accent transition-colors'
                )}
              >
                {r.representative_crop_url ? (
                  <img
                    src={r.representative_crop_url}
                    alt=""
                    className="w-14 h-14 object-cover rounded-lg flex-shrink-0"
                  />
                ) : (
                  <div className="w-14 h-14 rounded-lg bg-bg-tertiary flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {formatPersonDisplayLabel(r.person_id, r.label)}
                  </div>
                  <div className="text-[10px] text-text-muted">
                    Similarity {(r.similarity * 100).toFixed(1)}% ·{' '}
                    {r.sighting_count} sighting{r.sighting_count !== 1 ? 's' : ''}
                    {r.is_watchlist && ' · Watchlist'}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </Card>

      <PersonDetailModal
        personId={detailPersonId}
        isOpen={detailPersonId !== null}
        onClose={() => setDetailPersonId(null)}
        onMutate={loadPersons}
      />
    </>
  );
}

function PersonListCard({
  person,
  onOpen,
  showPromoteHint,
}: {
  person: Person;
  onOpen: () => void;
  showPromoteHint?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        'flex gap-3 p-2 rounded-xl border border-border text-left w-full',
        'hover:border-accent transition-colors bg-bg-card'
      )}
    >
      {person.representative_crop_url ? (
        <img
          src={person.representative_crop_url}
          alt=""
          className="w-14 h-14 object-cover rounded-lg flex-shrink-0"
        />
      ) : (
        <div className="w-14 h-14 rounded-lg bg-bg-tertiary flex-shrink-0" />
      )}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">
          {formatPersonDisplayLabel(person.person_id, person.label)}
        </div>
        <div className="text-[10px] text-text-muted">
          {person.sighting_count} sighting{person.sighting_count !== 1 ? 's' : ''}
          {person.is_watchlist && (
            <span className="text-error"> · Watchlist</span>
          )}
        </div>
        {showPromoteHint && !person.is_watchlist && (
          <div className="text-[10px] text-accent mt-0.5">Open to name or promote →</div>
        )}
      </div>
    </button>
  );
}

export default PersonsTab;
