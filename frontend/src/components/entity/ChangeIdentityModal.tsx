import { useEffect, useState, useMemo } from 'react';
import { Modal, Button } from '../ui';
import { listPersons, assignEntity } from '../../api/client';
import { useApp } from '../../context/AppContext';
import type { Person } from '../../types';
import { formatPersonDisplayLabel } from '../../utils/personDisplay';
import { cn } from '../../utils/cn';

interface ChangeIdentityModalProps {
  objectId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function ChangeIdentityModal({
  objectId,
  isOpen,
  onClose,
}: ChangeIdentityModalProps) {
  const { updateEntityIdentity } = useApp();
  const [persons, setPersons] = useState<Person[]>([]);
  const [loading, setLoading] = useState(false);
  const [assigning, setAssigning] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setFilter('');
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    listPersons(false, 200, 0)
      .then((r) => {
        if (!cancelled) setPersons(r.persons);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return persons;
    return persons.filter(
      (p) =>
        p.person_id.toLowerCase().includes(q) ||
        (p.label && p.label.toLowerCase().includes(q))
    );
  }, [persons, filter]);

  const handleAssign = async (personId: string) => {
    if (!objectId) return;
    setAssigning(personId);
    setError(null);
    try {
      const res = await assignEntity(objectId, personId);
      updateEntityIdentity(objectId, {
        person_id: res.person_id,
        person_label: res.person_label,
        is_watchlist: res.is_watchlist,
        match_status: 'matched',
        match_score: 1,
      });
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Assignment failed');
    } finally {
      setAssigning(null);
    }
  };

  if (!isOpen) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Assign to person"
      panelClassName="max-w-[480px] max-h-[85vh] overflow-y-auto"
      actions={
        <Button variant="secondary" onClick={onClose}>
          Cancel
        </Button>
      }
    >
      <p className="text-text-muted text-xs mb-3">
        Link this detection to an existing identity cluster.
      </p>
      <input
        type="text"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter by name or ID…"
        className="w-full px-3 py-2 mb-3 bg-bg-tertiary border border-border rounded-lg text-sm outline-none focus:border-accent"
      />
      {error && <p className="text-error text-xs mb-2">{error}</p>}
      {loading && <p className="text-text-muted text-sm">Loading persons…</p>}
      {!loading && (
        <ul className="max-h-[50vh] overflow-y-auto space-y-2 pr-1">
          {filtered.map((p) => (
            <li key={p.person_id}>
              <button
                type="button"
                disabled={!!assigning}
                onClick={() => handleAssign(p.person_id)}
                className={cn(
                  'w-full flex items-center gap-3 p-2 rounded-xl border border-border text-left',
                  'hover:border-accent hover:bg-bg-tertiary transition-colors',
                  assigning === p.person_id && 'opacity-60'
                )}
              >
                {p.representative_crop_url ? (
                  <img
                    src={p.representative_crop_url}
                    alt=""
                    className="w-12 h-12 object-cover rounded-lg flex-shrink-0"
                  />
                ) : (
                  <div className="w-12 h-12 rounded-lg bg-bg-tertiary flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">
                    {formatPersonDisplayLabel(p.person_id, p.label)}
                  </div>
                  <div className="text-[10px] text-text-muted">
                    {p.sighting_count} sighting{p.sighting_count !== 1 ? 's' : ''}
                    {p.is_watchlist && ' · Watchlist'}
                  </div>
                </div>
              </button>
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="text-text-muted text-sm text-center py-6">
              No matching persons
            </li>
          )}
        </ul>
      )}
    </Modal>
  );
}

export default ChangeIdentityModal;
