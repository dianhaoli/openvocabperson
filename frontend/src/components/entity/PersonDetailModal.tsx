import { useEffect, useState, useCallback } from 'react';
import { Modal, DeleteModal, Button } from '../ui';
import { cn } from '../../utils/cn';
import {
  getPerson,
  patchPerson,
  deletePerson,
  getSession,
  getSessionImageUrl,
} from '../../api/client';
import { useApp } from '../../context/AppContext';
import type { PersonDetail } from '../../types';
import type { Entity } from '../../types';
import { formatPersonDisplayLabel } from '../../utils/personDisplay';

interface PersonDetailModalProps {
  personId: string | null;
  isOpen: boolean;
  onClose: () => void;
  onMutate: () => void;
}

export function PersonDetailModal({
  personId,
  isOpen,
  onClose,
  onMutate,
}: PersonDetailModalProps) {
  const { setEntities, setCurrentImage, setActiveTab, selectEntity } = useApp();

  const [detail, setDetail] = useState<PersonDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState('');
  const [notes, setNotes] = useState('');
  const [isWatchlist, setIsWatchlist] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (!isOpen || !personId) {
      setDetail(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    getPerson(personId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        setLabel(d.label ?? '');
        setNotes(d.notes ?? '');
        setIsWatchlist(d.is_watchlist);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Load failed');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen, personId]);

  const handleSave = async () => {
    if (!personId) return;
    setSaving(true);
    setError(null);
    try {
      await patchPerson(personId, {
        label: label.trim() || null,
        notes: notes.trim() || null,
        is_watchlist: isWatchlist,
      });
      onMutate();
      const d = await getPerson(personId);
      setDetail(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!personId) return;
    setDeleting(true);
    try {
      await deletePerson(personId);
      onMutate();
      setDeleteOpen(false);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  const openSighting = useCallback(
    async (sessionId: string, objectId: string) => {
      try {
        const session = await getSession(sessionId);
        const img = new Image();
        img.onload = () => {
          setCurrentImage(img, getSessionImageUrl(sessionId));
          const entities: Entity[] = session.entities.map((e, idx) => ({
            object_id: e.object_id,
            index: idx,
            class: e.class_name,
            confidence: e.confidence,
            box: e.box,
            stage: e.stage,
            analysis: e.analysis,
            crop_image: e.crop_image || '',
            person_id: e.person_id,
            person_label: e.person_label,
            is_watchlist: e.is_watchlist,
            match_score: e.match_score,
            match_status: e.match_status,
          }));
          setEntities(entities);
          setActiveTab('upload');
          onClose();
          setTimeout(() => selectEntity(objectId), 80);
        };
        img.src = getSessionImageUrl(sessionId);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to open session');
      }
    },
    [
      setCurrentImage,
      setEntities,
      setActiveTab,
      selectEntity,
      onClose,
    ]
  );

  if (!isOpen || !personId) return null;

  return (
    <>
      <Modal
        isOpen={isOpen}
        onClose={onClose}
        title={
          detail
            ? formatPersonDisplayLabel(detail.person_id, detail.label)
            : 'Person'
        }
        panelClassName="max-w-[560px] max-h-[90vh] overflow-y-auto"
        actions={
          <>
            <Button variant="secondary" onClick={onClose}>
              Close
            </Button>
            <Button
              onClick={handleSave}
              loading={saving}
              disabled={loading || !detail}
            >
              Save
            </Button>
          </>
        }
      >
        {loading && (
          <p className="text-text-muted">Loading…</p>
        )}
        {error && (
          <p className="text-error text-sm mb-2">{error}</p>
        )}
        {!loading && detail && (
          <div className="space-y-4">
            <div className="flex gap-4">
              {detail.representative_crop_url && (
                <img
                  src={detail.representative_crop_url}
                  alt=""
                  className="w-24 h-24 object-cover rounded-xl border border-border flex-shrink-0"
                />
              )}
              <div className="flex-1 min-w-0 text-xs text-text-muted">
                <div>
                  <span className="text-text-secondary font-medium">ID</span>{' '}
                  <code className="text-text-primary">{detail.person_id}</code>
                </div>
                <div className="mt-1">
                  Sightings:{' '}
                  <span className="text-text-primary">{detail.sighting_count}</span>
                </div>
              </div>
            </div>

            <label className="block">
              <span className="text-[10px] uppercase text-text-muted">Label</span>
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="Suspect name…"
                className="mt-1 w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-sm text-text-primary outline-none focus:border-accent"
              />
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={isWatchlist}
                onChange={(e) => setIsWatchlist(e.target.checked)}
                className="rounded border-border accent-accent"
              />
              <span className="text-sm text-text-secondary">Watchlist (suspect)</span>
            </label>

            <label className="block">
              <span className="text-[10px] uppercase text-text-muted">Notes</span>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                className="mt-1 w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-sm text-text-primary outline-none focus:border-accent resize-y"
              />
            </label>

            <div>
              <h4 className="text-[10px] uppercase text-text-muted mb-2">
                Sightings
              </h4>
              <div className="flex gap-2 overflow-x-auto pb-2 -mx-1 px-1">
                {detail.sightings.length === 0 ? (
                  <span className="text-text-muted text-xs">None linked</span>
                ) : (
                  detail.sightings.map((s) => (
                    <button
                      key={`${s.session_id}-${s.object_id}`}
                      type="button"
                      onClick={() => openSighting(s.session_id, s.object_id)}
                      className={cn(
                        'flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden border border-border',
                        'hover:border-accent transition-colors'
                      )}
                    >
                      <img
                        src={s.crop_image}
                        alt=""
                        className="w-full h-full object-cover"
                      />
                    </button>
                  ))
                )}
              </div>
            </div>

            <div className="pt-2 border-t border-border">
              <Button
                variant="danger"
                className="w-full"
                onClick={() => setDeleteOpen(true)}
              >
                Delete person cluster
              </Button>
              <p className="text-[10px] text-text-muted mt-1">
                Unlinks sightings; does not delete session images.
              </p>
            </div>
          </div>
        )}
      </Modal>

      <DeleteModal
        isOpen={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        onConfirm={handleDelete}
        loading={deleting}
        itemName="person"
        entityCount={detail?.sighting_count ?? 0}
      />
    </>
  );
}

export default PersonDetailModal;
