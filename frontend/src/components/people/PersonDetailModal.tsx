import { useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import type { PersonDetail } from '../../types';
import { formatDate, personName } from '../../utils/format';
import { EntityTile } from '../common/EntityTile';
import { Img } from '../common/Img';
import { PersonPicker } from '../common/PersonPicker';
import { Button } from '../ui/Button';
import { Badge, ErrorState, Loading, inputClass } from '../ui/Misc';
import { ConfirmModal, Modal } from '../ui/Modal';

function PersonEditor({ person, onChanged }: { person: PersonDetail; onChanged: (p: Partial<PersonDetail>) => void }) {
  const actions = useActions();
  const [label, setLabel] = useState(person.label ?? '');
  const [notes, setNotes] = useState(person.notes ?? '');
  const [saving, setSaving] = useState(false);
  const [mergeOpen, setMergeOpen] = useState(false);
  const [merging, setMerging] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [watchBusy, setWatchBusy] = useState(false);

  const dirty = label !== (person.label ?? '') || notes !== (person.notes ?? '');

  const save = async () => {
    setSaving(true);
    try {
      const updated = await api.patchPerson(person.id, { label: label.trim() || null, notes: notes.trim() || null });
      onChanged(updated);
      actions.bumpPersons();
      actions.toast('Person saved', 'success');
    } catch (err) {
      actions.toastError(err, 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const toggleWatch = async () => {
    setWatchBusy(true);
    try {
      const updated = await api.patchPerson(person.id, { is_watchlist: !person.is_watchlist });
      onChanged(updated);
      actions.bumpPersons();
    } catch (err) {
      actions.toastError(err, 'Watchlist update failed');
    } finally {
      setWatchBusy(false);
    }
  };

  const mergeInto = async (targetId: string | null) => {
    if (!targetId) return;
    setMerging(true);
    try {
      // Merge this person (other_id) into the chosen target person.
      const target = await api.mergePerson(targetId, person.id);
      actions.toast(`Merged into ${personName(target)}`, 'success');
      setMergeOpen(false);
      actions.bumpPersons();
      actions.openPerson(target.id);
    } catch (err) {
      actions.toastError(err, 'Merge failed');
    } finally {
      setMerging(false);
    }
  };

  const remove = async () => {
    setDeleting(true);
    try {
      await api.deletePerson(person.id);
      actions.toast(`Deleted ${personName(person)}`, 'success');
      actions.bumpPersons();
      actions.openPerson(null);
    } catch (err) {
      actions.toastError(err, 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-4">
        <Img src={person.thumb_url ?? undefined} alt="" className="w-28 h-28 rounded-xl object-cover object-top border border-border shrink-0" />
        <div className="flex-1 min-w-0 flex flex-col gap-2">
          <input className={inputClass} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Name (e.g. Alice)" />
          <textarea className={`${inputClass} resize-y min-h-[60px]`} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Notes…" rows={2} />
          <div className="flex flex-wrap gap-1.5 items-center">
            <Button size="sm" onClick={() => void save()} disabled={!dirty} loading={saving}>
              Save
            </Button>
            <Button size="sm" variant={person.is_watchlist ? 'danger' : 'secondary'} onClick={() => void toggleWatch()} loading={watchBusy}>
              {person.is_watchlist ? '★ On watchlist' : '☆ Add to watchlist'}
            </Button>
            <Button size="sm" variant="secondary" onClick={() => setMergeOpen(true)}>
              Merge into…
            </Button>
            <Button size="sm" variant="ghost" className="text-error" onClick={() => setDeleteOpen(true)}>
              Delete
            </Button>
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 text-[11px]">
        <Badge>{person.sighting_count} sightings</Badge>
        <Badge>{person.media_count} media</Badge>
        <Badge>first seen {formatDate(person.first_seen)}</Badge>
        <Badge>last seen {formatDate(person.last_seen)}</Badge>
        <Badge className="font-mono">{person.id}</Badge>
      </div>

      <div className="flex flex-col gap-2">
        <div className="text-[11px] uppercase tracking-wider text-text-muted font-semibold">Sightings</div>
        {person.sightings.length === 0 ? (
          <div className="text-xs text-text-muted">No sightings linked.</div>
        ) : (
          <div className="grid grid-cols-5 gap-2">
            {person.sightings.map((s) => (
              <EntityTile
                key={s.id}
                entity={s}
                showPerson={false}
                onClick={() => {
                  actions.openMedia(s.media_id, s.id);
                  actions.openPerson(null);
                }}
                footer={<div className="text-[9.5px] text-text-muted truncate">{s.media?.name}</div>}
              />
            ))}
          </div>
        )}
      </div>

      <PersonPicker
        open={mergeOpen}
        title={`Merge ${personName(person)} into…`}
        excludeId={person.id}
        busy={merging}
        onPick={(id) => void mergeInto(id)}
        onClose={() => setMergeOpen(false)}
      />
      <ConfirmModal
        open={deleteOpen}
        title="Delete person?"
        message={`${personName(person)} will be deleted and its ${person.sighting_count} sighting(s) unlinked.`}
        loading={deleting}
        onConfirm={() => void remove()}
        onClose={() => setDeleteOpen(false)}
      />
    </div>
  );
}

export function PersonDetailModal() {
  const { personModalId, personsVersion } = useAppState();
  const { openPerson } = useActions();
  const q = useApi(personModalId ? `person:${personModalId}:${personsVersion}` : null, (s) => api.person(personModalId ?? '', s));
  const person = q.data ?? (q.stale?.id === personModalId ? q.stale : undefined);

  return (
    <Modal
      open={personModalId !== null}
      onClose={() => openPerson(null)}
      title={person ? personName(person) : 'Person'}
      className="max-w-[760px]"
    >
      {!person && q.loading && <Loading />}
      {!!q.error && <ErrorState error={q.error} onRetry={q.reload} />}
      {person && (
        <PersonEditor
          key={`${person.id}:${person.updated_at}`}
          person={person}
          onChanged={(patch) => q.mutate((prev) => ({ ...prev, ...patch }))}
        />
      )}
    </Modal>
  );
}
