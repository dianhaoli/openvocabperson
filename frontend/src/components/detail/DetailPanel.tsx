import { useState, type ReactNode } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import type { Entity, EntityDetail } from '../../types';
import { labelColor } from '../../utils/colors';
import { formatTime, personName, pct } from '../../utils/format';
import { IdentityBadge, VerifiedBadge } from '../common/IdentityBadge';
import { Img } from '../common/Img';
import { PersonPicker } from '../common/PersonPicker';
import { Button } from '../ui/Button';
import { Badge, ErrorState, Loading, Section } from '../ui/Misc';
import { ConfirmModal } from '../ui/Modal';
import { EntityQA } from './EntityQA';

function MetaRow({ k, children }: { k: string; children: ReactNode }) {
  return (
    <>
      <div className="text-text-muted">{k}</div>
      <div className="text-text-primary min-w-0 break-words">{children}</div>
    </>
  );
}

function attrValue(v: unknown): string {
  if (v == null) return '–';
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) return v.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(', ');
  return JSON.stringify(v);
}

function EntityDetails({ entityId, cached }: { entityId: string; cached: Entity | undefined }) {
  const actions = useActions();
  const { personsVersion } = useAppState();
  // Refetch when identities change elsewhere (rename, merge, watchlist); keep showing the previous data meanwhile.
  const q = useApi<EntityDetail>(`entity:${entityId}:${personsVersion}`, (s) => api.entity(entityId, s));
  const detail = q.data ?? q.stale;
  const entity: Entity | undefined = detail ?? cached;
  const [pickerOpen, setPickerOpen] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const [watchBusy, setWatchBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (!entity) {
    if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />;
    return <Loading />;
  }

  const applyEntity = (updated: Entity) => {
    actions.upsertEntity(updated);
    q.mutate((prev) => ({ ...prev, ...updated }));
  };

  const assign = async (personId: string | null) => {
    setAssigning(true);
    try {
      const updated = await api.assignEntity(entity.id, personId);
      applyEntity(updated);
      actions.bumpPersons();
      actions.toast(personId ? `Assigned to ${personName(updated.person)}` : 'Created a new person', 'success');
      setPickerOpen(false);
    } catch (err) {
      actions.toastError(err, 'Assign failed');
    } finally {
      setAssigning(false);
    }
  };

  const toggleWatchlist = async () => {
    if (!entity.person) return;
    setWatchBusy(true);
    try {
      const p = await api.patchPerson(entity.person.id, { is_watchlist: !entity.person.is_watchlist });
      applyEntity({ ...entity, person: { id: p.id, label: p.label, is_watchlist: p.is_watchlist } });
      actions.bumpPersons();
    } catch (err) {
      actions.toastError(err, 'Watchlist update failed');
    } finally {
      setWatchBusy(false);
    }
  };

  const remove = async () => {
    setDeleting(true);
    try {
      await api.deleteEntity(entity.id);
      actions.removeEntity(entity.id, entity.media_id);
      actions.bumpPersons();
      actions.toast('Detection deleted', 'success');
    } catch (err) {
      actions.toastError(err, 'Delete failed');
      setDeleting(false);
    }
  };

  const isVideo = entity.t_start != null;
  const attributes = entity.attributes ? Object.entries(entity.attributes) : [];
  const color = labelColor(entity.label);

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex flex-col gap-2">
        <Img src={entity.crop_url} alt={entity.label} className="w-full max-h-[260px] object-contain rounded-xl border border-border bg-black" />
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
          <span className="font-semibold">{entity.label}</span>
          <span className="font-mono text-xs text-text-muted ml-auto">{pct(entity.score, 1)}</span>
        </div>
        <div className="flex flex-wrap gap-1">
          <VerifiedBadge entity={entity} />
          <IdentityBadge entity={entity} />
          <Badge tone={entity.stage === 'rejected' ? 'error' : entity.stage === 'low_confidence' ? 'warning' : 'neutral'}>{entity.stage}</Badge>
        </div>
      </div>

      <div className="grid grid-cols-[86px_1fr] gap-x-3 gap-y-1 text-xs">
        <MetaRow k="Label">{entity.label}</MetaRow>
        <MetaRow k="Concept">{entity.concept ?? <span className="text-text-muted">default vocab</span>}</MetaRow>
        <MetaRow k="Score">{entity.score.toFixed(3)}</MetaRow>
        <MetaRow k="Stage">{entity.stage}</MetaRow>
        <MetaRow k="Box">
          <span className="font-mono text-[10.5px]">{entity.box.map((v) => Math.round(v)).join(', ')}</span>
        </MetaRow>
        {entity.mask_area != null && <MetaRow k="Mask area">{entity.mask_area.toLocaleString()} px</MetaRow>}
        {isVideo && (
          <>
            <MetaRow k="Track">#{entity.track_id ?? '–'}</MetaRow>
            <MetaRow k="Time">
              {formatTime(entity.t_start)} – {formatTime(entity.t_end)}
            </MetaRow>
            {entity.frame_index != null && <MetaRow k="Frame">{entity.frame_index}</MetaRow>}
          </>
        )}
        <MetaRow k="ID">
          <span className="font-mono text-[10.5px]">{entity.id}</span>
        </MetaRow>
      </div>

      {entity.verified != null && (
        <Section title="Verification">
          <div className={`text-xs rounded-lg border px-2.5 py-2 ${entity.verified ? 'border-success/40 bg-success-soft' : 'border-error/40 bg-error/10'}`}>
            <span className="font-medium">{entity.verified ? '✓ Confirmed' : '✗ Rejected'}</span>
            {entity.verify_reason && <span className="text-text-secondary"> — {entity.verify_reason}</span>}
          </div>
        </Section>
      )}

      {(entity.description || attributes.length > 0) && (
        <Section title="Description">
          {entity.description && <div className="text-sm text-text-secondary">{entity.description}</div>}
          {attributes.length > 0 && (
            <table className="w-full text-xs">
              <tbody>
                {attributes.map(([k, v]) => (
                  <tr key={k} className="border-t border-border">
                    <td className="py-1 pr-2 text-text-muted capitalize align-top whitespace-nowrap">{k.replace(/_/g, ' ')}</td>
                    <td className="py-1 text-text-primary break-words">{attrValue(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      )}

      <Section title="Identity">
        <div className="rounded-xl border border-border bg-bg-card p-2.5 flex flex-col gap-2">
          {entity.person ? (
            <div className="flex items-center gap-2">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium truncate">{personName(entity.person)}</div>
                <div className="text-[11px] text-text-muted">
                  {entity.match_status ?? 'linked'}
                  {entity.match_score != null ? ` · ${pct(entity.match_score)}` : ''}
                </div>
              </div>
              {entity.person.is_watchlist && <Badge tone="error">★ watchlist</Badge>}
            </div>
          ) : (
            <div className="text-xs text-text-muted">
              Not linked to a person{entity.match_status ? ` (${entity.match_status})` : ''}.
              {entity.suggested_person && <> Suggested: {personName(entity.suggested_person)}.</>}
            </div>
          )}
          <div className="flex flex-wrap gap-1.5">
            <Button size="xs" variant="secondary" onClick={() => setPickerOpen(true)}>
              Change identity
            </Button>
            {entity.person && (
              <>
                <Button size="xs" variant="secondary" onClick={() => void toggleWatchlist()} loading={watchBusy}>
                  {entity.person.is_watchlist ? 'Remove from watchlist' : 'Add to watchlist'}
                </Button>
                <Button size="xs" variant="secondary" onClick={() => entity.person && actions.openPerson(entity.person.id)}>
                  Open person
                </Button>
              </>
            )}
          </div>
        </div>
      </Section>

      <Section title="Ask about this entity">
        {detail ? (
          <EntityQA entityId={entity.id} history={detail.qa ?? []} onAnswered={(item) => q.mutate((prev) => ({ ...prev, qa: [...(prev.qa ?? []), item] }))} />
        ) : q.error ? (
          <ErrorState error={q.error} onRetry={q.reload} />
        ) : (
          <Loading />
        )}
      </Section>

      <div className="pt-2 border-t border-border">
        <Button size="xs" variant="ghost" className="text-error" onClick={() => setConfirmDelete(true)}>
          Delete detection
        </Button>
      </div>

      <PersonPicker
        open={pickerOpen}
        title="Assign identity"
        excludeId={entity.person?.id}
        allowNew
        busy={assigning}
        onPick={(id) => void assign(id)}
        onClose={() => setPickerOpen(false)}
      />
      <ConfirmModal
        open={confirmDelete}
        title="Delete detection?"
        message="This removes the entity (and unlinks it from its person)."
        loading={deleting}
        onConfirm={() => void remove()}
        onClose={() => setConfirmDelete(false)}
      />
    </div>
  );
}

export function DetailPanel() {
  const { selectedEntityId, currentMediaId, mediaCache } = useAppState();
  const { selectEntity } = useActions();
  if (!selectedEntityId) return null;
  const cached = currentMediaId ? mediaCache[currentMediaId]?.entities.find((e) => e.id === selectedEntityId) : undefined;
  return (
    <aside className="w-[320px] xl:w-[360px] shrink-0 border-l border-border bg-bg-secondary flex flex-col min-h-0">
      <div className="flex items-center justify-between px-4 h-11 border-b border-border shrink-0">
        <span className="text-[11px] uppercase tracking-wider text-text-muted font-semibold">Entity</span>
        <button type="button" onClick={() => selectEntity(null)} className="text-text-muted hover:text-text-primary text-lg leading-none" aria-label="Close panel">
          ×
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <EntityDetails key={selectedEntityId} entityId={selectedEntityId} cached={cached} />
      </div>
    </aside>
  );
}
