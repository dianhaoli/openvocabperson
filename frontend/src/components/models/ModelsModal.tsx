import { useState } from 'react';
import { api, errorMessage } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import { MODEL_ROLES, type ModelRole, type ModelsInfo } from '../../types';
import { Badge, ErrorState, Loading } from '../ui/Misc';
import { Modal } from '../ui/Modal';
import { Spinner } from '../ui/Spinner';

const ROLE_NAME: Record<ModelRole, string> = {
  detector: 'Detector',
  segmenter: 'Segmenter',
  vlm: 'VLM',
  llm: 'LLM',
  embedder: 'Embedder',
  reid: 'Re-ID',
};

const ROLE_INFO: Record<ModelRole, string> = {
  detector: 'Open-vocabulary object detector',
  segmenter: 'Segmentation masks',
  vlm: 'Vision-language model (describe, verify, Q&A)',
  llm: 'Language model (query expansion, agent)',
  embedder: 'Image / text embeddings (CLIP)',
  reid: 'Person re-identification',
};

/** Option labels stay short; the full note is shown below the select and in the tooltip. */
const NOTE_MAX = 48;
const shortNote = (note: string) => (note.length > NOTE_MAX ? `${note.slice(0, NOTE_MAX - 1)}…` : note);

export function ModelsModal() {
  const { modelsOpen, health } = useAppState();
  const { setModelsOpen, setHealth, toast } = useActions();
  const { data, error, loading, reload, mutate } = useApi<ModelsInfo>(modelsOpen ? 'models' : null, (s) => api.models(s));
  const [saving, setSaving] = useState<ModelRole | null>(null);
  const [roleErrors, setRoleErrors] = useState<Partial<Record<ModelRole, string>>>({});

  const apply = async (role: ModelRole, spec: string) => {
    setSaving(role);
    setRoleErrors((prev) => ({ ...prev, [role]: undefined }));
    try {
      const next = await api.patchConfig({ [role]: spec });
      mutate(() => next);
      // Keep the header's capability chips in sync without waiting for the next health poll.
      if (health) setHealth({ ...health, capabilities: next.capabilities }, false);
      toast(`${ROLE_NAME[role]} → ${spec}`, 'success');
    } catch (err) {
      setRoleErrors((prev) => ({ ...prev, [role]: errorMessage(err) }));
      // The backend may have partially applied the change: show its actual state.
      reload();
    } finally {
      setSaving(null);
    }
  };

  return (
    <Modal open={modelsOpen} onClose={() => setModelsOpen(false)} title="Models & capabilities" className="max-w-[680px]">
      {loading && <Loading />}
      {!!error && <ErrorState error={error} onRetry={reload} />}
      {data && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone="accent">device: {data.device}</Badge>
            {Object.entries(data.capabilities).map(([k, v]) => (
              <Badge key={k} tone={v ? 'success' : 'neutral'}>
                {k}
                {typeof v === 'string' ? `: ${v}` : v ? ' ✓' : ' ✗'}
              </Badge>
            ))}
          </div>
          <div className="flex flex-col gap-3">
            {MODEL_ROLES.map((role) => {
              const providers = data.providers[role] ?? [];
              const current = data.defaults[role] ?? '';
              const currentProvider = providers.find((p) => p.spec === current);
              return (
                <div key={role} className="grid grid-cols-[130px_minmax(0,1fr)] gap-3 items-start">
                  <div>
                    <div className="text-sm font-medium">{ROLE_NAME[role]}</div>
                    <div className="text-[10.5px] text-text-muted leading-tight">{ROLE_INFO[role]}</div>
                  </div>
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <select
                        className="flex-1 min-w-0 bg-bg-primary border border-border rounded-[10px] px-2.5 py-2 text-sm focus:outline-none focus:border-accent disabled:opacity-60"
                        value={current}
                        disabled={saving !== null || providers.length === 0}
                        onChange={(e) => void apply(role, e.target.value)}
                      >
                        {!currentProvider && <option value={current}>{current || '(none)'}</option>}
                        {providers.map((p) => (
                          <option key={p.spec} value={p.spec} disabled={!p.available} title={p.note ?? undefined}>
                            {p.label}
                            {p.local ? ' · local' : ''}
                            {!p.available ? ` — unavailable${p.note ? `: ${shortNote(p.note)}` : ''}` : ''}
                          </option>
                        ))}
                      </select>
                      {saving === role && <Spinner />}
                    </div>
                    <div className="flex flex-wrap gap-1 items-center">
                      <span className="font-mono text-[10.5px] text-text-muted">{current || '—'}</span>
                      {currentProvider?.open_vocab && <Badge>open-vocab</Badge>}
                      {currentProvider?.masks && <Badge>masks</Badge>}
                      {currentProvider?.note && <span className="text-[10.5px] text-warning break-words min-w-0">{currentProvider.note}</span>}
                    </div>
                    {roleErrors[role] && (
                      <div className="text-[11px] text-error rounded-lg border border-error/40 bg-error/10 px-2 py-1 break-words">{roleErrors[role]}</div>
                    )}
                    {providers.some((p) => !p.available) && (
                      <details className="text-[10.5px] text-text-muted">
                        <summary className="cursor-pointer">Unavailable providers</summary>
                        <ul className="pl-3 mt-1">
                          {providers
                            .filter((p) => !p.available)
                            .map((p) => (
                              <li key={p.spec} className="break-words">
                                <span className="font-mono">{p.spec}</span>
                                {p.note ? ` — ${p.note}` : ''}
                              </li>
                            ))}
                        </ul>
                      </details>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          {data.defaults.text_embedder && (
            <div className="text-[11px] text-text-muted">
              Text embedder: <span className="font-mono">{data.defaults.text_embedder}</span>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
