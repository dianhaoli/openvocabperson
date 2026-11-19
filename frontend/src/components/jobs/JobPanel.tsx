import { useEffect, useRef, useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { TERMINAL_STATUSES, type JobTrackerState } from '../../hooks/useJobTracker';
import type { JobStatus } from '../../types';
import { cn } from '../../utils/cn';
import { jobSummaryText } from '../../utils/format';
import { Img } from '../common/Img';
import { Button } from '../ui/Button';
import { Badge, ProgressBar } from '../ui/Misc';
import { Spinner } from '../ui/Spinner';
import { PlanView } from './PlanView';

const STATUS_TONE: Record<JobStatus, 'neutral' | 'accent' | 'success' | 'warning' | 'error'> = {
  queued: 'neutral',
  running: 'accent',
  done: 'success',
  error: 'error',
  cancelled: 'warning',
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return <Badge tone={STATUS_TONE[status] ?? 'neutral'}>{status}</Badge>;
}

interface Props {
  state: JobTrackerState;
  onCancel: () => Promise<void>;
  onDismiss: () => void;
}

export function JobPanel({ state, onCancel, onDismiss }: Props) {
  const { currentMediaId } = useAppState();
  const { openMedia, toastError, toast, removeMedia, bumpLibrary } = useActions();
  const [cancelling, setCancelling] = useState(false);
  const [deletedIds, setDeletedIds] = useState<string[]>([]);
  const [deletingLeftovers, setDeletingLeftovers] = useState(false);
  const [showLog, setShowLog] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [state.log.length, showLog]);

  if (!state.jobId) return null;
  const status = state.status ?? 'queued';
  const terminal = TERMINAL_STATUSES.includes(status);
  const frac = state.total > 0 ? state.done / state.total : 0;
  const statusLine = !terminal
    ? [state.stage, state.current].filter(Boolean).join(' · ')
    : status === 'error'
      ? (state.job?.error ?? 'Job failed')
      : status === 'cancelled'
        ? 'Cancelled before finishing'
        : state.job
          ? jobSummaryText(state.job)
          : '';

  // Media rows that failed / cancelled items left behind; hidden from the default library listing.
  const leftovers = terminal ? (state.job?.failed_media_ids ?? []).filter((id) => !deletedIds.includes(id)) : [];

  const deleteLeftovers = async () => {
    setDeletingLeftovers(true);
    try {
      for (const id of leftovers) {
        await api.deleteMedia(id);
        removeMedia(id);
        setDeletedIds((prev) => [...prev, id]);
      }
      bumpLibrary();
      toast(`Deleted ${leftovers.length} leftover media`, 'success');
    } catch (err) {
      toastError(err, 'Delete failed');
    } finally {
      setDeletingLeftovers(false);
    }
  };

  const cancel = async () => {
    setCancelling(true);
    try {
      await onCancel();
    } catch (err) {
      toastError(err, 'Cancel failed');
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div className="rounded-[14px] border border-border bg-bg-card p-3 flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        {state.streaming && !terminal && <Spinner size={14} className="text-accent" />}
        <span className="font-mono text-[11px] text-text-muted truncate">{state.jobId}</span>
        <StatusBadge status={status} />
        <div className="ml-auto flex gap-1">
          {!terminal && (
            <Button size="xs" variant="secondary" onClick={() => void cancel()} loading={cancelling}>
              Cancel
            </Button>
          )}
          {(terminal || !state.streaming) && (
            <Button size="xs" variant="ghost" onClick={onDismiss} title="Dismiss">
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <ProgressBar
          value={status === 'done' ? 1 : frac}
          indeterminate={!terminal && state.total === 0}
          tone={status === 'cancelled' ? 'warning' : status === 'error' ? 'error' : undefined}
        />
        <div className="flex justify-between gap-2 text-[11px] text-text-muted">
          <span className={cn('truncate', status === 'error' && 'text-error')} title={statusLine}>
            {statusLine}
          </span>
          {state.total > 0 && (
            <span className="font-mono shrink-0">
              {state.done}/{state.total}
            </span>
          )}
        </div>
      </div>

      {leftovers.length > 0 && (
        <div className="flex items-center gap-2 rounded-lg border border-border bg-bg-primary px-2 py-1.5 text-[11px] text-text-muted">
          <span className="flex-1">
            {leftovers.length} media left failed/cancelled (hidden from the library; see Library ▸ Status ▸ Failed / Cancelled)
          </span>
          <Button size="xs" variant="secondary" onClick={() => void deleteLeftovers()} loading={deletingLeftovers}>
            Delete
          </Button>
        </div>
      )}

      {state.plan && (
        <details className="text-xs">
          <summary className="cursor-pointer text-text-secondary">Plan · {state.plan.prompts.length} prompt(s)</summary>
          <div className="mt-2">
            <PlanView plan={state.plan} compact />
          </div>
        </details>
      )}

      {state.media.length > 0 && (
        <div className="flex gap-1.5 overflow-x-auto pb-1">
          {state.media.map((m) => (
            <button
              key={m.media.id}
              type="button"
              disabled={!m.done}
              onClick={() => openMedia(m.media.id)}
              title={`${m.media.name}${m.done ? ` · ${m.entityCount} entities` : ' · processing…'}`}
              className={cn(
                'relative shrink-0 w-14 h-14 rounded-lg overflow-hidden border-2 transition-colors',
                currentMediaId === m.media.id ? 'border-accent' : 'border-border hover:border-accent-border',
                !m.done && 'opacity-60 cursor-wait',
              )}
            >
              <Img src={m.media.thumb_url} alt={m.media.name} className="w-full h-full object-cover" />
              {m.media.kind === 'video' && <span className="absolute top-0.5 left-0.5 text-[9px] bg-black/70 rounded px-0.5">▶</span>}
              <span className="absolute bottom-0 right-0 text-[9px] bg-black/75 px-1 rounded-tl">
                {m.done ? m.entityCount : '…'}
              </span>
            </button>
          ))}
        </div>
      )}

      <div>
        <button type="button" className="text-[11px] text-text-muted hover:text-text-primary" onClick={() => setShowLog((v) => !v)}>
          {showLog ? '▾' : '▸'} Log ({state.log.length})
        </button>
        {showLog && (
          <div ref={logRef} className="mt-1 max-h-48 overflow-y-auto rounded-lg bg-bg-primary border border-border p-2 font-mono text-[10.5px] leading-relaxed">
            {state.log.length === 0 && <div className="text-text-muted">Waiting for events…</div>}
            {state.log.map((l) => (
              <div
                key={l.id}
                className={cn(
                  'whitespace-pre-wrap break-words',
                  l.level === 'error' && 'text-error',
                  l.level === 'warning' && 'text-warning',
                  l.level === 'success' && 'text-success',
                  l.level === 'info' && 'text-text-secondary',
                )}
              >
                <span className="text-text-muted">{new Date(l.ts * 1000).toLocaleTimeString()} </span>
                {l.message}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
