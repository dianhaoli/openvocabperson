import { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import { useApi } from '../../hooks/useApi';
import { TERMINAL_STATUSES } from '../../hooks/useJobTracker';
import { useWorkspaceJob } from '../../hooks/useWorkspaceJob';
import type { Job, Media, MediaKind } from '../../types';
import { cn } from '../../utils/cn';
import { relativeTime, shortId } from '../../utils/format';
import { Img } from '../common/Img';
import { JobPanel, StatusBadge } from '../jobs/JobPanel';
import { Button } from '../ui/Button';
import { EmptyState, ErrorState, Loading, Section, Segmented, inputClass } from '../ui/Misc';
import { ConfirmModal } from '../ui/Modal';

const PAGE = 24;
type Tri = 'auto' | 'on' | 'off';

function MediaCard({ m, active, onOpen, onDelete }: { m: Media; active: boolean; onOpen: () => void; onDelete: () => void }) {
  return (
    <div
      className={cn(
        'group relative rounded-xl overflow-hidden border bg-bg-card transition-colors',
        active ? 'border-accent' : 'border-border hover:border-accent-border',
      )}
    >
      <button type="button" onClick={onOpen} className="block w-full text-left" title={m.name}>
        <div className="relative aspect-square">
          <Img src={m.thumb_url} alt={m.name} className="absolute inset-0 w-full h-full object-cover" />
          {m.kind === 'video' && (
            <span className="absolute top-1 left-1 text-[9.5px] bg-black/75 rounded px-1">▶ {m.duration ? `${Math.round(m.duration)}s` : 'video'}</span>
          )}
          <div className="absolute bottom-0 inset-x-0 flex gap-1 px-1 py-0.5 bg-gradient-to-t from-black/85 to-transparent text-[10px]">
            <span title="entities">◻ {m.entity_count}</span>
            {m.person_count > 0 && <span title="persons">👤 {m.person_count}</span>}
          </div>
        </div>
        <div className="px-1.5 py-1">
          <div className="text-[10.5px] truncate">{m.name}</div>
          <div className="text-[9.5px] text-text-muted truncate">{m.query ? `“${m.query}”` : 'default vocab'}</div>
        </div>
      </button>
      <button
        type="button"
        onClick={onDelete}
        className="absolute top-1 right-1 w-6 h-6 rounded-md bg-black/70 text-[11px] opacity-0 group-hover:opacity-100 hover:bg-error transition"
        title="Delete media"
        aria-label={`Delete ${m.name}`}
      >
        🗑
      </button>
    </div>
  );
}

export function LibraryTab() {
  const { libraryVersion, currentMediaId } = useAppState();
  const actions = useActions();
  const scanJob = useWorkspaceJob('Scan', { autoOpen: false });

  const [kind, setKind] = useState<'all' | MediaKind>('all');
  const [offset, setOffset] = useState(0);
  const [scanQuery, setScanQuery] = useState('');
  const [scanVerify, setScanVerify] = useState<Tri>('auto');
  const [scanning, setScanning] = useState(false);
  const [toDelete, setToDelete] = useState<Media | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [openingJob, setOpeningJob] = useState<string | null>(null);

  const mediaQ = useApi(`media:${kind}:${offset}:${libraryVersion}`, (s) =>
    api.listMedia({ limit: PAGE, offset, kind: kind === 'all' ? null : kind }, s),
  );
  const jobsQ = useApi(`jobs:${libraryVersion}`, (s) => api.jobs(s));
  const page = mediaQ.data ?? mediaQ.stale;
  const jobs = (jobsQ.data ?? jobsQ.stale)?.jobs ?? [];
  const anyRunning = jobs.some((j) => !TERMINAL_STATUSES.includes(j.status));
  const { reload: reloadJobs } = jobsQ;
  const total = page?.total ?? 0;
  const items = page?.media ?? [];

  useEffect(() => {
    if (!anyRunning) return;
    const t = window.setInterval(reloadJobs, 4000);
    return () => window.clearInterval(t);
  }, [anyRunning, reloadJobs]);

  const startScan = async () => {
    const q = scanQuery.trim();
    if (!q) return;
    setScanning(true);
    try {
      const res = await api.scan({ query: q, options: { verify: scanVerify === 'auto' ? 'auto' : scanVerify === 'on' } });
      scanJob.track(res.job.id, res.job);
      jobsQ.reload();
    } catch (err) {
      actions.toastError(err, 'Scan failed');
    } finally {
      setScanning(false);
    }
  };

  const confirmDelete = async () => {
    if (!toDelete) return;
    setDeleting(true);
    try {
      await api.deleteMedia(toDelete.id);
      // Deleting the last item of a page moves back to the previous page instead of showing an empty one.
      if (items.length === 1 && offset > 0) setOffset(Math.max(0, offset - PAGE));
      actions.removeMedia(toDelete.id);
      actions.bumpPersons();
      actions.toast(`Deleted ${toDelete.name}`, 'success');
      setToDelete(null);
    } catch (err) {
      actions.toastError(err, 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  const openJob = async (job: Job) => {
    if (!TERMINAL_STATUSES.includes(job.status)) {
      scanJob.track(job.id, job);
      return;
    }
    setOpeningJob(job.id);
    try {
      const res = await api.job(job.id);
      const media = res.media ?? [];
      media.forEach((m) => actions.cacheMedia(m));
      actions.setCollection({ title: `${job.kind === 'scan' ? 'Scan' : 'Analysis'} ${shortId(job.id)}`, media });
      if (media[0]) actions.openMedia(media[0].id);
      else actions.toast('This job produced no media', 'info');
    } catch (err) {
      actions.toastError(err, 'Could not load job');
    } finally {
      setOpeningJob(null);
    }
  };


  return (
    <div className="flex flex-col gap-4 p-4">
      <Section title="Scan library">
        <form
          className="flex flex-col gap-2 rounded-[12px] border border-border bg-bg-card p-3"
          onSubmit={(e) => {
            e.preventDefault();
            void startScan();
          }}
        >
          <div className="text-[11px] text-text-muted">Re-run open-vocabulary detection over all stored media and add new entities.</div>
          <input className={inputClass} value={scanQuery} onChange={(e) => setScanQuery(e.target.value)} placeholder='e.g. "dangerous things", "bicycles"' />
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">Verify</span>
            <Segmented
              value={scanVerify}
              options={[
                { value: 'auto', label: 'Auto' },
                { value: 'on', label: 'On' },
                { value: 'off', label: 'Off' },
              ]}
              onChange={setScanVerify}
              size="xs"
            />
            <Button type="submit" size="sm" className="ml-auto" loading={scanning} disabled={!scanQuery.trim() || scanJob.state.streaming}>
              Scan library
            </Button>
          </div>
        </form>
        {scanJob.state.jobId && <JobPanel state={scanJob.state} onCancel={scanJob.cancel} onDismiss={scanJob.reset} />}
      </Section>

      <Section
        title={`Media${total ? ` (${total})` : ''}`}
        right={
          <Segmented
            value={kind}
            options={[
              { value: 'all', label: 'All' },
              { value: 'image', label: 'Images' },
              { value: 'video', label: 'Videos' },
            ]}
            onChange={(v) => {
              setKind(v);
              setOffset(0);
            }}
            size="xs"
          />
        }
      >
        {mediaQ.loading && !page && <Loading />}
        {!!mediaQ.error && <ErrorState error={mediaQ.error} onRetry={mediaQ.reload} />}
        {page && items.length === 0 && <EmptyState icon="🗂" title="No media yet" hint="Analyze some images or videos to fill the library." />}
        {items.length > 0 && (
          <div className={cn('grid grid-cols-3 gap-2', mediaQ.loading && 'opacity-60')}>
            {items.map((m) => (
              <MediaCard
                key={m.id}
                m={m}
                active={m.id === currentMediaId}
                onOpen={() => {
                  actions.setCollection({ title: 'Library', media: items });
                  actions.openMedia(m.id);
                }}
                onDelete={() => setToDelete(m)}
              />
            ))}
          </div>
        )}
        {total > PAGE && (
          <div className="flex items-center justify-between text-xs text-text-muted">
            <Button size="xs" variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
              ← Prev
            </Button>
            <span>
              {offset + 1}–{Math.min(total, offset + PAGE)} of {total}
            </span>
            <Button size="xs" variant="secondary" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
              Next →
            </Button>
          </div>
        )}
      </Section>

      <Section
        title="Recent jobs"
        right={
          <Button size="xs" variant="ghost" onClick={jobsQ.reload} loading={jobsQ.loading}>
            ↻
          </Button>
        }
      >
        {!!jobsQ.error && <ErrorState error={jobsQ.error} onRetry={jobsQ.reload} />}
        {jobsQ.data && jobs.length === 0 && <div className="text-xs text-text-muted">No jobs yet.</div>}
        <div className="flex flex-col gap-1">
          {jobs.slice(0, 25).map((j) => {
            const running = !TERMINAL_STATUSES.includes(j.status);
            return (
              <button
                key={j.id}
                type="button"
                onClick={() => void openJob(j)}
                disabled={openingJob === j.id}
                className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-border bg-bg-card hover:border-accent text-left text-xs"
                title={running ? 'Follow live progress' : 'Open the media from this job'}
              >
                <StatusBadge status={j.status} />
                <span className="text-text-secondary">{j.kind}</span>
                <span className="truncate flex-1 text-text-muted">
                  {j.options?.query ? `“${j.options.query}”` : ''} {j.media_ids.length ? `· ${j.media_ids.length} media` : ''}
                </span>
                {running && j.progress && j.progress.total > 0 && (
                  <span className="font-mono text-[10px] text-text-muted">
                    {j.progress.done}/{j.progress.total}
                  </span>
                )}
                <span className="text-[10px] text-text-muted shrink-0">{relativeTime(j.created_at)}</span>
              </button>
            );
          })}
        </div>
      </Section>

      <ConfirmModal
        open={toDelete !== null}
        title="Delete media?"
        message={
          <>
            This permanently deletes <b className="text-text-primary">{toDelete?.name}</b>
            {toDelete?.entity_count ? ` and its ${toDelete.entity_count} entities` : ''}, and updates person clusters.
          </>
        }
        loading={deleting}
        onConfirm={() => void confirmDelete()}
        onClose={() => setToDelete(null)}
      />
    </div>
  );
}
