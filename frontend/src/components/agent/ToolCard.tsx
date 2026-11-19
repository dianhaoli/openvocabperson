import { useState } from 'react';
import { useActions } from '../../context/hooks';
import { useWorkspaceJob } from '../../hooks/useWorkspaceJob';
import type { AgentToolResult } from '../../types';
import { formatJson, personName } from '../../utils/format';
import { EntityTile } from '../common/EntityTile';
import { Img } from '../common/Img';
import { JobPanel } from '../jobs/JobPanel';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';

const GALLERY_LIMIT = 12;

function JobFollow({ jobId }: { jobId: string }) {
  const job = useWorkspaceJob('Agent job', { autoOpen: false });
  if (job.state.jobId !== jobId) {
    return (
      <Button size="xs" variant="secondary" onClick={() => job.track(jobId)}>
        ▶ Follow job {jobId}
      </Button>
    );
  }
  return <JobPanel state={job.state} onCancel={job.cancel} onDismiss={job.reset} />;
}

export function ToolCard({ name, args, result }: { name: string; args: unknown; result: AgentToolResult | null }) {
  const { openMedia, openPerson } = useActions();
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const entities = result?.entities ?? [];
  const media = result?.media ?? [];
  const persons = result?.persons ?? [];
  const limit = showAll ? Infinity : GALLERY_LIMIT;
  const hidden = Math.max(0, entities.length - limit) + Math.max(0, media.length - limit) + Math.max(0, persons.length - limit);

  return (
    <div className="self-stretch rounded-xl border border-border bg-bg-secondary/80 text-xs overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-bg-tertiary/60"
      >
        <span className="text-text-muted">{open ? '▾' : '▸'}</span>
        <span className="font-mono text-[11px] text-[#a5a8ff]">{name}</span>
        {!result && <Spinner size={11} className="text-text-muted" />}
        {result?.summary && <span className="text-text-secondary truncate flex-1">{result.summary}</span>}
      </button>
      {open && (
        <div className="px-3 pb-2 flex flex-col gap-1.5">
          <div className="text-[10px] uppercase tracking-wider text-text-muted">Arguments</div>
          <pre className="font-mono text-[10.5px] bg-bg-primary border border-border rounded-lg p-2 overflow-x-auto max-h-48">
            {args == null ? '{}' : formatJson(args)}
          </pre>
          {result?.summary && (
            <>
              <div className="text-[10px] uppercase tracking-wider text-text-muted">Summary</div>
              <div className="text-text-secondary whitespace-pre-wrap">{result.summary}</div>
            </>
          )}
        </div>
      )}

      {(entities.length > 0 || media.length > 0 || persons.length > 0 || result?.job_id) && (
        <div className="px-3 pb-3 pt-1 flex flex-col gap-2">
          {entities.length > 0 && (
            <div className="grid grid-cols-4 gap-1.5">
              {entities.slice(0, limit).map((e) => (
                <EntityTile key={e.id} entity={e} onClick={() => openMedia(e.media_id, e.id)} />
              ))}
            </div>
          )}
          {media.length > 0 && (
            <div className="grid grid-cols-4 gap-1.5">
              {media.slice(0, limit).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => openMedia(m.id)}
                  title={m.name}
                  className="relative aspect-square rounded-lg overflow-hidden border border-border hover:border-accent"
                >
                  <Img src={m.thumb_url} alt={m.name} className="w-full h-full object-cover" />
                  {m.kind === 'video' && <span className="absolute top-1 left-1 text-[9px] bg-black/70 rounded px-1">▶</span>}
                  <span className="absolute bottom-0 inset-x-0 text-[9.5px] bg-black/70 px-1 truncate">{m.name}</span>
                </button>
              ))}
            </div>
          )}
          {persons.length > 0 && (
            <div className="grid grid-cols-4 gap-1.5">
              {persons.slice(0, limit).map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => openPerson(p.id)}
                  className="flex flex-col rounded-lg overflow-hidden border border-border bg-bg-card hover:border-accent"
                >
                  <div className="relative aspect-square">
                    <Img src={p.thumb_url ?? undefined} alt="" className="absolute inset-0 w-full h-full object-cover object-top" />
                    {p.is_watchlist && <span className="absolute top-1 right-1 text-[9px] bg-error text-white rounded px-1">★</span>}
                  </div>
                  <div className="px-1 py-0.5 text-[10.5px] truncate">{personName(p)}</div>
                  <div className="px-1 pb-1 text-[9.5px] text-text-muted">{p.sighting_count} sightings</div>
                </button>
              ))}
            </div>
          )}
          {hidden > 0 && (
            <button type="button" className="text-[11px] text-accent self-start" onClick={() => setShowAll(true)}>
              Show {hidden} more
            </button>
          )}
          {result?.job_id && <JobFollow jobId={result.job_id} />}
        </div>
      )}
    </div>
  );
}
