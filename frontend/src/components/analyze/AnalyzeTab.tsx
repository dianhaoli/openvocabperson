import { useState } from 'react';
import { api } from '../../api/client';
import { useActions } from '../../context/hooks';
import { useObjectUrl } from '../../hooks/useObjectUrl';
import { useWorkspaceJob } from '../../hooks/useWorkspaceJob';
import type { AnalyzeOptions, DescribeMode, Plan, Tristate } from '../../types';
import { Button } from '../ui/Button';
import { Dropzone } from '../ui/Dropzone';
import { Chip, Field, Section, Segmented, inputClass } from '../ui/Misc';
import { JobPanel } from '../jobs/JobPanel';
import { PlanView } from '../jobs/PlanView';

type Tri = 'auto' | 'on' | 'off';
const TRI_OPTIONS: { value: Tri; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'on', label: 'On' },
  { value: 'off', label: 'Off' },
];
const triToValue = (t: Tri): Tristate => (t === 'auto' ? 'auto' : t === 'on');

const EXAMPLES = ['people', 'dogs, cats', 'dangerous things', 'all living things', 'vehicles', 'people wearing red', 'food'];

function FileItem({ file, onRemove }: { file: File; onRemove: () => void }) {
  const isImage = file.type.startsWith('image/');
  const url = useObjectUrl(isImage ? file : null);
  return (
    <div className="relative group w-[60px] shrink-0" title={`${file.name} · ${(file.size / 1e6).toFixed(1)} MB`}>
      <div className="w-[60px] h-[60px] rounded-lg overflow-hidden border border-border bg-bg-primary flex items-center justify-center">
        {isImage && url ? (
          <img src={url} alt={file.name} className="w-full h-full object-cover" />
        ) : (
          <span className="text-xl">{file.type.startsWith('video/') ? '🎬' : '📄'}</span>
        )}
      </div>
      <div className="text-[9.5px] text-text-muted truncate mt-0.5">{file.name}</div>
      <button
        type="button"
        onClick={onRemove}
        className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-bg-tertiary border border-border text-[11px] leading-none opacity-0 group-hover:opacity-100 hover:bg-error hover:text-white transition"
        aria-label={`Remove ${file.name}`}
      >
        ×
      </button>
    </div>
  );
}

export function AnalyzeTab() {
  const { toastError } = useActions();
  const job = useWorkspaceJob('Analysis');

  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState('');
  const [expand, setExpand] = useState<Tri>('auto');
  const [verify, setVerify] = useState<Tri>('auto');
  const [segment, setSegment] = useState(false);
  const [describe, setDescribe] = useState<DescribeMode>('persons');
  const [conf, setConf] = useState(0.25);
  const [videoFps, setVideoFps] = useState(2);
  const [plan, setPlan] = useState<{ query: string; plan: Plan } | null>(null);
  const [planning, setPlanning] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const hasVideo = files.some((f) => f.type.startsWith('video/'));
  const jobActive = job.state.streaming;

  const addFiles = (incoming: File[]) => {
    setFiles((prev) => {
      const keys = new Set(prev.map((f) => `${f.name}:${f.size}:${f.lastModified}`));
      return [...prev, ...incoming.filter((f) => !keys.has(`${f.name}:${f.size}:${f.lastModified}`))];
    });
  };

  const previewPlan = async () => {
    setPlanning(true);
    try {
      const p = await api.plan(query.trim(), triToValue(expand));
      setPlan({ query: query.trim(), plan: p });
    } catch (err) {
      toastError(err, 'Plan preview failed');
    } finally {
      setPlanning(false);
    }
  };

  const analyze = async () => {
    if (!files.length) return;
    const options: AnalyzeOptions = {
      query: query.trim() || null,
      expand: triToValue(expand),
      verify: triToValue(verify),
      segment,
      describe,
      conf,
    };
    if (hasVideo) options.video_fps = videoFps;
    setSubmitting(true);
    try {
      const res = await api.analyze(files, options);
      job.track(res.job.id, res.job);
      setFiles([]);
    } catch (err) {
      toastError(err, 'Analyze failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <Section title="Media">
        <Dropzone accept="image/*,video/*" multiple onFiles={addFiles} compact={files.length > 0}>
          {files.length === 0 ? (
            <div className="flex flex-col items-center gap-1">
              <div className="text-2xl">⬆</div>
              <div className="text-sm text-text-secondary">Drop images or videos here</div>
              <div className="text-[11px] text-text-muted">or click to browse · multiple files supported</div>
            </div>
          ) : (
            <div className="text-xs text-text-secondary">+ Add more files</div>
          )}
        </Dropzone>
        {files.length > 0 && (
          <div className="flex flex-col gap-1">
            <div className="flex gap-2 overflow-x-auto pt-2 pb-1">
              {files.map((f, i) => (
                <FileItem key={`${f.name}:${f.size}:${f.lastModified}`} file={f} onRemove={() => setFiles((prev) => prev.filter((_, j) => j !== i))} />
              ))}
            </div>
            <div className="flex justify-between text-[11px] text-text-muted">
              <span>
                {files.length} file(s) · {(files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(1)} MB
              </span>
              <button type="button" className="hover:text-text-primary" onClick={() => setFiles([])}>
                Clear
              </button>
            </div>
          </div>
        )}
      </Section>

      <Section title="What to find">
        <input
          className={inputClass}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={'people, dogs — or "dangerous things", "all living things"'}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && files.length && !submitting) void analyze();
          }}
        />
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <Chip key={ex} onClick={() => setQuery(ex)} active={query === ex}>
              {ex}
            </Chip>
          ))}
        </div>
        <div className="text-[10.5px] text-text-muted">Leave empty to detect the default (COCO) vocabulary.</div>
      </Section>

      <Section title="Options">
        <div className="flex flex-col gap-1 rounded-[12px] border border-border bg-bg-card px-3 py-2">
          <Field label="Expand with LLM" hint="abstract concepts → prompts">
            <Segmented value={expand} options={TRI_OPTIONS} onChange={setExpand} size="xs" />
          </Field>
          <Field label="Verify with VLM" hint="yes/no check per candidate">
            <Segmented value={verify} options={TRI_OPTIONS} onChange={setVerify} size="xs" />
          </Field>
          <Field label="Segment (masks)">
            <Segmented
              value={segment ? 'on' : 'off'}
              options={[
                { value: 'off', label: 'Off' },
                { value: 'on', label: 'On' },
              ]}
              onChange={(v) => setSegment(v === 'on')}
              size="xs"
            />
          </Field>
          <Field label="Describe">
            <Segmented
              value={describe}
              options={[
                { value: 'persons', label: 'Persons' },
                { value: 'all', label: 'All' },
                { value: 'none', label: 'None' },
              ]}
              onChange={setDescribe}
              size="xs"
            />
          </Field>
          <Field label={`Confidence ≥ ${conf.toFixed(2)}`}>
            <input
              type="range"
              min={0.05}
              max={0.9}
              step={0.05}
              value={conf}
              onChange={(e) => setConf(Number(e.target.value))}
              className="w-32 accent-[#6366f1]"
            />
          </Field>
          {hasVideo && (
            <Field label={`Video sampling: ${videoFps} fps`}>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.5}
                value={videoFps}
                onChange={(e) => setVideoFps(Number(e.target.value))}
                className="w-32 accent-[#6366f1]"
              />
            </Field>
          )}
        </div>
      </Section>

      <div className="flex gap-2">
        <Button variant="secondary" size="md" onClick={() => void previewPlan()} loading={planning} className="flex-1">
          Preview plan
        </Button>
        <Button size="md" onClick={() => void analyze()} loading={submitting} disabled={!files.length || jobActive} className="flex-[1.4]">
          {jobActive ? 'Analyzing…' : `Analyze${files.length ? ` ${files.length}` : ''}`}
        </Button>
      </div>

      {plan && (
        <Section
          title={`Plan for “${plan.query || 'default vocabulary'}”`}
          right={
            <button type="button" className="text-[11px] text-text-muted hover:text-text-primary" onClick={() => setPlan(null)}>
              hide
            </button>
          }
        >
          <PlanView plan={plan.plan} />
        </Section>
      )}

      {job.state.jobId && (
        <Section title="Progress">
          <JobPanel state={job.state} onCancel={job.cancel} onDismiss={job.reset} />
        </Section>
      )}
    </div>
  );
}
