import { useCallback, useEffect, useReducer, useRef } from 'react';
import { api, errorMessage, isAbortError } from '../api/client';
import type { Job, JobEvent, JobStatus, Media, MediaDetail, Plan } from '../types';

export interface LogEntry {
  id: number;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
  ts: number;
}

export interface TrackedMedia {
  media: Media;
  done: boolean;
  entityCount: number;
}

export interface JobTrackerState {
  jobId: string | null;
  job: Job | null;
  status: JobStatus | null;
  plan: Plan | null;
  log: LogEntry[];
  done: number;
  total: number;
  stage: string | null;
  current: string | null;
  media: TrackedMedia[];
  streaming: boolean;
  connectionError: string | null;
}

const initial: JobTrackerState = {
  jobId: null,
  job: null,
  status: null,
  plan: null,
  log: [],
  done: 0,
  total: 0,
  stage: null,
  current: null,
  media: [],
  streaming: false,
  connectionError: null,
};

type TrackerAction =
  | { type: 'start'; jobId: string; job: Job | null }
  | { type: 'reconnect' }
  | { type: 'event'; evt: JobEvent }
  | { type: 'final'; job: Job; media: MediaDetail[] }
  | { type: 'streaming'; value: boolean }
  | { type: 'connectionError'; message: string | null }
  | { type: 'reset' };

let logSeq = 0;

function addLog(state: JobTrackerState, level: LogEntry['level'], message: string, ts?: number): JobTrackerState {
  const entry: LogEntry = { id: ++logSeq, level, message, ts: ts ?? Date.now() / 1000 };
  const log = state.log.length > 500 ? [...state.log.slice(-400), entry] : [...state.log, entry];
  return { ...state, log };
}

function upsertMedia(list: TrackedMedia[], media: Media, done: boolean, entityCount?: number): TrackedMedia[] {
  const idx = list.findIndex((m) => m.media.id === media.id);
  const item: TrackedMedia = {
    media,
    done: done || (idx >= 0 && list[idx].done),
    entityCount: entityCount ?? (idx >= 0 ? list[idx].entityCount : media.entity_count),
  };
  if (idx === -1) return [...list, item];
  const next = list.slice();
  next[idx] = item;
  return next;
}

function applyJob(state: JobTrackerState, job: Job): JobTrackerState {
  return {
    ...state,
    job,
    status: job.status,
    done: job.progress?.done ?? state.done,
    total: job.progress?.total ?? state.total,
    stage: job.progress?.stage ?? state.stage,
    current: job.progress?.message ?? state.current,
  };
}

function reduceEvent(state: JobTrackerState, evt: JobEvent): JobTrackerState {
  switch (evt.type) {
    case 'job_started':
      return addLog(applyJob(state, evt.job), 'info', `Job ${evt.job.id} started (${evt.job.kind})`, evt.ts);
    case 'plan': {
      const p = evt.plan;
      const msg = p.prompts.length
        ? `Plan (${p.source}): ${p.concepts.length} concept(s) → ${p.prompts.length} prompt(s)`
        : `Plan (${p.source}): default vocabulary`;
      return addLog({ ...state, plan: p }, 'info', msg, evt.ts);
    }
    case 'media_started':
      return addLog(
        { ...state, current: `${evt.name}`, total: Math.max(state.total, evt.total) },
        'info',
        `[${Math.min(evt.index + 1, evt.total)}/${evt.total}] ${evt.name} (${evt.kind})`,
        evt.ts,
      );
    case 'media_created':
      return { ...state, media: upsertMedia(state.media, evt.media, false) };
    case 'stage':
      return addLog({ ...state, stage: evt.stage, current: evt.message || state.current }, 'info', `${evt.stage}: ${evt.message}`, evt.ts);
    case 'detections': {
      const labels = Object.entries(evt.labels ?? {})
        .sort((a, b) => b[1] - a[1])
        .map(([l, c]) => `${l}×${c}`)
        .join(', ');
      return addLog(state, 'info', `${evt.count} detection(s)${labels ? `: ${labels}` : ''}`, evt.ts);
    }
    case 'entity':
      return state;
    case 'media_done':
      return addLog(
        { ...state, media: upsertMedia(state.media, evt.media, true, evt.media.entities?.length ?? evt.media.entity_count) },
        'success',
        `✓ ${evt.media.name} — ${evt.media.entities?.length ?? evt.media.entity_count} entit${(evt.media.entities?.length ?? evt.media.entity_count) === 1 ? 'y' : 'ies'}`,
        evt.ts,
      );
    case 'progress':
      return { ...state, done: evt.done, total: evt.total };
    case 'log':
      return addLog(state, evt.level, evt.message, evt.ts);
    case 'job_done': {
      const next = applyJob(state, evt.job);
      const level = evt.job.status === 'done' ? 'success' : evt.job.status === 'error' ? 'error' : 'warning';
      const msg =
        evt.job.status === 'error'
          ? `Job failed: ${evt.job.error ?? 'unknown error'}`
          : evt.job.status === 'cancelled'
            ? 'Job cancelled'
            : 'Job finished';
      return addLog({ ...next, streaming: false, stage: null }, level, msg, evt.ts);
    }
    default:
      return state;
  }
}

function trackerReducer(state: JobTrackerState, action: TrackerAction): JobTrackerState {
  switch (action.type) {
    case 'start':
      return {
        ...initial,
        jobId: action.jobId,
        job: action.job,
        status: action.job?.status ?? 'queued',
        streaming: true,
      };
    case 'reconnect':
      // The events endpoint replays past events, so start the derived state over.
      return { ...initial, jobId: state.jobId, job: state.job, status: state.status, streaming: true };
    case 'event':
      return action.evt.job_id && state.jobId && action.evt.job_id !== state.jobId ? state : reduceEvent(state, action.evt);
    case 'final': {
      let next = applyJob(state, action.job);
      for (const m of action.media) next = { ...next, media: upsertMedia(next.media, m, true, m.entities?.length) };
      return { ...next, streaming: false };
    }
    case 'streaming':
      return { ...state, streaming: action.value };
    case 'connectionError':
      return action.message ? addLog({ ...state, connectionError: action.message }, 'error', action.message) : { ...state, connectionError: null };
    case 'reset':
      return initial;
    default:
      return state;
  }
}

export const TERMINAL_STATUSES: JobStatus[] = ['done', 'error', 'cancelled'];

export interface JobTrackerCallbacks {
  onMediaCreated?: (media: Media) => void;
  onMediaDone?: (media: MediaDetail) => void;
  onEntity?: (evt: Extract<JobEvent, { type: 'entity' }>) => void;
  onFinished?: (job: Job) => void;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = window.setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(t);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

/**
 * Follows a job's SSE stream (`GET /api/jobs/{id}/events`), reconnecting on
 * dropped connections and aborting on unmount / reset.
 */
export function useJobTracker(callbacks: JobTrackerCallbacks = {}) {
  const [state, dispatch] = useReducer(trackerReducer, initial);
  const controllerRef = useRef<AbortController | null>(null);
  const cbRef = useRef(callbacks);
  useEffect(() => {
    cbRef.current = callbacks;
  });

  useEffect(() => () => controllerRef.current?.abort(), []);

  const run = useCallback(async (jobId: string, controller: AbortController) => {
    const { signal } = controller;
    let attempt = 0;
    let first = true;
    while (!signal.aborted) {
      if (!first) dispatch({ type: 'reconnect' });
      first = false;
      let finished = false;
      try {
        for await (const evt of api.jobEvents(jobId, signal)) {
          dispatch({ type: 'event', evt });
          attempt = 0;
          if (evt.type === 'media_created') cbRef.current.onMediaCreated?.(evt.media);
          else if (evt.type === 'media_done') cbRef.current.onMediaDone?.(evt.media);
          else if (evt.type === 'entity') cbRef.current.onEntity?.(evt);
          else if (evt.type === 'job_done') {
            finished = true;
            cbRef.current.onFinished?.(evt.job);
            break;
          }
        }
      } catch (err) {
        if (signal.aborted || isAbortError(err)) return;
        dispatch({ type: 'connectionError', message: `Event stream error: ${errorMessage(err)}` });
      }
      if (finished || signal.aborted) {
        dispatch({ type: 'streaming', value: false });
        return;
      }
      // Stream ended without job_done: check the job before reconnecting.
      try {
        const res = await api.job(jobId, signal);
        if (TERMINAL_STATUSES.includes(res.job.status)) {
          dispatch({ type: 'final', job: res.job, media: res.media ?? [] });
          for (const m of res.media ?? []) cbRef.current.onMediaDone?.(m);
          cbRef.current.onFinished?.(res.job);
          return;
        }
      } catch (err) {
        if (signal.aborted || isAbortError(err)) return;
      }
      attempt += 1;
      if (attempt > 6) {
        dispatch({ type: 'connectionError', message: 'Lost connection to job events; giving up.' });
        dispatch({ type: 'streaming', value: false });
        return;
      }
      try {
        await sleep(Math.min(8000, 750 * attempt), signal);
      } catch {
        return;
      }
    }
  }, []);

  const track = useCallback(
    (jobId: string, job: Job | null = null) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      dispatch({ type: 'start', jobId, job });
      void run(jobId, controller);
    },
    [run],
  );

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    dispatch({ type: 'streaming', value: false });
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    dispatch({ type: 'reset' });
  }, []);

  const cancel = useCallback(async () => {
    const id = state.jobId;
    if (!id) return;
    await api.cancelJob(id);
    // job_done (status cancelled) will arrive on the stream.
  }, [state.jobId]);

  return { state, track, stop, reset, cancel };
}
