import { useCallback, useRef } from 'react';
import { useActions } from '../context/hooks';
import type { Job } from '../types';
import { useJobTracker } from './useJobTracker';

/**
 * A job tracker wired to the workspace: processed media are cached, added to the
 * filmstrip collection, and the first finished media is opened automatically.
 */
export function useWorkspaceJob(label: string, opts: { autoOpen?: boolean } = {}) {
  const actions = useActions();
  const titleRef = useRef<string>('');
  const openedRef = useRef(false);
  const autoOpen = opts.autoOpen ?? true;

  const tracker = useJobTracker({
    onMediaCreated: (media) => actions.addToCollection(titleRef.current, media),
    onMediaDone: (media) => {
      actions.cacheMedia(media);
      actions.addToCollection(titleRef.current, media);
      if (autoOpen && !openedRef.current) {
        openedRef.current = true;
        actions.openMedia(media.id);
      }
    },
    onEntity: (evt) => actions.upsertEntity(evt.entity),
    onFinished: (job) => {
      actions.bumpLibrary();
      actions.bumpPersons();
      if (job.status === 'done') actions.toast(`${label} finished`, 'success');
      else if (job.status === 'error') actions.toast(`${label} failed: ${job.error ?? 'unknown error'}`, 'error');
      else if (job.status === 'cancelled') actions.toast(`${label} cancelled`, 'info');
    },
  });

  const { track: rawTrack } = tracker;
  const track = useCallback(
    (jobId: string, job: Job | null = null) => {
      titleRef.current = `${label} · ${jobId}`;
      openedRef.current = false;
      actions.setCollection({ title: titleRef.current, media: [] });
      rawTrack(jobId, job);
    },
    [actions, label, rawTrack],
  );

  return { ...tracker, track };
}
