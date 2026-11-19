import { useCallback, useRef } from 'react';
import { useActions } from '../context/hooks';
import type { Job } from '../types';
import { shortId } from '../utils/format';
import { useJobTracker } from './useJobTracker';

/**
 * A job tracker wired to the workspace: processed media are cached, added to the
 * filmstrip collection and flagged as processing until done, and the first media
 * is opened as soon as it is created.
 */
export function useWorkspaceJob(label: string, opts: { autoOpen?: boolean } = {}) {
  const actions = useActions();
  const titleRef = useRef<string>('');
  const openedRef = useRef(false);
  /** Media of the followed job that are created but not finished yet. */
  const processingRef = useRef<Set<string>>(new Set());
  const autoOpen = opts.autoOpen ?? true;

  const clearProcessing = useCallback(() => {
    actions.setProcessing([...processingRef.current], false);
    processingRef.current.clear();
  }, [actions]);

  const tracker = useJobTracker({
    onMediaCreated: (media) => {
      processingRef.current.add(media.id);
      actions.setProcessing([media.id], true);
      actions.addToCollection(titleRef.current, media);
      // Show the first media as soon as it exists; its entities stream in via `entity` events.
      if (autoOpen && !openedRef.current) {
        openedRef.current = true;
        actions.cacheMedia({ ...media, entities: [], plan: null });
        actions.openMedia(media.id);
      }
    },
    onMediaDone: (media) => {
      processingRef.current.delete(media.id);
      actions.setProcessing([media.id], false);
      actions.cacheMedia(media);
      actions.addToCollection(titleRef.current, media);
      if (autoOpen && !openedRef.current) {
        openedRef.current = true;
        actions.openMedia(media.id);
      }
    },
    onEntity: (evt) => actions.upsertEntity(evt.entity),
    onFinished: (job) => {
      clearProcessing();
      actions.bumpLibrary();
      actions.bumpPersons();
      if (job.status === 'done') actions.toast(`${label} finished`, 'success');
      else if (job.status === 'error') actions.toast(`${label} failed: ${job.error ?? 'unknown error'}`, 'error');
      else if (job.status === 'cancelled') actions.toast(`${label} cancelled`, 'info');
    },
  });

  const { track: rawTrack, reset: rawReset } = tracker;
  const track = useCallback(
    (jobId: string, job: Job | null = null) => {
      clearProcessing();
      titleRef.current = `${label} ${shortId(jobId)}`;
      openedRef.current = false;
      actions.setCollection({ title: titleRef.current, media: [] });
      rawTrack(jobId, job);
    },
    [actions, clearProcessing, label, rawTrack],
  );

  const reset = useCallback(() => {
    clearProcessing();
    rawReset();
  }, [clearProcessing, rawReset]);

  return { ...tracker, track, reset };
}
