import { useCallback, useEffect, useRef, useState } from 'react';
import { isAbortError } from '../api/client';

interface Settled<T> {
  key: string;
  data?: T;
  error?: unknown;
}

export interface ApiState<T> {
  /** Data for the current key (undefined while loading a new key). */
  data: T | undefined;
  /** Most recent data for any key (useful to avoid flashing while reloading). */
  stale: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
  /** Locally patch the current data (optimistic updates). */
  mutate: (fn: (prev: T) => T) => void;
}

/**
 * Fetch data for `key` (null = idle). Changing the key refetches and aborts the
 * previous request. The fetcher is read through a ref so it may close over fresh values.
 */
export function useApi<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>): ApiState<T> {
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const [nonce, setNonce] = useState(0);
  const [state, setState] = useState<Settled<T> | null>(null);
  const fullKey = key == null ? null : `${key}#${nonce}`;

  useEffect(() => {
    if (fullKey == null) return;
    const controller = new AbortController();
    fetcherRef.current(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ key: fullKey, data });
      },
      (error: unknown) => {
        if (!controller.signal.aborted && !isAbortError(error)) setState({ key: fullKey, error });
      },
    );
    return () => controller.abort();
  }, [fullKey]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const mutate = useCallback((fn: (prev: T) => T) => {
    setState((prev) => (prev && prev.data !== undefined ? { ...prev, data: fn(prev.data) } : prev));
  }, []);

  const settled = state != null && state.key === fullKey;
  return {
    data: settled ? state.data : undefined,
    stale: state?.data,
    error: settled ? state.error : undefined,
    loading: fullKey != null && !settled,
    reload,
    mutate,
  };
}
