import { useEffect, useRef } from 'react';
import { checkHealth } from '../api/client';
import { useApp } from '../context/AppContext';

/**
 * Hook to monitor pipeline health status.
 * Polls the health endpoint until pipeline is ready.
 */
export function useHealth() {
  const { isPipelineReady, setPipelineReady } = useApp();
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const data = await checkHealth();
        if (!cancelled) {
          if (data.pipeline_ready) {
            setPipelineReady(true);
          } else {
            // Retry in 3 seconds
            timeoutRef.current = window.setTimeout(check, 3000);
          }
        }
      } catch {
        if (!cancelled) {
          // Retry in 5 seconds on error
          timeoutRef.current = window.setTimeout(check, 5000);
        }
      }
    }

    if (!isPipelineReady) {
      check();
    }

    return () => {
      cancelled = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [isPipelineReady, setPipelineReady]);

  return { isPipelineReady };
}

export default useHealth;

