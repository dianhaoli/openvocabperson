import { useEffect, useMemo, useReducer, type ReactNode } from 'react';
import { api, errorMessage, isAbortError } from '../api/client';
import { ActionsContext, StateContext, initialState, reducer, type AppActions } from './state';

let toastSeq = 0;
const HEALTH_INTERVAL_MS = 15000;

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const actions = useMemo<AppActions>(() => {
    const toast: AppActions['toast'] = (message, kind = 'info') => {
      const id = ++toastSeq;
      dispatch({ type: 'pushToast', toast: { id, kind, message } });
      window.setTimeout(() => dispatch({ type: 'dismissToast', id }), kind === 'error' ? 7000 : 4000);
    };
    return {
      setTab: (tab) => dispatch({ type: 'setTab', tab }),
      openMedia: (mediaId, entityId) => dispatch({ type: 'openMedia', mediaId, entityId }),
      closeMedia: () => dispatch({ type: 'closeMedia' }),
      cacheMedia: (media) => dispatch({ type: 'cacheMedia', media }),
      upsertEntity: (entity) => dispatch({ type: 'upsertEntity', entity }),
      removeEntity: (entityId, mediaId) => dispatch({ type: 'removeEntity', entityId, mediaId }),
      removeMedia: (mediaId) => dispatch({ type: 'removeMedia', mediaId }),
      selectEntity: (entityId) => dispatch({ type: 'selectEntity', entityId }),
      setCollection: (collection) => dispatch({ type: 'setCollection', collection }),
      addToCollection: (title, media) => dispatch({ type: 'addToCollection', title, media }),
      openPerson: (personId) => dispatch({ type: 'openPerson', personId }),
      setModelsOpen: (open) => dispatch({ type: 'setModelsOpen', open }),
      toast,
      toastError: (err, prefix) => {
        if (isAbortError(err)) return;
        const msg = errorMessage(err);
        toast(prefix ? `${prefix}: ${msg}` : msg, 'error');
      },
      dismissToast: (id) => dispatch({ type: 'dismissToast', id }),
      setHealth: (health, error) => dispatch({ type: 'setHealth', health, error }),
      bumpPersons: () => dispatch({ type: 'bumpPersons' }),
      bumpLibrary: () => dispatch({ type: 'bumpLibrary' }),
      setProcessing: (mediaIds, processing) => dispatch({ type: 'setProcessing', mediaIds, processing }),
    };
  }, []);

  // Health polling
  useEffect(() => {
    let cancelled = false;
    let controller: AbortController | null = null;
    const poll = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const h = await api.health(controller.signal);
        if (!cancelled) dispatch({ type: 'setHealth', health: h, error: false });
      } catch (err) {
        if (!cancelled && !isAbortError(err)) dispatch({ type: 'setHealth', health: null, error: true });
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), HEALTH_INTERVAL_MS);
    return () => {
      cancelled = true;
      controller?.abort();
      window.clearInterval(timer);
    };
  }, []);

  return (
    <ActionsContext.Provider value={actions}>
      <StateContext.Provider value={state}>{children}</StateContext.Provider>
    </ActionsContext.Provider>
  );
}
