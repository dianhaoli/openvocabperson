import { useContext } from 'react';
import { ActionsContext, StateContext, type AppActions, type AppState } from './state';

export function useAppState(): AppState {
  const ctx = useContext(StateContext);
  if (!ctx) throw new Error('useAppState must be used inside <AppProvider>');
  return ctx;
}

export function useActions(): AppActions {
  const ctx = useContext(ActionsContext);
  if (!ctx) throw new Error('useActions must be used inside <AppProvider>');
  return ctx;
}
