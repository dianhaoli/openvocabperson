import { createContext } from 'react';
import type { Entity, Health, Media, MediaDetail } from '../types';

export type TabId = 'analyze' | 'agent' | 'search' | 'library' | 'people';

export type ToastKind = 'error' | 'success' | 'info';
export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

export interface Collection {
  title: string;
  media: Media[];
}

export interface AppState {
  tab: TabId;
  /** Latest known MediaDetail per media id */
  mediaCache: Record<string, MediaDetail>;
  currentMediaId: string | null;
  selectedEntityId: string | null;
  /** Filmstrip shown above the workspace */
  collection: Collection | null;
  personModalId: string | null;
  modelsOpen: boolean;
  toasts: Toast[];
  health: Health | null;
  healthError: boolean;
  /** Bumped whenever persons/identities may have changed */
  personsVersion: number;
  /** Bumped whenever the media library / jobs may have changed */
  libraryVersion: number;
}

export const initialState: AppState = {
  tab: 'analyze',
  mediaCache: {},
  currentMediaId: null,
  selectedEntityId: null,
  collection: null,
  personModalId: null,
  modelsOpen: false,
  toasts: [],
  health: null,
  healthError: false,
  personsVersion: 0,
  libraryVersion: 0,
};

export type Action =
  | { type: 'setTab'; tab: TabId }
  | { type: 'openMedia'; mediaId: string; entityId?: string | null }
  | { type: 'closeMedia' }
  | { type: 'cacheMedia'; media: MediaDetail }
  | { type: 'upsertEntity'; entity: Entity }
  | { type: 'removeEntity'; entityId: string; mediaId: string }
  | { type: 'removeMedia'; mediaId: string }
  | { type: 'selectEntity'; entityId: string | null }
  | { type: 'setCollection'; collection: Collection | null }
  | { type: 'addToCollection'; title: string; media: Media }
  | { type: 'openPerson'; personId: string | null }
  | { type: 'setModelsOpen'; open: boolean }
  | { type: 'pushToast'; toast: Toast }
  | { type: 'dismissToast'; id: number }
  | { type: 'setHealth'; health: Health | null; error: boolean }
  | { type: 'bumpPersons' }
  | { type: 'bumpLibrary' };

function upsertInList(list: Entity[], entity: Entity): Entity[] {
  const idx = list.findIndex((e) => e.id === entity.id);
  if (idx === -1) return [...list, entity];
  const next = list.slice();
  next[idx] = entity;
  return next;
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'setTab':
      return { ...state, tab: action.tab };
    case 'openMedia':
      return {
        ...state,
        currentMediaId: action.mediaId,
        selectedEntityId: action.entityId ?? null,
      };
    case 'closeMedia':
      return { ...state, currentMediaId: null, selectedEntityId: null };
    case 'cacheMedia':
      return { ...state, mediaCache: { ...state.mediaCache, [action.media.id]: action.media } };
    case 'upsertEntity': {
      const cached = state.mediaCache[action.entity.media_id];
      if (!cached) return state;
      return {
        ...state,
        mediaCache: {
          ...state.mediaCache,
          [cached.id]: { ...cached, entities: upsertInList(cached.entities, action.entity) },
        },
      };
    }
    case 'removeEntity': {
      const cached = state.mediaCache[action.mediaId];
      const selectedEntityId = state.selectedEntityId === action.entityId ? null : state.selectedEntityId;
      if (!cached) return { ...state, selectedEntityId };
      return {
        ...state,
        selectedEntityId,
        mediaCache: {
          ...state.mediaCache,
          [cached.id]: {
            ...cached,
            entities: cached.entities.filter((e) => e.id !== action.entityId),
            entity_count: Math.max(0, cached.entity_count - 1),
          },
        },
      };
    }
    case 'removeMedia': {
      const mediaCache = { ...state.mediaCache };
      delete mediaCache[action.mediaId];
      const isCurrent = state.currentMediaId === action.mediaId;
      return {
        ...state,
        mediaCache,
        currentMediaId: isCurrent ? null : state.currentMediaId,
        selectedEntityId: isCurrent ? null : state.selectedEntityId,
        collection: state.collection
          ? { ...state.collection, media: state.collection.media.filter((m) => m.id !== action.mediaId) }
          : null,
        libraryVersion: state.libraryVersion + 1,
      };
    }
    case 'selectEntity':
      return { ...state, selectedEntityId: action.entityId };
    case 'setCollection':
      return { ...state, collection: action.collection };
    case 'addToCollection': {
      const base =
        state.collection && state.collection.title === action.title
          ? state.collection
          : { title: action.title, media: [] };
      const idx = base.media.findIndex((m) => m.id === action.media.id);
      const media = base.media.slice();
      if (idx === -1) media.push(action.media);
      else media[idx] = action.media;
      return { ...state, collection: { title: base.title, media } };
    }
    case 'openPerson':
      return { ...state, personModalId: action.personId };
    case 'setModelsOpen':
      return { ...state, modelsOpen: action.open };
    case 'pushToast':
      return { ...state, toasts: [...state.toasts.slice(-4), action.toast] };
    case 'dismissToast':
      return { ...state, toasts: state.toasts.filter((t) => t.id !== action.id) };
    case 'setHealth':
      return { ...state, health: action.health ?? state.health, healthError: action.error };
    case 'bumpPersons':
      return { ...state, personsVersion: state.personsVersion + 1 };
    case 'bumpLibrary':
      return { ...state, libraryVersion: state.libraryVersion + 1 };
    default:
      return state;
  }
}

export interface AppActions {
  setTab: (tab: TabId) => void;
  openMedia: (mediaId: string, entityId?: string | null) => void;
  closeMedia: () => void;
  cacheMedia: (media: MediaDetail) => void;
  upsertEntity: (entity: Entity) => void;
  removeEntity: (entityId: string, mediaId: string) => void;
  removeMedia: (mediaId: string) => void;
  selectEntity: (entityId: string | null) => void;
  setCollection: (collection: Collection | null) => void;
  addToCollection: (title: string, media: Media) => void;
  openPerson: (personId: string | null) => void;
  setModelsOpen: (open: boolean) => void;
  toast: (message: string, kind?: ToastKind) => void;
  toastError: (err: unknown, prefix?: string) => void;
  dismissToast: (id: number) => void;
  setHealth: (health: Health | null, error: boolean) => void;
  bumpPersons: () => void;
  bumpLibrary: () => void;
}

export const StateContext = createContext<AppState | null>(null);
export const ActionsContext = createContext<AppActions | null>(null);
