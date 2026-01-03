import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import type { Entity, SidebarTab, LogEntry } from '../types';

// ══════════════════════════════════════════════════════════════════════════════
// Context Types
// ══════════════════════════════════════════════════════════════════════════════

interface AppState {
  // File & Image
  selectedFile: File | null;
  currentImage: HTMLImageElement | null;
  currentImageUrl: string | null;
  sessionId: string | null;

  // Analysis
  entities: Entity[];
  selectedEntityId: string | null;
  isAnalyzing: boolean;
  analysisLogs: LogEntry[];
  analysisTime: number | null;

  // UI
  activeTab: SidebarTab;
  isPipelineReady: boolean;
}

interface AppActions {
  // File handling
  setSelectedFile: (file: File | null) => void;
  setCurrentImage: (img: HTMLImageElement | null, url: string | null) => void;
  setSessionId: (id: string | null) => void;

  // Analysis
  setEntities: (entities: Entity[]) => void;
  selectEntity: (id: string | null) => void;
  setIsAnalyzing: (analyzing: boolean) => void;
  addLog: (type: LogEntry['type'], message: string) => void;
  clearLogs: () => void;
  setAnalysisTime: (time: number | null) => void;

  // UI
  setActiveTab: (tab: SidebarTab) => void;
  setPipelineReady: (ready: boolean) => void;

  // Utilities
  reset: () => void;
  getSelectedEntity: () => Entity | undefined;
}

type AppContextValue = AppState & AppActions;

// ══════════════════════════════════════════════════════════════════════════════
// Initial State
// ══════════════════════════════════════════════════════════════════════════════

const initialState: AppState = {
  selectedFile: null,
  currentImage: null,
  currentImageUrl: null,
  sessionId: null,
  entities: [],
  selectedEntityId: null,
  isAnalyzing: false,
  analysisLogs: [],
  analysisTime: null,
  activeTab: 'upload',
  isPipelineReady: false,
};

// ══════════════════════════════════════════════════════════════════════════════
// Context
// ══════════════════════════════════════════════════════════════════════════════

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(initialState);

  // File handling
  const setSelectedFile = useCallback((file: File | null) => {
    setState((s) => ({ ...s, selectedFile: file }));
  }, []);

  const setCurrentImage = useCallback(
    (img: HTMLImageElement | null, url: string | null) => {
      setState((s) => ({ ...s, currentImage: img, currentImageUrl: url }));
    },
    []
  );

  const setSessionId = useCallback((id: string | null) => {
    setState((s) => ({ ...s, sessionId: id }));
  }, []);

  // Analysis
  const setEntities = useCallback((entities: Entity[]) => {
    setState((s) => ({ ...s, entities }));
  }, []);

  const selectEntity = useCallback((id: string | null) => {
    setState((s) => ({ ...s, selectedEntityId: id }));
  }, []);

  const setIsAnalyzing = useCallback((isAnalyzing: boolean) => {
    setState((s) => ({ ...s, isAnalyzing }));
  }, []);

  const addLog = useCallback((type: LogEntry['type'], message: string) => {
    const time = new Date()
      .toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
      .split(':')
      .slice(1)
      .join(':');

    const entry: LogEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      type,
      message,
      time,
    };

    setState((s) => ({ ...s, analysisLogs: [...s.analysisLogs, entry] }));
  }, []);

  const clearLogs = useCallback(() => {
    setState((s) => ({ ...s, analysisLogs: [] }));
  }, []);

  const setAnalysisTime = useCallback((analysisTime: number | null) => {
    setState((s) => ({ ...s, analysisTime }));
  }, []);

  // UI
  const setActiveTab = useCallback((activeTab: SidebarTab) => {
    setState((s) => ({ ...s, activeTab }));
  }, []);

  const setPipelineReady = useCallback((isPipelineReady: boolean) => {
    setState((s) => ({ ...s, isPipelineReady }));
  }, []);

  // Utilities
  const reset = useCallback(() => {
    setState((s) => ({
      ...initialState,
      isPipelineReady: s.isPipelineReady,
      activeTab: s.activeTab,
    }));
  }, []);

  const getSelectedEntity = useCallback(() => {
    return state.entities.find((e) => e.object_id === state.selectedEntityId);
  }, [state.entities, state.selectedEntityId]);

  const value: AppContextValue = {
    ...state,
    setSelectedFile,
    setCurrentImage,
    setSessionId,
    setEntities,
    selectEntity,
    setIsAnalyzing,
    addLog,
    clearLogs,
    setAnalysisTime,
    setActiveTab,
    setPipelineReady,
    reset,
    getSelectedEntity,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

export default AppContext;

