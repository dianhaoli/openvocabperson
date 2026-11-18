import { useEffect } from 'react';
import { DetailPanel } from './components/detail/DetailPanel';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { ModelsModal } from './components/models/ModelsModal';
import { PersonDetailModal } from './components/people/PersonDetailModal';
import { Toasts } from './components/ui/Toasts';
import { Workspace } from './components/workspace/Workspace';
import { AppProvider } from './context/AppProvider';
import { useActions, useAppState } from './context/hooks';

function Shell() {
  const { selectedEntityId, personModalId, modelsOpen } = useAppState();
  const { selectEntity } = useActions();

  // Escape clears the entity selection when no modal is open.
  useEffect(() => {
    if (!selectedEntityId || personModalId || modelsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (e.key === 'Escape' && !(target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) && !document.querySelector('[role="dialog"]')) {
        selectEntity(null);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [selectedEntityId, personModalId, modelsOpen, selectEntity]);

  return (
    <div className="h-screen min-w-[1024px] flex flex-col bg-bg-primary overflow-hidden">
      <Header />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 min-h-0">
          <Workspace />
        </main>
        <DetailPanel />
      </div>
      <ModelsModal />
      <PersonDetailModal />
      <Toasts />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
