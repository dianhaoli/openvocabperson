import { AppProvider, useApp } from './context/AppContext';
import { Header, Sidebar, MainLayout } from './components/layout';
import { UploadTab, SearchTab, HistoryTab } from './components/sidebar';
import { CanvasSection } from './components/canvas';
import { ResultsGrid } from './components/results';
import { EntityPanel } from './components/entity';
import { useHealth } from './hooks';

function AppContent() {
  const { activeTab, selectedEntityId } = useApp();
  
  // Initialize health check to monitor pipeline readiness
  useHealth();

  return (
    <div className="min-h-screen bg-bg-primary">
      <Header />
      <MainLayout
        sidebar={
          <Sidebar>
            {activeTab === 'upload' && <UploadTab />}
            {activeTab === 'search' && <SearchTab />}
            {activeTab === 'history' && <HistoryTab />}
          </Sidebar>
        }
        canvas={<CanvasSection />}
        results={<ResultsGrid />}
        entityPanel={<EntityPanel />}
        showEntityPanel={selectedEntityId !== null}
      />
    </div>
  );
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
