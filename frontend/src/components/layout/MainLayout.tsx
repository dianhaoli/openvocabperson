import type { ReactNode } from 'react';

interface MainLayoutProps {
  sidebar: ReactNode;
  canvas: ReactNode;
  results: ReactNode;
  entityPanel: ReactNode;
  showEntityPanel: boolean;
}

export function MainLayout({
  sidebar,
  canvas,
  results,
  entityPanel,
  showEntityPanel,
}: MainLayoutProps) {
  return (
    <main className="flex h-[calc(100vh-70px)]">
      {/* Left Sidebar */}
      {sidebar}

      {/* Center - Canvas + Results */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Canvas Area */}
        {canvas}

        {/* Results Grid */}
        {results}
      </div>

      {/* Right Panel - Entity Details */}
      {showEntityPanel && entityPanel}
    </main>
  );
}

export default MainLayout;

