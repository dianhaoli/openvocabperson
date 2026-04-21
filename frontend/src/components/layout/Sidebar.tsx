import type { ReactNode } from 'react';
import { useApp } from '../../context/AppContext';
import type { SidebarTab } from '../../types';
import { cn } from '../../utils/cn';

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const { activeTab, setActiveTab } = useApp();

  const tabs: { id: SidebarTab; label: string }[] = [
    { id: 'upload', label: 'Upload' },
    { id: 'search', label: 'Search' },
    { id: 'history', label: 'History' },
    { id: 'persons', label: 'Persons' },
  ];

  return (
    <aside className="w-[340px] bg-bg-secondary border-r border-border flex flex-col overflow-hidden">
      {/* Tab buttons */}
      <div className="flex border-b border-border bg-bg-tertiary">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'flex-1 px-2 py-3 text-xs sm:text-sm font-medium transition-all duration-200',
              'flex items-center justify-center gap-1',
              activeTab === tab.id
                ? 'text-accent bg-bg-secondary border-b-2 border-accent'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-secondary'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 p-6 flex flex-col gap-4 overflow-y-auto">
        {children}
      </div>
    </aside>
  );
}

export default Sidebar;

