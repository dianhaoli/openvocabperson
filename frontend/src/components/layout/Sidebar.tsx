import type { ReactNode } from 'react';
import { useActions, useAppState } from '../../context/hooks';
import type { TabId } from '../../context/state';
import { cn } from '../../utils/cn';
import { AgentTab } from '../agent/AgentTab';
import { AnalyzeTab } from '../analyze/AnalyzeTab';
import { LibraryTab } from '../library/LibraryTab';
import { PeopleTab } from '../people/PeopleTab';
import { SearchTab } from '../search/SearchTab';

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: 'analyze', label: 'Analyze', icon: '⬆' },
  { id: 'agent', label: 'Agent', icon: '✦' },
  { id: 'search', label: 'Search', icon: '⌕' },
  { id: 'library', label: 'Library', icon: '▦' },
  { id: 'people', label: 'People', icon: '👤' },
];

export function Sidebar() {
  const { tab } = useAppState();
  const { setTab } = useActions();

  // All tabs stay mounted (hidden when inactive) so running jobs, chats and
  // form state survive tab switches.
  const panel = (id: TabId, scroll: boolean, node: ReactNode) => (
    <div className={cn('flex-1 min-h-0', tab === id ? (scroll ? 'overflow-y-auto' : 'flex flex-col') : 'hidden')}>{node}</div>
  );

  return (
    <aside className="w-[360px] xl:w-[420px] shrink-0 border-r border-border bg-bg-secondary flex flex-col min-h-0">
      <nav className="flex border-b border-border shrink-0">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={cn(
              'flex-1 flex flex-col items-center gap-0.5 py-2 text-[11px] border-b-2 transition-colors',
              tab === t.id ? 'border-accent text-text-primary bg-bg-tertiary/40' : 'border-transparent text-text-muted hover:text-text-secondary',
            )}
          >
            <span className="text-sm leading-none">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </nav>
      {panel('analyze', true, <AnalyzeTab />)}
      {panel('agent', false, <AgentTab />)}
      {panel('search', true, <SearchTab />)}
      {panel('library', true, <LibraryTab />)}
      {panel('people', true, <PeopleTab />)}
    </aside>
  );
}
