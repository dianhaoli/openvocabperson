import { useHealth } from '../../hooks';
import { cn } from '../../utils/cn';

export function Header() {
  const { isPipelineReady } = useHealth();

  return (
    <header className="bg-bg-secondary border-b border-border px-8 py-4 sticky top-0 z-50 backdrop-blur-xl">
      <div className="max-w-[1800px] mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 gradient-accent rounded-[10px] flex items-center justify-center text-xl">
          </div>
          <h1 className="text-xl font-semibold text-gradient">
            Human Analysis System
          </h1>
        </div>

        <div className="flex items-center gap-2 px-4 py-2 bg-bg-tertiary border border-border rounded-full text-sm text-text-secondary">
          <div
            className={cn(
              'w-2 h-2 rounded-full',
              isPipelineReady
                ? 'bg-success shadow-[0_0_8px_theme(colors.success.DEFAULT)]'
                : 'bg-text-muted'
            )}
          />
          <span>
            {isPipelineReady ? 'Pipeline Ready' : 'Loading Models...'}
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;

